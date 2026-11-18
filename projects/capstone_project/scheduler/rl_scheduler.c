#include "rl_scheduler.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ─────────────────────────────────────────────
   AIROS RL Scheduler — Q-Learning Engine
   
   State space: 3 features per task, each binned
     s0 = urgency      (time_to_deadline / deadline_ms)
     s1 = miss_rate    (misses / total_executions)
     s2 = burst_ratio  (avg_burst / worst_case_et)
   
   Action space: assign priority 0–7

   Reward (measured over the epoch the action was in force):
     (hits − misses) / (hits + misses + 1)   → normalised to [−1, +1]
     −0.5  if starved (task waiting > 2× its own period)

   Update rule (Q-Learning / Bellman):
     Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') − Q(s,a)]
     where s' is the state observed at the NEXT epoch, i.e. a state that
     actually resulted from taking a in s.

   Policy: ε-greedy with random tie-breaking, warm-started from the
   static rate-monotonic priority assignment.
   ───────────────────────────────────────────── */

static uint32_t rand_state = 12345;

static uint32_t _rand(void) {
    rand_state ^= rand_state << 13;
    rand_state ^= rand_state >> 17;
    rand_state ^= rand_state << 5;
    return rand_state;
}

static float _randf(void) { return (float)(_rand() & 0xFFFF) / 65536.0f; }

/* ── State Discretization ──
   Clamp feature [0,1] into one of RL_STATE_BINS integer bins.
*/
static int _bin(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;
    int b = (int)(value * RL_STATE_BINS);
    if (b >= RL_STATE_BINS) b = RL_STATE_BINS - 1;
    return b;
}

/* ── Feature Extraction ── */
static void _extract_state(TCB* t, uint32_t tick, int* s0, int* s1, int* s2) {
    /* Urgency: how close to deadline (0=far, 1=at deadline) */
    float time_remaining = 0.0f;
    if (tick < t->absolute_deadline)
        time_remaining = (float)(t->absolute_deadline - tick);
    float urgency = (t->deadline_ms > 0)
        ? 1.0f - (time_remaining / (float)t->deadline_ms)
        : 1.0f;
    *s0 = _bin(urgency);

    /* Miss rate */
    uint32_t total = t->deadline_hits + t->deadline_misses;
    float miss_rate = (total > 0) ? (float)t->deadline_misses / (float)total : 0.0f;
    *s1 = _bin(miss_rate);

    /* Burst ratio */
    float burst_ratio = (t->worst_case_et > 0)
        ? (float)t->avg_burst / (float)t->worst_case_et
        : 0.5f;
    *s2 = _bin(burst_ratio);
}

/* ── Init ── */
void rl_scheduler_init(RLScheduler* rl) {
    memset(rl->q_table, 0, sizeof(rl->q_table));
    rl->epsilon          = RL_EPSILON;
    rl->alpha            = RL_ALPHA;
    rl->gamma            = RL_GAMMA;
    rl->epoch_count      = 0;
    rl->total_updates    = 0;
    rl->cumulative_reward = 0.0f;
    memset(rl->last_state,  0, sizeof(rl->last_state));
    memset(rl->last_action, 0, sizeof(rl->last_action));
    memset(rl->has_last,    0, sizeof(rl->has_last));
    memset(rl->last_hits,   0, sizeof(rl->last_hits));
    memset(rl->last_misses, 0, sizeof(rl->last_misses));

    /* Optimistic initialization — encourage exploration */
    for (int t = 0; t < AIROS_MAX_TASKS; t++)
        for (int s0 = 0; s0 < RL_STATE_BINS; s0++)
            for (int s1 = 0; s1 < RL_STATE_BINS; s1++)
                for (int s2 = 0; s2 < RL_STATE_BINS; s2++)
                    for (int a = 0; a < RL_ACTIONS; a++)
                        rl->q_table[t][s0][s1][s2][a] = 0.5f;
}

/* ── Warm start from the static priority assignment ──
   A uniform Q-table means every argmax is a tie, and a first-index tie
   break hands every task priority 0 — at which point the selector always
   runs task 0 and everything else starves. Seeding each task's row so
   that Q is highest at its rate-monotonic base priority and falls off
   linearly with distance means the greedy policy STARTS as fixed-priority
   scheduling and learns deviations from it, rather than learning a
   scheduler from scratch in 500 epochs. This is standard practice for
   RL in systems: bootstrap from a known-good controller. */
void rl_scheduler_warmstart(RLScheduler* rl, KernelCB* kernel) {
    for (int i = 0; i < kernel->task_count; i++) {
        int base = kernel->task_list[i]->base_priority;
        if (base >= RL_ACTIONS) base = RL_ACTIONS - 1;

        for (int s0 = 0; s0 < RL_STATE_BINS; s0++)
            for (int s1 = 0; s1 < RL_STATE_BINS; s1++)
                for (int s2 = 0; s2 < RL_STATE_BINS; s2++)
                    for (int a = 0; a < RL_ACTIONS; a++) {
                        int d = a - base;
                        if (d < 0) d = -d;
                        /* 1.0 at the base priority → 0.5 at the far end */
                        rl->q_table[i][s0][s1][s2][a] =
                            1.0f - 0.5f * ((float)d / (float)(RL_ACTIONS - 1));
                    }
    }
}

/* ── Q-Learning Update (Bellman equation) ── */
static void _q_update(RLScheduler* rl, int task_id,
                       int s0, int s1, int s2, int action,
                       float reward,
                       int ns0, int ns1, int ns2) {
    /* Find max Q(s', a') */
    float max_next = -1e9f;
    for (int a = 0; a < RL_ACTIONS; a++) {
        float q = rl->q_table[task_id][ns0][ns1][ns2][a];
        if (q > max_next) max_next = q;
    }

    float* q_cur = &rl->q_table[task_id][s0][s1][s2][action];
    *q_cur += rl->alpha * (reward + rl->gamma * max_next - *q_cur);
    rl->total_updates++;
    rl->cumulative_reward += reward;
}

/* ── Compute Reward for Task ──
   Measured over the epoch that just elapsed, not from a single instant.
   t->reward_signal only ever held the LAST hit/miss written to it, so a
   task that missed forty deadlines and then met one looked identical to a
   task that met one. Differencing the kernel's own hit/miss counters
   against the previous epoch's snapshot credits the action with
   everything that actually happened while it was in force. */
static float _compute_reward(RLScheduler* rl, int idx, TCB* t, KernelCB* kernel) {
    uint32_t dh = t->deadline_hits   - rl->last_hits[idx];
    uint32_t dm = t->deadline_misses - rl->last_misses[idx];
    rl->last_hits[idx]   = t->deadline_hits;
    rl->last_misses[idx] = t->deadline_misses;
    t->reward_signal     = 0.0f;   /* consume any instantaneous signal */

    /* Normalised deadline performance over the interval, in [-1, +1] */
    float reward = ((float)dh - (float)dm) / ((float)(dh + dm) + 1.0f);

    /* Starvation penalty: a task held off for more than two of its own
       periods is being harmed regardless of its deadline arithmetic.
       NOTE: release_tick can sit in the FUTURE (a blocked task's next
       release), so the subtraction must be guarded — as unsigned it
       wrapped to ~4e9 and fired this penalty on essentially every task,
       every epoch, which made the starvation signal pure noise. */
    uint32_t wait = (kernel->tick_count > t->release_tick)
                  ? (kernel->tick_count - t->release_tick) : 0u;
    if (t->period_ms > 0 && wait > 2 * t->period_ms)
        reward -= 0.5f;

    return reward;
}

/* ── Main Update Pass (called every AIROS_SCHED_EPOCH ticks) ──
   Each task's Q-table is updated once per epoch using the transition
   that actually occurred between the *previous* epoch's decision and
   now: (last_state[i], last_action[i]) -> reward accrued since then ->
   current_state. That current_state then becomes the state we choose
   the next action from, and is stashed as last_state[i] for the epoch
   after this one. This is what makes it a real one-step TD update
   instead of bootstrapping a state off of itself. */
void rl_scheduler_update(RLScheduler* rl, KernelCB* kernel) {
    rl->epoch_count++;

    for (int i = 0; i < kernel->task_count; i++) {
        TCB* t = kernel->task_list[i];
        if (t->state == TASK_DEAD) continue;

        int cs0, cs1, cs2;
        _extract_state(t, kernel->tick_count, &cs0, &cs1, &cs2);

        float reward = _compute_reward(rl, i, t, kernel);

        /* Update using the PRIOR epoch's (state, action) -> this epoch's
           state as s'. Skipped on a task's first epoch, since there is
           no prior transition yet to credit. */
        if (rl->has_last[i]) {
            _q_update(rl, i,
                      rl->last_state[i][0], rl->last_state[i][1], rl->last_state[i][2],
                      rl->last_action[i],
                      reward,
                      cs0, cs1, cs2);
        }

        /* ── ε-greedy action selection → new priority, chosen from the
           CURRENT state (cs), which is what the update above just
           treated as s' ── */
        int new_priority;
        if (_randf() < rl->epsilon) {
            /* Explore: random priority */
            new_priority = _rand() % RL_ACTIONS;
        } else {
            /* Exploit: best Q-value action, with RANDOM tie-breaking.
               Deterministic first-index tie breaking is what collapsed the
               old policy: untouched cells all hold the same value, argmax
               always returned action 0, every task was handed priority 0,
               and the selector then ran whichever task sat first in the
               list while the rest starved. Reservoir-sampling among ties
               keeps the policy spread out until the Q-values genuinely
               separate. */
            float best_q  = -1e9f;
            int   n_ties  = 0;
            new_priority  = t->base_priority;
            for (int a = 0; a < RL_ACTIONS; a++) {
                float q = rl->q_table[i][cs0][cs1][cs2][a];
                float d = q - best_q;
                if (d > 1e-6f) {                 /* strictly better */
                    best_q = q; new_priority = a; n_ties = 1;
                } else if (d > -1e-6f) {         /* tied */
                    n_ties++;
                    if ((_rand() % (uint32_t)n_ties) == 0) new_priority = a;
                }
            }
        }
        t->priority = (uint8_t)new_priority;
        t->q_value  = rl->q_table[i][cs0][cs1][cs2][new_priority];

        /* Stash this epoch's state/action as "last" for next epoch's update */
        rl->last_state[i][0] = cs0;
        rl->last_state[i][1] = cs1;
        rl->last_state[i][2] = cs2;
        rl->last_action[i]   = new_priority;
        rl->has_last[i]      = 1;
    }

    /* Decay epsilon (reduce exploration over time → converge) */
    if (rl->epsilon > 0.01f)
        rl->epsilon *= 0.995f;
}

/* ── Task Selection ── */
TCB* rl_scheduler_select(RLScheduler* rl, KernelCB* kernel) {
    TCB* chosen   = NULL;
    uint8_t best  = 255;
    for (int i = 0; i < kernel->task_count; i++) {
        TCB* t = kernel->task_list[i];
        if (t->state == TASK_READY && t->priority < best) {
            best   = t->priority;
            chosen = t;
        }
    }
    return chosen;
}

/* ── Diagnostics ── */
float rl_get_avg_reward(RLScheduler* rl) {
    if (rl->total_updates == 0) return 0.0f;
    return rl->cumulative_reward / (float)rl->total_updates;
}
