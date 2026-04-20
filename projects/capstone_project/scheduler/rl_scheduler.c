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
   
   Reward:
     +1.0  deadline met
     -1.0  deadline missed
     +0.3  bonus if CPU utilization < 80% (efficiency)
     -0.5  penalty if starvation detected (task waiting > 2× period)
   
   Update rule (Q-Learning / Bellman):
     Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') − Q(s,a)]
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

    /* Optimistic initialization — encourage exploration */
    for (int t = 0; t < AIROS_MAX_TASKS; t++)
        for (int s0 = 0; s0 < RL_STATE_BINS; s0++)
            for (int s1 = 0; s1 < RL_STATE_BINS; s1++)
                for (int s2 = 0; s2 < RL_STATE_BINS; s2++)
                    for (int a = 0; a < RL_ACTIONS; a++)
                        rl->q_table[t][s0][s1][s2][a] = 0.5f;
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

/* ── Compute Reward for Task ── */
static float _compute_reward(TCB* t, KernelCB* kernel) {
    float reward = t->reward_signal;   /* +1 hit / -1 miss set by kernel */
    t->reward_signal = 0.0f;           /* consume the signal */

    /* Efficiency bonus */
    float util = (kernel->task_count > 0)
        ? (float)kernel->context_switch_count / (float)(kernel->tick_count + 1)
        : 0.0f;
    if (util < 0.8f) reward += 0.3f;

    /* Starvation penalty */
    uint32_t wait = kernel->tick_count - t->release_tick;
    if (t->period_ms > 0 && wait > 2 * t->period_ms)
        reward -= 0.5f;

    return reward;
}

/* ── Main Update Pass (called every AIROS_SCHED_EPOCH ticks) ── */
void rl_scheduler_update(RLScheduler* rl, KernelCB* kernel) {
    rl->epoch_count++;

    for (int i = 0; i < kernel->task_count; i++) {
        TCB* t = kernel->task_list[i];
        if (t->state == TASK_DEAD) continue;

        int s0, s1, s2;
        _extract_state(t, kernel->tick_count, &s0, &s1, &s2);

        float reward = _compute_reward(t, kernel);

        /* Current action = current priority */
        int action = t->priority;
        if (action >= RL_ACTIONS) action = RL_ACTIONS - 1;

        /* Next state (same task, post-reward state) */
        int ns0, ns1, ns2;
        _extract_state(t, kernel->tick_count, &ns0, &ns1, &ns2);

        _q_update(rl, i, s0, s1, s2, action, reward, ns0, ns1, ns2);

        /* ── ε-greedy action selection → new priority ── */
        int new_priority;
        if (_randf() < rl->epsilon) {
            /* Explore: random priority */
            new_priority = _rand() % RL_ACTIONS;
        } else {
            /* Exploit: best Q-value action */
            float best_q = -1e9f;
            new_priority = t->base_priority;
            for (int a = 0; a < RL_ACTIONS; a++) {
                float q = rl->q_table[i][ns0][ns1][ns2][a];
                if (q > best_q) { best_q = q; new_priority = a; }
            }
        }
        t->priority = (uint8_t)new_priority;
        t->q_value  = rl->q_table[i][ns0][ns1][ns2][new_priority];
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
