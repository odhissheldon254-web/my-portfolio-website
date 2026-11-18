#ifndef AIROS_RL_SCHEDULER_H
#define AIROS_RL_SCHEDULER_H

/* ─────────────────────────────────────────────
   AIROS — RL Scheduler Header
   Q-Learning Agent for Dynamic Priority Control
   ───────────────────────────────────────────── */

#include "../kernel/airos_kernel.h"

/* Hyperparameters */
#define RL_ALPHA        0.1f    /* learning rate                  */
#define RL_GAMMA        0.9f    /* discount factor                */
#define RL_EPSILON      0.15f   /* exploration rate (ε-greedy)    */
#define RL_STATE_BINS   4       /* discretization bins per feature */
#define RL_ACTIONS      AIROS_PRIORITY_LEVELS
#define RL_STATE_DIM    3       /* features: urgency, miss_rate, burst_ratio */
#define RL_TABLE_SIZE   (RL_STATE_BINS * RL_STATE_BINS * RL_STATE_BINS * RL_ACTIONS)

typedef struct {
    float   q_table[AIROS_MAX_TASKS][RL_STATE_BINS][RL_STATE_BINS][RL_STATE_BINS][RL_ACTIONS];
    float   epsilon;
    float   alpha;
    float   gamma;
    uint32_t epoch_count;
    uint32_t total_updates;
    float   cumulative_reward;

    /* Previous epoch's (state, action) per task — needed so the Bellman
       update at epoch N+1 can use a state that actually followed from
       taking last_action[i] in last_state[i], instead of reusing the
       same tick's state as both s and s'. */
    int      last_state[AIROS_MAX_TASKS][RL_STATE_DIM];
    int      last_action[AIROS_MAX_TASKS];
    uint8_t  has_last[AIROS_MAX_TASKS];

    /* Deadline counters snapshotted at the previous epoch, so reward is
       computed from what actually happened DURING the transition rather
       than from a single stale reward_signal value. */
    uint32_t last_hits[AIROS_MAX_TASKS];
    uint32_t last_misses[AIROS_MAX_TASKS];
} RLScheduler;

void    rl_scheduler_init(RLScheduler* rl);

/* Seed the Q-table from the static priority assignment already known to
   work (rate-monotonic / fixed priority). Must be called AFTER tasks are
   created. Without it the agent starts from a uniform table, every argmax
   ties on action 0, and the policy degenerates to "give everything
   priority 0" — which starves every task but the first. */
void    rl_scheduler_warmstart(RLScheduler* rl, KernelCB* kernel);

void    rl_scheduler_update(RLScheduler* rl, KernelCB* kernel);
TCB*    rl_scheduler_select(RLScheduler* rl, KernelCB* kernel);
float   rl_get_avg_reward(RLScheduler* rl);

#endif
