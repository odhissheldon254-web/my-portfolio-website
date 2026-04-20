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
} RLScheduler;

void    rl_scheduler_init(RLScheduler* rl);
void    rl_scheduler_update(RLScheduler* rl, KernelCB* kernel);
TCB*    rl_scheduler_select(RLScheduler* rl, KernelCB* kernel);
float   rl_get_avg_reward(RLScheduler* rl);

#endif
