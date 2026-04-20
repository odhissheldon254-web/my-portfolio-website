/*  ─────────────────────────────────────────────────────────────────────────
    AIROS — Simulation Harness & Benchmark Engine
    
    Simulates 3 scheduler modes over identical workloads:
      MODE 0 — Fixed Priority (baseline)
      MODE 1 — Round Robin
      MODE 2 — AIROS RL-Adaptive (our system)
    
    Metrics collected per run:
      • Total deadline misses
      • Total deadline hits
      • CPU utilisation
      • Context switch overhead
      • Scheduler convergence (RL only: ticks to stable miss rate < 5%)
      • Average Q-value reward signal
    
    Output: CSV file for Python dashboard + human-readable summary
    ─────────────────────────────────────────────────────────────────────── */

#include "../kernel/airos_kernel.h"
#include "../scheduler/rl_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIM_TICKS        5000
#define NUM_TASKS        8

/* Task workload profile */
typedef struct {
    const char* name;
    uint8_t     base_priority;
    uint32_t    period_ms;
    uint32_t    deadline_ms;
    uint32_t    wcet;          /* worst-case execution time (ticks) */
    uint32_t    bcet;          /* best-case execution time  (ticks) */
} TaskProfile;

static TaskProfile profiles[NUM_TASKS] = {
    { "CONTROL_LOOP",   0, 10,  10,  4,  2 },   /* hard RT, highest priority */
    { "SENSOR_READ",    1, 20,  20,  5,  3 },   /* hard RT                   */
    { "COMM_TX",        2, 50,  50,  8,  4 },   /* soft RT                   */
    { "DATA_LOGGER",    3, 100, 100, 15, 8 },   /* background                */
    { "FAULT_MONITOR",  1, 15,  15,  3,  2 },   /* hard RT, safety-critical  */
    { "UI_UPDATE",      4, 200, 200, 20, 10 },  /* low priority              */
    { "SELF_TEST",      5, 500, 500, 30, 15 },  /* maintenance               */
    { "IDLE",           7, 1,   1,   1,  1  },  /* idle task                 */
};

/* Per-tick snapshot for CSV export */
typedef struct {
    uint32_t tick;
    uint32_t deadline_misses;
    uint32_t deadline_hits;
    uint32_t context_switches;
    float    rl_avg_reward;
    uint8_t  task_priorities[NUM_TASKS];
} Snapshot;

#define MAX_SNAPSHOTS (SIM_TICKS / 10)
static Snapshot snapshots[3][MAX_SNAPSHOTS];
static int      snap_count[3];

/* Benchmark results per scheduler */
typedef struct {
    const char* name;
    uint32_t    total_misses;
    uint32_t    total_hits;
    uint32_t    context_switches;
    float       miss_rate;
    float       avg_reward;
    uint32_t    convergence_tick;  /* RL only */
} BenchResult;

/* ── Pseudo-random execution time between bcet and wcet ── */
static uint32_t rand_state_sim = 98765;
static uint32_t sim_rand(void) {
    rand_state_sim ^= rand_state_sim << 13;
    rand_state_sim ^= rand_state_sim >> 17;
    rand_state_sim ^= rand_state_sim << 5;
    return rand_state_sim;
}
static uint32_t rand_et(uint32_t bcet, uint32_t wcet) {
    if (wcet <= bcet) return bcet;
    return bcet + (sim_rand() % (wcet - bcet + 1));
}

/* ── Run a full simulation for one scheduler mode ── */
static BenchResult run_simulation(int mode, const char* mode_name) {
    airos_init();
    KernelCB* k = airos_get_kernel();
    k->scheduler_mode = (mode == 2) ? 1 : 0;

    RLScheduler rl;
    if (mode == 2) rl_scheduler_init(&rl);

    /* Create tasks */
    TCB* tasks[NUM_TASKS];
    for (int i = 0; i < NUM_TASKS; i++) {
        tasks[i] = airos_task_create(
            profiles[i].name, NULL, NULL,
            profiles[i].base_priority,
            profiles[i].period_ms,
            profiles[i].deadline_ms
        );
        tasks[i]->worst_case_et = profiles[i].wcet;
    }

    snap_count[mode] = 0;
    uint32_t convergence_tick = 0;
    int      converged = 0;

    /* ── Main simulation loop ── */
    for (uint32_t tick = 0; tick < SIM_TICKS; tick++) {
        airos_tick();

        /* Select task based on scheduler mode */
        TCB* running = NULL;
        if (mode == 0) {
            /* Fixed priority */
            running = NULL;
            uint8_t best = 255;
            for (int i = 0; i < k->task_count; i++) {
                if (k->task_list[i]->state == TASK_READY &&
                    k->task_list[i]->priority < best) {
                    best    = k->task_list[i]->priority;
                    running = k->task_list[i];
                }
            }
        } else if (mode == 1) {
            /* Round Robin */
            static int rr_idx = 0;
            for (int i = 0; i < k->task_count; i++) {
                int idx = (rr_idx + i) % k->task_count;
                if (k->task_list[idx]->state == TASK_READY) {
                    running = k->task_list[idx];
                    rr_idx  = (idx + 1) % k->task_count;
                    break;
                }
            }
        } else {
            /* RL adaptive */
            if (tick % AIROS_SCHED_EPOCH == 0 && tick > 0)
                rl_scheduler_update(&rl, k);
            running = rl_scheduler_select(&rl, k);
        }

        /* Simulate execution */
        if (running) {
            running->state   = TASK_RUNNING;
            uint32_t et      = rand_et(profiles[running->id].bcet,
                                       profiles[running->id].wcet);
            running->last_burst = et;
            running->avg_burst  = (running->avg_burst == 0)
                ? et : (running->avg_burst * 7 + et * 3) / 10;

            /* Check deadline */
            if (tick + et <= running->absolute_deadline) {
                running->deadline_hits++;
                k->total_deadline_hits++;
                running->reward_signal = 1.0f;
            } else {
                running->deadline_misses++;
                k->total_deadline_misses++;
                running->reward_signal = -1.0f;
            }
            running->exec_count++;
            running->state       = TASK_BLOCKED;
            running->release_tick = tick + running->period_ms;
            running->absolute_deadline = running->release_tick + running->deadline_ms;
            k->context_switch_count++;
        }

        /* Snapshot every 10 ticks */
        if (tick % 10 == 0 && snap_count[mode] < MAX_SNAPSHOTS) {
            Snapshot* s = &snapshots[mode][snap_count[mode]++];
            s->tick              = tick;
            s->deadline_misses   = k->total_deadline_misses;
            s->deadline_hits     = k->total_deadline_hits;
            s->context_switches  = k->context_switch_count;
            s->rl_avg_reward     = (mode == 2) ? rl_get_avg_reward(&rl) : 0.0f;
            for (int i = 0; i < NUM_TASKS; i++)
                s->task_priorities[i] = k->task_list[i]->priority;
        }

        /* Convergence detection (RL only) */
        if (mode == 2 && !converged && tick > 500) {
            uint32_t total = k->total_deadline_hits + k->total_deadline_misses;
            float mr = (total > 0) ? (float)k->total_deadline_misses / total : 1.0f;
            if (mr < 0.05f) {
                convergence_tick = tick;
                converged = 1;
            }
        }
    }

    uint32_t total = k->total_deadline_hits + k->total_deadline_misses;
    BenchResult r;
    r.name             = mode_name;
    r.total_misses     = k->total_deadline_misses;
    r.total_hits       = k->total_deadline_hits;
    r.context_switches = k->context_switch_count;
    r.miss_rate        = (total > 0) ? (float)r.total_misses / (float)total * 100.0f : 0.0f;
    r.avg_reward       = (mode == 2) ? rl_get_avg_reward(&rl) : 0.0f;
    r.convergence_tick = convergence_tick;

    /* Free tasks */
    for (int i = 0; i < k->task_count; i++) free(k->task_list[i]);

    return r;
}

/* ── Export CSV for Python Dashboard ── */
static void export_csv(BenchResult results[3]) {
    FILE* f = fopen("benchmark_results.csv", "w");
    if (!f) return;

    fprintf(f, "tick,fp_misses,fp_hits,rr_misses,rr_hits,rl_misses,rl_hits,"
               "fp_switches,rr_switches,rl_switches,rl_reward,"
               "rl_p0,rl_p1,rl_p2,rl_p3,rl_p4,rl_p5,rl_p6,rl_p7\n");

    int n = snap_count[0];
    for (int i = 0; i < n; i++) {
        Snapshot* fp = &snapshots[0][i];
        Snapshot* rr = &snapshots[1][i];
        Snapshot* rl = &snapshots[2][i];
        fprintf(f, "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%.4f,"
                   "%u,%u,%u,%u,%u,%u,%u,%u\n",
            fp->tick,
            fp->deadline_misses, fp->deadline_hits,
            rr->deadline_misses, rr->deadline_hits,
            rl->deadline_misses, rl->deadline_hits,
            fp->context_switches, rr->context_switches, rl->context_switches,
            rl->rl_avg_reward,
            rl->task_priorities[0], rl->task_priorities[1],
            rl->task_priorities[2], rl->task_priorities[3],
            rl->task_priorities[4], rl->task_priorities[5],
            rl->task_priorities[6], rl->task_priorities[7]);
    }
    fclose(f);
    printf("[AIROS] Benchmark CSV exported: benchmark_results.csv\n");
}

/* ── Main ── */
int main(void) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   AIROS — Adaptive Intelligent RTOS Benchmark    ║\n");
    printf("║   Simulation: %d ticks | %d tasks               ║\n", SIM_TICKS, NUM_TASKS);
    printf("╚══════════════════════════════════════════════════╝\n\n");

    BenchResult results[3];
    printf("[SIM] Running Fixed Priority scheduler...\n");
    results[0] = run_simulation(0, "Fixed Priority");
    printf("[SIM] Running Round Robin scheduler...\n");
    results[1] = run_simulation(1, "Round Robin");
    printf("[SIM] Running AIROS RL-Adaptive scheduler...\n");
    results[2] = run_simulation(2, "AIROS RL-Adaptive");

    printf("\n┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-20s %8s %8s %10s %10s %10s │\n",
           "Scheduler", "Misses", "Hits", "Miss%", "Switches", "RL Reward");
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    for (int i = 0; i < 3; i++) {
        printf("│ %-20s %8u %8u %9.2f%% %10u %10.4f │\n",
               results[i].name,
               results[i].total_misses,
               results[i].total_hits,
               results[i].miss_rate,
               results[i].context_switches,
               results[i].avg_reward);
    }
    printf("└─────────────────────────────────────────────────────────────────┘\n");

    if (results[2].convergence_tick > 0)
        printf("\n[RL] Scheduler converged at tick: %u (miss rate < 5%%)\n",
               results[2].convergence_tick);

    float improvement = results[0].miss_rate - results[2].miss_rate;
    printf("[RL] Deadline miss reduction vs Fixed Priority: %.2f%%\n\n", improvement);

    export_csv(results);
    return 0;
}
