/*  ─────────────────────────────────────────────────────────────────────────
    AIROS — Simulation Harness & Benchmark Engine  (v2: preemptive)

    WHAT CHANGED FROM v1, AND WHY IT MATTERS
    ----------------------------------------
    v1 ran a task for exactly one tick regardless of the execution time it
    drew, so `et` fed only the hit/miss arithmetic and never occupied the
    processor. Nothing was ever preempted, no two tasks ever competed for
    the CPU, and the workload's utilisation was irrelevant — which is why
    Fixed Priority and Round Robin returned byte-identical results across
    every random seed. The harness measured a queueing policy, not a
    scheduler.

    v2 is a proper preemptive tick-driven simulation:
      • a released job carries `remaining` ticks of work and occupies the
        CPU until that reaches zero, or until something preempts it
      • the scheduler re-decides every tick, so a higher-priority release
        preempts the running task — real preemption, not cooperative yield
      • a deadline miss is counted ONCE PER JOB, which is what real-time
        analysis means by a miss (v1 charged a miss every tick a task sat
        past its deadline, which made starvation dominate everything and
        is why Round Robin looked unbeatable)
      • IDLE is a true background task: always runnable, no deadline,
        never scored — rather than a periodic task claiming 100% CPU
      • utilisation is computed and reported, because it determines
        whether the workload is feasible at all

    TWO WORKLOAD REGIMES
    --------------------
      OVERLOAD — the originally documented task set. U_avg = 1.02, so
        demand exceeds the processor and misses are mathematically forced.
        The question is not whether deadlines are missed but WHICH ones.
      FEASIBLE — the same task shape scaled under the Liu-Layland bound,
        where a correct scheduler should approach zero misses.
    ─────────────────────────────────────────────────────────────────── */

#include "../kernel/airos_kernel.h"
#include "../scheduler/rl_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SIM_TICKS
#define SIM_TICKS   20000
#endif
#define NUM_TASKS   8
#define IDLE_TASK   7

#ifndef RL_SEED
#define RL_SEED  12345u
#endif
#ifndef SIM_SEED
#define SIM_SEED 98765u
#endif

typedef struct {
    const char* name;
    uint8_t     base_priority;
    uint32_t    period;
    uint32_t    deadline;
    uint32_t    wcet;
    uint32_t    bcet;
    uint8_t     is_idle;     /* always runnable, unscored */
    uint8_t     critical;    /* hard real-time / safety */
} TaskProfile;

/* OVERLOAD: the task set as originally documented. U_avg = 1.022 */
static TaskProfile wl_overload[NUM_TASKS] = {
    { "CONTROL_LOOP",  0,  10,  10,  4,  2, 0, 1 },
    { "SENSOR_READ",   1,  20,  20,  5,  3, 0, 1 },
    { "COMM_TX",       2,  50,  50,  8,  4, 0, 0 },
    { "DATA_LOGGER",   3, 100, 100, 15,  8, 0, 0 },
    { "FAULT_MONITOR", 1,  15,  15,  3,  2, 0, 1 },
    { "UI_UPDATE",     4, 200, 200, 20, 10, 0, 0 },
    { "SELF_TEST",     5, 500, 500, 30, 15, 0, 0 },
    { "IDLE",          7,   0,   0,  1,  1, 1, 0 },
};

/* FEASIBLE: periods stretched so the set sits under the RM bound (~0.70) */
static TaskProfile wl_feasible[NUM_TASKS] = {
    { "CONTROL_LOOP",  0,  20,  20,  4,  2, 0, 1 },
    { "SENSOR_READ",   1,  40,  40,  5,  3, 0, 1 },
    { "COMM_TX",       2,  80,  80,  8,  4, 0, 0 },
    { "DATA_LOGGER",   3, 200, 200, 15,  8, 0, 0 },
    { "FAULT_MONITOR", 1,  30,  30,  3,  2, 0, 1 },
    { "UI_UPDATE",     4, 400, 400, 20, 10, 0, 0 },
    { "SELF_TEST",     5, 800, 800, 30, 15, 0, 0 },
    { "IDLE",          7,   0,   0,  1,  1, 1, 0 },
};

/* ── Per-job simulation state ── */
typedef struct {
    uint32_t remaining;      /* ticks of work left in the current job   */
    uint32_t abs_deadline;
    uint32_t next_release;
    uint8_t  active;         /* a job is pending                        */
    uint8_t  miss_counted;   /* this job already charged a miss         */
    uint32_t released, completed, missed;
    uint32_t response_sum;   /* completion - release, over completed    */
    uint32_t start_tick;
} Job;

typedef struct {
    const char* name;
    uint32_t    misses, hits, released;
    uint32_t    crit_misses;      /* misses on safety-critical tasks    */
    uint32_t    preemptions;
    uint32_t    idle_ticks;
    float       miss_rate;
    float       avg_response;
    float       avg_reward;
    uint32_t    per_task_miss[NUM_TASKS];
} BenchResult;

/* ── Deterministic PRNG for execution times ── */
static uint32_t rand_state_sim = SIM_SEED;
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

/* ── Utilisation of a workload (excluding the idle task) ── */
static void workload_utilisation(TaskProfile* wl, float* u_wcet, float* u_avg) {
    *u_wcet = 0.0f; *u_avg = 0.0f;
    for (int i = 0; i < NUM_TASKS; i++) {
        if (wl[i].is_idle || wl[i].period == 0) continue;
        *u_wcet += (float)wl[i].wcet / (float)wl[i].period;
        *u_avg  += ((float)wl[i].wcet + (float)wl[i].bcet) * 0.5f / (float)wl[i].period;
    }
}

/* ── One simulation run ──
   mode 0 = Fixed Priority, 1 = Round Robin, 2 = AIROS RL-Adaptive
   All three are preemptive: the choice is re-made every tick. */
static BenchResult run_simulation(int mode, const char* mode_name, TaskProfile* wl) {
    airos_init();
    KernelCB* k = airos_get_kernel();
    k->scheduler_mode = (mode == 2) ? 1 : 0;

    rand_state_sim = SIM_SEED;   /* identical workload for every scheduler */

    RLScheduler rl;
    if (mode == 2) rl_scheduler_init(&rl);

    TCB* tasks[NUM_TASKS];
    Job  job[NUM_TASKS];
    memset(job, 0, sizeof(job));

    for (int i = 0; i < NUM_TASKS; i++) {
        tasks[i] = airos_task_create(wl[i].name, NULL, NULL,
                                     wl[i].base_priority,
                                     wl[i].period, wl[i].deadline);
        tasks[i]->worst_case_et = wl[i].wcet;
        job[i].next_release = 0;
        job[i].active       = wl[i].is_idle ? 1 : 0;   /* idle always runnable */
        job[i].remaining    = wl[i].is_idle ? 1 : 0;
    }

    if (mode == 2) rl_scheduler_warmstart(&rl, k);

    BenchResult r;
    memset(&r, 0, sizeof(r));
    r.name = mode_name;

    int      prev_run = -1;
    int      rr_cursor = 0;
    uint32_t response_total = 0, completed_total = 0;

    for (uint32_t tick = 0; tick < SIM_TICKS; tick++) {
        k->tick_count = tick;

        /* ── 1. Release periodic jobs ── */
        for (int i = 0; i < NUM_TASKS; i++) {
            if (wl[i].is_idle || wl[i].period == 0) continue;
            if (tick >= job[i].next_release) {
                if (job[i].active && !job[i].miss_counted) {
                    /* previous job never finished before its successor
                       arrived — an overrun, charged once */
                    job[i].missed++; r.misses++; r.per_task_miss[i]++;
                    if (wl[i].critical) r.crit_misses++;
                    tasks[i]->deadline_misses++;
                }
                job[i].remaining    = rand_et(wl[i].bcet, wl[i].wcet);
                job[i].abs_deadline = tick + wl[i].deadline;
                job[i].start_tick   = tick;
                job[i].active       = 1;
                job[i].miss_counted = 0;
                job[i].released++;
                job[i].next_release = tick + wl[i].period;
                r.released++;

                tasks[i]->state             = TASK_READY;
                tasks[i]->release_tick      = tick;
                tasks[i]->absolute_deadline = job[i].abs_deadline;
            }
        }

        /* ── 2. Charge a deadline miss, once per job ── */
        for (int i = 0; i < NUM_TASKS; i++) {
            if (wl[i].is_idle) continue;
            if (job[i].active && !job[i].miss_counted && tick > job[i].abs_deadline) {
                job[i].miss_counted = 1;
                job[i].missed++; r.misses++; r.per_task_miss[i]++;
                if (wl[i].critical) r.crit_misses++;
                tasks[i]->deadline_misses++;
                k->total_deadline_misses++;
            }
        }

        /* ── 3. Let the RL agent revise priorities on its epoch ── */
        if (mode == 2 && tick > 0 && (tick % AIROS_SCHED_EPOCH) == 0)
            rl_scheduler_update(&rl, k);

        /* ── 4. Select which runnable task gets this tick ──
           Re-decided every tick, so a higher-priority release preempts
           whatever is running. */
        int chosen = -1;
        if (mode == 1) {
            /* Round Robin over runnable tasks, skipping idle unless
               nothing else is runnable */
            for (int n = 0; n < NUM_TASKS; n++) {
                int i = (rr_cursor + n) % NUM_TASKS;
                if (wl[i].is_idle) continue;
                if (job[i].active) { chosen = i; rr_cursor = (i + 1) % NUM_TASKS; break; }
            }
        } else {
            /* Fixed Priority (base) or RL-Adaptive (learned dynamic
               priority); both pick the numerically lowest priority among
               runnable tasks. */
            uint8_t best = 255;
            for (int i = 0; i < NUM_TASKS; i++) {
                if (wl[i].is_idle || !job[i].active) continue;
                uint8_t p = (mode == 2) ? tasks[i]->priority : wl[i].base_priority;
                if (p < best) { best = p; chosen = i; }
            }
        }
        if (chosen < 0) { chosen = IDLE_TASK; r.idle_ticks++; }

        /* ── 5. Execute one tick of the chosen task ── */
        if (chosen != prev_run) {
            k->context_switch_count++;
            if (prev_run >= 0 && prev_run != IDLE_TASK && job[prev_run].active)
                r.preemptions++;      /* displaced with work outstanding */
            if (prev_run >= 0) tasks[prev_run]->state = TASK_READY;
            prev_run = chosen;
        }
        tasks[chosen]->state = TASK_RUNNING;

        if (!wl[chosen].is_idle && job[chosen].active) {
            job[chosen].remaining--;
            if (job[chosen].remaining == 0) {
                job[chosen].active = 0;
                job[chosen].completed++;
                uint32_t resp = tick - job[chosen].start_tick + 1;
                response_total += resp; completed_total++;

                /* rolling average burst, for the RL burst_ratio feature */
                uint32_t burst = resp;
                tasks[chosen]->avg_burst = (tasks[chosen]->avg_burst == 0)
                    ? burst : (tasks[chosen]->avg_burst * 7 + burst * 3) / 10;
                tasks[chosen]->exec_count++;
                tasks[chosen]->state = TASK_BLOCKED;

                if (tick <= job[chosen].abs_deadline) {
                    r.hits++;
                    tasks[chosen]->deadline_hits++;
                    k->total_deadline_hits++;
                }
            }
        }
    }

    uint32_t scored = r.hits + r.misses;
    r.miss_rate    = (scored > 0) ? (float)r.misses / (float)scored * 100.0f : 0.0f;
    r.avg_response = (completed_total > 0)
                   ? (float)response_total / (float)completed_total : 0.0f;
    r.avg_reward   = (mode == 2) ? rl_get_avg_reward(&rl) : 0.0f;

    for (int i = 0; i < k->task_count; i++) free(k->task_list[i]);
    return r;
}

static void run_workload(const char* label, TaskProfile* wl) {
    float uw, ua;
    workload_utilisation(wl, &uw, &ua);

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    printf(" WORKLOAD: %s\n", label);
    printf(" Utilisation: worst-case %.3f   average %.3f   (RM bound for 7 tasks 0.729)\n",
           uw, ua);
    printf(" %s\n", (ua > 1.0f)
        ? "OVERLOADED — demand exceeds the CPU; some deadlines MUST be missed."
        : "Feasible — a correct scheduler should approach zero misses.");
    printf("══════════════════════════════════════════════════════════════════════\n");

    BenchResult res[3];
    res[0] = run_simulation(0, "Fixed Priority",    wl);
    res[1] = run_simulation(1, "Round Robin",       wl);
    res[2] = run_simulation(2, "AIROS RL-Adaptive", wl);

    printf("\n%-20s %8s %8s %9s %10s %11s %10s\n",
           "Scheduler", "Misses", "Hits", "Miss%", "CritMiss", "AvgResp", "Preempt");
    printf("---------------------------------------------------------------------------------\n");
    for (int i = 0; i < 3; i++)
        printf("%-20s %8u %8u %8.2f%% %10u %11.2f %10u\n",
               res[i].name, res[i].misses, res[i].hits, res[i].miss_rate,
               res[i].crit_misses, res[i].avg_response, res[i].preemptions);

    printf("\nPer-task deadline misses (safety-critical marked *):\n");
    printf("%-16s", "");
    for (int i = 0; i < 3; i++) printf("%18s", res[i].name);
    printf("\n");
    for (int t = 0; t < NUM_TASKS; t++) {
        if (wl[t].is_idle) continue;
        printf("%-14s%s ", wl[t].name, wl[t].critical ? "*" : " ");
        for (int i = 0; i < 3; i++) printf("%18u", res[i].per_task_miss[t]);
        printf("\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════
   PRIORITY INVERSION EXPERIMENT

   The textbook three-task scenario, and the reason the Mars Pathfinder
   kept resetting on the surface in 1997:

     LOW    acquires a shared resource and holds it for a while
     HIGH   wakes up, needs the same resource, and blocks on LOW
     MEDIUM wakes up, needs nothing, and preempts LOW

   Without priority inheritance, MEDIUM (which HIGH outranks) delays LOW,
   which delays HIGH — so a high-priority task is blocked by a
   medium-priority one for an unbounded time. With inheritance, LOW runs
   at HIGH's priority while holding the resource, MEDIUM cannot preempt
   it, and HIGH's blocking time is bounded by the critical section.

   This measures HIGH's blocking directly, both ways.
   ═══════════════════════════════════════════════════════════════════ */
#define PI_TICKS      400
#define PI_CRIT_LEN    30    /* ticks LOW spends inside the critical section */
#define PI_MED_WORK    60    /* ticks of medium-priority work available      */

static uint32_t priority_inversion_test(int use_inheritance) {
    airos_init();
    KernelCB* k = airos_get_kernel();

    TCB* high = airos_task_create("HIGH",   NULL, NULL, 0, 0, 0);
    TCB* med  = airos_task_create("MEDIUM", NULL, NULL, 1, 0, 0);
    TCB* low  = airos_task_create("LOW",    NULL, NULL, 2, 0, 0);

    Mutex m; mutex_init(&m);

    uint32_t low_crit_left  = PI_CRIT_LEN;
    uint32_t med_work_left  = PI_MED_WORK;
    uint32_t high_blocked   = 0;
    int      high_released  = 0, high_done = 0;
    int      low_holds      = 0;

    for (uint32_t tick = 0; tick < PI_TICKS && !high_done; tick++) {
        k->tick_count = tick;

        /* LOW starts immediately and takes the resource at tick 0.
           HIGH wakes at tick 10; MEDIUM wakes at tick 15 — after HIGH is
           already blocked, which is what creates the inversion. */
        if (tick == 10) high_released = 1;

        /* HIGH tries to take the resource each tick it is awake */
        int high_has = 0;
        if (high_released && !high_done) {
            high_has = mutex_try_acquire(&m, high, use_inheritance);
            if (!high_has) high_blocked++;
        }

        if (high_has) { high_done = 1; mutex_release(&m, high); break; }

        /* Otherwise pick the highest-priority runnable task.
           low->priority may have been raised by inheritance. */
        int med_runnable = (tick >= 15) && (med_work_left > 0);
        int low_runnable = (low_crit_left > 0);

        if (low_runnable && (!med_runnable || low->priority <= med->priority)) {
            if (!low_holds) low_holds = mutex_try_acquire(&m, low, use_inheritance);
            if (low_holds) {
                low_crit_left--;
                if (low_crit_left == 0) mutex_release(&m, low);
            }
        } else if (med_runnable) {
            med_work_left--;
        }
    }
    return high_blocked;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   AIROS — Adaptive Intelligent RTOS Benchmark    ║\n");
    printf("║   Preemptive simulation | %5d ticks | %d tasks  ║\n", SIM_TICKS, NUM_TASKS);
    printf("╚══════════════════════════════════════════════════╝\n");

    run_workload("OVERLOAD (as originally documented)", wl_overload);
    run_workload("FEASIBLE (periods scaled under the RM bound)", wl_feasible);

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    printf(" PRIORITY INVERSION TEST (exercises kernel mutex + inheritance)\n");
    printf("══════════════════════════════════════════════════════════════════════\n");
    uint32_t without = priority_inversion_test(0);
    uint32_t with    = priority_inversion_test(1);
    printf(" HIGH-priority task blocked for:\n");
    printf("   without priority inheritance : %3u ticks\n", without);
    printf("   with    priority inheritance : %3u ticks   (critical section = %d)\n",
           with, PI_CRIT_LEN);
    if (with < without)
        printf(" => Inheritance bounds the blocking to the critical section,\n"
               "    cutting HIGH's delay by %u ticks (%.0f%%).\n",
               without - with, 100.0 * (double)(without - with) / (double)without);
    else
        printf(" => No improvement measured — inheritance is NOT working.\n");

    printf("\n");
    return 0;
}
