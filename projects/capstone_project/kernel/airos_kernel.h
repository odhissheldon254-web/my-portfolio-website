#ifndef AIROS_KERNEL_H
#define AIROS_KERNEL_H

#include <stdint.h>
#include <stddef.h>

/* ─────────────────────────────────────────────
   AIROS — Adaptive Intelligent Real-Time OS
   Kernel Core Definitions
   ───────────────────────────────────────────── */

#define AIROS_MAX_TASKS        16
#define AIROS_STACK_SIZE       2048
#define AIROS_TICK_RATE_HZ     1000
#define AIROS_SCHED_EPOCH      50      /* ticks between RL scheduler updates */
#define AIROS_PRIORITY_LEVELS  8

/* Task states */
typedef enum {
    TASK_READY     = 0,
    TASK_RUNNING   = 1,
    TASK_BLOCKED   = 2,
    TASK_SUSPENDED = 3,
    TASK_DEAD      = 4
} TaskState;

/* Task Control Block */
typedef struct TCB {
    uint8_t         id;
    char            name[16];
    TaskState       state;
    uint8_t         priority;           /* current dynamic priority (0=highest) */
    uint8_t         base_priority;      /* original static priority */
    uint32_t        period_ms;          /* task period (periodic tasks) */
    uint32_t        deadline_ms;        /* relative deadline */
    uint32_t        worst_case_et;      /* worst-case execution time (ticks) */

    /* Runtime statistics */
    uint32_t        exec_count;         /* total executions */
    uint32_t        deadline_hits;      /* times met deadline */
    uint32_t        deadline_misses;    /* times missed deadline */
    uint32_t        last_burst;         /* last execution burst (ticks) */
    uint32_t        avg_burst;          /* rolling average burst */
    uint32_t        release_tick;       /* tick when task became ready */
    uint32_t        absolute_deadline;  /* absolute deadline tick */

    /* RL agent state */
    float           q_value;            /* Q-value assigned by RL scheduler */
    float           reward_signal;      /* last reward from scheduler */

    /* Stack */
    uint32_t        stack[AIROS_STACK_SIZE / sizeof(uint32_t)];
    uint32_t*       stack_ptr;

    struct TCB*     next;               /* linked list pointer */
} TCB;

/* Mutex */
typedef struct {
    uint8_t         locked;
    TCB*            owner;
    TCB*            wait_queue[AIROS_MAX_TASKS];
    uint8_t         wait_count;
} Mutex;

/* Semaphore */
typedef struct {
    int32_t         count;
    TCB*            wait_queue[AIROS_MAX_TASKS];
    uint8_t         wait_count;
} Semaphore;

/* Message Queue */
#define MSG_QUEUE_SIZE 16
typedef struct {
    void*           buffer[MSG_QUEUE_SIZE];
    uint8_t         head, tail, count;
    Semaphore       sem_full;
    Semaphore       sem_empty;
} MessageQueue;

/* Kernel control block */
typedef struct {
    TCB*            task_list[AIROS_MAX_TASKS];
    uint8_t         task_count;
    TCB*            current_task;
    uint32_t        tick_count;
    uint32_t        context_switch_count;
    uint32_t        total_deadline_misses;
    uint32_t        total_deadline_hits;
    uint8_t         scheduler_mode;    /* 0=fixed, 1=RL-adaptive */
} KernelCB;

/* ── Public API ── */
void     airos_init(void);
TCB*     airos_task_create(const char* name, void(*func)(void*), void* arg,
                           uint8_t priority, uint32_t period_ms, uint32_t deadline_ms);
void     airos_start(void);
void     airos_tick(void);
void     airos_yield(void);
void     airos_task_delay(uint32_t ticks);
void     airos_task_suspend(TCB* task);
void     airos_task_resume(TCB* task);

void     mutex_init(Mutex* m);
void     mutex_lock(Mutex* m);
void     mutex_unlock(Mutex* m);

void     semaphore_init(Semaphore* s, int32_t initial);
void     semaphore_wait(Semaphore* s);
void     semaphore_signal(Semaphore* s);

void     msgq_init(MessageQueue* q);
void     msgq_send(MessageQueue* q, void* msg);
void*    msgq_recv(MessageQueue* q);

KernelCB* airos_get_kernel(void);

#endif /* AIROS_KERNEL_H */
