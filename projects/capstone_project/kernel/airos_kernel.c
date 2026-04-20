#include "airos_kernel.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ─────────────────────────────────────────────
   AIROS Kernel Implementation
   Context Switch | Scheduler | IPC Primitives
   ───────────────────────────────────────────── */

static KernelCB kernel;

/* ── Kernel Init ── */
void airos_init(void) {
    memset(&kernel, 0, sizeof(KernelCB));
    kernel.scheduler_mode = 1;  /* RL-adaptive by default */
    kernel.tick_count     = 0;
    kernel.task_count     = 0;
    kernel.current_task   = NULL;
}

KernelCB* airos_get_kernel(void) { return &kernel; }

/* ── Stack Initialisation (simulated ARM Cortex-M frame) ──
   In real hardware this sets up the exception return frame:
   xPSR, PC, LR, R12, R3, R2, R1, R0 pushed by hardware;
   R4-R11 pushed by software (context save).
   Here we simulate the stack pointer arithmetic.
*/
static void _init_stack(TCB* task, void(*func)(void*), void* arg) {
    uint32_t* sp = &task->stack[AIROS_STACK_SIZE / sizeof(uint32_t) - 1];

    /* Simulated exception frame */
    *sp-- = 0x01000000;            /* xPSR: Thumb bit set          */
    *sp-- = (uint32_t)(uintptr_t)func; /* PC                       */
    *sp-- = 0xFFFFFFFD;            /* LR: EXC_RETURN               */
    *sp-- = 0x12121212;            /* R12                          */
    *sp-- = 0x03030303;            /* R3                           */
    *sp-- = 0x02020202;            /* R2                           */
    *sp-- = 0x01010101;            /* R1                           */
    *sp-- = (uint32_t)(uintptr_t)arg; /* R0: task argument         */

    /* Software-saved registers R4-R11 */
    *sp-- = 0x11111111; /* R11 */
    *sp-- = 0x10101010; /* R10 */
    *sp-- = 0x09090909; /* R9  */
    *sp-- = 0x08080808; /* R8  */
    *sp-- = 0x07070707; /* R7  */
    *sp-- = 0x06060606; /* R6  */
    *sp-- = 0x05050505; /* R5  */
    *sp   = 0x04040404; /* R4  */

    task->stack_ptr = sp;
}

/* ── Task Creation ── */
TCB* airos_task_create(const char* name, void(*func)(void*), void* arg,
                        uint8_t priority, uint32_t period_ms, uint32_t deadline_ms) {
    if (kernel.task_count >= AIROS_MAX_TASKS) return NULL;

    TCB* task = (TCB*)malloc(sizeof(TCB));
    if (!task) return NULL;
    memset(task, 0, sizeof(TCB));

    task->id             = kernel.task_count;
    task->priority       = priority;
    task->base_priority  = priority;
    task->period_ms      = period_ms;
    task->deadline_ms    = deadline_ms;
    task->state          = TASK_READY;
    task->q_value        = 0.0f;
    task->reward_signal  = 0.0f;
    task->release_tick   = 0;
    task->absolute_deadline = deadline_ms;

    strncpy(task->name, name, 15);
    _init_stack(task, func, arg);

    kernel.task_list[kernel.task_count++] = task;
    return task;
}

/* ── Fixed-Priority Scheduler (baseline) ──
   Returns highest-priority READY task.
   Lower priority number = higher priority (convention).
*/
static TCB* _schedule_fixed(void) {
    TCB* chosen = NULL;
    uint8_t best_pri = 255;
    for (int i = 0; i < kernel.task_count; i++) {
        TCB* t = kernel.task_list[i];
        if (t->state == TASK_READY && t->priority < best_pri) {
            best_pri = t->priority;
            chosen   = t;
        }
    }
    return chosen;
}

/* ── Context Switch (simulated) ── */
static void _context_switch(TCB* next) {
    if (kernel.current_task && kernel.current_task->state == TASK_RUNNING) {
        kernel.current_task->state = TASK_READY;
    }
    kernel.current_task        = next;
    next->state                = TASK_RUNNING;
    kernel.context_switch_count++;
}

/* ── Deadline Tracking ── */
static void _check_deadlines(void) {
    for (int i = 0; i < kernel.task_count; i++) {
        TCB* t = kernel.task_list[i];
        if (t->state == TASK_DEAD || t->state == TASK_SUSPENDED) continue;

        if (kernel.tick_count > t->absolute_deadline && t->exec_count > 0) {
            if (t->state != TASK_DEAD) {
                /* Mark deadline miss if task hasn't completed this period */
                if (t->state == TASK_READY || t->state == TASK_RUNNING) {
                    t->deadline_misses++;
                    kernel.total_deadline_misses++;
                    t->reward_signal = -1.0f;   /* negative reward to RL agent */
                }
            }
        }
    }
}

/* ── Tick Handler ──
   Called every 1ms (AIROS_TICK_RATE_HZ).
   Advances time, checks deadlines, releases RL epoch updates.
*/
void airos_tick(void) {
    kernel.tick_count++;
    _check_deadlines();

    /* Re-release periodic tasks */
    for (int i = 0; i < kernel.task_count; i++) {
        TCB* t = kernel.task_list[i];
        if (t->state == TASK_BLOCKED && t->period_ms > 0) {
            if (kernel.tick_count >= t->release_tick + t->period_ms) {
                t->state            = TASK_READY;
                t->release_tick     = kernel.tick_count;
                t->absolute_deadline = kernel.tick_count + t->deadline_ms;
            }
        }
    }
}

/* ── Yield ── */
void airos_yield(void) {
    if (kernel.current_task) {
        kernel.current_task->state     = TASK_BLOCKED;
        kernel.current_task->release_tick = kernel.tick_count;
        kernel.current_task->exec_count++;

        /* Check if deadline was met */
        if (kernel.tick_count <= kernel.current_task->absolute_deadline) {
            kernel.current_task->deadline_hits++;
            kernel.total_deadline_hits++;
            kernel.current_task->reward_signal = 1.0f;
        }
    }

    TCB* next = _schedule_fixed();
    if (next) _context_switch(next);
}

/* ── Task Delay ── */
void airos_task_delay(uint32_t ticks) {
    if (kernel.current_task) {
        kernel.current_task->state      = TASK_BLOCKED;
        kernel.current_task->release_tick = kernel.tick_count + ticks;
    }
    airos_yield();
}

/* ── Suspend / Resume ── */
void airos_task_suspend(TCB* task) { task->state = TASK_SUSPENDED; }
void airos_task_resume(TCB* task)  { task->state = TASK_READY; }

/* ════════════════════════════════════════
   IPC PRIMITIVES
   ════════════════════════════════════════ */

/* ── Mutex ── */
void mutex_init(Mutex* m) {
    m->locked     = 0;
    m->owner      = NULL;
    m->wait_count = 0;
}

void mutex_lock(Mutex* m) {
    if (!m->locked) {
        m->locked = 1;
        m->owner  = kernel.current_task;
        return;
    }
    /* Priority Inheritance: boost owner priority to prevent inversion */
    if (m->owner && kernel.current_task &&
        kernel.current_task->priority < m->owner->priority) {
        m->owner->priority = kernel.current_task->priority;
    }
    /* Enqueue current task */
    if (m->wait_count < AIROS_MAX_TASKS)
        m->wait_queue[m->wait_count++] = kernel.current_task;
    airos_task_suspend(kernel.current_task);
    airos_yield();
}

void mutex_unlock(Mutex* m) {
    if (!m->locked || m->owner != kernel.current_task) return;
    /* Restore priority */
    m->owner->priority = m->owner->base_priority;
    if (m->wait_count > 0) {
        TCB* next = m->wait_queue[0];
        for (int i = 0; i < m->wait_count - 1; i++)
            m->wait_queue[i] = m->wait_queue[i+1];
        m->wait_count--;
        m->owner = next;
        airos_task_resume(next);
    } else {
        m->locked = 0;
        m->owner  = NULL;
    }
}

/* ── Semaphore ── */
void semaphore_init(Semaphore* s, int32_t initial) {
    s->count      = initial;
    s->wait_count = 0;
}

void semaphore_wait(Semaphore* s) {
    if (s->count > 0) { s->count--; return; }
    if (s->wait_count < AIROS_MAX_TASKS)
        s->wait_queue[s->wait_count++] = kernel.current_task;
    airos_task_suspend(kernel.current_task);
    airos_yield();
}

void semaphore_signal(Semaphore* s) {
    if (s->wait_count > 0) {
        TCB* t = s->wait_queue[0];
        for (int i = 0; i < s->wait_count - 1; i++)
            s->wait_queue[i] = s->wait_queue[i+1];
        s->wait_count--;
        airos_task_resume(t);
    } else {
        s->count++;
    }
}

/* ── Message Queue ── */
void msgq_init(MessageQueue* q) {
    q->head = q->tail = q->count = 0;
    semaphore_init(&q->sem_full,  0);
    semaphore_init(&q->sem_empty, MSG_QUEUE_SIZE);
}

void msgq_send(MessageQueue* q, void* msg) {
    semaphore_wait(&q->sem_empty);
    q->buffer[q->tail] = msg;
    q->tail = (q->tail + 1) % MSG_QUEUE_SIZE;
    q->count++;
    semaphore_signal(&q->sem_full);
}

void* msgq_recv(MessageQueue* q) {
    semaphore_wait(&q->sem_full);
    void* msg = q->buffer[q->head];
    q->head = (q->head + 1) % MSG_QUEUE_SIZE;
    q->count--;
    semaphore_signal(&q->sem_empty);
    return msg;
}
