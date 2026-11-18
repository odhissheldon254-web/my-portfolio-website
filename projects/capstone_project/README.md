# 🧠 AIROS — Adaptive Intelligent Real-Time Operating System

> **Capstone Project** | Embedded Systems & AI Scheduling Research  
> **Author:** Sheldon Odhi  
> **Email:** odhissheldon254@gmail.com

---

## 📌 Project Overview

AIROS is a **simulated preemptive Real-Time Operating System (RTOS) kernel** with three interchangeable schedulers — Fixed Priority, Round Robin, and a **Q-Learning reinforcement learning agent** that adapts task priorities at runtime from deadline history, urgency and execution-burst statistics.

This project was developed as a capstone to test the feasibility of **applying reinforcement learning to OS-level scheduling** in resource-constrained embedded environments (ARM Cortex-M class targets).

**The finding is negative, and that is the result.** On a correctly preemptive
simulator with deadlines counted per job, rate-monotonic Fixed Priority beats the
learned scheduler decisively — as Liu & Layland's optimality proof predicts it
should for periodic task sets of this shape. An earlier version of this project
reported the opposite; that claim came from a harness in which execution time
never occupied the CPU, and it does not survive the harness being fixed. The
analysis of *why* RL loses here, and the specific problem class where it could
still win, is in the Benchmark Results section.

---

## 🏗️ Architecture

```
AIROS_project/
├── kernel/
│   ├── airos_kernel.h        # Core data structures: TCB, Mutex, Semaphore, MessageQueue, KernelCB
│   └── airos_kernel.c        # Kernel implementation: task creation, context switch, IPC primitives
├── scheduler/
│   ├── rl_scheduler.h        # RL agent definitions & hyperparameters
│   └── rl_scheduler.c        # Q-Learning engine: state extraction, Bellman update, ε-greedy policy
├── simulation/
│   └── benchmark.c           # Simulation harness: 3-mode benchmark + CSV export
├── benchmark_results.csv     # Auto-generated benchmark output
└── Makefile                  # Build system
```

---

## ⚙️ Key Features

### Kernel Core
| Feature | Details |
|---|---|
| **Task Management** | Up to 16 tasks, TCB with full runtime stats |
| **Scheduling** | Preemptive: Fixed Priority, Round Robin, RL-Adaptive |
| **Context Switch** | Simulated ARM Cortex-M stack frame (xPSR, PC, LR, R0–R12) |
| **IPC Primitives** | Mutex (priority inheritance, **measured**), Semaphore, Message Queue |
| **Deadline Tracking** | Per-task deadline hit/miss counters with reward signals |
| **Periodic Tasks** | Automatic re-release at task period boundaries |

### RL Scheduler (Q-Learning Agent)
| Parameter | Value |
|---|---|
| **Algorithm** | Q-Learning (Bellman equation) |
| **Learning Rate (α)** | 0.10 |
| **Discount Factor (γ)** | 0.90 |
| **Exploration Rate (ε)** | 0.15 → decays to 0.01 |
| **State Space** | 3 features × 4 bins = 64 states per task |
| **Action Space** | 8 priority levels |
| **Reward Signal** | (hits − misses) / (hits + misses + 1) over the epoch, −0.5 if starved |
| **Policy** | ε-greedy with random tie-breaking, warm-started from rate-monotonic |

### State Features
1. **Urgency** — normalized time-to-deadline ratio
2. **Miss Rate** — rolling fraction of missed deadlines
3. **Burst Ratio** — avg execution time vs. worst-case estimate

---

## 📊 Benchmark Results

> **These numbers replace every earlier figure in this project.** The original
> harness did not simulate CPU contention (see "Harness rebuild" below), so its
> results described a queueing policy rather than a scheduler. Everything below
> comes from the v2 preemptive simulator, 20,000 ticks, verified across five
> random seeds.

### Is the workload even schedulable?

The task set as originally documented demands **132% of the CPU at worst case
and 102% on average**, against a Liu-Layland bound of 0.729 for seven tasks. It
is infeasible — deadline misses are mathematically forced, not a scheduling
failure. This was invisible while execution time was free. Both regimes are now
benchmarked:

| Regime | U (worst) | U (avg) | Meaning |
|---|---|---|---|
| **OVERLOAD** (as documented) | 1.320 | 1.022 | Demand exceeds the CPU. The question is *which* deadlines are sacrificed. |
| **FEASIBLE** (periods scaled) | 0.688 | 0.531 | Under the RM bound; a correct scheduler should approach zero misses. |

### OVERLOAD regime

| Scheduler | Misses | Miss rate | Critical misses | Avg response | Preemptions |
|---|---|---|---|---|---|
| **Fixed Priority** | **50** | **0.99%** | **4** | 11.90 | 1,898 |
| AIROS RL-Adaptive | 457 | 9.01% | 436 | 12.74 | 1,597 |
| Round Robin | 1,260 | 24.83% | 1,257 | 15.96 | 14,139 |

### FEASIBLE regime

| Scheduler | Misses | Miss rate | Critical misses |
|---|---|---|---|
| **Fixed Priority** | **0** | **0.00%** | **0** |
| AIROS RL-Adaptive | 2 | 0.08% | 2 |
| Round Robin | 14 | 0.54% | 14 |

### The honest conclusion

**Fixed Priority wins decisively in both regimes, and the RL scheduler loses.**
Every earlier result claiming otherwise was an artefact of a harness in which
execution time did not occupy the processor.

This is the theoretically expected outcome, not a surprise: Liu & Layland proved
rate-monotonic priority assignment is *optimal* among fixed-priority schedulers
for feasible periodic task sets. In the FEASIBLE regime there is nothing left to
win — Fixed Priority already achieves zero misses, so a learned policy can only
add noise, and it does (2 misses, both on a safety-critical task).

The overload regime is more damning, and the per-task breakdown shows why:

| Task | Critical | Fixed Priority | RL-Adaptive |
|---|---|---|---|
| CONTROL_LOOP | ✔ | 0 | 22 |
| SENSOR_READ | ✔ | 0 | 75 |
| FAULT_MONITOR | ✔ | 4 | 339 |
| UI_UPDATE | | 20 | 6 |
| SELF_TEST | | 26 | 1 |

Under overload, Fixed Priority degrades *gracefully*: it sacrifices UI_UPDATE and
SELF_TEST — the long-period, non-critical background work — and protects the
control loop and sensor path almost perfectly. The RL agent does the opposite,
spending 436 of its 457 misses on safety-critical tasks in order to protect
background work. In a real vehicle or medical device that behaviour is not
merely worse, it is dangerous.

The cause is ε-greedy exploration. The agent is warm-started onto the
Fixed-Priority policy, which is already near-optimal here, so every exploratory
action can only move it away from the optimum — and under 102% load there is no
slack to absorb a bad decision. Exploration that was *beneficial* under the old
broken metric (where it mimicked round-robin fairness) is actively harmful once
deadlines are counted correctly.

### Where a learned scheduler could genuinely win

Rate-monotonic orders tasks purely by period, so it is structurally unable to
protect a *long-period but safety-critical* task from a *short-period but
unimportant* one. This task set does not have that property — criticality and
rate happen to correlate — which is precisely why RM is unbeatable on it. A
workload where criticality is anti-correlated with period is the problem class
where a learned policy has real headroom, and is the honest next experiment.

---

## 🔧 Harness rebuild (v2) and post-submission corrections

### Defects found and fixed

**1. Execution time never occupied the CPU.** A running task blocked after
exactly one tick regardless of the execution time drawn for it, so `rand_et()`
fed only the hit/miss arithmetic. Nothing was ever preempted, no two tasks ever
competed, and utilisation was irrelevant — which is why Fixed Priority and Round
Robin returned byte-identical results across 20 different workload seeds. Fixed:
a job now carries `remaining` ticks of work and holds the CPU until it completes
or is preempted.

**2. No preemption existed.** The scheduler now re-decides every tick, so a
higher-priority release genuinely displaces the running task.

**3. Deadline misses were counted per tick, not per job.** A task sitting past
its deadline was charged a fresh miss every tick, so the metric measured
accumulated lateness rather than deadline misses. This single choice is what made
Round Robin appear unbeatable — it never starves anyone. Corrected to one miss
per job, which is what real-time analysis means.

**4. IDLE was a periodic task demanding 100% CPU** (period 1, WCET 1), which
alone made every workload infeasible. It is now a true unscored background task.

**5. The Bellman update bootstrapped a state off itself.** `_extract_state()` was
called twice in the same tick, so `s'` always equalled `s`, collapsing the
Q-update into a contextual bandit. Fixed by carrying each task's
`(state, action)` across epochs.

**6. The greedy policy collapsed to "give every task priority 0".** Uniform
Q-values meant every `argmax` tied, and first-index tie-breaking returned action
0 for every task — visible in `benchmark_results.csv` as priority rows of
`0,0,0,0,0,0,4,0`. Fixed with random tie-breaking plus a warm start from the
static priority assignment.

**7. Unsigned underflow in the starvation penalty.** `tick_count - release_tick`
wrapped to ~4×10⁹ whenever `release_tick` held a future value, firing the −0.5
penalty almost unconditionally. Fixed with a guarded subtraction.

### Priority inheritance is now actually tested

The kernel shipped mutexes with priority inheritance, semaphores and message
queues — and the benchmark never called any of them. The inheritance branch had
never executed. `mutex_try_acquire()` / `mutex_release()` were added as
non-blocking entry points usable from the tick-driven simulation, and the
classic three-task inversion scenario (the Mars Pathfinder bug) is now measured:

| Configuration | HIGH task blocked for |
|---|---|
| Without priority inheritance | **80 ticks** |
| With priority inheritance | **20 ticks** |

The 20 ticks is exactly the remaining critical section — bounded, as the protocol
requires. The 80 is that same 20 plus MEDIUM's entire 60 ticks of unrelated work:
the unbounded component, which grows with however much medium-priority work
exists. The kernel's inheritance implementation is correct; nothing had ever
demonstrated it.

## 🚀 Build & Run

### Prerequisites
- GCC (C11 standard)
- `make`
- `libm` (math library — included via `-lm`)

### Build
```bash
make
```

### Run Benchmark
```bash
make run
# or
./airos_bench
```

### Clean
```bash
make clean
```

---

## 📈 Visualising Results

After running the benchmark, `benchmark_results.csv` is generated with per-tick snapshots of:
- Deadline misses & hits (per scheduler)
- Context switch count
- RL agent average reward
- Dynamic priority assignments for all 8 tasks

You can plot this with Python:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")
df.plot(x="tick", y=["fp_misses", "rr_misses", "rl_misses"], title="Deadline Misses Over Time")
plt.show()
```

---

## 🔬 Technical Highlights

- **Priority Inheritance** in mutex implementation prevents priority inversion — a critical correctness property for real-time systems.
- **Optimistic Q-table initialization** (0.5 for all state-action pairs) encourages early exploration before the agent exploits learned policies.
- **ε-decay** (×0.995 per epoch) causes the RL agent to shift from exploration to exploitation as it gains confidence in its Q-values.
- **Deterministic simulation** via XOR-shift PRNG ensures reproducible benchmark results.
- The system is structured to be **portable to bare-metal ARM Cortex-M** by replacing the simulation harness with a real hardware timer ISR.

---

## 📚 References & Further Reading

- Liu, C. L., & Layland, J. W. (1973). *Scheduling algorithms for multiprogramming in a hard real-time environment.* JACM.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- FreeRTOS documentation — [freertos.org](https://www.freertos.org)
- ARM Cortex-M Exception Model — ARMv7-M Architecture Reference Manual

---

## 📄 License

This project is submitted as an academic capstone and is open for educational use.
