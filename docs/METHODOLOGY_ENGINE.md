# The methodology engine — how the Calibration DAG is executed

The engine is the execution layer (B) that runs the science layer (A). It is a **thin adapter**, not
a new platform: it decides nothing scientific. Every transition between rungs comes from an immutable
result artifact and the frozen contract — never from the scheduler, spare CPU, another asset's
result, or a human nudging the queue.

```
Rung 0: freeze contract snapshot
        │
   ┌────┴─────────────────────────────────────────────────────────┐
   │  planner:  derive per-asset state ──► next allowed experiment │   (a pure function)
   └────┬─────────────────────────────────────────────────────────┘
        │ enqueue
   queue/pending ──claim(atomic rename)──► worker ──► science runner (one asset)
        │                                    │
        │                       immutable per-asset artifact
        │                                    │
        │            reducer (single writer) ──► panel inputs + run panels
        │                                    │
   exec_ledger + method_ledger (audit, not authority)
        │
   planner re-derives state ──► next experiment, or RESOLVED / RESOLVED_EMPTY / NEEDS_CONTRACT
```

## Source of truth

- **The contract** is the only source of scientific rules. `engine/contract.py` writes an immutable
  snapshot per run (`runs/<id>/contract.json`) — the assembled contract plus the hashes and
  environment that pin it. It does **not** create a second contract system.
- **Result artifacts** are the source of each asset's state. `engine/states.py` derives the state
  purely from the artifacts and the contract, so a run is reconstructible from its results alone.
- **The two ledgers are audit, not authority.** If a ledger ever disagrees with a result artifact,
  the artifact and the contract win.

## The state machine (`engine/states.py`)

```
PENDING_VIABILITY → VIABLE → OPERATING_POINT_VALID → UTILITY_REGISTERED → CONFIRMED
   → NULL_VALIDATED → LOCALLY_OPTIMIZED → INTERACTIONS_EVALUATED → RESOLVED

CONFIRMED          → NULL_REJECTED        → RESOLVED_EMPTY
UTILITY_REGISTERED → NO_CONFIRMED_FEATURE → RESOLVED_EMPTY
any scientific rung → NEEDS_CONTRACT     (a science stop — a human mints a new contract version)
any task            → FAILED_TECHNICAL   (an execution error — the same task is retried unchanged)
```

`NEEDS_CONTRACT` and `FAILED_TECHNICAL` are kept apart deliberately: the first means the rules do not
cover what was observed; the second means a worker crashed. Recalibration is never automatic — a
`NEEDS_CONTRACT` state records `{reason, observed_statistic, required_human_action}`, may propose a
`proposed_contract_patch.json`, and stops. Applying it needs a human, a new contract version and a
new `run_id`.

Rung 2 (operating-point transfer) is validated inside Rungs 3–4 on this panel, so the executable path
is `1 → 3 → 4 → 5 → 6`; the state machine still names `OPERATING_POINT_VALID` conceptually.

## The planner is a deterministic function (`engine/planner.py`)

```python
next_action = classify(contract, asset_artifacts)   # {asset, state, next_action, reason, ...}
```

Planning and enqueuing are separate, so the whole plan can be read before a single worker starts
(`make engine-plan`). The planner changes no scientific parameter and invents no rule. Before it
enqueues a rung whose runner reads a panel register, it checks that the register holds the asset —
the reducer, run as a planner pre-step, assembles those panels from per-asset artifacts.

## Immutable artifacts and one reducer (`engine/worker.py`, `engine/reducer.py`)

A worker writes **only its own immutable artifact**,
`runs/<id>/results/<rung>/<asset>/<task_hash>.json` — a new file per attempt, so a publish never
overwrites an earlier result and the newest valid file is the state. The **reducer is the single
writer of panels**: it reads per-asset artifacts and assembles the panel views that downstream
runners read as input and that the report reads. There is no concurrent write to a shared scientific
file and no `flock` on results — `flock` is used only for the atomic task claim and the append-only
ledgers.

## Contract enforcement (the worker's one scientific gate)

The worker refuses a task whose `contract_hash` does not match the run's frozen contract — the check
happens **before** any compute, so no result is ever produced under different rules.

## tmux and the scheduler (`ops/*.sh`)

```
golden-calibration:  planner | worker-01..N | guard | scheduler
```

- **`ops/engine.sh`** — supervisor: global `flock`, the Rung-0 snapshot, the tmux session, `9>&-` on
  every child (so the long-lived tmux server never inherits and pins the lock).
- **`ops/worker.sh`** — a dumb loop: claim, run one task, repeat; honours a worker count the guard
  may lower; writes a heartbeat.
- **`ops/scheduler.sh`** — a tmux loop with cron-like cadence (planner every 10 min, report every
  30 min). Not system cron — self-contained, survives with the tmux server, decides nothing
  scientific.
- **`ops/guard.sh`** — technical watchdog only: requeue a stale task, lower the worker count under
  memory pressure (RAM + swap), hard-stop at the deadline. It may not touch a threshold, the HPO
  space, the null, the data boundary, or which asset runs. A requeued task is retried **unchanged**.

## Resume

State lives in artifacts and the queue, not in memory. A killed worker's stale task is requeued by
the guard and retried under the same contract; Rung 5 additionally keeps its own internal
fold/permutation ledger, so its finest granularity resumes without the top-level queue knowing about
it. The rungs without a ledger (1–3B) are cheap enough that a requeue simply recomputes the asset,
deterministically under the frozen seed.
