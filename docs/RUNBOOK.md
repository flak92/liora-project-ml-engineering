# Runbook

Two modes. **Presentation** reads frozen artifacts and prints results in a blink. **Reproduction**
runs the Calibration DAG per asset in a detached tmux session and produces new artifacts.

## Presentation (no compute)

```bash
make methodology-report          # funnel + per-asset descriptions from results/methodology_snapshot/
```
The funnel is derived from the artifacts every time; `--parity 26 11 9 4` asserts the known
development-panel numbers (a snapshot-parity check, not a success condition — a fresh panel may
differ and be correct).

## Reproduction (the engine)

```bash
make engine-smoke                # full DAG on AZO ADBE GOOG, detached (a validation run, ~1–1.5 h)
make engine-start ASSETS="AZO ADBE GOOG …" WORKERS=4 HOURS=8    # a full run, detached
```
`engine-start` opens the tmux session `golden-calibration` and returns; the tmux server is the
daemon, so the terminal can be closed. The run ends when every asset is terminal, at the deadline, or
on a cooperative halt.

```bash
make engine-status               # session, queue counts, per-asset states, ledger integrity, memory
make engine-attach               # watch it live (detach: Ctrl-b d)
make engine-plan                 # the deterministic plan — read before anything runs
make engine-report               # rebuild the run report from a live/finished run
make engine-stop                 # cooperative halt — finishes the current tasks, then stops
```

## Reading the states

| state | meaning | next |
|---|---|---|
| `PENDING_VIABILITY … NULL_VALIDATED` | mid-ladder | the planner enqueues the next rung |
| `RESOLVED` | features confirmed and retained | terminal — read `results/compiled/<asset>.json` |
| `RESOLVED_EMPTY` | nothing survived — a valid, complete result | terminal |
| `NEEDS_CONTRACT` | the rules cannot honestly continue | **human**: mint a new contract version + new `run_id` |

`NEEDS_CONTRACT` is not a failure and not something the engine resolves. It means the search space,
the operating point, or the data does not fit the frozen contract for that asset. The engine records
the reason and stops; changing the rules is a deliberate human act (a new contract version, a new
`run_id`, and an explicit note of which prior results that invalidates), never a side effect of a run.

## When something breaks

- A `FAILED_TECHNICAL` task (a crash, an OOM) is requeued by the guard and retried **unchanged**. It
  is not a scientific event.
- Under memory pressure the guard lowers the worker count (RAM + swap headroom); it never touches the
  science. Add swap if the machine has none — the guard is swap-aware.
- `make engine-selftest` proves the execution guarantees (atomic claim, contract enforcement,
  immutable publish, state-from-artifacts, two ledgers, OOS boundary) without running any science.

## Contract changes

Never edit a threshold, the HPO space, the folds, the null, the seeds, or the objective as a side
effect of running the engine. A scientific change is a separate commit and a new contract version.
After editing `config/contract/*`, run `make lint-contract` (from `main`) to regenerate the monolith
and confirm no constant drifted. Then a new run snapshots the new contract under a fresh `run_id`.
