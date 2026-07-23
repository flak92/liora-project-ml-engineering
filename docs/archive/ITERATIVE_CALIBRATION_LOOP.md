# The Iterative Calibration Loop

An automatic, results-dependent process that carries each asset through the smallest sequence of
computations needed to reach a methodologically decisive answer, and then keeps going — under one
pre-authorized frozen contract version after another — until a new attempt no longer changes the
answer. It runs detached: start it, close the terminal, and it computes to completion, then writes
its own summary of the methodology it found and the corrections it made along the way.

It is the outer loop above the Calibration DAG (`engine/`). The DAG drives one asset to a terminal
verdict under one frozen contract; this loop chooses which frozen contract to try next, compares what
each one confirmed, and stops when improvements flatten. It builds no new execution platform — it
reuses the engine per epoch and adds only what was missing: a global integrity gate, convergence
detection, a technical-repair accounting, and a self-summary.

## The one invariant that makes autonomy honest

**The proof standard is frozen across the whole ladder; only the hypothesis space may vary.**

Letting a loop advance on its own is only defensible if it cannot, on its own, make the bar easier.
So a ladder rung may propose a different *hypothesis* — a different model search space, a different
operating point, different arms — but it may never touch the standard by which a feature is judged
real. `engine/contract_patch.py` enforces this mechanically:

- **admissible** (a rung may vary): `operating_point`, `model_space`, `arms`, `rung_6_survivor_hpo`,
  `rung_7_interactions` — the hypothesis space.
- **frozen** (a rung may never touch): `viability`, `acceptance`, `cross_fitting`, `stop_conditions`,
  `max_null`, `rotation_level_null`, `data_boundary`, `certification` — the proof standard.

A patch that reaches into any frozen key, or into anything outside the admissible set, is rejected
before any compute. A negative or empty result therefore never triggers loosening — it triggers
"advance to the next pre-declared hypothesis space", or stop. The ladder is finite and human-authored
in `config/iteration_loop_policy.json`; the loop never invents a rung. `_why`: an autonomous loop that
could weaken its own null would prove nothing, however many epochs it ran.

## Three layers, kept apart

- **Scientific Decision Loop** — `planner.next_action`: `frozen contract + immutable artifact →
  statistic → verdict → next admissible experiment`. Unchanged science.
- **Execution Loop** — tmux, queue, workers, scheduler, reducer, guard, checkpoints, resume. Reused
  as-is (`ops/engine.sh` per epoch); an in-process backend exists for deterministic tests.
- **Repair Loop** — `engine/repair.py`: retries transient technical failures, quarantines
  non-reproducible ones, stops retrying what a retry cannot fix. It is forbidden, structurally, from
  touching a threshold, a fold, a null, or the OOS boundary. `FAILED_TECHNICAL` (a broken machine) is
  kept apart from `NEEDS_CONTRACT` (a science stop) on purpose.

## Two nested loops

**Inner (one frozen contract → fixpoint).** Each cycle: `verify_global_integrity` (`engine/integrity.py`,
which refuses to build on a corrupted run) → repair technical failures → derive each asset's state
purely from artifacts → plan the smallest next experiment → enqueue → drain → until every asset is
terminal. This is the engine's own loop, gated and instrumented.

**Outer (walk the ladder).** For each pre-authorized version: snapshot it through the safety kernel
(a fresh, self-consistent `contract_hash`), run the inner loop to a fixpoint, compile what it
confirmed, fold that into the cumulative confirmed set, and decide: converge, advance, or halt.

## Terminal outcomes

Per asset: `RESOLVED_RETAINED`, `RESOLVED_EMPTY`, `NEEDS_CONTRACT` (science stop — a human mints a new
contract version), `FAILED_TECHNICAL` (execution stop). For the whole loop: `CONVERGED`,
`LADDER_EXHAUSTED`, `HALTED_BUDGET` (a core-hour cap from the policy), `HALTED` (cooperative stop),
`INTEGRITY_FAILED`.

## Convergence — "no visible improvement after further attempts"

A confirmed feature is a null-validated stable survivor (A1∩A2∩B): a pair `(asset, unit)`. Rung 6
tunes survivors but adds no new feature, so the cumulative confirmed set is monotone. After at least
one variant epoch (`k ≥ 1`), if `patience` consecutive epochs add nothing new to that set, the loop
declares `CONVERGED`: a new hypothesis space no longer changes the answer. A barren base epoch does
not declare victory — the next hypothesis is still tried. The result is not one best configuration but
an auditable methodology: the smallest confirmed set of OHLCV relationships, and an explicit statement
of where the contract does not permit going further.

## The ladder mechanism

Each rung is `base contract + a pre-authorized, human-authored patch` over the admissible splits.
`contract_patch.apply` deep-merges the patch, runs the guard, and re-hashes — so every version is a
genuinely distinct, self-consistent frozen snapshot, addressable exactly like a hand-minted one. Each
epoch is a standard run directory under `runs/<ladder_id>/epochs/eK_<version>/`, driven to completion
by the existing supervisor; the ladder manifest (`ladder.json`), the hash-chained trace
(`iteration_trace.jsonl` → `iteration_trace.json`) and the summary tie them together at the ladder
root.

**Vocabulary — "variation", not "widening".** A rung does *human-pre-authorized hypothesis-space
variation*, which is not the same as widening. The shipped default θ move
`[0.75, 0.82, 0.88, 0.92, 0.96, 0.99] → [0.80, 0.90, 0.95]` is **coarser / alternative** — a different
decision-threshold hypothesis, not a mathematical superset of the old grid. "Widen" is accurate only when
the new range truly contains the old (e.g. `hpo_trials 30 → 60`). And the two terminal responses to a
negative result are distinct: `RESOLVED_EMPTY` means the current contract finished with a valid empty answer
(nothing is loosened); `NEEDS_CONTRACT` means it cannot proceed honestly, so a **human** may author the next
pre-declared version. The guard is the same in both: only the ADMISSIBLE hypothesis may change — now
enforced field by field, so an admissible section cannot smuggle a frozen leaf (`rung_6_survivor_hpo.alpha`,
`own_null.permutations`) — never the FROZEN proof standard, and a later version never alters a prior
version's result; it starts a new study.

## The deliverable: `iteration_summary.md`

When the loop stops, `engine/iteration_report.py` reads everything it left behind and writes one Polish
document: the methodology as a set of rules (the frozen proof standard every epoch shared and the
hypothesis each varied), the funnel per epoch, the convergence trail, and — gathered in one place — every
CORRECTION: the technical repairs this run performed, the scientific walls it hit (each with a proposed,
human-gated `proposed_contract_patch_<asset>.json`), and the corrections made while the engine itself
was built. It trains nothing and decides nothing; every number is copied from an artifact that already
computed it. OOS reads stay 0 in every epoch.

## Commands

```
make iteration-plan        # print the guard-checked ladder; compute nothing
make iteration-selftest    # engine guarantees + safety kernel, convergence, repair, budget
make iteration-start ASSETS="AZO ADBE GOOG" WORKERS=3   # detached ladder walk (survives terminal close)
make iteration-status      # epochs, convergence, budget, tmux liveness
make iteration-stop        # cooperative halt — finishes the current epoch, then stops (never pkill)
make iteration-report      # (re)generate iteration_summary.md
make iteration-smoke       # 2-version ladder on three assets, in-process (deterministic validation)
```

## Built vs specified

**Built:** the nested loop, the ladder with its policy and safety kernel, the integrity gate, the
repair accounting, convergence, the trace and self-summary, per-epoch per-asset compilation, the five
terminal outcomes, the `iteration-*` commands, and a two-version default ladder (base + a coarser
operating-point hypothesis). **Specified, unchanged:** Rung 7/9 remain `SPECIFIED / UNVALIDATED`;
longer ladders are user-extensible in the policy. The point of the loop is methodological knowledge and
an auditable record of its corrections — not the existence of models.

## Limitations — what the fast gate does and does not cover

**The fast smoke gate (`make iteration-smoke`, `RESEARCH_SMOKE_PERMS` / `FOLDS`) no longer reaches Rung 6.**
After the fail-closed fix — a reduced-strength null verdicts `smoke_pass`, never scientific `passed` — a smoke
run leaves every asset at `RESOLVED_EMPTY` after Rung 5: its `passed_arms` set is empty, so nothing reaches
`NULL_VALIDATED`, so the Rung-6 survivor-HPO dispatch is a no-op. This is correct (a smoke must not
manufacture a confirmation it had no power to make), but it means the **last gate of the chain is exercised
only by a full-strength run**, not by the minutes-long smoke. The smoke validates orchestration (rungs 1–5
dispatch, the reducer, the state machine, byte-level integrity); Rung 6's own scientific behavior is covered
by a full golden run plus the field-level guard tests in `engine/iteration_selftest.py` — stated here rather
than left implicit.

**Estimator scope.** Everything above is `XGB = VALIDATED`. `LSTM = PENDING`: the ladder is written to carry
to an LSTM (`docs/FEATURE_DISCOVERY_METHODOLOGY.md` §6), but no LSTM result has been produced through it.
