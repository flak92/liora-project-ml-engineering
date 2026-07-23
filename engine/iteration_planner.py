#!/usr/bin/env python3
"""The Iterative Calibration Loop's orchestrator — the planner of ITERATIONS, not of experiments.

`planner.py` names the smallest next EXPERIMENT for one asset under one frozen contract. This module
sits a level above it: it names the smallest next HYPOTHESIS. It walks a human-pre-authorized ladder
of frozen contract versions, drives each version's inner loop to a fixpoint (every asset terminal),
compiles what that hypothesis confirmed, and stops the moment a new hypothesis space adds no new
confirmed features — "brak widocznych popraw po kolejnych próbach". It never invents a rung and never
loosens a criterion: the ladder is finite and human-authored, and every version passes through the
`contract_patch` safety kernel, which forbids touching the proof standard.

Two nested loops, three layers:

  outer (this file)   for each pre-authorized contract version: snapshot it (kernel) -> run inner ->
                      compile -> compare to the cumulative confirmed set -> converge or advance.
  inner (per epoch)   verify_global_integrity -> repair technical failures -> derive states ->
                      plan smallest experiment -> enqueue -> drain -> until every asset terminal.

  Scientific Decision Loop  planner.next_action        (unchanged science)
  Execution Loop            engine.sh / worker / queue (reused; inproc backend for tests)
  Repair Loop               repair.py                  (technical only, never scientific)

Terminal outcomes of an asset: RESOLVED_RETAINED, RESOLVED_EMPTY, NEEDS_CONTRACT (science stop),
FAILED_TECHNICAL (execution stop). Terminal outcomes of the whole loop: CONVERGED, LADDER_EXHAUSTED,
HALTED_BUDGET, HALTED (cooperative stop), INTEGRITY_FAILED.

Execution backends:
  inproc    the orchestrator drains the queue itself with one worker — deterministic, what the smoke
            and selftest use, no tmux, no engine.sh.
  external  the orchestrator shells `ops/engine.sh` per epoch (the proven detached supervisor) and
            waits for it to bring every asset terminal — what the detached tmux deployment uses.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                       # noqa: E402
import contract_patch as CP                                                 # noqa: E402
import integrity as IG                                                      # noqa: E402
import planner as PL                                                        # noqa: E402
import reducer as RD                                                        # noqa: E402
import repair as RP                                                         # noqa: E402
import report as RE                                                         # noqa: E402
import states as ST                                                         # noqa: E402
import asset_driver as AD                                                   # noqa: E402
import worker as WK                                                         # noqa: E402
from artifact_io import read_json, write_json_atomic                        # noqa: E402
from ledger import Ledger                                                   # noqa: E402

SCI_TERMINAL = {"RESOLVED", "RESOLVED_EMPTY", "NEEDS_CONTRACT"}
POLICY = ROOT / "config" / "iteration_loop_policy.json"


# ---- policy + ladder ------------------------------------------------------------------------------

def load_policy(path=None):
    return json.loads(Path(path or POLICY).read_text(encoding="utf-8"))


def ladder_from_policy(policy):
    """The ordered list of versions to walk. Always starts at the base contract (empty patch); each
    later rung is a pre-authorized hypothesis patch. Every patch is guard-checked HERE, before any
    compute, so an inadmissible ladder fails fast instead of mid-run."""
    rungs = [{"version_id": "base", "patch": {}, "rationale": "kontrakt bazowy"}]
    for v in policy.get("ladder", []):
        patch = v.get("patch", {})
        CP.guard(patch)                                     # fail fast on an inadmissible ladder
        rungs.append({"version_id": v["version_id"], "patch": patch,
                      "rationale": v.get("rationale", "")})
    return rungs


# ---- state, evidence, budget ----------------------------------------------------------------------

def states_map(epoch_dir, assets, max_retries):
    """Per-asset state, with a FAILED_TECHNICAL overlay from the Repair Loop. The scientific state
    comes purely from artifacts (states.derive_state); a terminal technical failure is layered on top
    and kept explicitly distinct — a broken machine is never a scientific verdict."""
    tech = RP.technical_terminal(epoch_dir, max_retries)
    out = {}
    for a in assets:
        if a in tech:
            out[a] = "FAILED_TECHNICAL"
        else:
            out[a] = ST.derive_state(epoch_dir, a)["state"]
    return out


def is_loop_terminal(state):
    return state in SCI_TERMINAL or state == "FAILED_TECHNICAL"


def label_outcome(epoch_dir, asset, state):
    """RESOLVED with a retained survivor is reported as RESOLVED_RETAINED — the loop's success state."""
    if state == "RESOLVED":
        art = ST.read_artifact(epoch_dir, 6, asset)
        ev = ST.derive_state(epoch_dir, asset).get("evidence", {})
        if ev.get("retained_any") or (art and art.get("result", {}).get("retained")):
            return "RESOLVED_RETAINED"
    return state


def epoch_full_strength(epoch_dir):
    """Did this epoch's null EARN the right to confirm? rung5_verdict.full_strength on the panels vs the
    run's FROZEN permutations_max. A fast smoke (--permutations/--folds) is not full strength — its
    survivors validate orchestration, never science. Returns (ok, reason)."""
    import rung5_verdict as RV
    src = Path(epoch_dir) / "results" / "panels"
    try:
        frozen_max = int(CT.load(epoch_dir)["contract"]["max_null"]["permutations_max"])
    except Exception as e:                                                  # noqa: BLE001
        # Fail-closed: an unverifiable proof standard confirms NOTHING (never guess a max).
        return False, f"nie odczytano frozen permutations_max ({type(e).__name__}) — fail-closed"
    return RV.full_strength(read_json(src / "procedure_null_a1.json"),
                            read_json(src / "crossfit_selection.json"), frozen_max)


def confirmed_pairs(epoch_dir, exclude_assets=()):
    """The (asset, feature) pairs this epoch confirmed = null-validated stable survivors (A1∩A2∩B).
    Rung 6 refines survivors but adds no new feature, so this is the monotone 'what did this hypothesis
    space prove' set the convergence metric compares across epochs.

    Two filters guard the confirmed set:
    - A run below FULL strength (a --permutations/--folds smoke) confirms NOTHING — it had no power to.
    - An asset in `exclude_assets` (FAILED_TECHNICAL — its null was quarantined) contributes nothing.
    The funnel is built from the first published artifact and is unaware of either, so both are applied
    here, at the point the confirmed set is formed."""
    ok, _reason = epoch_full_strength(epoch_dir)
    if not ok:
        return set()                                    # smoke / capped run — mechanics only, no science
    src = Path(epoch_dir) / "results" / "panels"
    excl = set(exclude_assets)
    pairs = set()
    for s in RE.funnel(src).get("stable_units", []):
        parts = s.split("/")
        if len(parts) >= 2 and parts[0] not in excl:
            pairs.add((parts[0], parts[-1]))                # (ticker, unit)
    return pairs


def core_seconds(epoch_dir):
    """Core-seconds actually spent in an epoch = sum of completed exec-ledger durations (measurement,
    the convention this tree already uses; there is no core-hour cap anywhere else)."""
    total = 0.0
    for r in RP.ExecLedger(epoch_dir).read_all():
        if r.get("status") == "completed":
            total += float(r.get("payload", {}).get("seconds", 0) or 0)
    return total


# ---- cooperative stop + trace ---------------------------------------------------------------------

def halt_requested(ladder_dir):
    ctl = Path(ladder_dir) / "control.json"
    if not ctl.exists():
        return False
    try:
        return bool(json.loads(ctl.read_text(encoding="utf-8")).get("halt"))
    except (json.JSONDecodeError, OSError):
        return False


def _trace(ladder_dir, kind, payload):
    """Append one hash-chained line to iteration_trace.jsonl (audit) — the same ledger primitive the
    exec/method ledgers use. Kept compact to stay under the 4 KiB per-line budget."""
    Ledger(Path(ladder_dir) / "iteration_trace.jsonl").append("iteration", {"kind": kind}, "completed",
                                                              payload=payload)


def render_trace(ladder_dir):
    """Render the append-only trace into iteration_trace.json for a human/report to read at a glance."""
    led = Ledger(Path(ladder_dir) / "iteration_trace.jsonl")
    ok, bad = led.verify_chain()
    doc = {"ladder_dir": str(ladder_dir), "trace_chain_ok": ok, "trace_first_bad": bad,
           "events": [{"kind": r["unit"].get("kind"), "ts_utc": r["ts_utc"], **r["payload"]}
                      for r in led.read_all()]}
    write_json_atomic(Path(ladder_dir) / "iteration_trace.json", doc)
    return doc


# ---- inner loop (one frozen contract to fixpoint) -------------------------------------------------

def run_epoch(epoch_dir, assets, policy, ladder_dir, k, version_id, budget_spent_before):
    """Drive one epoch to a fixpoint with the parallel-over-assets driver (engine/asset_driver): each
    asset runs its DAG 1→3→4→5→6 in a PRIVATE workspace (no queue, no shared write, no guard/scheduler),
    the pool is RAM-capped and honours the epoch wall deadline and the cooperative halt. The integrity
    gate (oos_reads==0, boundary markers, self-consistent hash) is PRESERVED — it runs AFTER the panel
    and fails the epoch closed. The Repair Loop's technical-terminal overlay still applies via
    states_map; a dead child is re-run once inside the driver (idempotent via immutable artifacts).
    Byte-identical to the old queue path (validated by a per-asset result_sha256 parity harness on a
    three-asset smoke). Returns (outcome, states)."""
    max_retries = int(policy.get("repair", {}).get("max_retries", 2))
    if halt_requested(ladder_dir):
        return "HALTED", states_map(epoch_dir, assets, max_retries)
    control = str(Path(ladder_dir) / "control.json")
    wall_h = policy.get("budget", {}).get("wall_hours_per_epoch")
    deadline = (time.time() + float(wall_h) * 3600.0) if wall_h else None
    seed = int(CT.load(epoch_dir).get("seed", 42))
    max_parallel = policy.get("execution", {}).get("workers")
    AD.run_panel(str(epoch_dir), assets, seed=seed, control=control,
                 deadline_epoch=deadline, max_parallel=max_parallel)
    if halt_requested(ladder_dir):
        return "HALTED", states_map(epoch_dir, assets, max_retries)
    ok, ig = IG.verify(epoch_dir)
    if not ok:
        _trace(ladder_dir, "integrity", {"epoch": k, "version_id": version_id,
                                         "problems": ig["problems"]})
        return "INTEGRITY_FAILED", states_map(epoch_dir, assets, max_retries)
    st = states_map(epoch_dir, assets, max_retries)
    return ("EPOCH_DONE" if all(is_loop_terminal(s) for s in st.values()) else "EPOCH_STALLED"), st


# ---- outer loop (walk the ladder) -----------------------------------------------------------------

def _core_cap_seconds(policy):
    cap = policy.get("budget", {}).get("core_hours_cap")
    return None if cap in (None, 0) else float(cap) * 3600.0


def convergence_update(cumulative, pairs, no_improve):
    """Fold one epoch's confirmed pairs into the cumulative set. Returns (new_cumulative, delta,
    no_improve_streak). Pure — the selftest checks it without running an epoch."""
    delta = sorted(pairs - cumulative)
    return (cumulative | pairs), delta, (no_improve + 1 if not delta else 0)


def should_converge(k, no_improve, patience):
    """Stop only after at least one variant (k>=1) has added nothing for `patience` epochs — so a
    barren base epoch still gets the next hypothesis tried rather than declaring victory early."""
    return k >= 1 and no_improve >= patience


def run_ladder(ladder_dir, assets, seed=42, mode="inproc", allow_dirty=False, policy=None):
    """Walk the pre-authorized ladder. Returns a result dict; writes iteration_trace.{jsonl,json}."""
    ladder_dir = Path(ladder_dir)
    ladder_dir.mkdir(parents=True, exist_ok=True)
    policy = policy or load_policy()
    rungs = ladder_from_policy(policy)                      # guard-checks every patch up front
    patience = int(policy.get("convergence", {}).get("patience", 1))

    manifest = {"ladder_dir": str(ladder_dir), "mode": mode, "assets": list(assets),
                "rungs": [r["version_id"] for r in rungs], "epochs": []}
    write_json_atomic(ladder_dir / "ladder.json", manifest)
    _trace(ladder_dir, "ladder_start", {"assets": list(assets), "rungs": manifest["rungs"],
                                        "patience": patience})

    cumulative = set()
    no_improve = 0
    budget_spent = 0.0
    outcome = "LADDER_EXHAUSTED"

    for k, rung in enumerate(rungs):
        if halt_requested(ladder_dir):
            outcome = "HALTED"
            break
        epoch_dir = ladder_dir / "epochs" / f"e{k}_{rung['version_id']}"
        snap = CP.snapshot_version(epoch_dir, assets, rung["version_id"], rung["patch"],
                                   seed=seed, allow_dirty=allow_dirty)
        _trace(ladder_dir, "epoch_start", {"epoch": k, "version_id": rung["version_id"],
                                           "contract_hash": snap["contract_hash"][:16],
                                           "patch_touched": snap["iteration_patch_touched"],
                                           "rationale": rung["rationale"]})

        ep_outcome, st = run_epoch(epoch_dir, assets, policy, ladder_dir, k, rung["version_id"],
                                   budget_spent)
        budget_spent += core_seconds(epoch_dir)

        # A FAILED_TECHNICAL asset (non-reproducible/quarantined null) contributes no confirmed feature.
        failed_tech = {a for a, s in st.items() if s == "FAILED_TECHNICAL"}
        pairs = confirmed_pairs(epoch_dir, exclude_assets=failed_tech)
        # Only a fixpoint epoch (every asset terminal) may update the convergence state. A stalled or
        # halted epoch's confirmed set is incomplete, so folding it in could undercount a variant's
        # features and fire a false CONVERGED. Record its delta for audit, but leave the state alone.
        if ep_outcome == "EPOCH_DONE":
            cumulative, delta, no_improve = convergence_update(cumulative, pairs, no_improve)
        else:
            delta = sorted(pairs - cumulative)
        labeled = {a: label_outcome(epoch_dir, a, s) for a, s in st.items()}

        epoch_rec = {"epoch": k, "version_id": rung["version_id"],
                     "contract_hash": snap["contract_hash"][:16],
                     "epoch_outcome": ep_outcome, "states": labeled,
                     "confirmed_this_epoch": sorted(f"{a}/{u}" for a, u in pairs),
                     "delta_new": [f"{a}/{u}" for a, u in delta],
                     "cumulative_confirmed": len(cumulative),
                     "no_improve_streak": no_improve,
                     "core_seconds": round(core_seconds(epoch_dir), 1),
                     "budget_spent_core_seconds": round(budget_spent, 1)}
        manifest["epochs"].append(epoch_rec)
        write_json_atomic(ladder_dir / "ladder.json", manifest)
        # The trace line has a 4 KiB cap (ledger); the full record lives in ladder.json. Store a
        # compact form here (counts + a capped delta sample) so a 20-asset panel never overflows.
        _trace(ladder_dir, "epoch_end", {
            "epoch": k, "version_id": rung["version_id"], "contract_hash": snap["contract_hash"][:16],
            "epoch_outcome": ep_outcome, "n_confirmed": len(pairs), "delta_sample": [f"{a}/{u}" for a, u in delta][:12],
            "delta_new": len(delta), "cumulative_confirmed": len(cumulative),
            "no_improve_streak": no_improve, "core_seconds": round(core_seconds(epoch_dir), 1),
            "n_terminal": sum(1 for s in st.values() if is_loop_terminal(s))})

        # Any non-fixpoint outcome stops the ladder BEFORE the convergence check — we never declare
        # convergence on an epoch that did not resolve every asset.
        if ep_outcome != "EPOCH_DONE":
            outcome = "STALLED" if ep_outcome == "EPOCH_STALLED" else ep_outcome
            break
        # Budget at the epoch boundary — so the cap halts the external backend too (engine.sh runs a
        # whole epoch to completion; its own wall deadline is separate from this core-hour cap).
        cap = _core_cap_seconds(policy)
        if cap is not None and budget_spent >= cap:
            outcome = "HALTED_BUDGET"
            break
        # Convergence: after at least one variant (k>=1), `patience` epochs adding nothing new -> stop.
        if should_converge(k, no_improve, patience):
            outcome = "CONVERGED"
            break

    result = {"outcome": outcome, "epochs_run": len(manifest["epochs"]),
              "cumulative_confirmed": sorted(f"{a}/{u}" for a, u in cumulative),
              "budget_spent_core_seconds": round(budget_spent, 1)}
    _trace(ladder_dir, "ladder_end", result)
    render_trace(ladder_dir)
    manifest["result"] = result
    write_json_atomic(ladder_dir / "ladder.json", manifest)
    return result


# ---- CLI ------------------------------------------------------------------------------------------

def status(ladder_dir):
    """Read-only snapshot of a ladder's progress from its manifest — for make iteration-status."""
    ladder_dir = Path(ladder_dir)
    mf = ladder_dir / "ladder.json"
    if not mf.exists():
        print(f"brak drabiny w {ladder_dir}")
        return 1
    m = json.loads(mf.read_text(encoding="utf-8"))
    print(f"drabina {ladder_dir.name}  tryb={m.get('mode')}  wersje={m.get('rungs')}  "
          f"halt={halt_requested(ladder_dir)}")
    for e in m.get("epochs", []):
        st = e.get("states", {})
        print(f"  e{e['epoch']} {e['version_id']:<20} {e.get('epoch_outcome',''):<14} "
              f"Δ={len(e.get('delta_new', []))} skum={e.get('cumulative_confirmed',0)} "
              f"streak={e.get('no_improve_streak',0)} koszt={e.get('core_seconds',0)}s")
        print(f"      stany: {st}")
    r = m.get("result")
    if r:
        print(f"  WYNIK: {r['outcome']}  potwierdzone={len(r.get('cumulative_confirmed', []))}  "
              f"koszt={r.get('budget_spent_core_seconds')}s")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Iterative Calibration Loop — ladder orchestrator.")
    ap.add_argument("--ladder-dir", required=True)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=("inproc", "external"), default="inproc")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--plan-only", action="store_true", help="wypisz drabinę i wyjdź (nic nie licz)")
    ap.add_argument("--status", action="store_true", help="wypisz stan drabiny (read-only) i wyjdź")
    args = ap.parse_args()

    if args.status:
        return status(args.ladder_dir)

    policy = load_policy(args.policy)
    assets = args.assets or json.loads((ROOT / "config" / "sample_20.json").read_text())["sample"]

    if args.plan_only:
        rungs = ladder_from_policy(policy)
        print(f"drabina ({len(rungs)} wersji), assety={len(assets)}, "
              f"patience={policy.get('convergence', {}).get('patience', 1)}, "
              f"core_hours_cap={policy.get('budget', {}).get('core_hours_cap')}")
        for k, r in enumerate(rungs):
            touched = ",".join(sorted(CP.guard(r["patch"]))) or "∅"
            print(f"  e{k} {r['version_id']:<20} touched={touched:<20} {r['rationale'][:70]}")
        return 0

    t0 = time.time()
    res = run_ladder(args.ladder_dir, assets, seed=args.seed, mode=args.mode,
                     allow_dirty=args.allow_dirty, policy=policy)
    print(f"\n=== drabina zakończona ({time.time()-t0:.0f}s): {res['outcome']} ===")
    print(f"  epok: {res['epochs_run']}  |  potwierdzone (asset/feature): "
          f"{len(res['cumulative_confirmed'])}  |  koszt: {res['budget_spent_core_seconds']} rdz-s")
    for p in res["cumulative_confirmed"]:
        print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
