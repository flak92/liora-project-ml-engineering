#!/usr/bin/env python3
"""Prove the Iterative Calibration Loop's guarantees — fast, without running a science runner.

It runs the engine selftest (the inner loop's guarantees) and then adds the outer loop's: the safety
kernel refuses to loosen the proof standard, convergence stops exactly when a hypothesis adds nothing,
the Repair Loop classifies technical failures correctly, the budget cap and ladder guard behave, and
integrity actually catches tampering. None of it touches XGBoost.

    make iteration-selftest
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract_patch as CP                                                 # noqa: E402
import integrity as IG                                                      # noqa: E402
import iteration_planner as IP                                              # noqa: E402
import repair as RP                                                         # noqa: E402
from exec_ledger import ExecLedger                                          # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_engine_selftest():
    """The inner loop's 8 guarantees must still hold — run the existing engine selftest as a subprocess."""
    print("\n0. engine-selftest (gwarancje pętli wewnętrznej)")
    p = subprocess.run([sys.executable, str(ROOT / "engine" / "selftest.py")],
                       capture_output=True, text=True, timeout=300)
    # match the status-line marker "  FAIL  " (2+2 spaces), not the substring inside FAILED_INTEGRITY
    ok = p.returncode == 0 and "WSZYSTKO ZDANE" in p.stdout and "  FAIL  " not in p.stdout
    check("engine-selftest przechodzi (8 grup)", ok,
          "" if ok else "rc=%s; ostatnie: %s" % (p.returncode, p.stdout.strip().splitlines()[-3:]))


def test_patch_guard():
    """The kernel admits only hypothesis-space patches and reproduces the base hash on an empty one."""
    print("\n1. strażnik patcha: tylko przestrzeń hipotez, standard dowodu zamrożony")
    import contract as CT
    _p, h0, t0 = CP.apply({})
    check("pusty patch == fingerprint bazy", h0 == CT.contract_fingerprint() and not t0)
    for k in CP.FROZEN:
        try:
            CP.guard({k: {}}); check(f"odrzuca zamrożony {k}", False, "PRZYJĄŁ!")
        except CP.PatchRejected:
            check(f"odrzuca zamrożony {k}", True)
    for k in ("runtime", "certification", "brand_new_top"):
        try:
            CP.guard({k: {}}); check(f"odrzuca poza-hipotezę {k}", k not in CP.ADMISSIBLE and False)
        except CP.PatchRejected:
            check(f"odrzuca poza-hipotezę {k}", True)
    for k in CP.ADMISSIBLE:
        try:
            CP.guard({k: {}}); check(f"przyjmuje hipotezę {k}", True)
        except CP.PatchRejected as e:
            check(f"przyjmuje hipotezę {k}", False, str(e))
    _pv, hv, _tv = CP.apply({"operating_point": {"grid": [0.8, 0.9]}})
    check("patch operating_point zmienia hash i jest self-consistent",
          hv != h0 and CP._hash(_pv) == hv)

    # FIELD-LEVEL guard: a FROZEN leaf nested inside an ADMISSIBLE section must be rejected, while an
    # admissible leaf and an empty section-variant placeholder pass. (regression: the hole where
    # rung_6_survivor_hpo being admissible let a patch set own_null.permutations=5, loosening M=50.)
    def _guard_is(name, patch, want_pass):
        try:
            CP.guard(patch); got = True
        except CP.PatchRejected:
            got = False
        check(name, got == want_pass, "PRZESZŁO" if got else "ODRZUCONE")
    _guard_is("pole: rung_6.alpha ODRZUCONE (była dziura)", {"rung_6_survivor_hpo": {"alpha": 0.5}}, False)
    _guard_is("pole: rung_6.own_null.permutations=5 ODRZUCONE (2 poziomy w głąb)",
              {"rung_6_survivor_hpo": {"own_null": {"permutations": 5}}}, False)
    _guard_is("pole: operating_point.mode ODRZUCONE (structural)", {"operating_point": {"mode": "topk"}}, False)
    _guard_is("pole: operating_point.grid PRZECHODZI", {"operating_point": {"grid": [0.8, 0.9, 0.95]}}, True)
    _guard_is("pole: model_space.hpo_trials=60 PRZECHODZI", {"model_space": {"hpo_trials": 60}}, True)
    _guard_is("pole: puste rung_6 (extension placeholder) PRZECHODZI", {"rung_6_survivor_hpo": {}}, True)
    _guard_is("pole: data_boundary top-level nadal ODRZUCONE", {"data_boundary": {"oos_start": "9999-01-01"}}, False)
    # a hypothesis patch must not WRITE into a result/status/provenance subtree — those are the fields the
    # inventory skips, so without this the skip would be a hole into the audit record. (regression: a patch
    # could set rung_6_survivor_hpo.result.own_null_permutations, the record verify_calibration_docs reads.)
    _guard_is("rekord: rung_6.result.own_null_permutations ODRZUCONE (edycja rekordu audytu)",
              {"rung_6_survivor_hpo": {"result": {"own_null_permutations": 5}}}, False)
    _guard_is("rekord: model_space._note (proweniencja) ODRZUCONE", {"model_space": {"_note": "x"}}, False)

    # self-policing inventory: the field-level allowlist cannot silently rot. The current contract is
    # fully classified; a NEW field under an admissible section is unclassified (contract_lint reddens),
    # so a human must declare it — e.g. a future LSTM viability probe must not default to tunable.
    import contract_loader as CL2
    unc0, stale0 = CP.classify_admissible_leaves(CL2.assemble())
    check("inwentarz: bieżący kontrakt w pełni sklasyfikowany", not unc0 and not stale0,
          f"unc={sorted(unc0)} stale={sorted(stale0)}")
    probed = CL2.assemble(); probed["model_space"]["lstm_probe_threshold"] = 0.5
    unc1, _ = CP.classify_admissible_leaves(probed)
    check("inwentarz: nowe pole pod model_space → niesklasyfikowane (lint by zczerwieniał)",
          "model_space.lstm_probe_threshold" in unc1)
    frozen_probe = CL2.assemble(); frozen_probe.setdefault("viability", {})["lstm_probe_threshold"] = 0.5
    unc2, _ = CP.classify_admissible_leaves(frozen_probe)
    check("inwentarz: sonda pod viability (mit#1) → NIE flagowana (warstwa 1 top-level FROZEN chroni)",
          "viability.lstm_probe_threshold" not in unc2)


def test_convergence():
    """convergence_update + should_converge stop exactly at the first variant that adds nothing."""
    print("\n2. zbieżność: stop gdy kolejna hipoteza nie dodaje cechy")
    cum = set()
    cum, d0, ni0 = IP.convergence_update(cum, {("A", "x")}, 0)
    check("e0 dodaje (A,x): delta=1, streak=0", d0 == [("A", "x")] and ni0 == 0)
    check("e0 nie kończy (baza nigdy nie zbiega)", not IP.should_converge(0, ni0, 1))
    cum1, d1, ni1 = IP.convergence_update(cum, {("A", "x")}, ni0)     # variant adds nothing
    check("e1 bez nowości: delta=0, streak=1", d1 == [] and ni1 == 1)
    check("e1 CONVERGED przy patience=1", IP.should_converge(1, ni1, 1))
    _c2, d2, ni2 = IP.convergence_update(cum, {("A", "y")}, ni0)      # variant adds a new one
    check("wariant z nową cechą NIE zbiega", d2 == [("A", "y")] and not IP.should_converge(1, ni2, 1))
    _cb, _db, nib = IP.convergence_update(set(), set(), 0)            # barren base
    check("jałowa baza nie ogłasza zbieżności (k=0)", not IP.should_converge(0, nib, 1))


def test_repair_mining():
    """The Repair Loop classifies a synthetic exec ledger into repaired / retry / quarantine / failed."""
    print("\n3. Repair Loop: klasyfikacja awarii z exec_ledger")
    with tempfile.TemporaryDirectory() as d:
        led = ExecLedger(d)
        def task(a, h): return {"asset": a, "rung": 1, "task_hash": h}
        led.failed(task("A", "h1"), "w", 1.0, 1, "transient"); led.done(task("A", "h1"), "w", 1.0, 0, "sha")
        led.failed(task("B", "h2"), "w", 1.0, 95, "integrity")
        led.failed(task("C", "h3"), "w", 1.0, 1, "t"); led.failed(task("C", "h3"), "w", 1.0, 1, "t")
        led.failed(task("D", "h4"), "w", 1.0, 1, "t")
        # E: completed, THEN re-run to a divergent non-reproducible artifact (exit 95). The latched
        # `completed` must NOT mask the integrity failure — last event decides. (regression: review)
        led.done(task("E", "h5"), "w", 1.0, 0, "sha"); led.failed(task("E", "h5"), "w", 1.0, 95, "integrity")
        cls = RP.classify(d)
        diag = {rec["asset"]: RP.diagnose(rec, 2) for rec in cls.values()}
        check("A failed→completed = repaired", diag.get("A") == "repaired", str(diag))
        check("B exit95 = quarantine_integrity", diag.get("B") == "quarantine_integrity")
        check("C 2×fail (>=max_retries) = failed_technical", diag.get("C") == "failed_technical")
        check("D 1×fail (<max_retries) = safe_retry", diag.get("D") == "safe_retry")
        check("E completed-THEN-exit95 = quarantine_integrity (nie 'repaired')",
              diag.get("E") == "quarantine_integrity", "diag_E=%s" % diag.get("E"))
        # handle() requeues D: create its failed queue file, expect it moved to pending
        q = Path(d) / "queue"
        (q / "failed").mkdir(parents=True, exist_ok=True); (q / "pending").mkdir(parents=True, exist_ok=True)
        (q / "failed" / "h4.json").write_text("{}")
        RP.handle(d, 2)
        check("safe_retry przenosi failed→pending", (q / "pending" / "h4.json").exists()
              and not (q / "failed" / "h4.json").exists())


def test_budget_cap():
    """core_hours_cap converts to seconds; null means no cap."""
    print("\n4. budżet: limit rdzeniogodzin → HALTED_BUDGET")
    check("cap 0.001h = 3.6s", IP._core_cap_seconds({"budget": {"core_hours_cap": 0.001}}) == 3.6)
    check("cap null = brak limitu", IP._core_cap_seconds({"budget": {"core_hours_cap": None}}) is None)
    with tempfile.TemporaryDirectory() as d:
        led = ExecLedger(d)
        led.done({"asset": "A", "rung": 1, "task_hash": "h"}, "w", 10.0, 0, "sha")
        spent = IP.core_seconds(d)
        check("core_seconds sumuje completed (10s)", spent == 10.0)
        check("spent przekracza cap → halt-warunek", spent >= IP._core_cap_seconds({"budget": {"core_hours_cap": 0.001}}))


def test_ladder_guard():
    """ladder_from_policy prepends base and rejects an inadmissible ladder up front."""
    print("\n5. drabina: baza + wyłącznie dopuszczalne wersje, fail-fast")
    good = {"ladder": [{"version_id": "v1", "patch": {"operating_point": {"grid": [0.9]}}}]}
    rungs = IP.ladder_from_policy(good)
    check("baza doklejona, 2 wersje", len(rungs) == 2 and rungs[0]["version_id"] == "base")
    bad = {"ladder": [{"version_id": "bad", "patch": {"acceptance": {"min_win_rate": 0.0}}}]}
    try:
        IP.ladder_from_policy(bad); check("odrzuca niedopuszczalną drabinę", False, "PRZYJĄŁ!")
    except CP.PatchRejected:
        check("odrzuca niedopuszczalną drabinę (fail-fast)", True)


def test_integrity_tampering():
    """integrity.verify is green on a real snapshot and red when the contract hash or an artifact is tampered."""
    print("\n6. integralność wykrywa manipulację")
    with tempfile.TemporaryDirectory() as d:
        rd = Path(d) / "run"
        CP.snapshot_version(rd, ["A"], "base", {}, seed=42, allow_dirty=True)
        ok, rep = IG.verify(rd)
        check("świeży snapshot: integralność zielona", ok, str(rep["problems"]))
        snap = json.loads((rd / "contract.json").read_text())
        snap["contract_hash"] = "deadbeef"
        (rd / "contract.json").write_text(json.dumps(snap))
        ok2, rep2 = IG.verify(rd)
        check("podmieniony contract_hash → czerwona", not ok2 and "contract_self_consistent" in rep2["problems"])


def test_contract_injection():
    """A ladder patch must actually REACH the runners: RESEARCH_CONTRACT overrides the operating-point
    grid, the base snapshot reproduces the canonical grid bit-identically, and unset changes nothing.
    (regression: the review blocker — patches were provenance-only and every variant was a placebo.)"""
    import os
    import run_contract as RC
    import contract_loader as CL
    print("\n7. wstrzykiwanie kontraktu: patch drabiny realnie dociera do runnerów")
    Q = [float(x) for x in CL.assemble()["operating_point"]["grid"]]
    os.environ.pop("RC_ENV_SET", None)
    saved = os.environ.pop(RC.ENV, None)
    try:
        check("unset -> grid = kanoniczny default (bit-identyczny)",
              RC.operating_point_grid(Q) == Q and RC.contract() is None)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base"
            CP.snapshot_version(base, ["A"], "base", {}, allow_dirty=True)
            os.environ[RC.ENV] = str(base / "contract.json")
            check("baza snapshot -> grid == Q_GRID (epoka bazowa niezmieniona)",
                  RC.operating_point_grid(Q) == Q)
            pat = Path(d) / "pat"
            CP.snapshot_version(pat, ["A"], "coarser", {"operating_point": {"grid": [0.8, 0.9, 0.95]}},
                                allow_dirty=True)
            os.environ[RC.ENV] = str(pat / "contract.json")
            g = RC.operating_point_grid(Q)
            check("patch operating_point REALNIE zmienia grid runnera", g == [0.8, 0.9, 0.95] and g != Q)
    finally:
        os.environ.pop(RC.ENV, None)
        if saved is not None:
            os.environ[RC.ENV] = saved


def test_task_heartbeat():
    """A live long task refreshes its running-file mtime (worker heartbeat), so the guard's stale sweep
    (age >= STALE_TASK) never reclaims it into a concurrent duplicate; os.utime must NOT recreate a file
    the guard just moved. (regression: the 2h45m-null FAILED_INTEGRITY caused by stale-requeue.)"""
    import os
    import time
    print("\n8. heartbeat: żywy long-task nie requeue'owany + brak re-kreacji przeniesionego pliku")
    STALE = 5400
    with tempfile.TemporaryDirectory() as d:
        rf = Path(d) / "running.json"
        rf.write_text("{}")
        os.utime(rf, None)
        check("świeży mtime → nie stale", (time.time() - rf.stat().st_mtime) < STALE)
        os.utime(rf, (time.time() - 9900, time.time() - 9900))          # symuluj 2h45m bez heartbeatu
        check("2h45m bez heartbeatu → stale (age>=STALE)", (time.time() - rf.stat().st_mtime) >= STALE)
        os.utime(rf, None)                                              # heartbeat bije
        check("po heartbeacie → znów świeży, nie stale", (time.time() - rf.stat().st_mtime) < STALE)
        rf.unlink()                                                     # guard przeniósł plik
        try:
            os.utime(rf, None)
            recreated = rf.exists()
        except OSError:
            recreated = False
        check("os.utime na przeniesionym pliku NIE rekreuje (brak phantom running)", not recreated)


def test_confirmed_excludes_failed():
    """A survivor drawn from a FAILED_TECHNICAL asset (quarantined, non-reproducible null) must NOT count
    as a confirmed feature. (regression: iteration-smoke counted ADBE's quarantined survivor as the 4th.)"""
    import iteration_planner as IP
    import report as RE
    print("\n9. księgowanie: survivor z FAILED_TECHNICAL nie liczy się jako potwierdzony")
    orig_f, orig_s = RE.funnel, IP.epoch_full_strength
    RE.funnel = lambda src: {"stable_units": ["ADBE/momentum_return", "GOOG/205"]}
    IP.epoch_full_strength = lambda ed: (True, "ok")     # izoluj filtr wykluczeń od guardrailu siły
    try:
        allp = IP.confirmed_pairs("/nonexistent")
        check("bez wykluczeń: obie pary", allp == {("ADBE", "momentum_return"), ("GOOG", "205")}, str(allp))
        filt = IP.confirmed_pairs("/nonexistent", exclude_assets={"ADBE"})
        check("ADBE FAILED_TECHNICAL wykluczony → tylko GOOG", filt == {("GOOG", "205")}, str(filt))
    finally:
        RE.funnel, IP.epoch_full_strength = orig_f, orig_s


def test_determinism_guard():
    """Byte-level reproducibility (RATIFIED, FROZEN) holds iff the determinism conditions hold. Assert the
    code enforces exactly what the contract declares — a regression that unpinned any of these would
    reintroduce the max-null false-alarm (PYTHONHASHSEED-driven bytes)."""
    import contract_loader as CL
    import runtime_init
    print("\n10. determinizm: kod egzekwuje warunki byte-reprodukowalności z kontraktu")
    rt = CL.assemble()["runtime"]
    applied = runtime_init.apply()
    check("runtime_init pinuje pule wątków = 1", all(applied[v] == "1" for v in runtime_init.THREAD_VARS))
    check("kontrakt: thread_pools = 1", all(v == 1 for v in rt["thread_pools"].values()))
    check("kontrakt: python_hash_seed = 0", rt.get("python_hash_seed") == 0)
    src = (ROOT / "engine" / "dispatch.py").read_text(encoding="utf-8")
    check("dispatch ustawia PYTHONHASHSEED w env runnera", 'PYTHONHASHSEED="0"' in src)
    check("kontrakt: standard = BYTE-LEVEL (guard nie luzowany)",
          "BYTE-LEVEL" in rt.get("_reproducibility_standard", ""))


def test_full_strength_guardrail():
    """A --permutations/--folds smoke may NEVER count as a confirmation. rung5_verdict.full_strength
    gates BOTH axes: full permutation budget AND full fold scope. (regression: the fast dev-gate #1.)"""
    import rung5_verdict as RV
    print("\n11. guardrail siły: smoke (capped null) nie może potwierdzać")
    cross = {"tables": {"A": {"folds": [{"outer_fold": 0, "verdict": {
        "flat": {"accepted": True, "T": 1.0}, "hierarchical": {"accepted": False, "T": 0}}}]}}}
    full = {"contract": {"permutations_max": 50},
            "tables": {"A": {"folds": [{"outer_fold": 0, "arms": {}}]}}}
    ok, _ = RV.full_strength(full, cross, 50)
    check("pełny (perms=50, foldy pokryte) → potwierdza", ok)
    capped = {"contract": {"permutations_max": 5},
              "tables": {"A": {"folds": [{"outer_fold": 0, "arms": {}}]}}}
    ok, r = RV.full_strength(capped, cross, 50)
    check("perms=5 < 50 → odrzucony (nie potwierdza)", not ok, r)
    foldcap = {"contract": {"permutations_max": 50}, "tables": {}}
    ok, r = RV.full_strength(foldcap, cross, 50)
    check("foldy niepokryte (--folds cap) → odrzucony", not ok, r)


def test_seal_science_only():
    """A sealed artifact holds ONLY science — operational fields (timing) never enter the hashed result,
    so result_sha256 is byte-identical across re-runs. (regression: `seconds` was in the hashed result.)"""
    import json as _json
    import dispatch as DP
    print("\n12. pieczęć = tylko nauka: dispatch._entry usuwa 'seconds' (byte-repro result_sha256)")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "null.json"
        p.write_text(_json.dumps({"tables": {"X": {"ticker": "X", "seconds": 9.9, "folds": [1, 2]}}}))
        e = DP._entry(str(p), "X")
        check("'seconds' (timing) usunięte z entry", "seconds" not in e, str(e))
        check("nauka zachowana (folds)", e.get("folds") == [1, 2])


def test_rung6_stable_survivors():
    """Rung 6 is fed the STABLE set A1∩A2∩B, never A1 alone: dispatch._stable_entry keeps only stable
    arms, dropping one that passed A1 but was rejected by A2/B. (regression: P0-1 — Rung 6 fed the A1
    panel retained IDXX/2 208 and BKNG/0 volume, which A2/B had already rejected.)"""
    import dispatch as DP
    import rung5_verdict as RV
    print("\n13. Rung 6 dostaje A1∩A2∩B, nie samo A1 (P0-1)")

    def entry(arms):     # arms: {(outer_fold, arm, unit): verdict}
        folds = {}
        for (of, arm, unit), verd in arms.items():
            folds.setdefault(of, {"ticker": "A", "outer_fold": of, "arms": {}})
            folds[of]["arms"][arm] = {"verdict": verd, "unit": unit}
        return {"folds": list(folds.values())}

    a1 = entry({(0, "flat", "111"): "passed", (0, "hierarchical", "222"): "passed"})
    a2 = entry({(0, "flat", "111"): "passed", (0, "hierarchical", "222"): "rejected_early"})
    b = entry({(0, "flat", "111"): "passed", (0, "hierarchical", "222"): "passed"})
    stable = RV.stable_survivors(a1, a2, b)
    check("stable = tylko arm w A1∩A2∩B (flat/111)", stable == {(0, "flat", "111")}, str(stable))
    arms_fed = {(f["outer_fold"], arm) for f in DP._stable_entry(a1, stable)["folds"] for arm in f["arms"]}
    check("_stable_entry karmi TYLKO flat/111, odrzuca hierarchical/222 (padł A2)",
          arms_fed == {(0, "flat")}, str(arms_fed))
    check("brak a2/b → fail-closed pusty (nic do Rung 6)", RV.stable_survivors(a1, None, None) == set())


def test_rung6_threshold_from_alpha():
    """Rung 6's own-null pass threshold is DERIVED from alpha and the permutation count, not a constant
    calibrated to a different M. (regression: P0-2 — b<=4 was calibrated for M=50 but ran at M=20, where
    it admitted p_mc=5/21=0.238 against a declared alpha of 0.10.)"""
    import rung6_survivor_hpo as R6
    print("\n14. próg Rung 6 wywiedziony z alfy, nie stała pod inne M (P0-2)")
    check("alpha czytana z kontraktu = 0.10", R6.ALPHA == 0.1, f"ALPHA={R6.ALPHA}")
    check("M=20, alpha=0.10 → pass_b=1 (p_max=2/21=0.095 ≤ 0.10)", R6._pass_b(0.1, 20) == 1)
    check("M=50, alpha=0.10 → pass_b=4 (odtwarza skalibrowaną wartość)", R6._pass_b(0.1, 50) == 4)
    check("stary błąd udokumentowany: b≤4 przy M=20 dopuszczał p=5/21=0.238 > alpha", round(5 / 21, 3) == 0.238)


def test_report_retained_intersect():
    """The funnel's retained is stable ∩ Rung-6-retained, and stable is fail-closed — a missing a2/b
    empties it, never falls back to A1. (regression: P1-4 — retained counted A2/B-rejected arms and
    stable fell OPEN to pa1 when a2/b were absent.)"""
    import report as RE
    print("\n15. raport: retained = stable ∩ rung6_retained, stable fail-closed (P1-4)")
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)

        def null(name, passed):
            (dd / name).write_text(json.dumps({"tables": {"A": {"folds": [
                {"ticker": "A", "outer_fold": of, "arms": {arm: {"verdict": "passed", "unit": unit}}}
                for (of, arm, unit) in passed]}}}))

        (dd / "crossfit_selection.json").write_text(json.dumps({"tables": {"A": {"folds": [
            {"outer_fold": 0, "verdict": {"flat": {"accepted": True}, "hierarchical": {"accepted": True}}}]}}}))
        null("procedure_null_a1.json", [(0, "flat", "111"), (0, "hierarchical", "222")])
        null("procedure_null_a2.json", [(0, "flat", "111")])                       # 222 fails A2
        null("procedure_null_b.json", [(0, "flat", "111")])
        (dd / "rung6_survivor_hpo.json").write_text(json.dumps({"results": [
            {"ticker": "A", "outer_fold": 0, "arm": "flat", "unit": "111", "verdict": "retained"},
            {"ticker": "A", "outer_fold": 0, "arm": "hierarchical", "unit": "222", "verdict": "retained"}]}))
        fn = RE.funnel(dd)
        check("stable = 1 (flat/111; hierarchical/222 padł A2)", fn["stable_a1_a2_b"] == 1, str(fn["stable_units"]))
        check("retained = stable ∩ rung6 = 1 (222 'retained' NIE liczony)", fn["retained_rung6"] == 1,
              str(fn["retained_units"]))
        (dd / "procedure_null_a2.json").unlink()                                    # a2 znika
        fn2 = RE.funnel(dd)
        check("brak a2 → stable=0 (fail-closed, NIE fallback do A1=2)", fn2["stable_a1_a2_b"] == 0,
              str(fn2["stable_units"]))


def test_smoke_pass_not_science():
    """A reduced (smoke) null verdicts `smoke_pass`, never `passed`; rung5_verdict.passed_arms keys
    strictly on `passed`, so a smoke never enters the science as a survivor and cannot reach
    NULL_VALIDATED. (regression: P1-5 — a --permutations 5 run emitted `passed`.)"""
    import rung5_verdict as RV
    print("\n16. smoke ≠ passed: reduced → smoke_pass, selektor nauki go pomija (P1-5)")
    smoke = {"folds": [{"outer_fold": 0, "arms": {"flat": {"verdict": "smoke_pass", "unit": "111"}}}]}
    full = {"folds": [{"outer_fold": 0, "arms": {"flat": {"verdict": "passed", "unit": "111"}}}]}
    check("passed_arms POMIJA smoke_pass (0 survivorów)", RV.passed_arms(smoke) == set())
    check("passed_arms liczy passed (1 survivor)", RV.passed_arms(full) == {(0, "flat", "111")})
    check("smoke a1 → brak stabilnych → NIE NULL_VALIDATED",
          not RV.null_validated({"a1": smoke, "a2": full, "b": full}))
    check("pełny a1∩a2∩b → NULL_VALIDATED",
          RV.null_validated({"a1": full, "a2": full, "b": full}))


def main():
    print("iteration-selftest — gwarancje Iterative Calibration Loop (bez uruchamiania nauki)")
    for t in (test_engine_selftest, test_patch_guard, test_convergence, test_repair_mining,
              test_budget_cap, test_ladder_guard, test_integrity_tampering, test_contract_injection,
              test_task_heartbeat, test_confirmed_excludes_failed, test_determinism_guard,
              test_full_strength_guardrail, test_seal_science_only,
              test_rung6_stable_survivors, test_rung6_threshold_from_alpha,
              test_report_retained_intersect, test_smoke_pass_not_science):
        try:
            t()
        except Exception as e:                                              # noqa: BLE001
            check(f"{t.__name__} rzucił wyjątek", False, f"{type(e).__name__}: {e}")
    print("\n" + ("WSZYSTKO ZDANE" if not FAILS else f"NIEZDANE: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
