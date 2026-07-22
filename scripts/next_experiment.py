#!/usr/bin/env python3
"""The automaton — reads the ladder's state and names the smallest experiment that could still change
the verdict. It advises; it never acts.

The objective the whole method serves is not "run more trials" but "reach the same verdict for the
least compute". That makes the central question, at every point, *what is the smallest experiment
that could still move the decision?* — and if none can within budget, stop. This file answers that
question by reading the artifacts each rung has already written and classifying the state.

Two rules keep it honest.

It only reads. It reports the next admissible experiment as a command a person runs, with the
evidence behind the recommendation. It does not launch anything.

It never auto-recalibrates. The owner's decision chain includes states like "the model cannot learn
— recalibrate the search space". Recalibrating the space to the panel is exactly the co-adaptation
the method forbids, so those states do not loop back into an automatic retune: they STOP and ask for
human authorization, because widening the space is a decision that mints a new contract version, not
a step in a loop. The automaton can drive the cheap deterministic forward path (viability → utility
→ cross-fit → null → survivor HPO); it halts at anything that would change the rules.

    python3 scripts/next_experiment.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init  # noqa: E402,F401
runtime_init.apply()
from artifact_io import read_json                                          # noqa: E402

DATA = XGB / "data"
PY = ".venv/bin/python3"

# Each state carries: whether it needs human authorization, the reason, and the next command.
HUMAN = "STOP — wymaga autoryzacji człowieka (nowa wersja kontraktu)"


def _viability_ok():
    d = read_json(DATA / "model_viability.json")
    if d is None:
        return None
    # Enough tables produced a learnable model under the v2 space? Read the register: a table is
    # viable if its median split-node count over v2 trials clears the floor.
    tables = d.get("tables", {})
    if not tables:
        return None
    viable = 0
    for rec in tables.values():
        recs = rec.get("spaces", {}).get("v2_hessian_relative", [])
        sn = sorted(x["split_nodes"] for x in recs) if recs else []
        med = sn[len(sn) // 2] if sn else 0
        if med >= 20:
            viable += 1
    return viable, len(tables)


def _has_marginal_candidates():
    d = read_json(DATA / "feature_utility.json")
    if d is None:
        return None
    for rec in d.get("tables", {}).values():
        for f in rec.get("folds", []):
            for s in f.get("singles", {}).values():
                if s.get("viable") and s.get("gain", 0) > 0.004:
                    return True
    return False


def _provisional_acceptances():
    d = read_json(DATA / "crossfit_selection.json")
    if d is None:
        return None
    n = 0
    for rec in d["tables"].values():
        for f in rec["folds"]:
            for arm in ("flat", "hierarchical"):
                if f["verdict"][arm].get("accepted"):
                    n += 1
    return n


def _null_state():
    """(survivors, rejected, still_provisional, authority) from the strongest null present."""
    for kind in ("procedure_null_a1.json", "procedure_null_a1_smoke.json"):
        d = read_json(DATA / kind)
        if d is None:
            continue
        surv = rej = 0
        for t in d["tables"].values():
            for f in t["folds"]:
                for v in f["arms"].values():
                    if v["verdict"] == "passed":
                        surv += 1
                    elif v["verdict"] == "rejected_early":
                        rej += 1
        return surv, rej, kind.replace("procedure_null_", "").replace(".json", "")
    return None


def _rung6_done():
    return (DATA / "rung6_survivor_hpo.json").exists()


def _cross_asset_done():
    d = read_json(DATA / "cross_asset_matrix.json")
    if d is None:
        return None
    return d.get("counts", {})


def classify():
    """Walk the decision chain in order; the first state that holds is the answer."""
    via = _viability_ok()
    if via is None:
        return {"state": "no_viability_register", "action": HUMAN,
                "why": "brak model_viability.json — uruchom Rung 1 najpierw",
                "next": f"{PY} scripts/model_viability.py --jobs 4"}
    viable, total = via
    if viable < max(1, total // 2):
        return {"state": "model_not_viable", "action": HUMAN,
                "why": f"tylko {viable}/{total} tabel ma uczący się model — przestrzeń HPO wymaga "
                       "rekalibracji, a to zmienia kontrakt",
                "next": "rekalibracja config/xgb_search_space_v2.json → nowa wersja kontraktu"}

    if _has_marginal_candidates() is None:
        return {"state": "no_utility_register", "action": "forward",
                "why": "brak feature_utility.json", "next": f"{PY} scripts/feature_utility.py --jobs 4"}
    if not _has_marginal_candidates():
        return {"state": "no_marginal_candidates", "action": "STOP — pusty subset",
                "why": "żadna pojedyncza cecha nie przekracza kary za złożoność na viable modelu",
                "next": "koniec: obecna przestrzeń cech nie dostarcza kandydata (poprawny wynik)"}

    prov = _provisional_acceptances()
    if prov is None:
        return {"state": "no_crossfit", "action": "forward",
                "why": "brak crossfit_selection.json", "next": f"{PY} scripts/crossfit_selection.py --jobs 4"}
    if prov == 0:
        return {"state": "discovery_not_confirmed", "action": "STOP — pusty subset",
                "why": "cross-fitting nie przyjął niczego w żadnym ramieniu — null nie może nic uratować",
                "next": "koniec: brak prowizorycznego survivora (poprawny wynik)"}

    null = _null_state()
    if null is None:
        return {"state": "provisional_pending_null", "action": "forward",
                "why": f"{prov} prowizorycznych akceptacji czeka na procedure-level null",
                "next": f"{PY} scripts/procedure_null.py --null a1 --jobs 4"}
    surv, rej, authority = null
    if authority == "a1_smoke":
        return {"state": "null_smoke_only", "action": "forward",
                "why": f"tylko null smoke (3 foldy): {surv} survivor(ów), {rej} odrzuconych — "
                       "potrzeba pełnego A1 na 16 foldach",
                "next": f"{PY} scripts/procedure_null.py --null a1 --jobs 4"}
    if surv == 0:
        return {"state": "all_candidates_fail_max_null", "action": "STOP — pusty subset",
                "why": f"pełne A1: 0 survivorów, {rej} odrzuconych przy futility — przestrzeń cech "
                       "nie spełnia standardu dowodu",
                "next": "koniec: pusty subset po procedure-level null (poprawny wynik)"}

    if not _rung6_done():
        return {"state": "candidate_survives", "action": "forward",
                "why": f"pełne A1: {surv} survivor(ów) — przeznacz budżet HPO tylko dla nich",
                "next": f"{PY} scripts/rung6_survivor_hpo.py --jobs 3"}

    ca = _cross_asset_done()
    if ca is None:
        return {"state": "survivors_pending_transfer", "action": "forward",
                "why": "survivorzy po Rung 6 — zbuduj macierz cross-asset",
                "next": f"{PY} scripts/cross_asset_matrix.py"}
    if ca.get("universal", 0) or ca.get("conditional", 0):
        return {"state": "survivors_stable_cross_asset", "action": "forward",
                "why": f"rodziny stabilne między aktywami: {ca.get('universal',0)} universal, "
                       f"{ca.get('conditional',0)} conditional — zbuduj transfer prior i rozważ Rung 7",
                "next": f"{PY} scripts/cross_asset_matrix.py  (+ Rung 7 interakcje na survivorach)"}
    return {"state": "survivors_asset_specific", "action": "STOP — lokalny wynik",
            "why": "survivorzy nie generalizują między aktywami — wynik ważny lokalnie, bez transferu",
            "next": "koniec: cechy asset-specific, każdy asset wymaga własnego potwierdzenia"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    d = classify()
    if args.json:
        print(json.dumps(d, indent=1, ensure_ascii=False))
        return 0
    print("Automat — najmniejszy następny eksperyment (czyta stan, nie działa)\n")
    print(f"  stan   : {d['state']}")
    print(f"  akcja  : {d['action']}")
    print(f"  powód  : {d['why']}")
    print(f"  krok   : {d['next']}")
    if d["action"] == HUMAN:
        print(f"\n  UWAGA: {HUMAN} — automat nie rekalibruje przestrzeni sam, bo to co-adaptacja "
              "do panelu; rozszerzenie przestrzeni mintuje nową wersję kontraktu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
