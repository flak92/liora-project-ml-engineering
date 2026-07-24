"""Read-only access to the frozen methodology snapshot for the native (Plotly) presentation.

The methodology branch presents THE METHOD, not the sealed product models — so these pages read the
committed `results/methodology_snapshot/` JSONs (and the frozen contract) directly, and nothing else.
Nothing trains at runtime; every number is a read of a frozen artifact. Distinct from app/data.py,
which serves the sealed-model product console over data/results.db on the main branch.
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "results" / "methodology_snapshot"
CONTRACT = ROOT / "config" / "feature_discovery_contract.json"
FAMILIES = ROOT / "config" / "feature_families_xgb.json"
PANEL6 = ROOT / "config" / "panel_6.json"

OK, MISSING = "OK", "SNAPSHOT_MISSING"


@lru_cache(maxsize=None)
def _read(path_str, _stamp):
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load(path):
    p = Path(path)
    return _read(str(p), p.stat().st_mtime if p.exists() else 0.0)


def health():
    need = [SNAP / "family_transfer.json", SNAP / "crossfit_selection.json", CONTRACT, FAMILIES]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        return {"status": MISSING, "detail": "run `make family-transfer` — missing: " + ", ".join(missing)}
    return {"status": OK, "detail": ""}


def contract():
    return _load(CONTRACT) or {}


def data_boundary():
    c = contract()
    db = c.get("data_boundary", {})
    return {"train_end": db.get("train_end"), "oos_start": db.get("oos_start"),
            "oos_reads": db.get("oos_reads"), "label_horizon_bars": db.get("label_horizon_bars"),
            "embargo_bars": db.get("embargo_bars")}


def contract_identity():
    idy = (contract().get("identity") or {})
    return {"contract_version": idy.get("contract_version"),
            "sample_sha256_prefix": idy.get("sample_sha256_prefix"),
            "bar_store": idy.get("bar_store")}


def families_config():
    fam = (_load(FAMILIES) or {}).get("families", {})
    return {k: [int(i) for i in v] for k, v in fam.items()}


def family_transfer(panel="full"):
    """The Rung 8 artifact. panel='full' = canonical 20-asset (parity); panel='panel6' = study view."""
    name = "family_transfer.json" if panel == "full" else "family_transfer_panel6.json"
    return _load(SNAP / name)


def panel6_assets():
    return (_load(PANEL6) or {}).get("assets", [])


def compiled():
    d = SNAP / "compiled"
    if not d.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}


# Rung ladder (SSOT docs/SMART_METHODOLOGY.md §3), with implementation status.
LADDER = [
    (0, "Freeze the problem", "run", "implemented"),
    (1, "Can the model learn?", "asset", "implemented"),
    (2, "Does the operating point transfer?", "asset", "folded into 3–4"),
    (3, "Does a feature improve a learnable model?", "asset", "implemented"),
    (4, "Does the choice survive data that did not choose it?", "asset", "implemented"),
    (5, "Edge > the maximum a search produces itself?", "asset", "implemented"),
    (6, "How much is a survivor worth under its own tuned model?", "asset", "implemented"),
    (7, "Do survivors combine (interactions)?", "asset", "specified, unvalidated"),
    (8, "Which OHLCV families travel across assets?", "panel", "implemented (this work)"),
    (9, "Does the method hold on a fresh panel?", "new panel", "not started"),
]

STATUS_ORDER = ["PANEL_STABLE", "ASSET_CONDITIONAL", "TUNING_DEPENDENT", "CORE_CONDITIONAL_FAILURE",
                "REGIME_DEPENDENT", "SEARCH_INFLATED", "CROSSFIT_UNSTABLE", "NO_POSITIVE_UTILITY",
                "INSUFFICIENT_EVIDENCE"]

STATUS_MEANING = {
    "PANEL_STABLE": "passed all nulls, kept by Rung 6, on > 1 asset-fold",
    "ASSET_CONDITIONAL": "passed the full procedure, but one asset / narrow folds",
    "TUNING_DEPENDENT": "passed A1×A2×B, demoted by Rung 6 own-null",
    "CORE_CONDITIONAL_FAILURE": "passed marginal null, not conditional residual B",
    "REGIME_DEPENDENT": "passed A1, not A2",
    "SEARCH_INFLATED": "passed cross-fit, not the A1 max-null",
    "CROSSFIT_UNSTABLE": "good in discovery, not on rotating confirmation folds",
    "NO_POSITIVE_UTILITY": "no candidate shows positive marginal utility",
    "INSUFFICIENT_EVIDENCE": "not enough correct folds / viable models",
}
