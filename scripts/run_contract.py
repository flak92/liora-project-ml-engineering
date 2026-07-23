#!/usr/bin/env python3
"""Run-scoped contract accessor — how a frozen contract SNAPSHOT reaches the science runners.

The runners take their canonical defaults from `config/` (the quantile grid, the model space). That
is correct for a normal run. But the Iterative Calibration Loop walks a ladder of frozen contract
VERSIONS: each epoch snapshots a contract whose ADMISSIBLE hypothesis keys (operating_point,
model_space, …) may differ, and the runners must compute under THAT version, not the canonical one —
otherwise a ladder rung is a placebo (it changes only the contract_hash, never the science).

This is the single, additive channel: if the environment variable `RESEARCH_CONTRACT` names a run's
`contract.json`, a runner reads the admissible hypothesis values from its embedded contract; if it is
unset, every accessor returns the caller's canonical default unchanged. So behaviour with the variable
unset is bit-identical to before — the change is invisible to a normal run and to a direct runner
invocation.

The name is deliberately generic (not project-branded): this is a universal research-engine
convention, the contract-side twin of the data-side `*_RESEARCH_DATA_DIR`.

Only ADMISSIBLE keys are ever read here. FROZEN proof-standard keys (viability, acceptance, max_null,
…) are identical across the ladder by construction (the safety kernel forbids patching them), so the
runners keep reading those from their canonical source; nothing here can vary the proof standard.
"""
import json
import os

ENV = "RESEARCH_CONTRACT"                    # path to a run's frozen contract.json (universal name)


def contract():
    """The run's embedded contract dict if RESEARCH_CONTRACT points at a snapshot, else None. None
    means 'no override' — every accessor below then returns the caller's canonical default."""
    path = os.environ.get(ENV)
    if not path:
        return None
    try:
        doc = json.loads(open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None
    return doc.get("contract", doc)          # a snapshot embeds the assembled contract under "contract"


def operating_point_grid(default):
    """The run's operating_point.grid (a list of quantiles) if overridden, else `default`. Used at the
    single choke point every runner funnels through (nested_validation.choose_operating_point)."""
    c = contract()
    if c:
        g = (c.get("operating_point") or {}).get("grid")
        if g:
            return [float(x) for x in g]
    return default


def model_space(default=None):
    """The run's model_space dict if overridden, else `default`. Consumed where a runner reads its
    search-space file / hpo_trials (wired with the estimator-parity work)."""
    c = contract()
    if c and c.get("model_space"):
        return c["model_space"]
    return default
