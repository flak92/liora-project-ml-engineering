#!/usr/bin/env python3
"""Universal-backbone helpers for the warm-start LSTM (docs/METHODOLOGY.md).

The reconciliation of "fast" with "per-asset feature selection kept": train ONE universal LSTM on
the pooled panel over the feature SUPERSET at a FIXED global architecture, then warm-start every
per-asset training (feature-search evals + the sealed refit) from that checkpoint. Feature
selection stays a separate per-asset mechanism (feature_search prefilter+forward), so which
features a ticker keeps is untouched — only the weight INITIALISATION is shared.

This module holds the two pure, unit-testable pieces (no bars, no heavy training):
  - `pooled_train_scaler`  : one z-score fit over EVERY ticker's TRAIN-region rows (leakage-guarded
                             by the caller's masks), the universal analogue of pipeline.train_norm_stats.
  - `transfer_init`        : load a universal state_dict into a per-asset LSTMClassifier —
                             hidden/bias/head verbatim, the input weight COLUMN-ALIGNED by feature id,
                             so a subset manifest inherits exactly the universal columns it uses.

The heavy `pretrain_backbone` orchestration (pooled panel build + the single fit at torch_threads
= cores-1) lives in `pretrain_universal.py`; it must run on a FREE box (it saturates all cores).
"""
import base64
import io
import json
from pathlib import Path

import numpy as np

_BB_CACHE = {}


def load_backbone(path):
    """Lazy, per-process cached load of the committed universal backbone JSON. Returns None when
    the file is absent (cold-start pipelines keep working untouched)."""
    key = str(path)
    if key in _BB_CACHE:
        return _BB_CACHE[key]
    p = Path(path)
    if not p.exists():
        _BB_CACHE[key] = None
        return None
    import torch
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {"superset": d["superset"], "arch": d["arch"], "eval_arch": d.get("eval_arch"),
           "refit_epochs": int(d.get("refit_epochs", 5)), "scaler": d.get("scaler"),
           "state_sha256": d.get("state_sha256"),
           "state": torch.load(io.BytesIO(base64.b64decode(d["state_b64"]))),
           "eval_state": (torch.load(io.BytesIO(base64.b64decode(d["eval_state_b64"])))
                          if d.get("eval_state_b64") else None)}
    _BB_CACHE[key] = out
    return out


def pooled_train_scaler(rows, names):
    """Pooled μ/σ per feature over TRAIN-only rows of every ticker.

    rows: iterable of (values[n_i, F], train_mask[n_i]) — one pair per ticker. `train_mask` is
          True only for bars inside the CV/train window AND outside every test/holdout interval
          (the caller applies the same leakage guard as the per-fold scaler: no warmup, no OOS,
          no test-group span). NaN rows are ignored per feature.
    Returns {name: {"mean": float, "std": float}} with std floored so a constant column ⇒ 1.0.
    """
    F = len(names)
    acc = [np.empty(0, dtype=np.float64) for _ in range(F)]
    for values, mask in rows:
        v = np.asarray(values, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        if v.shape[1] != F:
            raise ValueError(f"pooled_train_scaler: {v.shape[1]} cols != {F} names")
        sel = v[m]
        for j in range(F):
            col = sel[:, j]
            acc[j] = np.concatenate([acc[j], col[np.isfinite(col)]])
    out = {}
    for j, nm in enumerate(names):
        a = acc[j]
        mu = float(a.mean()) if a.size else 0.0
        sd = float(a.std()) if a.size else 1.0
        out[nm] = {"mean": mu, "std": sd if sd > 1e-8 else 1.0}
    return out


def transfer_init(dst_module, src_state, superset_names, manifest_names):
    """Warm-start `dst_module` (a per-asset LSTMClassifier at n_features=len(manifest_names)) from a
    universal `src_state` (state_dict trained at n_features=len(superset_names)). Same hidden /
    num_layers / seq_len are REQUIRED (a global fixed arch — that is what makes transfer legal).

    Copied verbatim: every recurrent weight/bias and the head — they do NOT depend on the input
    feature set. Column-aligned: `lstm.weight_ih_l0` (shape 4H×F) — its columns ARE the input
    features, so each manifest column is copied from the universal column of the SAME feature id;
    any manifest feature absent from the superset keeps the fresh init (defensive; manifest ⊆
    superset in practice). Layer-1+ input weights take hidden-sized input, so they transfer verbatim.
    Returns dst_module (mutated in place). Deterministic — no RNG here.
    """
    import torch
    col = {n: i for i, n in enumerate(superset_names)}
    dst = dst_module.state_dict()
    with torch.no_grad():
        for k, sv in src_state.items():
            if k not in dst:
                continue
            dv = dst[k]
            if k == "lstm.weight_ih_l0":
                if sv.shape[0] != dv.shape[0]:
                    raise ValueError(f"transfer_init: hidden mismatch on {k}: {sv.shape} vs {dv.shape}")
                for j, nm in enumerate(manifest_names):
                    si = col.get(nm)
                    if si is not None:
                        dv[:, j] = sv[:, si]
            elif sv.shape == dv.shape:
                dv.copy_(sv)
            # else: shape-incompatible (e.g. an unexpected feature-dependent tensor) — leave fresh init
        dst_module.load_state_dict(dst)
    return dst_module
