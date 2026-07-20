"""Out-of-fold SHAP over the candidate superset — measurement only, never a gate.

The feature search decides what enters a model by Train-OOF trading log-growth: it trains on a
fold's training rows, replays the trade simulator on that fold's validation rows, and asks
whether adding a candidate lifted the mean. Answering that costs one full evaluation per
candidate, and there are 45 of them — about 85% of the search's budget for a table.

This module asks a cheaper question of the same folds: inside ONE model that holds every
candidate at once, how much does each feature move the prediction? If that ordering agrees with
which candidates survive the unchanged gates, the search could put the promising ones first and
stop asking the rest — and if it does not agree, we have learned that before building anything.

The leak discipline is the whole point, so it is stated rather than assumed:

  * every attribution comes from a booster trained ONLY on a fold's training rows, never on the
    rows it is then asked about;
  * the frame handed in must already be the fold-indexed matrix — the pipeline's full 1h frame
    contains OOS bars, and safety there rests entirely on indexing, so nothing global is ever
    computed here;
  * the labels are not read at all. `pred_contribs` decomposes a prediction; y never enters.

Two variants are produced side by side, because which rows the attribution is read on is exactly
the question the safeguard turns on:

  validation rows — the model has not fitted them, so the attribution is honest about unseen
      data; the cost is that a statistic computed on those rows then informs a selection whose
      acceptance gate is scored on the same rows.
  training rows — the validation fold stays untouched by any selection statistic whatsoever;
      the cost is that in-sample attributions flatter whatever the booster memorised.

If both orderings agree, the distinction stops mattering and we will have shown it rather than
argued it.
"""
import numpy as np


def _contribs(booster, X, names, xgb):
    """Exact TreeSHAP from the booster itself, bias column dropped.

    binary:logistic returns (n_rows, n_features + 1). The three-dimensional indexing that suits
    multi:softprob would raise IndexError here, so the shape is checked rather than trusted.
    """
    C = np.asarray(booster.predict(xgb.DMatrix(X, feature_names=names), pred_contribs=True))
    if C.ndim != 2 or C.shape[1] != len(names) + 1:
        raise ValueError(f"unexpected contribution shape {C.shape} for {len(names)} features")
    return C[:, :-1]


def rank_superset(P, xgb, df_b, bounds, params, names, seed):
    """Per-feature mean |SHAP| across the purged folds, on validation rows and on training rows.

    Trains one booster per fold on the full candidate superset — four fits, no trade simulation,
    which is where the search's real cost sits. Returns None when no fold is usable, matching
    how the search itself treats a degenerate table.
    """
    X = df_b[names].to_numpy(float)
    y = df_b["Y_outcome"].to_numpy(int)
    w = df_b["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(sid.split(":")[1]) for sid in df_b["setup_id"]]
    folds = P.purged_wf_folds(t0s, bounds["train_start_idx"], bounds["train_end_idx"])

    # The same fail-closed invariant the search asserts on every evaluation. Repeated here rather
    # than inherited, because this path builds its own folds and must not be safe by accident.
    assert all(max(t0s[j] for j in va) < bounds["oos_start_idx"] for _, va in folds if va), \
        "oof_shap: a CV val fold reaches OOS (purge invariant violated)"

    per_fold_va, per_fold_tr, used = [], [], 0
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        per_fold_va.append(np.abs(_contribs(bst, X[va], names, xgb)).mean(axis=0))
        per_fold_tr.append(np.abs(_contribs(bst, X[tr], names, xgb)).mean(axis=0))
        used += 1

    if not used:
        return None

    # Validation folds are disjoint, training folds overlap, so both are averaged fold-by-fold
    # rather than pooled — one table's long fold must not outvote the others in either variant.
    va_mean = np.mean(per_fold_va, axis=0)
    tr_mean = np.mean(per_fold_tr, axis=0)

    def as_ranking(vals):
        total = float(vals.sum())
        order = np.argsort(-vals)
        rank = np.empty(len(names), int)
        rank[order] = np.arange(1, len(names) + 1)
        return {n: {"mean_abs_shap": round(float(vals[i]), 10),
                    "share": round(float(vals[i] / total), 10) if total > 0 else 0.0,
                    "rank": int(rank[i])}
                for i, n in enumerate(names)}

    return {"folds_used": used, "n_features": len(names),
            "val_rows": as_ranking(va_mean), "train_rows": as_ranking(tr_mean)}
