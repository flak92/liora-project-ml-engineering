#!/usr/bin/env python3
"""D7-D8: the LSTM classifier, deterministic CPU training, Optuna HPO, Kelly calibration.

Architecture: one LSTM layer over the (SEQ_LEN × n_features) z-scored window, dropout on the
last hidden state, a linear head → logit of Y=1 (Triple-Barrier win). Loss = BCE-with-logits
weighted by pos_weight (class balance) × label-uniqueness weight. Everything is deterministic:
seeds are re-planted before every fold/refit, torch runs 2 CPU threads with deterministic
algorithms, batching order comes from a seeded numpy Generator — a rerun reproduces
best_params and the OOS row exactly.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import pipeline as P

# Shared operating-point selection (src/shared/op_select.py): ONE accumulate-then-select
# implementation for both pipelines, so theta spectrum / trade floor / robustness can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import op_select                                    # noqa: E402

SEED = int(P.CONFIG["RANDOM_SEED"])
TR = P.CONFIG["TRAIN"]
HPO = P.CONFIG["HPO"]
OPS = P.CONFIG["OPERATING_SPACE"]


def op_lambdas():
    """The lambda dimension of the operating space; collapses to [None] (f = 1) under
    all-in, so every scorer trades with the sizing it will deploy."""
    if P.CONFIG["CAPITAL_MODE"].startswith("all_in"):
        return [None]
    return [float(x) for x in OPS["lambda"]]


def op_grid_scores(df, fold_data, thetas, lambdas, dir_modes=("both",)):
    """Accumulate step of accumulate-then-select: replay every fold's OOF predictions
    through run_engine at each (direction, theta, lambda) and keep the PER-FOLD log-growth
    and trade counts. fold_data = [(scored, lo, hi)] with hi = the fold's last val t0 (the
    engine window extends to hi + H — the full label horizon, provably pre-OOS by the purge
    invariant). Selection happens ONCE, on the whole grid, in
    op_select.select_operating_point — never per fold (the fold-oracle bias)."""
    H, E0 = P.CONFIG["H"], P.CONFIG["INITIAL_CAPITAL_USD"]
    grid = []
    for mode in dir_modes:
        filtered = [(_dir_filter(sc, mode), lo, hi) for sc, lo, hi in fold_data]
        for th in thetas:
            for lam in lambdas:
                gs, ns = [], []
                for sc, lo, hi in filtered:
                    summ = P.run_engine(df, sc, lo, hi + H, th, kelly_fraction=lam)[0]
                    gs.append(math.log(max(summ["end_capital"], P.EPS) / E0))
                    ns.append(int(summ["trades"]))
                grid.append({"theta": float(th), "lambda": lam, "direction": mode,
                             "fold_growth": gs, "fold_trades": ns})
    return grid


def score_shared_operating_point(df, fold_data):
    """Shared Train-OOF score for HPO / feature-search: ONE floor-respecting operating
    point over the FULL OPERATING_SPACE theta spectrum (shared with D8 calibration),
    chosen on the ACCUMULATED folds. Returns the op_select selection dict."""
    thetas = [float(t) for t in OPS["theta"]]
    grid = op_grid_scores(df, fold_data, thetas, op_lambdas())
    return op_select.select_operating_point(grid, min_oof_trades=int(OPS["min_oof_trades"]),
                                            theta_spectrum=thetas)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(TR["torch_threads"]))
    return seed


class LSTMClassifier(nn.Module):
    """LSTM over (B, SEQ_LEN, F) windows -> logit of Y=1. num_layers in {1,2}; inter-layer LSTM
    dropout applies only when num_layers>1 (torch ignores it for a single layer). num_layers=1
    reproduces the original 1-layer architecture exactly."""

    def __init__(self, n_features, hidden, dropout, num_layers=1):
        super().__init__()
        num_layers = int(num_layers)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True,
                            dropout=float(dropout) if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


def _pos_weight(y):
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    return torch.tensor(neg / pos, dtype=torch.float32)


def _epoch(model, opt, loss_fn, X, y, w, batch, rng):
    model.train()
    order = rng.permutation(len(X))
    for i in range(0, len(order), batch):
        b = order[i:i + batch]
        opt.zero_grad()
        loss = (loss_fn(model(X[b]), y[b]) * w[b]).mean()
        loss.backward()
        opt.step()


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    out = []
    for i in range(0, len(X), 1024):
        out.append(torch.sigmoid(model(X[i:i + 1024])))
    return torch.cat(out).numpy() if out else np.empty(0)


def _tensors(Xtr, ytr, wtr):
    return (torch.from_numpy(np.ascontiguousarray(Xtr)),
            torch.from_numpy(np.asarray(ytr, np.float32)),
            torch.from_numpy(np.asarray(wtr, np.float32)))


def train_model(Xtr, ytr, wtr, hidden, lr, dropout, epochs=None, Xva=None, yva=None, seed=SEED,
                weight_decay=0.0, num_layers=1, init_state=None, init_names=None, feat_names=None):
    """Deterministic training. With (Xva, yva): early stopping on validation AUC-PR
    (patience/min_delta from config) and the best state is restored — returns
    (model, best_ap, best_epoch). Without: fixed `epochs`, returns (model, None, epochs).
    `seed` plants both the torch init and the batch-shuffle RNG; it defaults to the global
    SEED so every existing caller is byte-identical, and the feature-search evaluator passes
    distinct seeds to average out weight-init luck (a harder overfit gate). `weight_decay`
    (Adam L2) and `num_layers` regularize / size the net; the defaults (0.0, 1) reproduce the
    original unregularized 1-layer model exactly.

    Warm-start (universal backbone): when `init_state` (a state_dict trained at the SAME
    hidden/num_layers on the feature SUPERSET) is given, the freshly-seeded random init is
    overwritten by it — recurrent/head verbatim, input weight column-aligned from `init_names`
    (superset order) to `feat_names` (this training's manifest). The seed still governs the
    batch-shuffle RNG and the optimizer, so training stays deterministic; init_state=None keeps
    every existing caller byte-identical."""
    seed_everything(seed)
    Xt, yt, wt = _tensors(Xtr, ytr, wtr)
    model = LSTMClassifier(Xt.shape[-1], int(hidden), float(dropout), int(num_layers))
    if init_state is not None:
        import universal
        universal.transfer_init(model, init_state, init_names, feat_names)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=_pos_weight(np.asarray(ytr)))
    rng = np.random.default_rng(seed)
    if Xva is None:
        for _ in range(int(epochs)):
            _epoch(model, opt, loss_fn, Xt, yt, wt, int(TR["batch_size"]), rng)
        return model, None, int(epochs)
    Xv = torch.from_numpy(np.ascontiguousarray(Xva))
    best_ap, best_epoch, best_state, since = -1.0, 0, None, 0
    for ep in range(1, int(TR["max_epochs"]) + 1):
        _epoch(model, opt, loss_fn, Xt, yt, wt, int(TR["batch_size"]), rng)
        ap = P.average_precision(yva, predict_proba(model, Xv))
        if ap > best_ap + float(TR["min_delta"]):
            best_ap, best_epoch, since = ap, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= int(TR["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_ap), int(best_epoch)


def hpo(df, events, y, w, folds_data, bounds):
    """D7: seeded Optuna study that selects the LSTM architecture by TRADEABLE OOF LOG-GROWTH,
    not by AUC-PR — so the searched hyper-parameters are chosen for profitability, the fix for
    "Optuna calibrates the wrong objective". Per trial, per fold: train (early stopping on val
    AUC-PR, a smooth stop signal over the whole val set), predict out-of-fold, then score the
    fold by the BEST out-of-fold log-growth over a COARSE (θ, λ) grid via run_engine — the exact
    OOF→engine machinery D8 uses, so model selection and the trading objective are aligned. The
    objective is the mean fold log-growth; the fine (θ, λ, direction) point is chosen at D8. The
    search also spans weight_decay (Adam L2) and num_layers for regularization/capacity — the
    levers that most reduce the Train→OOS gap. Each fold's OOF stays strictly inside
    [min(val t0), max(val t0)+H], pre-OOS by the purge invariant. The winning trial's mean AUC-PR
    is still reported (cv_auc_pr) for continuity. folds_data = [(tr, va, X_fold)] with fold-causal
    normalization. Returns (best_params, mean_best_epochs, best_cv_auc_pr)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    H = P.CONFIG["H"]
    t0s = [e["t0"] for e in events]
    # ENFORCE (not just assume) the one-way OOS boundary: every CV val fold's log-growth is scored
    # over [min val t0, max val t0 + H], which must stay strictly inside Train — the profit HPO can
    # never see an OOS bar. Mirrors the D8 calibration assert; fails closed if the purge is ever wrong.
    oos0 = bounds["oos_start_idx"]
    assert all(max(t0s[j] for j in va) + H < oos0 for _, va, _ in folds_data if len(va)), \
        "hpo: a CV val fold's label horizon reaches OOS (purge invariant violated)"

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", HPO["hidden_choices"])
        lr = trial.suggest_float("lr", HPO["lr_low"], HPO["lr_high"], log=True)
        dropout = trial.suggest_float("dropout", HPO["dropout_low"], HPO["dropout_high"])
        weight_decay = trial.suggest_float("weight_decay", HPO["weight_decay_low"],
                                           HPO["weight_decay_high"], log=True)
        num_layers = trial.suggest_categorical("num_layers", HPO["num_layers_choices"])
        fold_data, aps, eps_, mean_g = [], [], [], -1e9
        for step, (tr, va, Xf) in enumerate(folds_data):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            model, ap, best_ep = train_model(Xf[tr], y[tr], w[tr], hidden, lr, dropout,
                                             Xva=Xf[va], yva=y[va],
                                             weight_decay=weight_decay, num_layers=num_layers)
            oof = predict_proba(model, torch.from_numpy(np.ascontiguousarray(Xf[va])))
            scored = [(events[j], float(oof[k])) for k, j in enumerate(va)]
            vt0 = [t0s[j] for j in va]
            fold_data.append((scored, min(vt0), max(vt0)))
            aps.append(ap)
            eps_.append(max(1, best_ep))
            # accumulate-then-select over the folds seen so far: ONE shared floor-respecting
            # point (never a per-fold best theta — that oracle upward-biased every trial)
            sel = score_shared_operating_point(df, fold_data)
            mean_g = sel["log_growth"] / len(fold_data)
            trial.report(mean_g, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not fold_data:
            return -1e9
        trial.set_user_attr("best_epochs", eps_)
        trial.set_user_attr("cv_auc_pr", float(np.mean(aps)))
        return float(mean_g)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(
                                    n_warmup_steps=int(HPO["pruner_warmup_steps"])))
    study.optimize(objective, n_trials=int(HPO["n_trials"]), show_progress_bar=False)
    best = study.best_trial
    epochs = best.user_attrs.get("best_epochs") or [int(TR["max_epochs"]) // 2]
    cv_ap = float(best.user_attrs.get("cv_auc_pr", 0.0))
    return dict(best.params), int(round(float(np.mean(epochs)))), cv_ap


def _dir_filter(scored, mode):
    """long_only acts on the up-signals only (direction == +1); both keeps every signal.
    Restricting which signals we ACT on is a Train-only operating choice — the model was
    trained on both sides and only outputs a win-probability."""
    if mode == "long_only":
        return [(ev, p) for ev, p in scored if ev["direction"] == 1]
    return scored


def calibrate_gate_kelly(df, events, y, w, folds_data, best_params, refit_epochs, bounds,
                         init_state=None, init_names=None, feat_names=None):
    """D8: choose the per-asset operating point (entry threshold θ, Kelly fraction λ,
    direction_mode) JOINTLY on Train out-of-fold log-growth. Per fold: train with the best
    params for the fixed refit epoch count (no validation peeking) on the fold's CAUSALLY
    normalized tensor and predict out-of-fold; then replay each fold through run_engine over
    the whole (θ, λ, direction) grid — the OOF predictions are computed once and reused, so
    the grid is nearly free. Full label horizon (end = last val t0 + H, provably pre-OOS by
    the purge invariant). Ties (equal OOF log-growth) resolve to the most conservative point:
    smaller λ, then higher θ, then both-sided. Returns a dict with θ, λ, direction_mode."""
    H = P.CONFIG["H"]
    gc = P.CONFIG["GATE_CALIBRATION"]                  # Kelly-epoch lambda geometry only
    E0 = P.CONFIG["INITIAL_CAPITAL_USD"]
    thetas = [float(t) for t in OPS["theta"]]          # SSOT: OPERATING_SPACE (#13-14)
    # all-in mode: the lambda dimension disappears — theta and direction are still calibrated,
    # but every grid point replays the engine at full capital (kelly_fraction=None -> f=1)
    lambdas = ([None] if P.CONFIG["CAPITAL_MODE"].startswith("all_in")
               else [float(x) for x in np.geomspace(gc["lambda_low"], gc["lambda_high"], int(gc["lambda_points"]))])
    dir_modes = list(OPS["direction_modes"])
    t0s = [e["t0"] for e in events]
    fold_data, fold_aps = [], []
    for tr, va, Xf in folds_data:
        if len(np.unique(y[tr])) < 2:
            continue
        model, _, _ = train_model(Xf[tr], y[tr], w[tr], best_params["hidden"],
                                  best_params["lr"], best_params["dropout"], epochs=refit_epochs,
                                  weight_decay=best_params.get("weight_decay", 0.0),
                                  num_layers=best_params.get("num_layers", 1),
                                  init_state=init_state, init_names=init_names,
                                  feat_names=feat_names)
        oof = predict_proba(model, torch.from_numpy(np.ascontiguousarray(Xf[va])))
        fold_aps.append(P.average_precision(y[va], oof))
        scored = [(events[j], float(oof[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        fold_data.append((scored, min(vt0), max(vt0)))
    default = {"theta_entry": max(thetas),
               "kelly_fraction": None if lambdas[0] is None else float(min(lambdas)),
               "direction_mode": "both", "oof_log_growth": 0.0,
               "cv_auc_pr": float(np.mean(fold_aps)) if fold_aps else 0.0}
    if not fold_data:
        return default
    assert all(hi + H < bounds["oos_start_idx"] for _, _, hi in fold_data), \
        "gate calibration must not reach OOS (even with the full label horizon)"
    # min-2-trades mandate (08-clean p_star_grid pattern): the operating point is chosen ONLY
    # among Train-OOF grid points clearing the trade floor — an ML strategy must DIFFER from
    # HODL. If no point qualifies, the most-trading point is taken and flagged (the OOS row will
    # then honestly show what that ticker can support). Train-OOF only; OOS untouched.
    min_tr = int(OPS["min_oof_trades"])
    grid = []
    for mode in dir_modes:
        filtered = [(_dir_filter(sc, mode), lo, hi) for sc, lo, hi in fold_data]
        for theta in thetas:
            for lam in lambdas:
                g, n_tr = 0.0, 0
                for sc, lo, hi in filtered:
                    summ = P.run_engine(df, sc, lo, hi + H, theta, kelly_fraction=lam)[0]
                    g += math.log(max(summ["end_capital"], P.EPS) / E0)
                    n_tr += int(summ["trades"])
                grid.append((g, n_tr, mode, theta, lam))
    viable = [t for t in grid if t[1] >= min_tr]
    pool = viable if viable else [max(grid, key=lambda t: (t[1], t[0]))]
    gmax = max(g for g, *_ in pool)
    tied = [(mode, theta, lam) for g, n, mode, theta, lam in pool if g >= gmax - 1e-9]
    # most conservative among ties: smaller λ (None sorts as 0 in all-in), then higher θ, then both
    tied.sort(key=lambda x: (0.0 if x[2] is None else x[2], -x[1], 0 if x[0] == "both" else 1))
    mode, theta, lam = tied[0]
    oof_trades = int(next(n for g, n, m2, t2, l2 in grid if (m2, t2, l2) == (mode, theta, lam)))
    return {"theta_entry": float(theta), "kelly_fraction": None if lam is None else float(lam),
            "direction_mode": mode, "oof_log_growth": float(gmax),
            "oof_trades": oof_trades, "trade_floor_met": bool(viable),
            "cv_auc_pr": float(np.mean(fold_aps)) if fold_aps else 0.0}
