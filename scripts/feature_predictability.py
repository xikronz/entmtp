#!/usr/bin/env python3
"""Quick predictability check for entropy/MACD features in `logs/outputs/science.ipynb`
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import silhouette_score as _silhouette_score
except ImportError:
    _silhouette_score = None  # type: ignore[misc, assignment]


BASE_FEATURE_COLS: List[str] = [
    "block_start_entropy",
    "entropy_smoothed_w4",
    "entropy_smoothed_w16",
    "entropy_smoothed_w32",
    "macd_4_16",
    "macd_4_32",
    "macd_16_32",
    "macd_4_16_signal",
    "macd_4_32_signal",
    "macd_16_32_signal",
    "macd_4_16_hist",
    "macd_4_32_hist",
    "macd_16_32_hist",
    "macd_1_4",
    "macdgit push -u origin main
git push origin --all
git push origin --tags_1_16",
    "macd_1_32",
    "macd_1_4_signal",
    "macd_1_16_signal",
    "macd_1_32_signal",
    "macd_1_4_hist",
    "macd_1_16_hist",
    "macd_1_32_hist",
]

PREV_ACCEPT_FEATURE_COLS: List[str] = [
    "prev_accept",
    "prev_2_accept",
    "prev_3_accept",
    "prev_3_accept_avg",
    "prev_5_accept_avg",
    "prev_5_accept_max",
    "prev_5_accept_min",
    "prev_accept_ema",
]

FEATURE_COLS: List[str] = BASE_FEATURE_COLS + PREV_ACCEPT_FEATURE_COLS


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    macd_df = df.copy()
    macd_df["macd_4_16"] = macd_df["entropy_smoothed_w4"] - macd_df["entropy_smoothed_w16"]
    macd_df["macd_4_32"] = macd_df["entropy_smoothed_w4"] - macd_df["entropy_smoothed_w32"]
    macd_df["macd_16_32"] = macd_df["entropy_smoothed_w16"] - macd_df["entropy_smoothed_w32"]

    macd_df["macd_1_4"] = macd_df["block_start_entropy"] - macd_df["entropy_smoothed_w4"]
    macd_df["macd_1_16"] = macd_df["block_start_entropy"] - macd_df["entropy_smoothed_w16"]
    macd_df["macd_1_32"] = macd_df["block_start_entropy"] - macd_df["entropy_smoothed_w32"]

    for col in [
        "macd_4_16",
        "macd_4_32",
        "macd_16_32",
        "macd_1_4",
        "macd_1_16",
        "macd_1_32",
    ]:
        macd_df[f"{col}_signal"] = macd_df.groupby("sample_idx")[col].transform(
            lambda x: x.ewm(span=4, adjust=False).mean()
        )
        macd_df[f"{col}_hist"] = macd_df[col] - macd_df[f"{col}_signal"]
    return macd_df


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    return out


def ensure_sample_idx(
    df: pd.DataFrame, *, group_col: str | None
) -> Tuple[pd.DataFrame, str]:
    """Ensure a `sample_idx` column exists (trajectory / prompt id for groupby)."""
    out = df.copy()
    if group_col is not None:
        if group_col not in out.columns:
            raise SystemExit(
                f"--group-col={group_col!r} not in columns: {list(out.columns)[:30]}..."
            )
        out["sample_idx"] = out[group_col].astype(int)
        return out, f"trajectory_id=column {group_col!r} (copied to sample_idx)"

    if "sample_idx" in out.columns:
        out["sample_idx"] = out["sample_idx"].astype(int)
        return out, "trajectory_id=sample_idx"

    for alias in ("sample_id", "prompt_idx", "prompt_id", "traj_id", "trajectory_id"):
        if alias in out.columns:
            out["sample_idx"] = out[alias].astype(int)
            return out, f"trajectory_id=alias {alias!r} (copied to sample_idx)"

    if "step" in out.columns:
        st = out["step"].astype(int)
        sid = (st == 0).astype(np.int64).cumsum() - 1
        out["sample_idx"] = sid.clip(lower=0).astype(int)
        return (
            out,
            "trajectory_id=derived from step==0 boundaries (no sample_idx in file); "
            "ensure rows are ordered as in generation",
        )

    raise SystemExit(
        "No trajectory id column (expected sample_idx or similar). "
        f"Columns: {list(out.columns)}. Pass --group-col <name>."
    )


def add_prev_accept_features(df: pd.DataFrame, history_col: str) -> pd.DataFrame:
    """Shifts/rollings/EMA of accept signal within each trajectory."""

    out = df.copy()
    gb = out.groupby("sample_idx", sort=False)[history_col]
    s1 = gb.shift(1).astype(float)
    out["prev_accept"] = s1
    out["prev_2_accept"] = gb.shift(2).astype(float)
    out["prev_3_accept"] = gb.shift(3).astype(float)

    out["_s1_for_roll"] = s1
    groll = out.groupby("sample_idx", sort=False)["_s1_for_roll"]
    out["prev_3_accept_avg"] = groll.transform(lambda x: x.rolling(3, min_periods=1).mean())
    out["prev_5_accept_avg"] = groll.transform(lambda x: x.rolling(5, min_periods=1).mean())
    out["prev_5_accept_max"] = groll.transform(lambda x: x.rolling(5, min_periods=1).max())
    out["prev_5_accept_min"] = groll.transform(lambda x: x.rolling(5, min_periods=1).min())
    out["prev_accept_ema"] = groll.transform(lambda x: x.ewm(span=8, adjust=False).mean())
    out.drop(columns=["_s1_for_roll"], inplace=True)
    return out


def resolve_accept_history_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise SystemExit(f"--accept-history-col={explicit!r} not in dataframe columns")
        return explicit
    if "accept_depth" in df.columns:
        return "accept_depth"
    if "accept_length" in df.columns:
        return "accept_length"
    raise SystemExit("Need accept_depth or accept_length column for accept history features")


def group_split_by_sample_idx(
    df: pd.DataFrame, *, test_frac: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = df["sample_idx"].dropna().astype(int).unique()
    rng.shuffle(ids)
    n_test = max(1, int(math.ceil(test_frac * len(ids))))
    test_ids = set(ids[:n_test].tolist())
    is_test = df["sample_idx"].astype(int).isin(test_ids)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def standardize_fit(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=0)
    sigma = x.std(dim=0).clamp_min(1e-6)
    return mu, sigma


def standardize_apply(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (x - mu) / sigma


def build_shallow_mlp(in_dim: int, hidden_dim: int, n_hidden_layers: int) -> torch.nn.Module:
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1")
    layers: List[torch.nn.Module] = []
    d = in_dim
    for _ in range(n_hidden_layers):
        layers.append(torch.nn.Linear(d, hidden_dim))
        layers.append(torch.nn.ReLU(inplace=False))
        d = hidden_dim
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


def fit_mlp_regression(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    hidden_dim: int,
    n_hidden_layers: int,
    l2: float,
    steps: int,
    lr: float,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    in_dim = x_tr.shape[1]
    model = build_shallow_mlp(in_dim, hidden_dim, n_hidden_layers)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2)
    for _ in range(steps):
        pred = model(x_tr).squeeze(1)
        loss = torch.nn.functional.mse_loss(pred, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def fit_mlp_classification(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    hidden_dim: int,
    n_hidden_layers: int,
    l2: float,
    steps: int,
    lr: float,
    seed: int,
    pos_weight: float | None,
) -> torch.nn.Module:
    """Shallow ReLU MLP, trained with BCE-with-logits + Adagrad."""
    torch.manual_seed(seed)
    in_dim = x_tr.shape[1]
    model = build_shallow_mlp(in_dim, hidden_dim, n_hidden_layers)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2)
    pw = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)
    for _ in range(steps):
        logits = model(x_tr).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tr, pos_weight=pw
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def fit_linear_regression(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    l2: float,
    steps: int,
    lr: float,
) -> torch.nn.Module:
    d = x_tr.shape[1]
    model = torch.nn.Linear(d, 1, bias=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        pred = model(x_tr).squeeze(1)
        loss = torch.nn.functional.mse_loss(pred, y_tr)
        if l2 > 0:
            loss = loss + l2 * (model.weight.square().sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def fit_logistic_regression(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    l2: float,
    steps: int,
    lr: float,
    pos_weight: float | None,
) -> torch.nn.Module:
    d = x_tr.shape[1]
    model = torch.nn.Linear(d, 1, bias=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
    for _ in range(steps):
        logits = model(x_tr).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tr, pos_weight=pw
        )
        if l2 > 0:
            loss = loss + l2 * (model.weight.square().sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.inference_mode()
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    err = y_pred - y_true
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err.square().mean())).item()
    y_mean = y_true.mean()
    ss_tot = (y_true - y_mean).square().sum().item()
    ss_res = err.square().sum().item()
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-9))
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    r = (yt * yp).mean() / (yt.std() * yp.std() + 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": float(r2), "pearson_r": float(r.item())}


@torch.inference_mode()
def _pairwise_sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X (n,d) and centers C (k,d) -> (n,k)."""
    xx = (X * X).sum(axis=1, keepdims=True)
    cc = (C * C).sum(axis=1)
    xc = X @ C.T
    return xx + cc - 2 * xc


def kmeans_fit(
    X: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Classic Lloyd K-means with random restarts (NumPy only)."""
    n = X.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n_samples, got k={k}, n={n}")
    best_inertia = float("inf")
    best_labels = np.zeros(n, dtype=np.int64)
    best_centers = np.zeros((k, X.shape[1]))
    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            dist2 = _pairwise_sq_dist(X, centers)
            labels = dist2.argmin(axis=1)
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    new_centers[j] = X[rng.integers(n)]
            if np.allclose(new_centers, centers, rtol=1e-6, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_inertia


def kmeans_assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return _pairwise_sq_dist(X, centers).argmin(axis=1)


def print_kmeans_summary(
    *,
    x_tr_np: np.ndarray,
    x_te_np: np.ndarray,
    al_tr: np.ndarray,
    al_te: np.ndarray,
    bin_tr: np.ndarray,
    bin_te: np.ndarray,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> None:
    rng = np.random.default_rng(seed)
    labels_tr, centers, inertia = kmeans_fit(
        x_tr_np, k, rng=rng, n_init=n_init, max_iter=max_iter
    )
    labels_te = kmeans_assign(x_te_np, centers)

    print(
        f"kmeans (train-fit): K={k} inertia={inertia:.2f} "
        f"n_init={n_init} max_iter={max_iter}"
    )
    if _silhouette_score is not None and k >= 2 and len(x_tr_np) > k:
        try:
            sil = float(_silhouette_score(x_tr_np, labels_tr, metric="euclidean"))
            print(f"  silhouette (train, sklearn): {sil:.4f}")
        except Exception as e:
            print(f"  silhouette skipped: {e}")

    def _block(name: str, labels: np.ndarray, al: np.ndarray, bn: np.ndarray) -> None:
        print(f"  [{name}]")
        for j in range(k):
            m = labels == j
            cnt = int(m.sum())
            if cnt == 0:
                print(f"    cluster {j}: n=0")
                continue
            print(
                f"    cluster {j}: n={cnt} "
                f"mean_accept_length={al[m].mean():.3f} "
                f"P(accept_len==1)={bn[m].mean():.3f}"
            )

    _block("train", labels_tr, al_tr, bin_tr)
    _block("test", labels_te, al_te, bin_te)


def classification_metrics(y_true: torch.Tensor, prob: torch.Tensor) -> Dict[str, float]:
    pred = (prob >= 0.5).to(y_true.dtype)
    acc = (pred == y_true).float().mean().item()
    pos = y_true == 1
    neg = ~pos
    tpr = (pred[pos] == 1).float().mean().item() if pos.any() else float("nan")
    tnr = (pred[neg] == 0).float().mean().item() if neg.any() else float("nan")
    bacc = (tpr + tnr) / 2.0 if (not math.isnan(tpr) and not math.isnan(tnr)) else float("nan")
    return {"acc": acc, "balanced_acc": bacc, "pos_rate": float(y_true.mean().item())}


def run_one_dataset(
    df_raw: pd.DataFrame,
    *,
    seed: int,
    test_frac: float,
    accept_history_col: str | None,
    group_col: str | None,
    mode: str,
    regressor: str,
    classifier: str,
    mlp_hidden: int,
    mlp_layers: int,
    train_steps: int,
    train_lr: float,
    train_l2: float,
    kmeans_k: int,
    kmeans_n_init: int,
    kmeans_max_iter: int,
) -> None:
    df0, traj_note = ensure_sample_idx(df_raw, group_col=group_col)
    if "step" in df0.columns:
        df0 = df0.sort_values(["sample_idx", "step"], kind="mergesort").reset_index(drop=True)

    hist_col = resolve_accept_history_col(df0, accept_history_col)
    df = add_macd_features(df0)
    df = add_prev_accept_features(df, hist_col)

    needed = ["sample_idx", "accept_length"] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    clean = df[needed].dropna().copy()
    clean["accept_length"] = clean["accept_length"].astype(float)
    clean["is_accept_1"] = (clean["accept_length"] == 1.0).astype(float)

    tr_df, te_df = group_split_by_sample_idx(clean, test_frac=test_frac, seed=seed)

    x_tr = torch.tensor(tr_df[FEATURE_COLS].to_numpy(), dtype=torch.float32)
    x_te = torch.tensor(te_df[FEATURE_COLS].to_numpy(), dtype=torch.float32)
    mu, sigma = standardize_fit(x_tr)
    x_tr = standardize_apply(x_tr, mu, sigma)
    x_te = standardize_apply(x_te, mu, sigma)

    y_tr_reg = torch.tensor(tr_df["accept_length"].to_numpy(), dtype=torch.float32)
    y_te_reg = torch.tensor(te_df["accept_length"].to_numpy(), dtype=torch.float32)
    y_tr_cls = torch.tensor(tr_df["is_accept_1"].to_numpy(), dtype=torch.float32)
    y_te_cls = torch.tensor(te_df["is_accept_1"].to_numpy(), dtype=torch.float32)

    x_tr_np = x_tr.numpy()
    x_te_np = x_te.numpy()
    al_tr = tr_df["accept_length"].to_numpy(dtype=np.float64)
    al_te = te_df["accept_length"].to_numpy(dtype=np.float64)
    bin_tr = tr_df["is_accept_1"].to_numpy(dtype=np.float64)
    bin_te = te_df["is_accept_1"].to_numpy(dtype=np.float64)

    pos = float(y_tr_cls.mean().item())
    pos_weight = ((1 - pos) / max(pos, 1e-6)) if pos > 0 else None

    print(
        f"rows: total={len(clean)} train={len(tr_df)} test={len(te_df)} | "
        f"{traj_note} | accept_history_col={hist_col} | mode={mode}"
    )

    if kmeans_k > 0:
        n_tr = len(tr_df)
        kk = min(int(kmeans_k), n_tr)
        if n_tr < 2 or kk < 2:
            print("kmeans: skipped (need at least 2 train rows and k >= 2)")
        else:
            print_kmeans_summary(
                x_tr_np=x_tr_np,
                x_te_np=x_te_np,
                al_tr=al_tr,
                al_te=al_te,
                bin_tr=bin_tr,
                bin_te=bin_te,
                k=kk,
                seed=seed,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
            )

    if mode in ("both", "regression"):
        if regressor == "linear":
            reg_model = fit_linear_regression(
                x_tr, y_tr_reg, l2=train_l2, steps=train_steps, lr=train_lr
            )
        elif regressor == "mlp":
            reg_model = fit_mlp_regression(
                x_tr,
                y_tr_reg,
                hidden_dim=mlp_hidden,
                n_hidden_layers=mlp_layers,
                l2=train_l2,
                steps=train_steps,
                lr=train_lr,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown regressor={regressor!r}")
        with torch.inference_mode():
            pred_te = reg_model(x_te).squeeze(1)
        m_reg = regression_metrics(y_te_reg, pred_te)
        print(
            "regress accept_length:",
            f"model={regressor}",
            f"MAE={m_reg['mae']:.3f}",
            f"RMSE={m_reg['rmse']:.3f}",
            f"R2={m_reg['r2']:.3f}",
            f"Pearson={m_reg['pearson_r']:.3f}",
        )

    if mode in ("both", "classification"):
        if classifier == "linear":
            clf = fit_logistic_regression(
                x_tr,
                y_tr_cls,
                l2=train_l2,
                steps=train_steps,
                lr=train_lr,
                pos_weight=pos_weight,
            )
        elif classifier == "mlp":
            clf = fit_mlp_classification(
                x_tr,
                y_tr_cls,
                hidden_dim=mlp_hidden,
                n_hidden_layers=mlp_layers,
                l2=train_l2,
                steps=train_steps,
                lr=train_lr,
                seed=seed + 1,
                pos_weight=pos_weight,
            )
        else:
            raise ValueError(f"Unknown classifier={classifier!r}")
        with torch.inference_mode():
            prob_te = torch.sigmoid(clf(x_te).squeeze(1))
        m_cls = classification_metrics(y_te_cls, prob_te)
        print(
            "classify accept_length==1:",
            f"model={classifier}",
            f"acc={m_cls['acc']:.3f}",
            f"balanced_acc={m_cls['balanced_acc']:.3f}",
            f"pos_rate={m_cls['pos_rate']:.3f}",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["human_eval", "gsm8k", "sharegpt", "arc", "litbench", "all"],
        default="all",
    )
    p.add_argument("--entropy-dir", default="entropy", help="Directory holding the CSVs.")
    p.add_argument(
        "--accept-history-col",
        choices=["auto", "accept_depth", "accept_length"],
        default="auto",
        help=(
            "Column used for prev_accept_* features (notebook uses accept_depth). "
            "`auto`: accept_depth if present, else accept_length."
        ),
    )
    p.add_argument(
        "--group-col",
        default=None,
        metavar="COL",
        help=(
            "Trajectory id column (default: sample_idx, or sample_id/prompt_idx aliases, "
            "else derive from step==0)."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument(
        "--mode",
        choices=["both", "classification", "regression"],
        default="both",
        help=(
            "`classification`: only neural/logistic classifier on is_accept_1 (no regression). "
            "`regression`: only predict accept_length. `both`: run both."
        ),
    )
    p.add_argument(
        "--regressor",
        choices=["linear", "mlp"],
        default="linear",
        help="Regressor for accept_length when mode is both or regression.",
    )
    p.add_argument(
        "--classifier",
        choices=["linear", "mlp"],
        default="linear",
        help="Classifier for is_accept_1 when mode is both or classification.",
    )
    p.add_argument(
        "--mlp-hidden",
        type=int,
        default=128,
        help="Hidden width for each hidden layer (MLP regressor or MLP classifier).",
    )
    p.add_argument(
        "--mlp-layers",
        type=int,
        default=1,
        help="Number of hidden ReLU layers (MLP regressor or MLP classifier).",
    )
    p.add_argument(
        "--train-steps",
        "--reg-steps",
        type=int,
        default=800,
        help="Optimization steps for whichever model(s) run.",
    )
    p.add_argument(
        "--train-lr",
        "--reg-lr",
        type=float,
        default=0.05,
        help="Learning rate (Adam for linear heads; Adagrad for MLP reg/class).",
    )
    p.add_argument(
        "--train-l2",
        "--reg-l2",
        type=float,
        default=1e-4,
        help="L2 / weight_decay depending on optimizer.",
    )
    p.add_argument(
        "--kmeans-k",
        type=int,
        default=0,
        help=(
            "If >0, run Lloyd K-means on standardized train features with this many clusters "
            "(0 disables)."
        ),
    )
    p.add_argument(
        "--kmeans-n-init",
        type=int,
        default=10,
        help="Number of random centroid initializations for K-means.",
    )
    p.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=100,
        help="Max Lloyd iterations per K-means run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entropy_dir = Path(args.entropy_dir)

    paths = {
        "human_eval": entropy_dir / "humanevak.csv",
        "gsm8k": entropy_dir / "gsm8k.csv",
        "sharegpt": entropy_dir / "sharegpt.csv",
        "arc": entropy_dir / "arc_challenge.csv",
        "litbench": entropy_dir / "litbench_train.csv",
    }
    todo = list(paths.keys()) if args.dataset == "all" else [args.dataset]

    for ds in todo:
        path = paths[ds]
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")
        df = strip_column_names(pd.read_csv(path, encoding="utf-8-sig"))
        print(f"\n== {ds} ({path}) ==")
        run_one_dataset(
            df,
            seed=args.seed,
            test_frac=args.test_frac,
            accept_history_col=(
                None if args.accept_history_col == "auto" else args.accept_history_col
            ),
            group_col=args.group_col,
            mode=args.mode,
            regressor=args.regressor,
            classifier=args.classifier,
            mlp_hidden=args.mlp_hidden,
            mlp_layers=args.mlp_layers,
            train_steps=args.train_steps,
            train_lr=args.train_lr,
            train_l2=args.train_l2,
            kmeans_k=args.kmeans_k,
            kmeans_n_init=args.kmeans_n_init,
            kmeans_max_iter=args.kmeans_max_iter,
        )


if __name__ == "__main__":
    main()