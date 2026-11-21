"""
Leave-one-attack-out detection (W2 Step 5).

Tests zero-shot generalization: can a model detect an attack family it has
never seen during training?

For each held_out_scenario in {7 attacks}:
  - Train: baseline + (all other 6 attacks), all 5 seeds
  - Test: held_out_scenario, evaluated per seed for variance
  - Report mean ± std of AUC / AP / F1 / Top-K F1 / per-obs median&p10

Then join with network_metrics.csv to surface DTN delivery/overhead per held-out.

Usage:
  python ML/run_leave_one_attack_out.py \
      --data ML/data/dataset_multiseed.parquet \
      --network_metrics ML/data/network_metrics.csv \
      --out result/leave_one_attack_out.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

LABEL = "label_malicious_neighbor"
ID_COLS = {"observer", "neighbor", "window_start", "window_end", "scenario", "seed"}

DEFAULT_ATTACK_SCENARIOS = [
    "attack_flooding",
    "attack_blackhole",
    "attack_grayhole",
    "attack_selfish",
    "attack_selective_forwarding",
    "attack_adaptive_attack",
    "attack_adaptive_attack_boost",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLS)
    excluded.update(c for c in df.columns if c.startswith("label_") or c.startswith("attack_"))
    return [c for c in df.columns if c not in excluded and df[c].dtype.kind in "biufc"]


def topk_f1(y: np.ndarray, p: np.ndarray) -> float:
    k = int(y.sum())
    if k == 0:
        return float("nan")
    order = np.argsort(-p)
    pred = np.zeros_like(y)
    pred[order[:k]] = 1
    _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return float(f1)


def per_observer_topk_distribution(
    df_test: pd.DataFrame, p_test: np.ndarray, min_evidence: int = 5
) -> tuple[float, float]:
    tmp = df_test[["observer", "neighbor", LABEL]].copy()
    tmp["p"] = p_test
    per_obs_f1 = []
    for _, sub in tmp.groupby("observer"):
        agg = sub.groupby("neighbor", as_index=False).agg(
            p_mean=("p", "mean"), y=(LABEL, "max"), n=("p", "size")
        )
        agg = agg[agg["n"] >= min_evidence]
        if len(agg) == 0:
            continue
        f1 = topk_f1(agg["y"].values, agg["p_mean"].values)
        if not np.isnan(f1):
            per_obs_f1.append(f1)
    if not per_obs_f1:
        return float("nan"), float("nan")
    arr = np.array(per_obs_f1)
    return float(np.median(arr)), float(np.percentile(arr, 10))


def build_pipeline(model_name: str, feature_cols: list[str], rng_seed: int):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_cols)], remainder="drop"
    )
    if model_name == "LogReg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=rng_seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def eval_test_seed(
    df_test_held_out: pd.DataFrame,
    p_test: np.ndarray,
) -> dict:
    y_test = df_test_held_out[LABEL].astype(int).values
    pred_test = (p_test >= 0.5).astype(int)

    if len(np.unique(y_test)) < 2:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "f1": float("nan"),
            "topk_f1": float("nan"),
            "per_obs_median_topk_f1": float("nan"),
            "per_obs_p10_topk_f1": float("nan"),
        }

    auc = float(roc_auc_score(y_test, p_test))
    ap = float(average_precision_score(y_test, p_test))
    _, _, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )

    agg = (
        df_test_held_out.assign(p=p_test)
        .groupby("neighbor", as_index=False)
        .agg(p_mean=("p", "mean"), y=(LABEL, "max"), n=("p", "size"))
    )
    agg = agg[agg["n"] >= 5] if (agg["n"] >= 5).any() else agg
    tk = topk_f1(agg["y"].values, agg["p_mean"].values)

    obs_median, obs_p10 = per_observer_topk_distribution(df_test_held_out, p_test)

    return {
        "auc": auc,
        "ap": ap,
        "f1": float(f1),
        "topk_f1": tk,
        "per_obs_median_topk_f1": obs_median,
        "per_obs_p10_topk_f1": obs_p10,
    }


def aggregate(per_seed: list[dict], held_out: str, model_name: str) -> dict:
    metrics = ["auc", "ap", "f1", "topk_f1", "per_obs_median_topk_f1", "per_obs_p10_topk_f1"]
    out = {"held_out": held_out, "model": model_name}
    for m in metrics:
        vals = np.array([r[m] for r in per_seed], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
        else:
            out[f"{m}_mean"] = float(vals.mean())
            out[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) >= 2 else 0.0
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to multi-seed dataset")
    p.add_argument(
        "--network_metrics",
        default="ML/data/network_metrics.csv",
        help="CSV produced by aggregate_network_metrics.py",
    )
    p.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ATTACK_SCENARIOS),
        help="Comma-separated attack scenarios (each will be held out once)",
    )
    p.add_argument("--baseline_scenario", default="baseline")
    p.add_argument("--models", default="LogReg,RF")
    p.add_argument("--rng_seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")
    if "seed" not in df.columns:
        print("ERROR: dataset has no 'seed' column", file=sys.stderr)
        return 1

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_seeds = sorted(df["seed"].unique().tolist())
    feature_cols = get_feature_cols(df)

    print(f"Held-out candidates: {scenarios}")
    print(f"Models: {models}")
    print(f"Seeds for variance: {all_seeds}")
    print(f"Features: {len(feature_cols)} columns")
    print()

    aggregated_rows: list[dict] = []
    for held_out in scenarios:
        train_scenarios = [args.baseline_scenario] + [s for s in scenarios if s != held_out]
        train_mask = df["scenario"].isin(train_scenarios)
        train_df = df[train_mask]
        test_df_all = df[df["scenario"] == held_out]

        print(f"\nHeld-out: {held_out}")
        print(f"  Train scenarios: {train_scenarios}")
        print(f"  Train rows: {len(train_df):,}, Test rows: {len(test_df_all):,}")

        for model_name in models:
            print(f"  Fitting {model_name} ...", end=" ", flush=True)
            pipe = build_pipeline(model_name, feature_cols, args.rng_seed)
            pipe.fit(train_df[feature_cols], train_df[LABEL].astype(int).values)
            print("done.")

            per_seed_results: list[dict] = []
            for test_seed in all_seeds:
                sub = test_df_all[test_df_all["seed"] == test_seed]
                if len(sub) == 0:
                    continue
                p_test = pipe.predict_proba(sub[feature_cols])[:, 1]
                fold = eval_test_seed(sub, p_test)
                per_seed_results.append(fold)
                print(f"    seed={test_seed}: AUC={fold['auc']:.3f}, topk_f1={fold['topk_f1']:.3f}")
            aggregated_rows.append(aggregate(per_seed_results, held_out, model_name))

    result = pd.DataFrame(aggregated_rows)

    # Join with network metrics by held-out scenario
    net_path = Path(args.network_metrics)
    if net_path.exists():
        net = pd.read_csv(net_path)
        net_summary = (
            net.groupby("scenario", as_index=False)
            .agg(
                delivery_prob_mean=("delivery_prob", "mean"),
                overhead_ratio_mean=("overhead_ratio", "mean"),
            )
            .rename(columns={"scenario": "held_out"})
        )
        result = result.merge(net_summary, on="held_out", how="left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print()
    print("=" * 80)
    print("LOAO summary: mean ± std across 5 test seeds")
    print("=" * 80)
    cols = [
        "held_out",
        "model",
        "auc_mean",
        "auc_std",
        "topk_f1_mean",
        "per_obs_p10_topk_f1_mean",
    ]
    if "delivery_prob_mean" in result.columns:
        cols.append("delivery_prob_mean")
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
