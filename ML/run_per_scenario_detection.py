"""
Per-scenario node-level detection (W2 Step 4).

For each attack scenario, train on (baseline + that scenario) using leave-one-seed-out
cross-validation. Report mean ± std for:
  - Sample-level: AUC, AP, F1@0.5
  - Node-level: Top-K F1
  - Per-observer aggregated: median Top-K F1, p10 Top-K F1

Then join with network_metrics.csv to add DTN columns (delivery_ratio, overhead_ratio)
so the final report stays anchored in network impact, not just ML metrics.

Usage:
  python ML/run_per_scenario_detection.py \
      --data ML/data/dataset_multiseed.parquet \
      --network_metrics ML/data/network_metrics.csv \
      --out result/per_scenario_detection.csv
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
    """Top-K precision/recall/F1 where K = number of true positives."""
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
    """For each observer, aggregate (neighbor, mean p) and compute Top-K F1.

    Returns (median, p10) of per-observer Top-K F1, only counting observers
    that saw at least one true positive (otherwise Top-K is undefined).
    """
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


def evaluate_one(
    df: pd.DataFrame,
    scenario: str,
    test_seed: int,
    model_name: str,
    baseline_scenario: str,
    rng_seed: int,
) -> dict:
    """Train on (baseline + scenario) all seeds except test_seed; test on test_seed."""
    subset = df[df["scenario"].isin([baseline_scenario, scenario])].copy()
    train_df = subset[subset["seed"] != test_seed]
    test_df = subset[subset["seed"] == test_seed]

    feature_cols = get_feature_cols(subset)

    pipe = build_pipeline(model_name, feature_cols, rng_seed)
    pipe.fit(train_df[feature_cols], train_df[LABEL].astype(int).values)

    y_test = test_df[LABEL].astype(int).values
    p_test = pipe.predict_proba(test_df[feature_cols])[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    if len(np.unique(y_test)) < 2:
        return {
            "scenario": scenario,
            "model": model_name,
            "test_seed": test_seed,
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

    # Node-level: aggregate neighbor mean prob across the test set
    agg = (
        test_df.assign(p=p_test)
        .groupby("neighbor", as_index=False)
        .agg(p_mean=("p", "mean"), y=(LABEL, "max"), n=("p", "size"))
    )
    agg = agg[agg["n"] >= 5] if (agg["n"] >= 5).any() else agg
    tk = topk_f1(agg["y"].values, agg["p_mean"].values)

    obs_median, obs_p10 = per_observer_topk_distribution(test_df, p_test)

    return {
        "scenario": scenario,
        "model": model_name,
        "test_seed": test_seed,
        "auc": auc,
        "ap": ap,
        "f1": float(f1),
        "topk_f1": tk,
        "per_obs_median_topk_f1": obs_median,
        "per_obs_p10_topk_f1": obs_p10,
    }


def aggregate_seeds(folds: list[dict]) -> dict:
    """Mean ± std across the seed folds for one (scenario, model)."""
    metrics = ["auc", "ap", "f1", "topk_f1", "per_obs_median_topk_f1", "per_obs_p10_topk_f1"]
    out = {"scenario": folds[0]["scenario"], "model": folds[0]["model"]}
    for m in metrics:
        vals = np.array([f[m] for f in folds], dtype=float)
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
    p.add_argument("--data", required=True, help="Path to multi-seed dataset (.parquet or .csv)")
    p.add_argument(
        "--network_metrics",
        default="ML/data/network_metrics.csv",
        help="Optional CSV produced by aggregate_network_metrics.py to join in",
    )
    p.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ATTACK_SCENARIOS),
        help="Comma-separated attack scenario names",
    )
    p.add_argument(
        "--baseline_scenario",
        default="baseline",
        help="Scenario name for benign data mixed into train/test",
    )
    p.add_argument(
        "--models",
        default="LogReg,RF",
        help="Comma-separated model names to evaluate (LogReg, RF)",
    )
    p.add_argument("--rng_seed", type=int, default=42, help="RNG seed for models")
    p.add_argument("--out", required=True, help="Output CSV")
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
    print(f"Scenarios: {scenarios}")
    print(f"Models: {models}")
    print(f"Seeds: {all_seeds}")
    print()

    aggregated_rows: list[dict] = []
    for scenario in scenarios:
        for model_name in models:
            folds: list[dict] = []
            for test_seed in all_seeds:
                print(f"  fit {model_name} on {scenario} (test_seed={test_seed}) ...", end=" ", flush=True)
                fold = evaluate_one(
                    df,
                    scenario=scenario,
                    test_seed=test_seed,
                    model_name=model_name,
                    baseline_scenario=args.baseline_scenario,
                    rng_seed=args.rng_seed,
                )
                folds.append(fold)
                print(f"AUC={fold['auc']:.3f}, topk_f1={fold['topk_f1']:.3f}")
            aggregated_rows.append(aggregate_seeds(folds))

    result = pd.DataFrame(aggregated_rows)

    # Optional join with network metrics (averaged over seeds per scenario)
    net_path = Path(args.network_metrics)
    if net_path.exists():
        net = pd.read_csv(net_path)
        net_summary = (
            net.groupby("scenario", as_index=False)
            .agg(
                delivery_prob_mean=("delivery_prob", "mean"),
                delivery_prob_std=("delivery_prob", "std"),
                overhead_ratio_mean=("overhead_ratio", "mean"),
                overhead_ratio_std=("overhead_ratio", "std"),
                latency_avg_mean=("latency_avg", "mean"),
            )
        )
        result = result.merge(net_summary, on="scenario", how="left")
    else:
        print(f"WARN: network metrics not found at {net_path}, skipping join")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(result)}")

    # Pretty print summary
    print()
    print("=" * 70)
    print("Summary: per (scenario, model) mean ± std across 5 seed folds")
    print("=" * 70)
    pretty_cols = [
        "scenario",
        "model",
        "auc_mean",
        "auc_std",
        "topk_f1_mean",
        "topk_f1_std",
        "per_obs_median_topk_f1_mean",
        "per_obs_p10_topk_f1_mean",
    ]
    if "delivery_prob_mean" in result.columns:
        pretty_cols.append("delivery_prob_mean")
    print(result[pretty_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
