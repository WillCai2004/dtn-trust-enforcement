"""
Attacker-ratio transfer evaluation (W2 Step 7).

Trains on the default 20%-attacker dataset (existing dataset_multiseed.parquet)
and tests on the 10% / 30% ratio datasets produced by run_ratio_sweep.py +
build_dataset.py. Reports AUC / AP / F1@0.5 / Top-K F1 per (attack, test_ratio)
to quantify how stable detection is under attacker-density shift.

Usage:
  python ML/run_ratio_transfer.py \
      --train_data ML/data/dataset_multiseed.parquet \
      --ratio_data ML/data/dataset_ratio_sweep.parquet \
      --out result/ratio_transfer.csv
"""

import argparse
import re
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


def build_pipeline(model_name: str, feature_cols: list[str], rng_seed: int):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    pre = ColumnTransformer(transformers=[("num", numeric_pipe, feature_cols)], remainder="drop")
    if model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=rng_seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "LogReg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        raise ValueError(model_name)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def parse_attack_and_ratio(scenario: str) -> tuple[str, int]:
    """attack_flooding_ratio30 -> ('attack_flooding', 30)"""
    m = re.match(r"^(.+)_ratio(\d+)$", scenario)
    if not m:
        return scenario, 20  # default ratio
    return m.group(1), int(m.group(2))


def evaluate_on_seed(
    df_test_seed: pd.DataFrame, p_test: np.ndarray
) -> dict:
    y = df_test_seed[LABEL].astype(int).values
    pred = (p_test >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "f1": float("nan"), "topk_f1": float("nan")}
    auc = float(roc_auc_score(y, p_test))
    ap = float(average_precision_score(y, p_test))
    _, _, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    agg = (
        df_test_seed.assign(p=p_test)
        .groupby("neighbor", as_index=False)
        .agg(p_mean=("p", "mean"), y=(LABEL, "max"), n=("p", "size"))
    )
    agg = agg[agg["n"] >= 5] if (agg["n"] >= 5).any() else agg
    tk = topk_f1(agg["y"].values, agg["p_mean"].values)
    return {"auc": auc, "ap": ap, "f1": float(f1), "topk_f1": tk}


def aggregate(per_seed: list[dict], attack: str, test_ratio: int, model_name: str) -> dict:
    out = {"attack": attack, "model": model_name, "test_ratio": test_ratio}
    for m in ("auc", "ap", "f1", "topk_f1"):
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
    p.add_argument(
        "--train_data",
        required=True,
        help="Dataset for training (e.g. multi-seed 20% dataset)",
    )
    p.add_argument(
        "--ratio_data",
        required=True,
        help="Dataset for testing at alternate ratios (built from run_ratio_sweep outputs)",
    )
    p.add_argument(
        "--attacks",
        default="attack_flooding,attack_grayhole,attack_adaptive_attack",
        help="Comma-separated attack scenarios (base names without _ratioN suffix)",
    )
    p.add_argument("--baseline_scenario", default="baseline")
    p.add_argument("--models", default="LogReg,RF")
    p.add_argument("--rng_seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    train_df = pd.read_parquet(args.train_data) if args.train_data.endswith(".parquet") else pd.read_csv(args.train_data)
    test_df = pd.read_parquet(args.ratio_data) if args.ratio_data.endswith(".parquet") else pd.read_csv(args.ratio_data)
    print(f"Train data: {len(train_df):,} rows from {args.train_data}")
    print(f"Test data:  {len(test_df):,} rows from {args.ratio_data}")

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Test ratios available in ratio_data
    test_scenarios = test_df["scenario"].unique().tolist()
    ratios_per_attack: dict[str, list[int]] = {}
    for scen in test_scenarios:
        base, ratio = parse_attack_and_ratio(scen)
        ratios_per_attack.setdefault(base, []).append(ratio)
    print(f"Detected test ratios: {ratios_per_attack}")

    # Feature columns must be consistent between train and test
    feature_cols = [c for c in get_feature_cols(train_df) if c in test_df.columns]
    print(f"Features (intersection): {len(feature_cols)} cols")

    aggregated_rows: list[dict] = []
    seeds_test = sorted(test_df["seed"].unique().tolist())

    for attack in attacks:
        train_sub = train_df[train_df["scenario"].isin([args.baseline_scenario, attack])]
        if train_sub.empty:
            print(f"WARN: no training data for {attack}")
            continue
        for model_name in models:
            print(f"\nTraining {model_name} on baseline + {attack} (20%, n={len(train_sub):,}) ...", flush=True)
            pipe = build_pipeline(model_name, feature_cols, args.rng_seed)
            pipe.fit(train_sub[feature_cols], train_sub[LABEL].astype(int).values)
            print("  trained.")

            for ratio in sorted(ratios_per_attack.get(attack, [])):
                test_scen = f"{attack}_ratio{ratio}"
                test_data_scen = test_df[test_df["scenario"] == test_scen]
                if test_data_scen.empty:
                    continue
                per_seed_results: list[dict] = []
                for s in seeds_test:
                    sub = test_data_scen[test_data_scen["seed"] == s]
                    if len(sub) == 0:
                        continue
                    p_t = pipe.predict_proba(sub[feature_cols])[:, 1]
                    fold = evaluate_on_seed(sub, p_t)
                    per_seed_results.append(fold)
                row = aggregate(per_seed_results, attack, ratio, model_name)
                aggregated_rows.append(row)
                print(f"  test_ratio={ratio}%: AUC={row['auc_mean']:.3f}±{row['auc_std']:.3f}, "
                      f"F1={row['f1_mean']:.3f}, topk_f1={row['topk_f1_mean']:.3f}")

    result = pd.DataFrame(aggregated_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print()
    print("=" * 80)
    print("Ratio transfer summary (train at 20%, test at variable ratio)")
    print("=" * 80)
    cols = ["attack", "model", "test_ratio", "auc_mean", "auc_std", "f1_mean", "topk_f1_mean"]
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
