"""
Feature importance analysis (W2 Step 6).

For each attack scenario, train RF on (baseline + scenario) and rank features
by `feature_importances_`. Also train LogReg and rank by |coefficient|.

Outputs three CSVs:
  - feature_importance_per_scenario.csv  (long format: scenario, model, feature, importance, rank, dtn_native)
  - top5_per_scenario.csv  (wide table: rows = features, cols = scenarios, values = rank if in top-5 else blank)
  - feature_overlap.csv  (Jaccard overlap of top-K feature sets between every pair of attack scenarios)

The overlap table directly explains LOAO patterns: high overlap pairs
generalize well, low overlap pairs do not.

Usage:
  python ML/run_feature_importance.py \
      --data ML/data/dataset_multiseed.parquet \
      --top_k 5 \
      --out_dir result/feature_importance/
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

# Hand-curated DTN-native feature classification. A feature is DTN-native if
# its semantic is specific to opportunistic / store-carry-forward networks
# and would not appear in conventional packet-switched networks.
DTN_NATIVE_FEATURES = {
    # Encounter / mobility-driven
    "contacts",
    "contact_time",
    # Multi-copy routing artifacts
    "rx_dup",
    "rx_dup_window",
    "feat_rx_dup_ratio",
    # Opportunistic forwarding semantics
    "rx_as_source",      # message looped back to originator via store-carry
    "rx_as_relay",       # asked to carry for others
    "tx_to_neighbor_as_relay",
    # DTN-specific buffer / TTL policies
    "tx_denied_old",     # DENIED_OLD = duplicate/already-seen
    "feat_tx_denied_old_ratio",
    # Opportunistic ACK timing
    "ack_delay_sum",
    "ack_delay_avg",
    "ack_delay_count",
}


def feature_class(name: str) -> str:
    return "DTN-native" if name in DTN_NATIVE_FEATURES else "generic"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLS)
    excluded.update(c for c in df.columns if c.startswith("label_") or c.startswith("attack_"))
    return [c for c in df.columns if c not in excluded and df[c].dtype.kind in "biufc"]


def build_pipeline(model_name: str, feature_cols: list[str], rng_seed: int):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    pre = ColumnTransformer(transformers=[("num", numeric_pipe, feature_cols)], remainder="drop")
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
        raise ValueError(model_name)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def importance_for_scenario(
    df: pd.DataFrame, scenario: str, baseline_scenario: str, feature_cols: list[str], rng_seed: int
) -> pd.DataFrame:
    sub = df[df["scenario"].isin([baseline_scenario, scenario])]
    X = sub[feature_cols]
    y = sub[LABEL].astype(int).values

    rows = []
    for model_name in ("RF", "LogReg"):
        pipe = build_pipeline(model_name, feature_cols, rng_seed)
        pipe.fit(X, y)
        clf = pipe.named_steps["clf"]
        if model_name == "RF":
            importances = clf.feature_importances_
        else:
            # LogReg coefficient magnitude (after StandardScaler this is comparable)
            importances = np.abs(clf.coef_[0])
        # Normalize so they sum to 1 within a model
        if importances.sum() > 0:
            importances = importances / importances.sum()
        order = np.argsort(-importances)
        for rank, idx in enumerate(order, 1):
            feat = feature_cols[idx]
            rows.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "feature": feat,
                    "importance": float(importances[idx]),
                    "rank": rank,
                    "dtn_native": feature_class(feat) == "DTN-native",
                }
            )
    return pd.DataFrame(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def build_top5_wide(per_scenario: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Wide table: features as rows, scenarios as cols, value = rank if in top-K else NaN.
    Uses RF importance ranking.
    """
    rf = per_scenario[per_scenario["model"] == "RF"]
    top = rf[rf["rank"] <= top_k]
    wide = top.pivot_table(index="feature", columns="scenario", values="rank", aggfunc="first")
    # Sort by how many scenarios where feature was top-k (most universal first)
    wide["appearances"] = wide.notna().sum(axis=1)
    wide = wide.sort_values("appearances", ascending=False)
    # Add DTN-native column
    wide["dtn_native"] = [feature_class(f) == "DTN-native" for f in wide.index]
    return wide


def build_overlap_matrix(per_scenario: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rf = per_scenario[per_scenario["model"] == "RF"]
    top = rf[rf["rank"] <= top_k]
    sets = {scen: set(group["feature"]) for scen, group in top.groupby("scenario")}
    scenarios = sorted(sets.keys())
    mat = pd.DataFrame(index=scenarios, columns=scenarios, dtype=float)
    for s1 in scenarios:
        for s2 in scenarios:
            mat.loc[s1, s2] = jaccard(sets[s1], sets[s2])
    return mat


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--scenarios", default=",".join(DEFAULT_ATTACK_SCENARIOS))
    p.add_argument("--baseline_scenario", default="baseline")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--rng_seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    feature_cols = get_feature_cols(df)
    n_dtn = sum(1 for f in feature_cols if feature_class(f) == "DTN-native")
    print(f"Features: {len(feature_cols)} total ({n_dtn} DTN-native, {len(feature_cols) - n_dtn} generic)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for scen in scenarios:
        print(f"  fitting on baseline + {scen} ...", flush=True)
        all_rows.append(
            importance_for_scenario(df, scen, args.baseline_scenario, feature_cols, args.rng_seed)
        )
    per_scenario = pd.concat(all_rows, ignore_index=True)

    per_scenario.to_csv(out_dir / "feature_importance_per_scenario.csv", index=False)
    print(f"Saved: {out_dir / 'feature_importance_per_scenario.csv'}")

    wide = build_top5_wide(per_scenario, args.top_k)
    wide.to_csv(out_dir / f"top{args.top_k}_per_scenario.csv")
    print(f"Saved: {out_dir / f'top{args.top_k}_per_scenario.csv'}")

    overlap = build_overlap_matrix(per_scenario, args.top_k)
    overlap.to_csv(out_dir / f"feature_overlap_top{args.top_k}.csv")
    print(f"Saved: {out_dir / f'feature_overlap_top{args.top_k}.csv'}")

    # Print summaries
    print()
    print("=" * 80)
    print(f"RF Top-{args.top_k} features per scenario (rank shown, blank = not in top-{args.top_k})")
    print("=" * 80)
    print(wide.to_string())

    print()
    print("=" * 80)
    print(f"Jaccard overlap of RF top-{args.top_k} feature sets between scenarios")
    print("=" * 80)
    print(overlap.to_string(float_format=lambda x: f"{x:.2f}"))

    # DTN-native vs generic share of top features
    print()
    print("=" * 80)
    print(f"DTN-native share among top-{args.top_k} features (RF)")
    print("=" * 80)
    rf_top = per_scenario[(per_scenario["model"] == "RF") & (per_scenario["rank"] <= args.top_k)]
    share = rf_top.groupby("scenario")["dtn_native"].agg(["sum", "count"])
    share["dtn_native_share"] = share["sum"] / share["count"]
    print(share.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
