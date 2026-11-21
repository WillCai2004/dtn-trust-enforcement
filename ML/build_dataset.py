# build_dataset.py
import argparse
import os
import glob
import re
from typing import Dict, Set, Tuple

import pandas as pd


ID_COLS = {"observer", "neighbor", "window_start", "window_end", "scenario", "seed"}


def parse_ranges(spec: str) -> Set[int]:
    """
    spec examples:
      "" (empty) -> empty set
      "80-99"
      "80-99,10,200-205"
    """
    out: Set[int] = set()
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(p))
    return out


def parse_malicious_by_scenario(spec: str) -> Dict[str, Set[int]]:
    """
    spec example:
      "attack_flooding=80-99;attack_blackhole=80-99;baseline="
    """
    out: Dict[str, Set[int]] = {}
    if not spec:
        return out
    pairs = [x.strip() for x in spec.split(";") if x.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad malicious_by_scenario entry: {pair} (expected name=ranges)")
        name, ranges = pair.split("=", 1)
        name = name.strip()
        ranges = ranges.strip()
        out[name] = parse_ranges(ranges)
    return out


def to_int_id(x):
    """
    Robustly convert observer/neighbor to int.
    If already numeric -> int
    If string contains digits (e.g., 'p12') -> 12
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(x)
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot convert id to int: {x!r}")


def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def empty_attack_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "node",
            "window_start",
            "window_end",
            "attack_mode_active",
            "attack_opportunity_total",
            "attack_drop_total",
            "attack_targeted_opportunity_total",
            "attack_targeted_drop_total",
        ]
    )


def load_scenario(
    logs_root: str,
    scenario: str,
    log_subdir: str = None,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load logs for one scenario (optionally one seed).

    - logs_root: parent dir containing per-run subdirs.
    - scenario: tag written to the scenario column (kept as the base name).
    - log_subdir: optional subdir name override (defaults to scenario). For
      multi-seed runs, pass `<scenario>_seed<N>` here.
    - seed: optional seed value to attach as a column.
    """
    subdir = log_subdir if log_subdir is not None else scenario
    scen_dir = os.path.join(logs_root, subdir)
    direct_files = sorted(glob.glob(os.path.join(scen_dir, "node_*_direct.csv")))
    self_files = sorted(glob.glob(os.path.join(scen_dir, "node_*_self.csv")))
    attack_files = sorted(glob.glob(os.path.join(scen_dir, "node_*_attack.csv")))

    if not direct_files:
        raise FileNotFoundError(f"No direct CSV files found under: {scen_dir}")
    if not self_files:
        raise FileNotFoundError(f"No self CSV files found under: {scen_dir}")

    def _tag(df):
        df = df.assign(scenario=scenario)
        if seed is not None:
            df = df.assign(seed=seed)
        return df

    direct = pd.concat([_tag(safe_read_csv(f)) for f in direct_files], ignore_index=True)
    self_df = pd.concat([_tag(safe_read_csv(f)) for f in self_files], ignore_index=True)
    if attack_files:
        attack_df = pd.concat([_tag(safe_read_csv(f)) for f in attack_files], ignore_index=True)
    else:
        attack_df = _tag(empty_attack_frame())
    return direct, self_df, attack_df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ID_COLS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace({0: 1})


def dedupe_logs(
    direct: pd.DataFrame, self_df: pd.DataFrame, attack_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop duplicate log rows before merge to avoid multiplicative joins."""
    seed_in_direct = ["seed"] if "seed" in direct.columns else []
    seed_in_self = ["seed"] if "seed" in self_df.columns else []
    seed_in_attack = ["seed"] if "seed" in attack_df.columns else []
    direct_keys = ["scenario"] + seed_in_direct + ["observer", "neighbor", "window_start", "window_end"]
    self_keys = ["scenario"] + seed_in_self + ["observer", "window_start", "window_end"]
    attack_keys = ["scenario"] + seed_in_attack + ["node", "window_start", "window_end"]
    direct = direct.drop_duplicates(subset=direct_keys, keep="first").copy()
    self_df = self_df.drop_duplicates(subset=self_keys, keep="first").copy()
    attack_df = attack_df.drop_duplicates(subset=attack_keys, keep="first").copy()
    return direct, self_df, attack_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", required=True, help="Path to logs/ directory (contains scenario subfolders)")
    ap.add_argument("--scenarios", required=True,
                    help="Comma-separated scenario names, e.g. baseline,attack_flooding")
    ap.add_argument("--seeds", default="",
                    help="Optional comma-separated seeds (e.g. 1,2,3,4,5). When set, "
                         "looks for <scenario>_seed<N>/ subdirs and emits a `seed` column.")
    ap.add_argument("--malicious", default="",
                    help="Global malicious neighbor id ranges (fallback), e.g. 80-99")
    ap.add_argument("--malicious_by_scenario", default="",
                    help="Per-scenario malicious ranges, e.g. "
                         "\"attack_flooding=80-99;attack_blackhole=80-99;baseline=\"")
    ap.add_argument("--out", required=True, help="Output dataset path (.parquet or .csv)")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds_spec = args.seeds.strip()
    if seeds_spec:
        seeds = [int(s.strip()) for s in seeds_spec.split(",") if s.strip()]
    else:
        seeds = [None]
    global_mal = parse_ranges(args.malicious)
    mal_map = parse_malicious_by_scenario(args.malicious_by_scenario)

    all_rows = []
    for scen in scenarios:
      for seed in seeds:
        log_subdir = scen if seed is None else f"{scen}_seed{seed}"
        direct, self_df, attack_df = load_scenario(
            args.logs_root, scen, log_subdir=log_subdir, seed=seed
        )

        direct.columns = [c.strip() for c in direct.columns]
        self_df.columns = [c.strip() for c in self_df.columns]
        attack_df.columns = [c.strip() for c in attack_df.columns]

        # Required columns
        for k in ["observer", "neighbor", "window_start", "window_end"]:
            if k not in direct.columns:
                raise KeyError(f"[{scen}] direct.csv missing column: {k}")
        for k in ["observer", "window_start", "window_end"]:
            if k not in self_df.columns:
                raise KeyError(f"[{scen}] self.csv missing column: {k}")
        for k in ["node", "window_start", "window_end"]:
            if k not in attack_df.columns:
                raise KeyError(f"[{scen}] attack.csv missing column: {k}")

        # Convert ids to int robustly
        direct["observer"] = direct["observer"].apply(to_int_id)
        direct["neighbor"] = direct["neighbor"].apply(to_int_id)
        self_df["observer"] = self_df["observer"].apply(to_int_id)
        attack_df["node"] = attack_df["node"].apply(to_int_id)

        # Defensive dedupe in case logs were appended across repeated runs
        direct, self_df, attack_df = dedupe_logs(direct, self_df, attack_df)

        seed_keys = ["seed"] if "seed" in direct.columns else []
        merged = direct.merge(
            self_df,
            on=["scenario"] + seed_keys + ["observer", "window_start", "window_end"],
            how="left",
            suffixes=("", "_selfdup")
        )
        attack_join = attack_df.rename(columns={"node": "neighbor"})
        merged = merged.merge(
            attack_join,
            on=["scenario"] + seed_keys + ["neighbor", "window_start", "window_end"],
            how="left",
            suffixes=("", "_attackdup"),
        )

        # filter out warmup/negative windows
        merged = merged[merged["window_end"] > 0].copy()

        # Label by scenario-specific malicious set (fallback to global)
        mal_set = mal_map.get(scen, global_mal)
        merged["label_malicious_neighbor"] = merged["neighbor"].isin(mal_set).astype(int)

        merged = ensure_numeric(merged)

        # Fill numeric NaNs with 0
        for c in merged.columns:
            if c in ID_COLS:
                continue
            if merged[c].dtype.kind in "biufc":
                merged[c] = merged[c].fillna(0)

        merged["label_attack_mode_window"] = (merged.get("attack_mode_active", 0) > 0).astype(int)
        merged["label_attack_action_window"] = (merged.get("attack_drop_total", 0) > 0).astype(int)

        # Derived features (optional but useful)
        if "rx_dup_window" in merged.columns and "rx_total" in merged.columns:
            merged["feat_rx_dup_ratio"] = safe_div(merged["rx_dup_window"], merged["rx_total"])
        if "tx_denied_old" in merged.columns and "tx_attempt" in merged.columns:
            merged["feat_tx_denied_old_ratio"] = safe_div(merged["tx_denied_old"], merged["tx_attempt"])
        if "tx_abort" in merged.columns and "tx_start" in merged.columns:
            merged["feat_tx_abort_ratio"] = safe_div(merged["tx_abort"], merged["tx_start"])
        if "ack_received" in merged.columns and "tx_ok" in merged.columns:
            merged["feat_ack_per_tx_ok"] = safe_div(merged["ack_received"], merged["tx_ok"])

        all_rows.append(merged)

    dataset = pd.concat(all_rows, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.lower().endswith(".parquet"):
        dataset.to_parquet(args.out, index=False)
    else:
        dataset.to_csv(args.out, index=False)

    print(f"Saved dataset: {args.out}")
    print(f"Rows: {len(dataset):,}")
    print("Scenarios:", dataset["scenario"].value_counts().to_dict())
    if "seed" in dataset.columns:
        print("Seeds:", dataset["seed"].value_counts().sort_index().to_dict())
    print("Malicious label ratio:", dataset["label_malicious_neighbor"].value_counts(normalize=True).to_dict())
    if "label_attack_mode_window" in dataset.columns:
        print("Attack-mode label ratio:", dataset["label_attack_mode_window"].value_counts(normalize=True).to_dict())
    if "label_attack_action_window" in dataset.columns:
        print("Attack-action label ratio:", dataset["label_attack_action_window"].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
