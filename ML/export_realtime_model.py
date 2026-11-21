import argparse
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLS = ["scenario", "observer", "neighbor", "window_start", "window_end", "seed"]
LABEL = "label_malicious_neighbor"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLS)
    excluded.update(c for c in df.columns if c.startswith("label_") or c.startswith("attack_"))
    return [c for c in df.columns if c not in excluded and df[c].dtype.kind in "biufc"]


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/dataset_all_attacks_with_adaptive.csv",
        help="Training dataset path (.csv or .parquet)",
    )
    ap.add_argument(
        "--out",
        default="models/realtime_logreg_all_attacks_with_adaptive.joblib",
        help="Output model artifact path",
    )
    ap.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "rf"],
        help="Classifier to train and export (logreg=logistic regression, rf=random forest)",
    )
    args = ap.parse_args()

    df = load_data(args.data)

    feature_cols = get_feature_cols(df)

    X = df[feature_cols]
    y = df[LABEL].astype(int).values

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", numeric_pipe, feature_cols)],
        remainder="drop",
    )

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model_name = "RandomForest"
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        model_name = "LogReg"

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", clf),
        ]
    )
    pipe.fit(X, y)

    artifact = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "pipeline": pipe,
        "training_rows": int(len(df)),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"Saved realtime model: {args.out}")
    print(f"Training rows: {len(df):,}")
    print(f"Feature columns: {len(feature_cols)}")


if __name__ == "__main__":
    main()
