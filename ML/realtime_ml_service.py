import argparse
import os
import sys
import warnings

import joblib
import pandas as pd


warnings.filterwarnings("ignore")


def parse_row(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5 or parts[0] != "ROW":
        raise ValueError(f"Bad ROW line: {line!r}")

    observer = int(parts[1])
    neighbor = int(parts[2])
    window_start = float(parts[3])
    window_end = float(parts[4])
    features = {}
    for token in parts[5:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        features[key] = float(value)

    return {
        "observer": observer,
        "neighbor": neighbor,
        "window_start": window_start,
        "window_end": window_end,
        "features": features,
    }


def build_frame(rows, feature_cols):
    data = []
    for row in rows:
        feature_row = {col: 0.0 for col in feature_cols}
        for key, value in row["features"].items():
            if key in feature_row:
                feature_row[key] = value
        data.append(feature_row)
    return pd.DataFrame(data, columns=feature_cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to exported realtime model artifact")
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["pipeline"]
    feature_cols = artifact["feature_cols"]
    model_name = artifact.get("model_name", "model")

    sys.stdout.write(f"READY\t{model_name}\n")
    sys.stdout.flush()

    while True:
        header = sys.stdin.readline()
        if not header:
            break
        header = header.rstrip("\n")
        if not header:
            continue
        if header == "QUIT":
            break

        header_parts = header.split("\t")
        if len(header_parts) < 4 or header_parts[0] != "BATCH":
            raise RuntimeError(f"Unexpected command: {header!r}")

        request_id = header_parts[1]
        n_rows = int(header_parts[3])

        rows = []
        for _ in range(n_rows):
            line = sys.stdin.readline()
            if not line:
                raise RuntimeError("EOF while reading batch rows")
            rows.append(parse_row(line))

        end = sys.stdin.readline()
        if not end or end.rstrip("\n") != "END":
            raise RuntimeError("Missing END after batch")

        if rows:
            X = build_frame(rows, feature_cols)
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = []

        sys.stdout.write(f"RESULT\t{request_id}\t{len(rows)}\n")
        for row, score in zip(rows, scores):
            sys.stdout.write(
                f"SCORE\t{row['observer']}\t{row['neighbor']}\t{float(score):.12f}\n"
            )
        sys.stdout.write("END\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
