import argparse
from pathlib import Path

import pandas as pd


def parse_weights(raw: str | None, n: int):
    if not raw:
        return [1 / n] * n
    weights = [float(x) for x in raw.split(",")]
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    total = sum(weights)
    if total == 0:
        raise ValueError("weights sum to zero")
    return [w / total for w in weights]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--subs", nargs="+", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output", default="blend.csv")
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    frames = [pd.read_csv(p) for p in args.subs]
    weights = parse_weights(args.weights, len(frames))
    id_col = sample.columns[0]
    pred_cols = list(sample.columns[1:])
    out = sample.copy()

    for path, frame in zip(args.subs, frames, strict=True):
        if list(frame.columns) != list(sample.columns):
            raise ValueError(f"{path}: columns do not match sample")
        if not frame[id_col].equals(sample[id_col]):
            raise ValueError(f"{path}: id order does not match sample")

    for col in pred_cols:
        out[col] = 0.0
        for weight, frame in zip(weights, frames, strict=True):
            out[col] += weight * frame[col]

    out.to_csv(args.output, index=False)
    print(f"saved {args.output}")
    print("weights:", dict(zip([Path(p).name for p in args.subs], weights, strict=True)))


if __name__ == "__main__":
    main()

