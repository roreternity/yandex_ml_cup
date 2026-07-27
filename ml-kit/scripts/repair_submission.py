import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", default="submission_repaired.csv")
    parser.add_argument("--fillna", type=float, default=0.0)
    parser.add_argument("--clip-min", type=float, default=None)
    parser.add_argument("--clip-max", type=float, default=None)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sub = pd.read_csv(args.submission)
    id_col = sample.columns[0]
    pred_cols = list(sample.columns[1:])

    if id_col not in sub.columns:
        raise ValueError(f"submission has no id column: {id_col}")

    sub = sample[[id_col]].merge(sub, on=id_col, how="left", validate="one_to_one")
    for col in pred_cols:
        if col not in sub.columns:
            sub[col] = sample[col]
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(args.fillna)
        if args.clip_min is not None or args.clip_max is not None:
            sub[col] = np.clip(
                sub[col],
                -np.inf if args.clip_min is None else args.clip_min,
                np.inf if args.clip_max is None else args.clip_max,
            )

    sub[list(sample.columns)].to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

