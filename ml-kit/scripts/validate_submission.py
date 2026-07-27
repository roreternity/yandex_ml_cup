import argparse
import sys

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sub = pd.read_csv(args.submission)
    errors = []

    if list(sample.columns) != list(sub.columns):
        errors.append(f"columns differ: expected {list(sample.columns)}, got {list(sub.columns)}")
    if len(sample) != len(sub):
        errors.append(f"row count differs: expected {len(sample)}, got {len(sub)}")
    if len(sample.columns) >= 1 and list(sample.iloc[:, 0]) != list(sub.iloc[:, 0]):
        errors.append("first/id column values or order differ from sample submission")
    if sub.isna().any().any():
        errors.append("submission contains missing values")

    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("OK: submission shape, columns, id order, and missing values are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

