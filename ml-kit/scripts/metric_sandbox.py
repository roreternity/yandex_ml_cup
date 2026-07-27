import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--metric", required=True, choices=["auc", "accuracy", "f1", "logloss", "rmse", "mae"])
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    pred = pd.read_csv(args.pred)
    y = truth[args.target]
    p = pred[args.prediction]

    if args.metric == "auc":
        score = roc_auc_score(y, p)
    elif args.metric == "accuracy":
        score = accuracy_score(y, p)
    elif args.metric == "f1":
        score = f1_score(y, p)
    elif args.metric == "logloss":
        score = log_loss(y, p)
    elif args.metric == "rmse":
        score = mean_squared_error(y, p, squared=False)
    elif args.metric == "mae":
        score = mean_absolute_error(y, p)
    else:
        raise ValueError(args.metric)
    print(f"{args.metric}={score:.8f}")


if __name__ == "__main__":
    main()

