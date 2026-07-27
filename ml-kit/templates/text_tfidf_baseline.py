import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline


def infer_task(y: pd.Series) -> str:
    if y.dtype.kind in "ifu" and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--text-col", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--metric", default="auto", choices=["auto", "auc", "accuracy", "logloss", "rmse"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    y = train[args.target]
    task = infer_task(y)
    x = train[args.text_col].fillna("").astype(str)
    x_test = test[args.text_col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(min_df=2, max_features=300_000, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)

    if task == "classification":
        classes = np.sort(y.unique())
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        oof = np.zeros((len(train), len(classes))) if len(classes) > 2 else np.zeros(len(train))
        test_pred = np.zeros((len(test), len(classes))) if len(classes) > 2 else np.zeros(len(test))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y), start=1):
        if task == "regression":
            model = make_pipeline(vectorizer, Ridge(alpha=10.0))
            model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
            oof[va_idx] = model.predict(x.iloc[va_idx])
            test_pred += model.predict(x_test) / args.folds
        else:
            model = make_pipeline(vectorizer, LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced", solver="liblinear"))
            model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict_proba(x.iloc[va_idx])
            model_classes = model[-1].classes_
            full_pred = np.zeros((len(va_idx), len(classes)))
            for idx, cls in enumerate(model_classes):
                full_pred[:, np.where(classes == cls)[0][0]] = pred[:, idx]
            if len(classes) == 2:
                oof[va_idx] = full_pred[:, 1]
                test_pred += model.predict_proba(x_test)[:, list(model_classes).index(classes[1])] / args.folds
            else:
                oof[va_idx] = full_pred
                fold_test = model.predict_proba(x_test)
                for idx, cls in enumerate(model_classes):
                    test_pred[:, np.where(classes == cls)[0][0]] += fold_test[:, idx] / args.folds
        print(f"fold {fold} done")

    metric = "rmse" if args.metric == "auto" and task == "regression" else "logloss" if args.metric == "auto" else args.metric
    if metric == "rmse":
        score = mean_squared_error(y, oof, squared=False)
    elif metric == "auc":
        score = roc_auc_score(y, oof)
    elif metric == "accuracy":
        labels = np.argmax(oof, axis=1) if getattr(oof, "ndim", 1) > 1 else (oof > 0.5).astype(int)
        score = accuracy_score(y, labels)
    elif metric == "logloss":
        score = log_loss(y, oof)
    else:
        raise ValueError(metric)
    print(f"task={task} metric={metric} cv={score:.6f}")

    out = pd.DataFrame()
    out[args.id_col or "id"] = test[args.id_col] if args.id_col else np.arange(len(test))
    if task == "classification" and len(classes) > 2:
        for i, cls in enumerate(classes):
            out[f"proba_{cls}"] = test_pred[:, i]
    else:
        out[args.target] = test_pred
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

