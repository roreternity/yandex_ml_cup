import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


def infer_task(y: pd.Series) -> str:
    if y.dtype.kind in "ifu" and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def metric_score(y_true, pred, task: str, metric: str) -> float:
    if metric == "auto":
        metric = "rmse" if task == "regression" else "logloss"
    if metric == "rmse":
        return mean_squared_error(y_true, pred, squared=False)
    if metric == "mae":
        return mean_absolute_error(y_true, pred)
    if metric == "auc":
        return roc_auc_score(y_true, pred)
    if metric == "accuracy":
        labels = np.argmax(pred, axis=1) if getattr(pred, "ndim", 1) > 1 else (pred > 0.5).astype(int)
        return accuracy_score(y_true, labels)
    if metric == "logloss":
        return log_loss(y_true, pred)
    raise ValueError(f"Unsupported metric: {metric}")


def encode_frames(train_x: pd.DataFrame, test_x: pd.DataFrame):
    cat_cols = train_x.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in train_x.columns if c not in cat_cols]
    train_out = train_x.copy()
    test_out = test_x.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        train_out[cat_cols] = enc.fit_transform(train_out[cat_cols].astype("string"))
        test_out[cat_cols] = enc.transform(test_out[cat_cols].astype("string"))
    for col in num_cols:
        median = train_out[col].median()
        train_out[col] = train_out[col].fillna(median)
        test_out[col] = test_out[col].fillna(median)
    return train_out, test_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--metric", default="auto", choices=["auto", "rmse", "mae", "auc", "accuracy", "logloss"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    y = train[args.target]
    task = infer_task(y)
    drop_cols = [args.target] + ([args.id_col] if args.id_col else [])
    features = [c for c in train.columns if c not in drop_cols]
    train_x, test_x = encode_frames(train[features], test[features])

    if task == "classification":
        classes = np.sort(y.unique())
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        oof = np.zeros((len(train), len(classes))) if len(classes) > 2 else np.zeros(len(train))
        test_pred = np.zeros((len(test), len(classes))) if len(classes) > 2 else np.zeros(len(test))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_x, y), start=1):
        if task == "regression":
            model = LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=64, random_state=fold, verbosity=-1)
            model.fit(train_x.iloc[tr_idx], y.iloc[tr_idx])
            oof[va_idx] = model.predict(train_x.iloc[va_idx])
            test_pred += model.predict(test_x) / args.folds
        else:
            model = LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=64, random_state=fold, verbosity=-1)
            model.fit(train_x.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict_proba(train_x.iloc[va_idx])
            full_pred = np.zeros((len(va_idx), len(classes)))
            for idx, cls in enumerate(model.classes_):
                full_pred[:, np.where(classes == cls)[0][0]] = pred[:, idx]
            if len(classes) == 2:
                oof[va_idx] = full_pred[:, 1]
                test_pred += model.predict_proba(test_x)[:, list(model.classes_).index(classes[1])] / args.folds
            else:
                oof[va_idx] = full_pred
                fold_test = model.predict_proba(test_x)
                for idx, cls in enumerate(model.classes_):
                    test_pred[:, np.where(classes == cls)[0][0]] += fold_test[:, idx] / args.folds
        print(f"fold {fold} done")

    score = metric_score(y, oof, task, args.metric)
    print(f"task={task} metric={args.metric} cv={score:.6f}")

    out = pd.DataFrame()
    out[args.id_col or "id"] = test[args.id_col] if args.id_col else np.arange(len(test))
    if task == "classification" and len(classes) > 2:
        for i, cls in enumerate(classes):
            out[f"proba_{cls}"] = test_pred[:, i]
    else:
        out[args.target] = test_pred
    out.to_csv(args.output, index=False)
    pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(Path(args.output).with_suffix(".feature_importance.csv"), index=False)


if __name__ == "__main__":
    main()

