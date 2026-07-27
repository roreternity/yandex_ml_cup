import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


def encode(train: pd.DataFrame, test: pd.DataFrame):
    data = pd.concat([train, test], axis=0, ignore_index=True)
    cat_cols = data.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in data.columns if c not in cat_cols]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        data[cat_cols] = enc.fit_transform(data[cat_cols].astype("string"))
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="adversarial_feature_importance.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    drop_cols = [c for c in [args.target, args.id_col] if c and c in train.columns]
    common = [c for c in train.columns if c in test.columns and c not in drop_cols]
    x = encode(train[common], test[common])
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(x))
    importances = np.zeros(len(common))
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y), start=1):
        model = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=64, random_state=fold, verbosity=-1)
        model.fit(x.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
        importances += model.feature_importances_ / args.folds
        print(f"fold {fold} done")

    auc = roc_auc_score(y, oof)
    print(f"adversarial_auc={auc:.6f}")
    if auc < 0.60:
        print("train/test look similar enough")
    elif auc < 0.75:
        print("moderate train/test shift: inspect top features")
    else:
        print("strong train/test shift: validation split may be unreliable")

    pd.DataFrame({"feature": common, "importance": importances}).sort_values("importance", ascending=False).to_csv(
        args.output, index=False
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

