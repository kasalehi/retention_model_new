# ==========================================================
# train_and_save.py — Train XGBoost churn model and save joblib
# ==========================================================
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from xgboost import XGBClassifier
import joblib


# ---- feature schema ----
FEATURES_CAT = ['MembershipSubCategory', 'MembershipTerm', 'MembershipType', 'PaymentFrequency', 'Gender']
FEATURES_NUM = ['RegularPayment', 'Age', 'TotalAttendance']
TARGET_COL   = 'Churned'

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only features + target if present (ignore extra columns gracefully)
    needed = set(FEATURES_CAT + FEATURES_NUM + [TARGET_COL])
    present = [c for c in df.columns if c in needed]
    missing = sorted(list(needed - set(present)))
    if TARGET_COL not in present:
        raise ValueError(f"Target column '{TARGET_COL}' not found in file.")
    if missing:
        print(f"[WARN] Missing columns in data (will use NaN + imputers): {missing}")
        # ensure missing columns exist so imputers can work
        for m in missing:
            df[m] = np.nan
    # reorder to consistent column order
    ordered_cols = FEATURES_CAT + FEATURES_NUM + [TARGET_COL]
    df = df[ordered_cols]
    return df

def build_pipeline():
    # numeric: impute median + scale
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])
    # categorical: impute most_frequent + OHE
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preproc = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, FEATURES_CAT),
            ("num", num_pipe, FEATURES_NUM),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preproc),
        ("model", model)
    ])
    return pipe

def main(args):
    df = load_data(args.data)
    X = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET_COL].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # handle potential imbalance
    pos = np.sum(y_tr == 1)
    neg = np.sum(y_tr == 0)
    spw = float(neg / max(pos, 1))

    pipe = build_pipeline()
    pipe.set_params(model__scale_pos_weight=spw)

    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("== Test Metrics ==")
    print(f"Accuracy   : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_te, y_pred):.4f}")
    print(f"Recall     : {recall_score(y_te, y_pred):.4f}")
    print(f"F1         : {f1_score(y_te, y_pred):.4f}")
    print(f"ROC AUC    : {roc_auc_score(y_te, y_prob):.4f}")
    print(f"scale_pos_weight used: {spw:.3f}")

    # save
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "retention_xgb.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "features_cat": FEATURES_CAT,
        "features_num": FEATURES_NUM,
        "target": TARGET_COL,
        "threshold_default": 0.5,
        "scale_pos_weight": spw
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved model to: {model_path}")
    print(f"✅ Saved metadata to: {out_dir/'model_meta.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--outdir", default="models", help="Directory to save model")
    args = parser.parse_args()
    main(args)
