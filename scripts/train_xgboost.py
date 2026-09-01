"""
Vanguardium - XGBoost Multi-Class Training (HTTP-layer attack detection)

Trains an XGBoost classifier on the non-leaky HTTP feature table produced
by extract_features.py.  All underscore-prefixed columns are dropped
before training (see extract_features.py's ANTI-LEAKAGE CONVENTION).

Usage (from project root, with venv active):
    python scripts/train_xgboost.py

Input:
    data/processed/http_features.csv

Output (all to data/models/):
    xgboost_model.pkl          - trained XGBClassifier
    label_encoder.pkl          - LabelEncoder (class name ↔ int)
    xgboost_feature_columns.json - ordered feature list used at training time
    xgboost_results.json       - hyperparams, metrics, classification report

Notes:
    - The model uses class weights (inverse frequency) to compensate for
      imbalanced attack-class counts.  This prioritises recall on minority
      classes at the cost of slightly more false positives on benign,
      consistent with IDS operational priorities.
    - If any class has < 5 samples, a warning is printed but training
      proceeds; expect that class's metrics to be unreliable.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

INPUT_PATH = os.path.join("data", "processed", "http_features.csv")
OUTPUT_DIR = os.path.join("data", "models")

# ───────────────────────────────────────────────────── hyper-parameters ──
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
TEST_RATIO = 0.2
CV_FOLDS = 10
RANDOM_STATE = 42


def load_and_prepare(path: str):
    """Load CSV, drop label-only columns, one-hot encode `method`."""
    df = pd.read_csv(path)

    # ── drop label-only (underscore-prefixed) columns ──
    label_only = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=label_only, inplace=True)

    # ── separate targets ──
    y_col = "label"
    drop_cols = [y_col, "label_binary"]

    # ── drop methodology-artifact columns ──
    # num_headers was confirmed (via manual groupby inspection) to separate
    # traffic-COLLECTION-METHOD rather than attack content: benign traffic
    # was captured via a real Chrome browser (~10.5 headers), while ALL
    # attack classes were generated via python-requests (~6.0-8.2 headers)
    # regardless of what the attack actually was. A model trained on this
    # would be fingerprinting "was this sent by python-requests" rather
    # than detecting malicious payloads, and would not generalize to a
    # real attacker using a browser, curl, or Burp Suite. Kept in the CSV
    # (not underscore-prefixed, since it's a legitimate structural feature
    # in principle) but excluded from training here specifically because
    # of how THIS dataset was collected.
    methodology_artifact_cols = ["num_headers"]
    drop_cols = drop_cols + [c for c in methodology_artifact_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[y_col]

    # ── one-hot encode 'method' ──
    X = pd.get_dummies(X, columns=["method"], prefix="method", dtype=int)

    return X, y


def compute_sample_weights(y_encoded: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weighting, applied per-sample."""
    counts = np.bincount(y_encoded, minlength=len(classes))
    weights = len(y_encoded) / (len(classes) * counts.astype(float))
    return weights[y_encoded]


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run extract_features.py first."
        )

    # ── load ──
    X, y = load_and_prepare(INPUT_PATH)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"Distribution:\n{y.value_counts()}\n")

    # ── warn on thin classes ──
    for cls_name in le.classes_:
        n = (y == cls_name).sum()
        if n < 5:
            warnings.warn(
                f"Class '{cls_name}' has only {n} sample(s) — metrics will "
                f"be unreliable. Consider generating more traffic."
            )

    # ── compute class weights ──
    sample_weights = compute_sample_weights(y_enc, le.classes_)

    # ── stratified train/test split (manual so we keep sample weights aligned) ──
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
        X, y_enc, sample_weights,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # ── train ──
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sw_train)

    # ── evaluate on held-out test set ──
    y_pred = model.predict(X_test)
    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, target_names=le.classes_
    )
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(report_text)
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")

    # ── stratified 10-fold CV on full dataset ──
    print(f"\nRunning {CV_FOLDS}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X, y_enc, cv=cv, scoring="f1_macro"
    )
    print(f"CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── feature importance ──
    importance = model.feature_importances_
    feat_imp = sorted(
        zip(X.columns, importance), key=lambda x: x[1], reverse=True
    )
    print("\nTop-15 feature importances:")
    for feat, imp in feat_imp[:15]:
        print(f"  {feat:30s} {imp:.4f}")

    # ── check for leakage red flags ──
    top_feat, top_imp = feat_imp[0]
    if top_imp > 0.85:
        print(f"\n⚠️  WARNING: Top feature '{top_feat}' has importance {top_imp:.3f} "
              f"(>0.85). This may indicate residual label leakage — investigate.")

    # ── save artifacts ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(OUTPUT_DIR, "xgboost_model.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    feature_cols = list(X.columns)
    with open(os.path.join(OUTPUT_DIR, "xgboost_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    results = {
        "model": "XGBoost (multi:softprob)",
        "params": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
        },
        "features": feature_cols,
        "num_classes": len(le.classes_),
        "classes": list(le.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_weighted_f1": weighted_f1,
        "test_macro_f1": macro_f1,
        f"cv{CV_FOLDS}_macro_f1_mean": cv_scores.mean(),
        f"cv{CV_FOLDS}_macro_f1_std": cv_scores.std(),
        "classification_report": report_dict,
        "feature_importance_top15": {f: float(i) for f, i in feat_imp[:15]},
        "notes": (
            "Non-leaky design: all label-signature features (_sqli_match, "
            "_xss_match, _is_admin_path, _is_cross_owner_basket_access, "
            "_cmdi_match, _path_traversal_match, _ssrf_match) dropped before "
            "training. Model uses only generic structural/statistical features. "
            "Class weights applied via inverse-frequency sample weighting. "
            "'num_headers' additionally excluded from training (kept in CSV): "
            "confirmed via groupby inspection to separate traffic-collection "
            "method (Chrome for benign ~10.5 headers vs python-requests for "
            "ALL attack classes ~6.0-8.2 headers) rather than attack content, "
            "a methodology artifact of this dataset's collection process."
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "xgboost_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to {OUTPUT_DIR}/")
    print("  - xgboost_model.pkl")
    print("  - label_encoder.pkl")
    print("  - xgboost_feature_columns.json")
    print("  - xgboost_results.json")


if __name__ == "__main__":
    main()
