"""
Vanguardium - CNN Training (character-level, HTTP-layer attack detection)

Third ensemble member alongside XGBoost and BiLSTM. Trains on the same
combined_text column XGBoost's engineered features are derived from
(see extract_features.py's build_combined_text()), so predictions can be
compared row-for-row with XGBoost on a shared held-out test set.

Usage (from project root, with venv active):
    python scripts/train_cnn.py

Input:
    data/processed/http_features.csv
    data/models/label_encoder.pkl   (reused from XGBoost - same class<->int
                                      mapping across models, not refit here)

Notes:
    - Test split is IDENTICAL to train_xgboost.py's (same random_state,
      test_size, stratify column, unmodified row order) so the two models'
      test-set predictions are directly comparable/combinable later during
      ensemble construction. BiLSTM trains on a structurally different
      dataset (CICIDS2017 network flows) and is NOT alignable this way.
    - A further train/val split (from the 80% train portion only) is used
      for Keras early stopping; this split is internal to the CNN and has
      no cross-model alignment requirement.
    - KNOWN LIMITATION (flagged before training, not after): several attack
      classes in this dataset are defined by a fixed vulnerable endpoint
      (e.g. cmdi -> /vulnerabilities/exec/, path_traversal/ssrf-rfi ->
      /vulnerabilities/fi/, ssrf -> /profile/image/url). Because the raw
      endpoint string sits directly in combined_text, the CNN may learn
      "path starts with X" as a near-complete shortcut for some classes
      rather than learning injected-payload syntax. This is a real property
      of how the vulnerable apps are structured, not a labeling bug - but
      high accuracy here should not be read as "the model understands
      attack syntax" without a follow-up check (e.g. filter-activation
      inspection, or held-out endpoint-variant evaluation).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from cnn_tokenizer import MAX_LEN, VOCAB_SIZE, texts_to_matrix, save_vocab

INPUT_PATH = os.path.join("data", "processed", "http_features.csv")
MODELS_DIR = os.path.join("data", "models")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

TEST_RATIO = 0.2          # must match train_xgboost.py exactly
VAL_RATIO_OF_TRAIN = 0.15 # internal to CNN only, no cross-model constraint
RANDOM_STATE = 42


def load_split_and_encode():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find {INPUT_PATH}. Run extract_features.py first.")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            f"Could not find {LABEL_ENCODER_PATH}. Run train_xgboost.py first "
            f"(CNN reuses XGBoost's fitted LabelEncoder for consistent class indices)."
        )

    df = pd.read_csv(INPUT_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    # Sanity check: the label set in this CSV must exactly match what the
    # loaded encoder was fit on. If extract_features.py's labeling logic
    # ever changes without XGBoost being retrained, this catches it instead
    # of silently mis-encoding a class.
    csv_classes = set(df["label"].unique())
    encoder_classes = set(le.classes_)
    if csv_classes != encoder_classes:
        raise ValueError(
            f"Label mismatch between http_features.csv and saved label_encoder.pkl.\n"
            f"CSV has: {sorted(csv_classes)}\n"
            f"Encoder has: {sorted(encoder_classes)}\n"
            f"Retrain XGBoost first so the encoder is in sync."
        )

    y_enc = le.transform(df["label"])

    # -- split 1: replicate train_xgboost.py's split exactly --
    # Splitting on row indices (not X/y arrays directly) so we can recover
    # which original CSV rows ended up in test, for later ensemble use.
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # -- split 2: carve validation out of the train portion (CNN-internal only) --
    y_train_full = y_enc[idx_train]
    idx_train2, idx_val = train_test_split(
        idx_train,
        test_size=VAL_RATIO_OF_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    texts = df["combined_text"].astype(str).values

    X_train = texts_to_matrix(texts[idx_train2], max_len=MAX_LEN)
    X_val = texts_to_matrix(texts[idx_val], max_len=MAX_LEN)
    X_test = texts_to_matrix(texts[idx_test], max_len=MAX_LEN)

    y_train = y_enc[idx_train2]
    y_val = y_enc[idx_val]
    y_test = y_enc[idx_test]

    return {
        "X_train": X_train, "y_train": y_train, "idx_train": idx_train2,
        "X_val": X_val, "y_val": y_val, "idx_val": idx_val,
        "X_test": X_test, "y_test": y_test, "idx_test": idx_test,
        "label_encoder": le,
        "df": df,
    }


def main():
    data = load_split_and_encode()
    print(f"Vocab size: {VOCAB_SIZE}, max_len: {MAX_LEN}")
    print(f"Classes ({len(data['label_encoder'].classes_)}): {list(data['label_encoder'].classes_)}")
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")
    print(f"Train label dist: {np.bincount(data['y_train'])}")
    print(f"Val label dist:   {np.bincount(data['y_val'])}")
    print(f"Test label dist:  {np.bincount(data['y_test'])}")

    # -- verify test split matches XGBoost's exactly --
    expected_test_size = round(len(data["df"]) * TEST_RATIO)
    print(f"\nTest set size: {len(data['idx_test'])} (expected ~{expected_test_size})")

    save_vocab()


if __name__ == "__main__":
    main()