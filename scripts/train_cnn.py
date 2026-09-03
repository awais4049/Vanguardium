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

def build_model(vocab_size: int, max_len: int, num_classes: int):
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=32),
        layers.SpatialDropout1D(0.2),
        layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> dict:
    """Inverse-frequency class weighting, same rationale as XGBoost/BiLSTM:
    false negatives (missed attacks) costlier than false positives in IDS."""
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    weights = total / (num_classes * counts.astype(float))
    return {i: float(w) for i, w in enumerate(weights)}

def main():
    data = load_split_and_encode()
    le = data["label_encoder"]
    num_classes = len(le.classes_)

    print(f"Vocab size: {VOCAB_SIZE}, max_len: {MAX_LEN}")
    print(f"Classes ({num_classes}): {list(le.classes_)}")
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")

    model = build_model(VOCAB_SIZE, MAX_LEN, num_classes)
    model.summary()

    class_weights = compute_class_weights(data["y_train"], num_classes)
    print(f"\nClass weights: {class_weights}")

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2,
    )

    # -- evaluate on the XGBoost-aligned held-out test set --
    from sklearn.metrics import classification_report, f1_score

    y_pred_proba = model.predict(data["X_test"])
    y_pred = np.argmax(y_pred_proba, axis=1)

    report = classification_report(
        data["y_test"], y_pred, target_names=le.classes_, output_dict=True
    )
    macro_f1 = f1_score(data["y_test"], y_pred, average="macro")
    weighted_f1 = f1_score(data["y_test"], y_pred, average="weighted")

    print(f"\nTest macro F1: {macro_f1:.4f}")
    print(f"Test weighted F1: {weighted_f1:.4f}")
    print(classification_report(data["y_test"], y_pred, target_names=le.classes_))

    # -- save artifacts --
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, "cnn_model.keras"))

    results = {
        "model": "CNN (char-level, Conv1D)",
        "architecture": {
            "embedding_dim": 32,
            "conv_blocks": [
                {"filters": 64, "kernel_size": 5, "pool": "MaxPooling1D(2)"},
                {"filters": 128, "kernel_size": 3, "pool": "GlobalMaxPooling1D"},
            ],
            "dense_head": [64],
            "dropout": {"spatial": 0.2, "dense": 0.5},
        },
        "vocab_size": VOCAB_SIZE,
        "max_len": MAX_LEN,
        "num_classes": num_classes,
        "classes": list(le.classes_),
        "train_samples": int(data["X_train"].shape[0]),
        "val_samples": int(data["X_val"].shape[0]),
        "test_samples": int(data["X_test"].shape[0]),
        "epochs_trained": len(history.history["loss"]),
        "test_weighted_f1": float(weighted_f1),
        "test_macro_f1": float(macro_f1),
        "classification_report": report,
        "class_weights": class_weights,
        "notes": (
            "Test split IDENTICAL to train_xgboost.py's (same random_state=42, "
            "test_size=0.2, stratify=label, unmodified row order) - verified "
            "matching test-set label distribution before training. Trained on "
            "combined_text (decoded path+query_params+body), same text XGBoost's "
            "engineered stats are derived from (extract_features.py's "
            "build_combined_text()). KNOWN LIMITATION: several classes are "
            "defined by a fixed vulnerable endpoint string (cmdi, path_traversal, "
            "ssrf); the CNN may partly learn endpoint-matching as a shortcut "
            "rather than injected-payload syntax - see module docstring."
        ),
    }
    with open(os.path.join(MODELS_DIR, "cnn_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR}/cnn_model.keras")
    print(f"Saved results to {MODELS_DIR}/cnn_results.json")


if __name__ == "__main__":
    main()