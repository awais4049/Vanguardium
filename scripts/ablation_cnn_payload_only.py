"""
Vanguardium - CNN Ablation: payload-only (no endpoint path) text

Diagnostic script, not part of the main pipeline. Tests whether the CNN's
near-perfect test score (macro F1 0.9933 on full combined_text) reflects
genuine attack-syntax learning or a shortcut on fixed vulnerable-endpoint
strings (e.g. cmdi -> /vulnerabilities/exec/, path_traversal/ssrf-rfi ->
/vulnerabilities/fi/, ssrf -> /profile/image/url) that sit at a fixed
position in combined_text.

Rebuilds each row's text as query_params + body ONLY (endpoint path
stripped), re-joins to labels via _capture_line_id, trains the SAME
architecture on the SAME split, and compares per-class F1 to the full
combined_text result in cnn_results.json.

Usage:
    python scripts/ablation_cnn_payload_only.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import build_combined_text, INPUT_PATH as CAPTURE_LOG_PATH
from cnn_tokenizer import MAX_LEN, VOCAB_SIZE, texts_to_matrix
from train_cnn import load_split_and_encode, build_model, compute_class_weights


def build_payload_only_lookup(capture_log_path: str) -> dict:
    """capture_line_id -> 'query_params_json decoded_body' (path excluded)."""
    lookup = {}
    with open(capture_log_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                flow = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = build_combined_text(flow)
            payload_only = f'{json.dumps(parsed["query_params"])} {parsed["decoded_body"]}'
            lookup[i] = payload_only
    return lookup


def main():
    data = load_split_and_encode()
    df = data["df"]
    le = data["label_encoder"]
    num_classes = len(le.classes_)

    print("Building payload-only (no path) text from raw capture log...")
    lookup = build_payload_only_lookup(CAPTURE_LOG_PATH)

    payload_texts = df["_capture_line_id"].map(lookup)
    missing = payload_texts.isna().sum()
    if missing:
        raise ValueError(f"{missing} rows could not be matched back to capture_log.jsonl - join key broken.")
    payload_texts = payload_texts.astype(str).values

    # sanity: confirm path is actually gone from a sample
    cmdi_idx = df.index[df["label"] == "cmdi"][0]
    print("Sample (cmdi row) payload-only text:")
    print(" ", repr(payload_texts[cmdi_idx][:150]))
    print("  (compare to original combined_text:)")
    print(" ", repr(df["combined_text"].iloc[cmdi_idx][:150]))

    X_train = texts_to_matrix(payload_texts[data["idx_train"]], max_len=MAX_LEN)
    X_val = texts_to_matrix(payload_texts[data["idx_val"]], max_len=MAX_LEN)
    X_test = texts_to_matrix(payload_texts[data["idx_test"]], max_len=MAX_LEN)

    model = build_model(VOCAB_SIZE, MAX_LEN, num_classes)
    class_weights = compute_class_weights(data["y_train"], num_classes)

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, data["y_train"],
        validation_data=(X_val, data["y_val"]),
        epochs=100, batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2,
    )

    from sklearn.metrics import classification_report, f1_score
    y_pred = np.argmax(model.predict(X_test), axis=1)

    macro_f1 = f1_score(data["y_test"], y_pred, average="macro")
    weighted_f1 = f1_score(data["y_test"], y_pred, average="weighted")
    print(f"\n[ABLATION: payload-only, no path] Test macro F1: {macro_f1:.4f}")
    print(f"[ABLATION: payload-only, no path] Test weighted F1: {weighted_f1:.4f}")
    print(classification_report(data["y_test"], y_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()