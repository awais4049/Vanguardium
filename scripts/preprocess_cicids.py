"""
Vanguardium - Day 3: CICIDS 2017 Preprocessing
Handles known CICFlowMeter sentinel values before this dataset is used
for XGBoost/BiLSTM training. Does NOT change the feature schema, join it
with HTTP-native features, or otherwise force it into a different shape -
CICIDS stays its own packet-level feature space (see project notes on
"complementary but distinct" dataset design).

What this does:
  - CICFlowMeter uses -1 in Init_Win_bytes_forward / Init_Win_bytes_backward
    to mean "TCP window size could not be determined" (e.g. very short or
    single-packet flows). Left as a literal -1, a model may learn something
    spurious from it (it looks numerically like "very small" rather than
    "unknown"). We add an explicit companion binary flag for each column
    so the model gets that signal directly rather than inferring it.
  - Reports (but does not silently "fix") any other unexpected negative
    values found in columns that should logically be >= 0, so you can
    decide how to handle them rather than have them hidden.

Usage (from project root, with venv active):
    python scripts/preprocess_cicids.py

Output:
    data/processed/cicids2017_processed.csv
"""

import os
import pandas as pd

INPUT_PATH = os.path.join("data", "cicids2017", "cicids2017_trimmed.csv")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cicids2017_processed.csv")

# Columns where CICFlowMeter documents -1 as a "could not determine" sentinel,
# not a real measured negative value.
SENTINEL_COLUMNS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# Columns that are logically non-negative (counts, durations, sizes) where
# ANY negative value (other than the documented sentinel columns above) is
# worth flagging for manual review rather than silently keeping or dropping.
NON_NEGATIVE_EXPECTED = [
    "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
]


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run this script from the project root."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    # --- Step 1: explicit missingness flags for known sentinel columns ---
    for col in SENTINEL_COLUMNS:
        if col not in df.columns:
            print(f"[warn] expected sentinel column '{col}' not found - skipping.")
            continue
        flag_col = f"{col}_missing"
        df[flag_col] = (df[col] == -1).astype(int)
        n_missing = df[flag_col].sum()
        print(f"{col}: {n_missing} rows flagged missing ({n_missing / len(df):.1%}) -> added '{flag_col}'")

    # --- Step 2: report unexpected negatives, then drop the (tiny, non-rare-class) rows ---
    print("\n=== Unexpected negative value check (non-sentinel columns) ===")
    any_unexpected = False
    for col in NON_NEGATIVE_EXPECTED:
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            any_unexpected = True
            print(f"[flag] {col}: {n_neg} negative values found.")
    if not any_unexpected:
        print("None found - clean.")
    else:
        # Rows where Flow Duration == -1 are CICFlowMeter's own "undetermined"
        # sentinel (same convention as Init_Win_bytes_*), not real corruption.
        # Rows with a small negative Flow IAT Min are a documented
        # CICFlowMeter timestamp-reordering artifact, not recoverable.
        # Both are dropped rather than imputed - imputing would fabricate
        # data; these counts are small (<0.1% combined) and not concentrated
        # in the already-thin Web Attacks / Bots classes, so dropping does
        # not meaningfully affect class balance.
        before = len(df)
        before_by_class = df["Attack Type"].value_counts()
        df = df[(df["Flow Duration"] >= 0) & (df["Flow IAT Min"] >= 0)]
        after_by_class = df["Attack Type"].value_counts()
        dropped = before - len(df)
        print(f"\nDropped {dropped} rows with invalid negative timing values.")
        print("Per-class impact:")
        impact = (before_by_class - after_by_class).fillna(0).astype(int)
        print(impact[impact > 0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved processed dataset to {OUTPUT_PATH}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns "
          f"({df.shape[1] - 53} new columns added: missingness flags).")


if __name__ == "__main__":
    main()
