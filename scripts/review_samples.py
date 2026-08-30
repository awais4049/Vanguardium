"""
review_samples.py

Prints a random sample of rows per label class so you can manually
spot-check whether the heuristic labeling looks correct before using
http_features.csv for training.

Usage:
    python scripts/review_samples.py
    python scripts/review_samples.py --n 15          # more rows per class
    python scripts/review_samples.py --label sqli     # just one class
"""

import argparse
import os

import pandas as pd

CSV_PATH = os.path.join("data", "processed", "http_features.csv")

REVIEW_COLUMNS = [
    "method",
    "_path",
    "status_code",
    "has_sql_keywords",
    "has_script_tags",
    "is_admin_path",
    "is_cross_owner_basket_access",
    "_body_preview",
    "label",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Rows to sample per class")
    parser.add_argument("--label", type=str, default=None, help="Only show this label")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    args = parser.parse_args()

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    df = pd.read_csv(CSV_PATH)

    labels = [args.label] if args.label else sorted(df["label"].unique())

    for lbl in labels:
        subset = df[df["label"] == lbl]
        sample_n = min(args.n, len(subset))
        sample = subset[REVIEW_COLUMNS].sample(sample_n, random_state=args.seed)

        print(f"\n{'=' * 70}")
        print(f"LABEL: {lbl}  (total rows: {len(subset)}, showing {sample_n})")
        print("=" * 70)
        print(sample.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("Review checklist:")
    print("  - sqli/xss rows: does _body_preview or _path actually contain an attack payload?")
    print("  - idor rows: does is_cross_owner_basket_access == 1? does _path look like a basket/profile access?")
    print("  - broken_auth rows: is_admin_path==1, OR path has security-question, OR brute-force page, OR failed login (401/403)?")
    print("  - benign rows: skim for anything that looks like it should have been flagged.")


if __name__ == "__main__":
    main()
