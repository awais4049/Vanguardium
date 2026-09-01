"""
review_samples.py

Prints a random sample of rows per label class across all 8 classes
(benign, sqli, xss, broken_auth, idor, cmdi, path_traversal, ssrf) so you can
manually spot-check whether the heuristic labeling and extracted features
look correct before using http_features.csv for model training.

Usage:
    python scripts/review_samples.py
    python scripts/review_samples.py --n 15          # more rows per class
    python scripts/review_samples.py --label sqli     # just one class
    python scripts/review_samples.py --seed 42        # custom random seed
"""

import argparse
import os

import pandas as pd

CSV_PATH = os.path.join("data", "processed", "http_features.csv")

REVIEW_COLUMNS = [
    "method",
    "_path",
    "status_code",
    "special_char_ratio",
    "payload_entropy",
    "_sqli_match",
    "_xss_match",
    "_cmdi_match",
    "_path_traversal_match",
    "_ssrf_match",
    "_is_admin_path",
    "_is_cross_owner_basket_access",
    "_body_preview",
    "label",
]


def main():
    parser = argparse.ArgumentParser(description="Review sampled rows per label class from http_features.csv.")
    parser.add_argument("--n", type=int, default=10, help="Rows to sample per class")
    parser.add_argument("--label", type=str, default=None, help="Only show this label")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}. Ensure feature extraction has been run.")

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)

    df = pd.read_csv(CSV_PATH)

    available_cols = [c for c in REVIEW_COLUMNS if c in df.columns]

    labels = [args.label] if args.label else sorted(df["label"].unique())

    for lbl in labels:
        subset = df[df["label"] == lbl]
        if subset.empty:
            print(f"\n{'=' * 70}")
            print(f"LABEL: {lbl} (no rows found)")
            print("=" * 70)
            continue

        sample_n = min(args.n, len(subset))
        sample = subset[available_cols].sample(sample_n, random_state=args.seed)

        print(f"\n{'=' * 70}")
        print(f"LABEL: {lbl}  (total rows: {len(subset)}, showing {sample_n})")
        print("=" * 70)
        print(sample.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("Review checklist (8 classes):")
    print("  - sqli: does _sqli_match == True? Check _body_preview / _path for SQL keywords (UNION, OR 1=1, etc.).")
    print("  - xss: does _xss_match == True? Check _body_preview / _path for script tags/event handlers (<script>, alert, etc.).")
    print("  - cmdi: does _cmdi_match == True? Check _path (/vulnerabilities/exec) and commands (whoami, id, cat, etc.).")
    print("  - path_traversal: does _path_traversal_match == True? Check _path (/vulnerabilities/fi) for traversal sequences (../, etc/passwd).")
    print("  - ssrf: does _ssrf_match == True? Check _body_preview / _path for internal endpoints/loopback IPs/schemes.")
    print("  - broken_auth: is _is_admin_path == True, path has security-question/brute-force page, or failed login (401/403)?")
    print("  - idor: is _is_cross_owner_basket_access == 1 or does _path target unauthorized user baskets/profiles?")
    print("  - benign: skim for anything that looks like it should have been flagged as an attack.")
    print("  - trained features: verify special_char_ratio and payload_entropy reflect payload complexity.")


if __name__ == "__main__":
    main()
