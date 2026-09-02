"""
Scratch analysis: distribution of combined payload lengths
(path + query_params + body) from captured traffic.

Purpose: pick an evidence-based max sequence length for the CNN's
raw-text input branch (Option A: char/byte-level 1D conv), rather
than guessing.

Not part of the reproducible training pipeline -- lives in scratch/,
not scripts/.
"""

import json
import statistics
from pathlib import Path
from urllib.parse import urlencode

CAPTURE_LOG = Path("data/captured_traffic/capture_log.jsonl")


def build_payload_string(record: dict) -> str:
    """
    Combine path + query_params + body into a single string,
    the same way the CNN's raw-text input will be constructed.
    """
    path = record.get("path", "") or ""
    query_params = record.get("query_params", {}) or {}
    body = record.get("body", "") or ""

    # Reconstruct query string from parsed params for a consistent
    # representation (captured data may or may not include raw query
    # string separately from parsed query_params).
    query_str = urlencode(query_params) if query_params else ""

    combined = path
    if query_str:
        combined += "?" + query_str
    if body:
        combined += " " + str(body)

    return combined


def main():
    if not CAPTURE_LOG.exists():
        print(f"ERROR: {CAPTURE_LOG} not found.")
        return

    lengths = []
    empty_body_count = 0
    total = 0
    parse_errors = 0

    with open(CAPTURE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            total += 1
            payload = build_payload_string(record)
            lengths.append(len(payload))

            if not record.get("body"):
                empty_body_count += 1

    if not lengths:
        print("No valid records found.")
        return

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p):
        idx = int(n * p)
        idx = min(idx, n - 1)
        return lengths_sorted[idx]

    print(f"Total records parsed:     {total}")
    print(f"JSON parse errors:        {parse_errors}")
    print(f"Records with empty body:  {empty_body_count} ({100*empty_body_count/total:.1f}%)")
    print()
    print("Combined payload length (chars) distribution:")
    print(f"  min:     {min(lengths)}")
    print(f"  max:     {max(lengths)}")
    print(f"  mean:    {statistics.mean(lengths):.1f}")
    print(f"  median:  {statistics.median(lengths)}")
    print(f"  p90:     {percentile(0.90)}")
    print(f"  p95:     {percentile(0.95)}")
    print(f"  p99:     {percentile(0.99)}")
    print(f"  p99.9:   {percentile(0.999)}")
    print()

    # Suggest candidate max_len values and show truncation impact
    print("Truncation impact at candidate max_len values:")
    for candidate in [128, 256, 512, 1024, 2048, 4096]:
        truncated = sum(1 for l in lengths if l > candidate)
        pct = 100 * truncated / n
        print(f"  max_len={candidate:5d} -> {truncated:6d} records truncated ({pct:.2f}%)")


if __name__ == "__main__":
    main()
