"""
Vanguardium - Day 3: Feature Extraction
Converts raw mitmproxy capture (data/captured_traffic/capture_log.jsonl)
into a structured HTTP-native feature table for XGBoost/BiLSTM training.

Usage (from project root, with venv active):
    python scripts/extract_features.py

Output:
    data/processed/http_features.csv

IMPORTANT: The 'label' and 'label_binary' columns are AUTO-GENERATED using
heuristic signature rules (see label_flow()). This is a first-pass label,
not ground truth. You know exactly which requests were your attack tests
(timestamps, target paths) - open the output CSV and correct any
mislabeled rows before using this for model training. Auto-labeling exists
to save you from labeling ~500+ rows from scratch, not to replace review.
"""

import json
import re
import math
import os
import base64
from collections import Counter

import pandas as pd

INPUT_PATH = os.path.join("data", "captured_traffic", "capture_log.jsonl")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "http_features.csv")

# ---- Signature patterns used both for content features AND heuristic labeling ----

SQL_PATTERN = re.compile(
    r"(\bOR\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|\bUNION\b\s+\bSELECT\b|\bSELECT\b.+\bFROM\b|\bDROP\b\s+\bTABLE\b|['\"]\s*--|\bINSERT\b\s+\bINTO\b|\bDELETE\b\s+FROM|'\s*OR\s*'|admin'--)",
    re.IGNORECASE,
)
XSS_PATTERN = re.compile(
    r"(<script|onerror\s*=|onload\s*=|javascript:|<img|<iframe|alert\()",
    re.IGNORECASE,
)
SPECIAL_CHARS = re.compile(r"['\";<>=\-]{1}")
ADMIN_PATH_PATTERN = re.compile(r"^/administration/?$", re.IGNORECASE)
SECURITY_QUESTION_PATTERN = re.compile(r"/rest/user/security-question", re.IGNORECASE)
IDOR_PATH_PATTERN = re.compile(r"/(basket|api/users|api/orders|profile)/\d+", re.IGNORECASE)
NUMERIC_ID_PATTERN = re.compile(r"/(\d+)(?:/|$)")
BRUTE_FORCE_PATH_PATTERN = re.compile(r"/vulnerabilities/brute", re.IGNORECASE)
LOGIN_PATH_PATTERN = re.compile(r"^/rest/user/login$", re.IGNORECASE)

# Non-target domains to exclude from benign counts (browser/OS telemetry,
# not traffic to your actual target apps). Extend this list if new noise
# domains show up in your captures.
NOISE_HOSTS = {
    "update.googleapis.com",
    "content-autofill.googleapis.com",
    "optimizationguide-pa.googleapis.com",
    "www.google.com",
    "android.clients.google.com",
    "clientservices.googleapis.com",
}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def safe_get(d: dict, key, default=""):
    return d.get(key, default) if isinstance(d, dict) else default


def extract_jwt_token(headers: dict) -> str | None:
    """Pull a raw JWT out of Authorization: Bearer <token> or a Cookie token=<token>."""
    auth = ""
    for k, v in (headers or {}).items():
        if k.lower() == "authorization":
            auth = str(v)
            break
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    cookie = str((headers or {}).get("Cookie", ""))
    match = re.search(r"(?:^|;\s*)token=([^;]+)", cookie)
    if match:
        return match.group(1).strip()
    return None


def decode_jwt_bid(headers: dict):
    """
    Decode the JWT payload (no signature verification - we only need the
    claimed basket id 'bid' to compare against the basket id in the URL,
    not to trust the token cryptographically) and return the 'bid' claim
    if present, else None.
    """
    token = extract_jwt_token(headers)
    if not token or token.count(".") != 2:
        return None
    try:
        payload_b64 = token.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_json)
        return payload.get("bid")
    except Exception:
        return None


def extract_features(flow: dict) -> dict:
    method = flow.get("method", "")
    path = flow.get("path", "")
    query_params = flow.get("query_params", {}) or {}
    headers = flow.get("headers", {}) or {}
    body = flow.get("body", "") or ""
    status_code = flow.get("status_code", 0)
    response_size = flow.get("response_size", 0)
    duration_ms = flow.get("duration_ms", 0)

    host = str(headers.get("Host", "")).lower()
    combined_text = f"{path} {json.dumps(query_params)} {body}"

    # IDOR ownership check: compare basket id in URL vs. basket id claimed
    # in the caller's own JWT. Mismatch = requesting someone else's resource.
    basket_match = re.search(r"/basket/(\d+)", path, re.IGNORECASE)
    requested_basket_id = int(basket_match.group(1)) if basket_match else None
    session_bid = decode_jwt_bid(headers)
    is_cross_owner_basket_access = int(
        requested_basket_id is not None
        and session_bid is not None
        and requested_basket_id != session_bid
    )

    special_char_count = len(SPECIAL_CHARS.findall(combined_text))
    text_len = max(len(combined_text), 1)

    row = {
        # timing / size
        "duration_ms": duration_ms,
        "response_size": response_size,
        "body_length": len(body),
        "status_code": status_code,
        # structural
        "method": method,
        "path_depth": path.count("/"),
        "num_query_params": len(query_params),
        "num_headers": len(headers),
        # content signatures
        "has_sql_keywords": int(bool(SQL_PATTERN.search(combined_text))),
        "has_script_tags": int(bool(XSS_PATTERN.search(combined_text))),
        "special_char_ratio": round(special_char_count / text_len, 4),
        "payload_entropy": round(shannon_entropy(body), 4),
        "numeric_id_in_path": int(bool(NUMERIC_ID_PATTERN.search(path))),
        # auth / session
        "has_auth_header": int(
            "authorization" in {k.lower() for k in headers.keys()}
            or "token" in str(headers.get("Cookie", "")).lower()
        ),
        "is_admin_path": int(bool(ADMIN_PATH_PATTERN.search(path))),
        "is_cross_owner_basket_access": is_cross_owner_basket_access,
        # raw fields kept for manual review / debugging (drop before training if not needed)
        "_path": path,
        "_host": host,
        "_body_preview": body[:120],
        "_requested_basket_id": requested_basket_id,
        "_session_bid": session_bid,
    }
    return row


def label_flow(row: dict) -> str:
    """Heuristic first-pass label. REVIEW MANUALLY before training."""
    if row["has_script_tags"]:
        return "xss"
    if row["has_sql_keywords"]:
        return "sqli"
    if row["is_admin_path"] or SECURITY_QUESTION_PATTERN.search(row["_path"]):
        return "broken_auth"
    if BRUTE_FORCE_PATH_PATTERN.search(row["_path"]):
        return "broken_auth"
    if LOGIN_PATH_PATTERN.search(row["_path"]) and row.get("status_code") in (401, 403):
        return "broken_auth"
    if IDOR_PATH_PATTERN.search(row["_path"]) and row["method"] in ("GET", "PUT", "DELETE"):
        # Only flag as IDOR if we can confirm the requester's own session
        # basket id differs from the basket id in the URL. If we can't
        # decode a session bid (e.g. no token), fall back to flagging any
        # basket-path access as a weaker signal - better to review than to
        # silently drop a potential attack.
        if row["_session_bid"] is not None:
            if row["is_cross_owner_basket_access"]:
                return "idor"
        else:
            return "idor"
    return "benign"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run this script from the project root."
        )

    rows = []
    skipped = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                flow = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            try:
                feats = extract_features(flow)
                if feats["_host"] in NOISE_HOSTS:
                    continue  # drop browser/OS telemetry - not target-app traffic
                feats["label"] = label_flow(feats)
                feats["label_binary"] = 0 if feats["label"] == "benign" else 1
                rows.append(feats)
            except Exception as e:
                print(f"[warn] line {i} failed feature extraction: {e}")
                skipped += 1

    df = pd.DataFrame(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed {len(df)} flows ({skipped} skipped/corrupted).")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nREMINDER: 'label' is auto-generated via heuristic rules.")
    print("Review data/processed/http_features.csv and correct mislabeled rows")
    print("before using it for model training.")


if __name__ == "__main__":
    main()
