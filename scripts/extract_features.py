"""
Vanguardium - Feature Extraction (8-class: 7 attacks + benign)
Converts raw mitmproxy capture (data/captured_traffic/capture_log.jsonl)
into a structured HTTP-native feature table for XGBoost/BiLSTM training.

Usage (from project root, with venv active):
    python scripts/extract_features.py

Output:
    data/processed/http_features.csv

IMPORTANT: The 'label' and 'label_binary' columns are AUTO-GENERATED using
heuristic signature rules (see label_flow()). This is a first-pass label,
not ground truth. Review data/processed/http_features.csv and spot-check
before using it for model training.

ANTI-LEAKAGE CONVENTION (applies to ALL 7 attack classes as of this version):
Every field whose name starts with an underscore (e.g. "_sqli_match",
"_cmdi_match", "_is_cross_owner_basket_access") is a LABEL-ONLY signal:
it is the exact rule used by label_flow() to assign a class, kept in the
output purely so you can manually spot-check *why* a row got its label.
These fields must be DROPPED before training. They are never returned
by extract_features() as top-level (non-underscore) columns, precisely
so a training script that naively does df.drop(columns=[c for c in df
if c.startswith('_')]) can't accidentally leave one in.

The model is trained only on generic structural/statistical features
(timing, size, entropy, counts, method, status code, etc.) that are not
themselves the literal rule that produced the label. This is a deliberate
choice: earlier iterations had classes (e.g. IDOR via
is_cross_owner_basket_access) where the trained feature WAS a 1:1
re-encoding of the label, which lets a classifier hit ~100% accuracy by
memorizing the labeling rule rather than learning a generalizable
detection pattern. Expect per-class accuracy to look "worse" than a
leaky baseline as a direct, correct consequence of this fix.
"""

import json
import re
import math
import os
import base64
from collections import Counter
from urllib.parse import unquote_plus

import pandas as pd

INPUT_PATH = os.path.join("data", "captured_traffic", "capture_log.jsonl")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "http_features.csv")

# ---- Signature patterns: LABEL-ONLY. Never exposed as trained features directly. ----

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

# --- new: command injection, path traversal, SSRF ---

CMDI_ENDPOINT_PATTERN = re.compile(r"^/vulnerabilities/exec/?$", re.IGNORECASE)
# operator immediately followed by a common recon/enumeration command —
# requires BOTH an operator and a command word, not just a bare ';' or '|',
# to avoid flagging incidental special characters as command injection.
CMDI_PATTERN = re.compile(
    r"(;\s*|\&\&\s*|\|\s*)(whoami|id|ls|cat|uname|ifconfig|netstat|ps|pwd|hostname)"
    r"|`[^`]+`"
    r"|\$\([^)]+\)",
    re.IGNORECASE,
)

FI_ENDPOINT_PATTERN = re.compile(r"^/vulnerabilities/fi/?(\?|$)", re.IGNORECASE)
URL_SCHEME_PATTERN = re.compile(r"^(https?|gopher|ftp|dict)://", re.IGNORECASE)
PATH_TRAVERSAL_PATTERN = re.compile(
    r"(\.\.[/\\]|%2e%2e|\.\.%2f|%252e%252e|\.\.\.\.//"
    r"|/?etc/passwd|/?etc/shadow|/?etc/hosts|proc/(self/)?environ|proc/version"
    r"|boot\.ini|win\.ini|file://|php://filter)",
    re.IGNORECASE,
)

SSRF_ENDPOINT_PATTERN = re.compile(r"^/profile/image/url$", re.IGNORECASE)
SSRF_INTERNAL_TARGET_PATTERN = re.compile(
    r"(https?://(127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|169\.254\."
    r"|localhost|juice-shop|dvwa|vanguardium_postgres|0x7f|2130706433|0177)"
    r"|file://|gopher://|dict://)",
    re.IGNORECASE,
)

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


def is_ssrf_signal(path: str, query_params: dict, combined_text: str) -> bool:
    """True for Juice Shop's /profile/image/url with an internal/non-http(s)
    target, OR DVWA's file-inclusion module used in RFI/SSRF mode (page=
    a URL rather than a traversal path)."""
    if SSRF_ENDPOINT_PATTERN.search(path):
        return bool(SSRF_INTERNAL_TARGET_PATTERN.search(combined_text))
    if FI_ENDPOINT_PATTERN.search(path):
        page_val = str(query_params.get("page", ""))
        return bool(URL_SCHEME_PATTERN.match(page_val))
    return False


def is_path_traversal_signal(path: str, query_params: dict, combined_text: str) -> bool:
    """True for DVWA's file-inclusion module with a traversal sequence or
    sensitive-file target in `page` — but NOT if `page` is itself a URL
    (that's SSRF via RFI, handled by is_ssrf_signal instead, mutually
    exclusive by construction)."""
    if not FI_ENDPOINT_PATTERN.search(path):
        return False
    page_val = str(query_params.get("page", ""))
    if URL_SCHEME_PATTERN.match(page_val):
        return False
    return bool(PATH_TRAVERSAL_PATTERN.search(page_val)) or bool(
        PATH_TRAVERSAL_PATTERN.search(combined_text)
    )


def is_cmdi_signal(path: str, combined_text: str) -> bool:
    """True for DVWA's command-exec module with an operator+command
    payload (e.g. '127.0.0.1;whoami')."""
    if not CMDI_ENDPOINT_PATTERN.search(path):
        return False
    return bool(CMDI_PATTERN.search(combined_text))

def build_combined_text(flow: dict) -> dict:
    """Single source of truth for decoding + combining request text.
    Used by extract_features() (for XGBoost's engineered stats) AND by
    the CNN's raw-text pipeline (for character-level input) so both
    models see byte-for-byte identical decoded text. Do not duplicate
    this decoding logic anywhere else."""
    path = unquote_plus(flow.get("path", "") or "")
    query_params = flow.get("query_params", {}) or {}
    headers = flow.get("headers", {}) or {}
    body = flow.get("body", "") or ""
    decoded_body = unquote_plus(body)
    combined_text = f"{path} {json.dumps(query_params)} {decoded_body}"
    return {
        "path": path,
        "query_params": query_params,
        "headers": headers,
        "body": body,
        "decoded_body": decoded_body,
        "combined_text": combined_text,
    }

def extract_features(flow: dict) -> dict:
    method = flow.get("method", "")
    status_code = flow.get("status_code", 0)
    response_size = flow.get("response_size", 0)
    duration_ms = flow.get("duration_ms", 0)

    parsed = build_combined_text(flow)
    path = parsed["path"]
    query_params = parsed["query_params"]
    headers = parsed["headers"]
    body = parsed["body"]
    combined_text = parsed["combined_text"]

    host = str(headers.get("Host", "")).lower()

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
        # content signatures (GENERIC, statistical — not tied 1:1 to any single label)
        "special_char_ratio": round(special_char_count / text_len, 4),
        "payload_entropy": round(shannon_entropy(body), 4),
        "numeric_id_in_path": int(bool(NUMERIC_ID_PATTERN.search(path))),
        # auth / session (generic — presence of auth isn't itself a label rule)
        "has_auth_header": int(
            "authorization" in {k.lower() for k in headers.keys()}
            or "token" in str(headers.get("Cookie", "")).lower()
        ),
        # ---- LABEL-ONLY fields below (underscore prefix). DROP before training. ----
        "_path": path,
        "_host": host,
        "_body_preview": body[:120],
        "_requested_basket_id": requested_basket_id,
        "_session_bid": session_bid,
        "_is_cross_owner_basket_access": is_cross_owner_basket_access,
        "_sqli_match": bool(SQL_PATTERN.search(combined_text)),
        "_xss_match": bool(XSS_PATTERN.search(combined_text)),
        "_is_admin_path": bool(ADMIN_PATH_PATTERN.search(path)),
        "_cmdi_match": is_cmdi_signal(path, combined_text),
        "_path_traversal_match": is_path_traversal_signal(path, query_params, combined_text),
        "_ssrf_match": is_ssrf_signal(path, query_params, combined_text),
    }
    return row


def label_flow(row: dict) -> str:
    """Heuristic first-pass label. Uses ONLY underscore-prefixed (label-only)
    fields plus raw _path — never a top-level trained feature. REVIEW
    MANUALLY before training; see module docstring for the anti-leakage
    convention this relies on."""
    if row["_xss_match"]:
        return "xss"
    if row["_sqli_match"]:
        return "sqli"
    if row["_cmdi_match"]:
        return "cmdi"
    if row["_path_traversal_match"]:
        return "path_traversal"
    if row["_ssrf_match"]:
        return "ssrf"
    if row["_is_admin_path"] or SECURITY_QUESTION_PATTERN.search(row["_path"]):
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
            if row["_is_cross_owner_basket_access"]:
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
                feats["_capture_line_id"] = i
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

    label_only_cols = [c for c in df.columns if c.startswith("_")]

    print(f"Processed {len(df)} flows ({skipped} skipped/corrupted).")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"\n{len(label_only_cols)} label-only columns present (drop before training): "
          f"{label_only_cols}")
    print("\nREMINDER: 'label' is auto-generated via heuristic rules.")
    print("Review data/processed/http_features.csv and correct mislabeled rows")
    print("before using it for model training.")


if __name__ == "__main__":
    main()