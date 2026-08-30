"""
topup_broken_auth.py

Re-sends the DVWA brute-force portion of broken_auth traffic, this
time with DVWA session cookies attached so the requests actually hit
the authenticated vulnerability page instead of bouncing off a 302
redirect to login.php.

Doesn't touch Juice Shop failed-login / security-question traffic --
those were already correctly authenticated (or intentionally
unauthenticated, in the case of login attempts) in the original run.
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

DVWA_BASE = "http://localhost:8081"
SESSION_STATE_PATH = os.path.join("data", "captured_traffic", "_test_session_state.json")

MIN_DELAY = 0.15
MAX_DELAY = 0.5

DVWA_BRUTE_USERNAMES = ["admin", "gordonb", "1337", "pablo", "smithy", "root", "test"]
DVWA_BRUTE_PASSWORDS = [
    "password", "123456", "letmein", "admin123", "qwerty",
    "abc123", "monkey", "dragon", "iloveyou", "welcome",
]


def sleep_briefly():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def load_session_state():
    with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def topup_broken_auth(dvwa_cookies):
    print("\n=== Topping up authenticated DVWA brute-force traffic ===")
    count = 0
    for username in DVWA_BRUTE_USERNAMES:
        for password in DVWA_BRUTE_PASSWORDS:
            try:
                resp = requests.get(
                    f"{DVWA_BASE}/vulnerabilities/brute/",
                    params={"username": username, "password": password, "Login": "Login"},
                    cookies=dvwa_cookies,
                    proxies=PROXIES,
                    verify=False,
                    timeout=10,
                )
                count += 1
                if count == 1:
                    # Sanity check on the very first request -- should be 200, not 302
                    print(f"  First request status: {resp.status_code} "
                          f"({'OK, authenticated' if resp.status_code == 200 else 'still redirecting!'})")
            except requests.RequestException as e:
                print(f"  [DVWA brute] failed: {e}")
            sleep_briefly()

    print(f"Authenticated broken_auth (DVWA brute) requests sent: {count}")
    return count


def main():
    state = load_session_state()
    dvwa_cookies = state["dvwa"]["cookies"]

    total = topup_broken_auth(dvwa_cookies)

    print(f"\nDone. Total requests sent: {total}")
    print("Now stop mitmdump (Ctrl+C) and re-run scripts/extract_features.py "
          "to check the updated label distribution.")


if __name__ == "__main__":
    main()
