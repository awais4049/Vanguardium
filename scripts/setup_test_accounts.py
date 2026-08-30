"""
setup_test_accounts.py

One-time setup for Vanguardium's synthetic attack traffic generation.
Creates a fresh Juice Shop test account and establishes a low-security
DVWA session. Saves session state (cookies + tokens) to a local JSON
file so scripts/generate_attack_traffic.py can reuse them without
logging in again.

NOT routed through the mitmproxy on purpose -- this is account
provisioning, not attack traffic, so there's no reason to pollute the
capture log with it.
"""

import json
import os
import re
import sys

import requests

JUICE_SHOP_BASE = "http://localhost:3000"
DVWA_BASE = "http://localhost:8081"

SESSION_STATE_PATH = os.path.join("data", "captured_traffic", "_test_session_state.json")

# A fixed test account -- deterministic so re-runs don't spam Juice Shop
# with new registrations every time this script is executed.
JUICE_SHOP_TEST_EMAIL = "vanguardium.tester@example.com"
JUICE_SHOP_TEST_PASSWORD = "TestPass123!"

DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"


def setup_juice_shop():
    print("\n[Juice Shop] Attempting registration...")
    session = requests.Session()

    register_payload = {
        "email": JUICE_SHOP_TEST_EMAIL,
        "password": JUICE_SHOP_TEST_PASSWORD,
        "passwordRepeat": JUICE_SHOP_TEST_PASSWORD,
        "securityQuestion": None,
        "securityAnswer": None,
    }

    reg_resp = session.post(
        f"{JUICE_SHOP_BASE}/api/Users",
        json=register_payload,
        timeout=10,
    )

    if reg_resp.status_code in (200, 201):
        print(f"[Juice Shop] Registered new account: {JUICE_SHOP_TEST_EMAIL}")
    elif reg_resp.status_code == 400:
        print("[Juice Shop] Account likely already exists (400) -- continuing to login.")
    else:
        print(f"[Juice Shop] Unexpected register response: {reg_resp.status_code} {reg_resp.text[:200]}")

    print("[Juice Shop] Logging in...")
    login_resp = session.post(
        f"{JUICE_SHOP_BASE}/rest/user/login",
        json={"email": JUICE_SHOP_TEST_EMAIL, "password": JUICE_SHOP_TEST_PASSWORD},
        timeout=10,
    )

    if login_resp.status_code != 200:
        print(f"[Juice Shop] LOGIN FAILED: {login_resp.status_code} {login_resp.text[:200]}")
        return None

    token = login_resp.json().get("authentication", {}).get("token")
    if not token:
        print("[Juice Shop] Login succeeded but no token found in response.")
        return None

    print("[Juice Shop] Login successful, token acquired.")
    return {
        "email": JUICE_SHOP_TEST_EMAIL,
        "token": token,
    }


def setup_dvwa():
    print("\n[DVWA] Fetching login page for CSRF token...")
    session = requests.Session()

    login_page = session.get(f"{DVWA_BASE}/login.php", timeout=10)
    match = re.search(r"user_token['\"]\s+value=['\"]([a-f0-9]+)['\"]", login_page.text)
    if not match:
        print("[DVWA] Could not find CSRF token on login page. Is DVWA running at "
              f"{DVWA_BASE}?")
        return None
    csrf_token = match.group(1)
    print(f"[DVWA] CSRF token acquired: {csrf_token[:10]}...")

    login_resp = session.post(
        f"{DVWA_BASE}/login.php",
        data={
            "username": DVWA_USERNAME,
            "password": DVWA_PASSWORD,
            "Login": "Login",
            "user_token": csrf_token,
        },
        timeout=10,
    )

    if "login.php" in login_resp.url and login_resp.status_code == 200:
        # DVWA redirects to index.php on success; if we're still on login.php, it failed
        print("[DVWA] LOGIN FAILED -- check credentials or DVWA setup (did you click "
              "'Create / Reset Database' in the DVWA setup page at least once?).")
        return None

    print("[DVWA] Login successful.")

    print("[DVWA] Setting security level to 'low'...")
    sec_page = session.get(f"{DVWA_BASE}/security.php", timeout=10)
    sec_match = re.search(r"user_token['\"]\s+value=['\"]([a-f0-9]+)['\"]", sec_page.text)
    sec_token = sec_match.group(1) if sec_match else csrf_token

    sec_resp = session.post(
        f"{DVWA_BASE}/security.php",
        data={
            "security": "low",
            "seclev_submit": "Submit",
            "user_token": sec_token,
        },
        timeout=10,
    )

    if sec_resp.status_code == 200:
        print("[DVWA] Security level set to low.")
    else:
        print(f"[DVWA] Unexpected response setting security level: {sec_resp.status_code}")

    cookies = session.cookies.get_dict()
    return {
        "cookies": cookies,
    }


def main():
    juice_state = setup_juice_shop()
    dvwa_state = setup_dvwa()

    if juice_state is None:
        print("\nJuice Shop setup FAILED -- fix this before running the attack generator.")
    if dvwa_state is None:
        print("\nDVWA setup FAILED -- fix this before running the attack generator.")

    if juice_state is None or dvwa_state is None:
        sys.exit(1)

    state = {
        "juice_shop": juice_state,
        "dvwa": dvwa_state,
    }

    os.makedirs(os.path.dirname(SESSION_STATE_PATH), exist_ok=True)
    with open(SESSION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"\nSession state saved to {SESSION_STATE_PATH}")
    print("Ready for scripts/generate_attack_traffic.py")


if __name__ == "__main__":
    main()
