"""
generate_benign_endpoint_traffic.py

Generates BENIGN (legitimate, non-attack) traffic against the two endpoints
that currently have zero benign examples in capture_log.jsonl:

  - DVWA file-inclusion page  (/vulnerabilities/fi/?page=<legit filename>)
  - Juice Shop profile image  (/profile/image/url with a real external image URL)

Why this exists: extract_features.py analysis showed path_traversal and ssrf
sitting at a suspicious 1.00 F1 because EVERY captured hit to these two
endpoints was an attack — the model was learning "which endpoint was hit"
rather than "is this payload malicious." Without benign examples AT THE
SAME endpoints, no amount of feature engineering fixes that; the model
needs a genuine same-endpoint contrast to learn from.

Run this WHILE mitmdump -s capture/capture_addon.py is running on port 8080,
and while juice-shop (3000) and dvwa (8081) containers are up.

    python scripts/generate_benign_endpoint_traffic.py
"""

import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- config ---------------------------------------------------------------

PROXIES = {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"}
DVWA_BASE = "http://localhost:8081"
JUICE_BASE = "http://localhost:3000"
SESSION_STATE_PATH = Path("data/captured_traffic/_test_session_state.json")

FI_SAMPLES = 200          # legitimate DVWA file-inclusion hits
SSRF_ENDPOINT_SAMPLES = 150  # legitimate Juice Shop profile-image hits
REQUEST_DELAY_RANGE = (0.15, 0.6)

DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"

_EXECUTOR = ThreadPoolExecutor(max_workers=20)


def call_with_hard_timeout(fn, *args, hard_timeout=8, **kwargs):
    """Belt-and-suspenders wrapper: requests' own timeout= can fail to fire
    in some proxy-in-the-middle hang scenarios (observed previously). Bounds
    every call to hard_timeout seconds no matter what, regardless of what
    the server does internally."""
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=hard_timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"hard timeout ({hard_timeout}s) exceeded")


# --- session bootstrap (same as generate_attack_traffic_extended.py) ------

def get_dvwa_session() -> requests.Session:
    s = requests.Session()
    s.proxies.update(PROXIES)
    s.verify = False

    login_page = s.get(f"{DVWA_BASE}/login.php")
    token = _extract_user_token(login_page.text)

    resp = s.post(
        f"{DVWA_BASE}/login.php",
        data={
            "username": DVWA_USERNAME,
            "password": DVWA_PASSWORD,
            "user_token": token,
            "Login": "Login",
        },
        allow_redirects=True,
    )
    if "login.php" in resp.url and "index.php" not in resp.url:
        raise RuntimeError("DVWA login failed — check credentials / container is up on :8081")

    sec_page = s.get(f"{DVWA_BASE}/security.php")
    token = _extract_user_token(sec_page.text)
    s.post(
        f"{DVWA_BASE}/security.php",
        data={"security": "low", "seclev_submit": "Submit", "user_token": token},
    )
    print("[dvwa] session established, security=low")
    return s


def _extract_user_token(html: str) -> str:
    marker = "user_token' value='"
    if marker not in html:
        marker = 'user_token" value="'
    if marker not in html:
        return ""
    start = html.index(marker) + len(marker)
    end = html.index("'" if "'" in marker else '"', start)
    return html[start:end]


def get_juice_jwt() -> str:
    if SESSION_STATE_PATH.exists():
        state = json.loads(SESSION_STATE_PATH.read_text())
        token = state.get("juice_shop", {}).get("token")
        if token:
            print(f"[juice-shop] loaded JWT from {SESSION_STATE_PATH} (juice_shop.token)")
            return token
        print(f"[juice-shop] WARNING: 'juice_shop.token' not found, "
              f"falling back to fresh login")

    r = requests.post(
        f"{JUICE_BASE}/rest/user/login",
        json={"email": "vanguardium.tester@example.com", "password": "TestPass123!"},
        proxies=PROXIES,
        verify=False,
    )
    r.raise_for_status()
    token = r.json()["authentication"]["token"]
    print("[juice-shop] fresh login succeeded")
    return token


# --- benign payload generators ---------------------------------------------

def benign_fi_pages(n: int):
    """Legitimate values for DVWA's file-inclusion `page` param — the
    filenames the module actually ships with / expects for normal use."""
    legit_pages = [
        "include.php", "file1.php", "file2.php", "file3.php", "file4.php",
    ]
    return random.choices(legit_pages, k=n)


def benign_image_urls(n: int):
    """Legitimate external image URLs — real public image-hosting domains,
    varied paths, no internal/private targets, no unusual schemes."""
    hosts_exts = [
        ("i.imgur.com", "jpg"),
        ("images.unsplash.com", "jpg"),
        ("upload.wikimedia.org", "png"),
        ("cdn.pixabay.com", "jpg"),
        ("www.gravatar.com", "jpg"),
        ("picsum.photos", "jpg"),
        ("images.pexels.com", "jpg"),
        ("avatars.githubusercontent.com", "png"),
    ]
    payloads = []
    for _ in range(n):
        host, ext = random.choice(hosts_exts)
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        payloads.append(f"https://{host}/{rand_id}.{ext}")
    return payloads


# --- traffic generation ----------------------------------------------------

def generate_benign_fi(session: requests.Session):
    print(f"[benign-fi] generating {FI_SAMPLES} samples...")
    ok = 0
    for i, page in enumerate(benign_fi_pages(FI_SAMPLES), 1):
        try:
            call_with_hard_timeout(
                session.get, f"{DVWA_BASE}/vulnerabilities/fi/", params={"page": page}
            )
            ok += 1
        except (requests.RequestException, TimeoutError) as e:
            print(f"  [!] ({i}/{FI_SAMPLES}) failed/timed out: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    print(f"[benign-fi] done: {ok}/{FI_SAMPLES} sent")


def generate_benign_ssrf_endpoint(jwt: str):
    print(f"[benign-ssrf-endpoint] generating {SSRF_ENDPOINT_SAMPLES} samples...")
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
    ok = 0
    for i, url in enumerate(benign_image_urls(SSRF_ENDPOINT_SAMPLES), 1):
        try:
            call_with_hard_timeout(
                requests.post,
                f"{JUICE_BASE}/profile/image/url",
                json={"imageUrl": url},
                headers=headers,
                proxies=PROXIES,
                verify=False,
                timeout=3,
                hard_timeout=5,
            )
            ok += 1
        except (requests.RequestException, TimeoutError) as e:
            print(f"  [!] ({i}/{SSRF_ENDPOINT_SAMPLES}) failed/timed out: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    print(f"[benign-ssrf-endpoint] done: {ok}/{SSRF_ENDPOINT_SAMPLES} sent")


def main():
    print("=" * 60)
    print("Vanguardium — benign same-endpoint traffic generation")
    print("=" * 60)

    dvwa_session = get_dvwa_session()
    generate_benign_fi(dvwa_session)

    jwt = get_juice_jwt()
    generate_benign_ssrf_endpoint(jwt)

    print("\nDone. Re-run extract_features.py, then verify with the")
    print("endpoint/label groupby check before retraining.")


if __name__ == "__main__":
    main()
