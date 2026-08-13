import requests
import json
import time

URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OUT = "data/nvd/nvd_starter_entries.json"
LIMIT = 100  # starter subset size

print("Fetching starter set of NVD CVE entries...")
params = {"resultsPerPage": LIMIT, "startIndex": 0}

resp = requests.get(URL, params=params, timeout=120)
resp.raise_for_status()
data = resp.json()

vulns = data.get("vulnerabilities", [])
print(f"Fetched {len(vulns)} entries")

simplified = []
for v in vulns:
    cve = v.get("cve", {})
    descriptions = cve.get("descriptions", [])
    eng_desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    metrics = cve.get("metrics", {})
    cvss_score = None
    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
        if key in metrics:
            cvss_score = metrics[key][0]["cvssData"].get("baseScore")
            break

    simplified.append({
        "id": cve.get("id"),
        "description": eng_desc[:500],
        "published": cve.get("published"),
        "cvss_score": cvss_score,
    })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(simplified, f, indent=2)

print(f"Saved {len(simplified)} CVE entries to {OUT}")