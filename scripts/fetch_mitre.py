import requests
import json

URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
OUT = "data/mitre/mitre_attack_techniques.json"
LIMIT = 100  # starter subset size

print("Downloading MITRE ATT&CK STIX bundle (this is a few MB, may take a moment)...")
resp = requests.get(URL, timeout=60)
resp.raise_for_status()
bundle = resp.json()

techniques = [
    obj for obj in bundle["objects"]
    if obj.get("type") == "attack-pattern" and not obj.get("revoked", False)
]

print(f"Total techniques found: {len(techniques)}")

subset = techniques[:LIMIT]
simplified = [
    {
        "id": t.get("external_references", [{}])[0].get("external_id", ""),
        "name": t.get("name"),
        "description": t.get("description", "")[:500],
        "tactics": [phase["phase_name"] for phase in t.get("kill_chain_phases", [])],
    }
    for t in subset
]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(simplified, f, indent=2)

print(f"Saved {len(simplified)} techniques to {OUT}")