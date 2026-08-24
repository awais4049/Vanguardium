"""
Vanguardium capture addon.
Hooks into mitmproxy's request/response lifecycle and extracts
structured fields from every flow that passes through, appending
each as a JSON line to a persistent capture file.
"""
from mitmproxy import http
import json
import os

# Output file: one JSON object per line (JSONL format)
OUTPUT_PATH = os.path.join(
    "D:\\", "Awais", "Important", "FYP - Vanguardium", "Project",
    "data", "captured_traffic", "capture_log.jsonl"
)

def response(flow: http.HTTPFlow) -> None:
    """
    Called by mitmproxy automatically once a full request/response
    pair (a 'flow') has completed. We pull out the fields we care
    about, print them for live visibility, and append them to disk
    as a JSON line so nothing is lost between sessions.
    """
    record = {
        "method": flow.request.method,
        "path": flow.request.path,
        "query_params": dict(flow.request.query),
        "headers": dict(flow.request.headers),
        "body": flow.request.get_text(strict=False),
        "status_code": flow.response.status_code,
        "response_size": len(flow.response.raw_content) if flow.response.raw_content else 0,
        "duration_ms": round((flow.response.timestamp_end - flow.request.timestamp_start) * 1000, 2),
    }
    print(record)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")