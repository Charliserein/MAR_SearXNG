#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config_dir="${project_dir}/deploy/searxng"
compose_file="${config_dir}/compose.yml"
operation="${1:-status}"

case "${operation}" in
  up)
    python3 - "${config_dir}/instance.env" <<'PY'
import os
import pathlib
import secrets
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write("SEARXNG_SECRET=" + secrets.token_hex(32) + "\n")
PY
    docker compose -f "${compose_file}" up -d --wait --wait-timeout 90
    ;;
  down)
    docker compose -f "${compose_file}" down
    ;;
  restart)
    docker compose -f "${compose_file}" restart
    docker compose -f "${compose_file}" up -d --wait --wait-timeout 90
    ;;
  status)
    docker compose -f "${compose_file}" ps
    ;;
  logs)
    docker compose -f "${compose_file}" logs --tail 100
    ;;
  check)
    python3 - "${project_dir}" <<'PY'
import datetime
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

root = pathlib.Path(sys.argv[1])
base = "http://127.0.0.1:18888"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
tests = []
for category, query in [
    ("general", "NASA Artemis I launch November 16 2022"),
    ("images", "NASA Artemis I launch November 16 2022"),
]:
    started = time.monotonic()
    url = base + "/search?" + urllib.parse.urlencode({
        "q": query, "format": "json", "categories": category, "language": "en",
    })
    item = {"category": category, "query": query}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MAR-Cache-Check/1.0"})
        with opener.open(request, timeout=60) as response:
            payload = json.load(response)
            item["http_status"] = response.status
        rows = payload.get("results", [])
        item.update({
            "status": "ok" if rows else "empty",
            "result_count": len(rows),
            "engines_returning_results": sorted({
                name for row in rows
                for name in row.get("engines", [row.get("engine", "")])
                if name
            }),
            "unresponsive_engines": payload.get("unresponsive_engines", []),
            "sample_results": [{
                key: row.get(key)
                for key in ("title", "url", "img_src", "engine", "engines")
                if row.get(key)
            } for row in rows[:3]],
        })
    except Exception as exc:
        item.update({"status": "error", "error": str(exc)})
    item["elapsed_seconds"] = round(time.monotonic() - started, 3)
    tests.append(item)
try:
    container = json.loads(subprocess.check_output([
        "docker", "inspect", "mar-cached-searxng", "--format",
        '{"image_id":{{json .Image}},"configured_image":{{json .Config.Image}},"status":{{json .State.Status}}}',
    ], text=True))
except Exception as exc:
    container = {"inspection_error": str(exc)}
report = {
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_url": base,
    "container": container,
    "native_reverse_image_search": False,
    "images_category_semantics": "text_query_to_image_results",
    "sources": [
        "https://docs.searxng.org/dev/search_api.html",
        "https://docs.searxng.org/admin/installation-docker.html",
        "https://github.com/searxng/searxng/issues/291",
        "https://github.com/searxng/searxng/blob/master/searx/engines/google_images.py",
    ],
    "checks": tests,
}
output = root / "outputs/searxng-check.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "report": str(output),
    "checks": [{key: item[key] for key in ("category", "status", "result_count", "unresponsive_engines") if key in item} for item in tests],
}, ensure_ascii=False, indent=2))
sys.exit(0 if all(item["status"] == "ok" for item in tests) else 1)
PY
    ;;
  *)
    echo "用法：$0 {up|down|restart|status|logs|check}" >&2
    exit 2
    ;;
esac
