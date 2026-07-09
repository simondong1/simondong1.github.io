#!/usr/bin/env bash
# Notify IndexNow participants (Bing, Yandex, Seznam, Naver, ...) about site URLs.
set -euo pipefail
KEY=d019f3c68a3557c811ecdc27f5ac0dc3
HOST=simondong1.github.io
KEYLOC="https://${HOST}/${KEY}.txt"
URLS=(
  "https://${HOST}/"
  "https://${HOST}/mla-decoding.html"
  "https://${HOST}/mla.html"
  "https://${HOST}/gqa.html"
  "https://${HOST}/rope.html"
  "https://${HOST}/llms.txt"
  "https://${HOST}/sitemap.xml"
  "https://${HOST}/feed.xml"
)
# Allow extra URLs as args
URLS+=("$@")
python3 - "$KEY" "$HOST" "$KEYLOC" "${URLS[@]}" <<'PY'
import json, sys, urllib.request
key, host, keyloc, *urls = sys.argv[1:]
payload = json.dumps({"host": host, "key": key, "keyLocation": keyloc, "urlList": urls}).encode()
for ep in [
  "https://api.indexnow.org/indexnow",
  "https://www.bing.com/indexnow",
  "https://yandex.com/indexnow",
  "https://search.seznam.cz/indexnow",
  "https://searchadvisor.naver.com/indexnow",
]:
  req = urllib.request.Request(ep, data=payload, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
  try:
    with urllib.request.urlopen(req, timeout=30) as r:
      print(ep, r.status)
  except Exception as e:
    print(ep, "ERR", getattr(e, "code", e))
PY
