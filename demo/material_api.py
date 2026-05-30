from __future__ import annotations

import argparse
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def match_query(row: dict, query: str) -> bool:
    if not query:
        return True
    blob = json.dumps(row, ensure_ascii=False).lower()
    return query.lower() in blob


def make_handler(records: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            if parsed.path == "/health":
                self.send_json({"ok": True, "records": len(records)})
                return
            if parsed.path == "/stats":
                self.send_json({
                    "records": len(records),
                    "by_property": dict(Counter(str(r.get("property") or "unknown") for r in records)),
                    "by_source_doc_type": dict(Counter(str(r.get("source_doc_type") or "unknown") for r in records)),
                    "docs": len({str(r.get("doc_id") or "unknown") for r in records}),
                })
                return
            if parsed.path in {"/records", "/search"}:
                prop = (qs.get("property") or [""])[0]
                doc_id = (qs.get("doc_id") or [""])[0]
                q = (qs.get("q") or [""])[0]
                limit = int((qs.get("limit") or ["50"])[0])
                out = []
                for row in records:
                    if prop and str(row.get("property") or "") != prop:
                        continue
                    if doc_id and str(row.get("doc_id") or "") != doc_id:
                        continue
                    if not match_query(row, q):
                        continue
                    out.append(row)
                    if len(out) >= limit:
                        break
                self.send_json({"count": len(out), "records": out})
                return
            self.send_json({"error": "not found", "paths": ["/health", "/stats", "/records", "/search"]}, 404)
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve MinerU material extraction dataset as a tiny HTTP API.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    records = load_jsonl(args.dataset)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(records))
    print(f"Serving {len(records)} records at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
