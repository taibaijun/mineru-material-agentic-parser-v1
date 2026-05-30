from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Search material extraction JSONL records.")
    parser.add_argument("query")
    parser.add_argument("--dataset", type=Path, default=Path("dataset.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    q = args.query.lower()
    count = 0
    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if q not in json.dumps(row, ensure_ascii=False).lower():
                continue
            print(json.dumps({
                "doc_id": row.get("doc_id"),
                "material": row.get("material"),
                "property": row.get("property"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "evidence_quote": row.get("evidence_quote"),
            }, ensure_ascii=False))
            count += 1
            if count >= args.limit:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
