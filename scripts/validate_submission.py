from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = ["doc_id", "material", "property", "value", "unit", "evidence_quote", "record_id"]
ALLOWED = {"process_temperature", "thermal_transition", "discharge_capacity", "ionic_conductivity", "adhesion_strength"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON error at {path}:{line_no}: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the public submission JSONL structure.")
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset.jsonl"))
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    errors = []
    ids = set()
    for i, row in enumerate(rows, 1):
        for field in REQUIRED:
            if row.get(field) in (None, ""):
                errors.append(f"line {i}: missing {field}")
        if row.get("property") not in ALLOWED:
            errors.append(f"line {i}: invalid property {row.get('property')}")
        rid = str(row.get("record_id") or "")
        if rid in ids:
            errors.append(f"line {i}: duplicate record_id {rid}")
        ids.add(rid)
    print(json.dumps({
        "records": len(rows),
        "errors": errors[:50],
        "error_count": len(errors),
        "by_property": dict(Counter(str(r.get("property") or "unknown") for r in rows)),
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
