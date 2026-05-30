from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl


STAGE2_MODEL_VERSION = "stage2_nuextract8b_v1"

VALUE_LIMITS = {
    "adhesion_strength": (-1e-9, 100000.0),
    "ionic_conductivity": (1e-15, 100.0),
    "discharge_capacity": (-1e-9, 5000.0),
    "thermal_transition": (-273.15, 5000.0),
    "process_temperature": (-273.15, 5000.0),
}

PROPERTY_DOMAINS = {
    "ionic_conductivity": "solid_electrolyte",
    "discharge_capacity": "battery",
    "process_temperature": "synthesis_process",
    "thermal_transition": "polymer_or_thermal_material",
    "adhesion_strength": "adhesive_or_interface",
}

MATERIAL_STOP = {
    "",
    "A",
    "AN",
    "THE",
    "NONE",
    "NULL",
    "REPORTED MATERIAL",
    "TEMPERATURE",
}


def normalize_number(value: Any, fallback_text: str | None = None) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value if value not in {None, ""} else fallback_text or "")
    text = text.replace(",", "").replace("$", "").strip()
    text = text.replace("×", "x").replace("\\times", "x")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    sci = re.search(r"(?P<m>[-+]?\d+(?:\.\d+)?)\s*x\s*10\s*(?:\^?\{?(?P<e>[-+]?\d+)\}?)", text, re.I)
    if sci:
        return float(sci.group("m")) * (10 ** int(sci.group("e")))
    pow10 = re.search(r"10\s*(?:\^|\{)\s*\{?(?P<e>[-+]?\d+)\}?", text, re.I)
    if pow10:
        return 10 ** int(pow10.group("e"))
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    if match:
        return float(match.group(0))
    return None


def range_max(raw: str, first_value: float | None) -> float | None:
    if re.search(r"(±|卤|\+/-)", raw):
        return None
    if re.search(r"\b(?:and|between)\b", raw, re.I):
        return None
    match = re.search(
        r"(?P<lo>[-+]?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(?P<hi>[-+]?\d+(?:\.\d+)?)",
        raw,
        re.I,
    )
    if match:
        hi = normalize_number(None, match.group("hi"))
        if hi is not None:
            return hi
    return None


def clean_material(material: Any, prop: str) -> str:
    text = str(material or "").replace("$", "").strip()
    text = re.sub(r"\s+", " ", text).strip(" ,.;:()[]")
    text = text.replace("\\mathrm", "").replace("{", "").replace("}", "")
    if len(text) > 90 and re.search(r"[;；。]|溶于|加入|搅拌|得到|条件下", text):
        text = ""
    if text.upper() in MATERIAL_STOP or len(text) < 2:
        if prop == "process_temperature":
            return "reported process/material"
        if prop == "thermal_transition":
            return "reported material"
        return "reported material"
    return text[:180]


def relation_from_raw(raw: str, value_max: float | None) -> str:
    stripped = raw.strip()
    if value_max is not None:
        return "range"
    if stripped.startswith(">"):
        return ">"
    if stripped.startswith("<"):
        return "<"
    if stripped.startswith("~") or "about" in stripped.lower() or "approx" in stripped.lower():
        return "approx"
    return "="


def normalize_unit(prop: str, unit: Any, value: float) -> tuple[str, float]:
    raw = str(unit or "")
    compact = raw.replace("$", "").replace("{", "").replace("}", "")
    compact = compact.replace("−", "-").replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", " ", compact).strip()
    lower = compact.lower()
    if prop == "ionic_conductivity":
        if "μs" in lower or "µs" in lower or re.search(r"\bus\b", lower):
            return "S cm^-1", value / 1_000_000.0
        if lower.startswith("ms") or " ms" in lower:
            return "S cm^-1", value / 1000.0
        return "S cm^-1", value
    if prop == "discharge_capacity":
        return "mAh g^-1", value
    if prop == "process_temperature" or prop == "thermal_transition":
        if re.search(r"\bK\b", compact) and not re.search(r"°|℃|C\b", compact):
            return "K", value
        return "C", value
    if prop == "adhesion_strength":
        if re.search(r"\bkPa\b", compact, re.I):
            return "kPa", value
        if re.search(r"N\s*/\s*mm", compact, re.I):
            return "N/mm", value
        if re.search(r"\bkN\s*/\s*m", compact, re.I):
            return "N/m", value * 1000.0
        if re.search(r"N\s*/\s*m", compact, re.I):
            return "N/m", value
        return "MPa", value
    return compact, value


def condition_obj(parsed: dict[str, Any]) -> dict[str, Any] | None:
    condition: dict[str, Any] = {}
    for source, key in [("condition", "condition_text"), ("method", "method"), ("quantity_name", "quantity_name")]:
        value = parsed.get(source)
        if isinstance(value, str) and value.strip():
            condition[key] = compact_text(value, 300)
    quote = parsed.get("evidence_quote")
    if isinstance(quote, str) and quote.strip():
        condition["evidence_quote"] = compact_text(quote, 500)
    return condition or None


def record_id(record: dict[str, Any]) -> str:
    payload = "|".join(
        [
            str(record.get("doc_id")),
            str(record.get("page_range")),
            str(record.get("material")).lower(),
            str(record.get("property")),
            str(round(float(record.get("value", 0.0)), 10)),
            str(round(float(record.get("value_max", record.get("value", 0.0))), 10)),
            str(record.get("unit")),
            json.dumps(record.get("condition", {}), sort_keys=True, ensure_ascii=False),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return None, ["missing_parsed"]
    prop = row.get("property_hint") or parsed.get("property")
    if isinstance(prop, list):
        prop = prop[0] if prop else None
    prop = str(prop or "")
    raw_value = str(parsed.get("raw_value") or parsed.get("value") or "")
    raw_first_value = normalize_number(None, raw_value)
    raw_value_max = range_max(raw_value, raw_first_value)
    value = normalize_number(parsed.get("value"), raw_value)
    if raw_value_max is not None and raw_first_value is not None:
        value = raw_first_value
    elif raw_first_value is not None and prop in {"process_temperature", "thermal_transition"} and value is not None:
        if abs(value - raw_first_value) > max(abs(raw_first_value) * 0.25, 20.0):
            value = raw_first_value
    if value is None:
        return None, ["missing_value"]
    unit, value = normalize_unit(prop, parsed.get("unit"), value)
    lo_hi = VALUE_LIMITS.get(prop)
    if lo_hi and not (lo_hi[0] <= value <= lo_hi[1]):
        errors.append("value_out_of_range")
    value_max_raw = raw_value_max
    value_max = None
    if value_max_raw is not None:
        _, value_max = normalize_unit(prop, parsed.get("unit"), value_max_raw)
    material = clean_material(parsed.get("material"), prop)
    evidence = compact_text(str(row.get("evidence") or ""), 1600)
    confidence = 0.86
    support = (row.get("checks") or {}).get("support_score")
    if isinstance(support, (int, float)):
        confidence = min(0.95, 0.72 + float(support) * 0.03)
    if row.get("strict_accepted"):
        confidence += 0.03
    if material.startswith("reported "):
        confidence -= 0.08
    record: dict[str, Any] = {
        "doc_id": str(row.get("doc_id") or ""),
        "page_range": str(row.get("page_range") or ""),
        "domain": PROPERTY_DOMAINS.get(prop, "materials"),
        "material": material,
        "property": prop,
        "value": value,
        "unit": unit,
        "value_text": compact_text(raw_value, 100),
        "relation": relation_from_raw(raw_value, value_max),
        "evidence": evidence,
        "confidence": round(max(0.0, min(confidence, 0.99)), 2),
        "source_type": "nuextract_focused_window",
        "extraction_method": STAGE2_MODEL_VERSION,
        "candidate_score": row.get("candidate_score"),
    }
    if value_max is not None:
        record["value_max"] = value_max
    cond = condition_obj(parsed)
    if cond:
        record["condition"] = cond
    if row.get("item_key"):
        record["source_item_key"] = row.get("item_key")
    if row.get("source_candidate_key"):
        record["source_candidate_key"] = row.get("source_candidate_key")
    if row.get("focus_span"):
        record["focus_span"] = row.get("focus_span")
    record["record_id"] = record_id(record)
    return record, errors


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["record_id"]
        if key not in best or record.get("confidence", 0.0) > best[key].get("confidence", 0.0):
            best[key] = record
    return sorted(
        best.values(),
        key=lambda r: (
            int(r["doc_id"]) if str(r["doc_id"]).isdigit() else 10**9,
            r["property"],
            r["page_range"],
            -float(r.get("confidence", 0.0)),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize accepted NuExtract trials into stage2 material records.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "stage2_nuextract")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    raw_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        record, errors = convert_row(row)
        if record and not errors:
            raw_records.append(record)
        else:
            rejected.append({"errors": errors, "row": row, "record": record})
    records = dedupe(raw_records)
    high_conf = [
        record
        for record in records
        if float(record.get("confidence", 0.0)) >= 0.85 and not str(record.get("material", "")).startswith("reported ")
    ]
    out_dir = ensure_dir(args.output_dir)
    write_jsonl(out_dir / "schema_records.raw.jsonl", raw_records)
    write_jsonl(out_dir / "schema_records.jsonl", records)
    write_jsonl(out_dir / "schema_records.high_confidence.jsonl", high_conf)
    write_jsonl(out_dir / "schema_records.rejected.jsonl", rejected)
    by_doc = ensure_dir(out_dir / "by_doc")
    for doc_id in sorted({record["doc_id"] for record in records}, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        write_jsonl(by_doc / f"{doc_id}_schema_records.jsonl", [record for record in records if record["doc_id"] == doc_id])
    write_json(
        out_dir / "summary.json",
        {
            "stage2_version": STAGE2_MODEL_VERSION,
            "input": str(args.input),
            "raw_input_rows": len(rows),
            "raw_record_count": len(raw_records),
            "deduped_record_count": len(records),
            "high_confidence_record_count": len(high_conf),
            "rejected_record_count": len(rejected),
            "doc_count": len({record["doc_id"] for record in records}),
            "high_confidence_doc_count": len({record["doc_id"] for record in high_conf}),
            "by_property": dict(Counter(record["property"] for record in records)),
            "high_confidence_by_property": dict(Counter(record["property"] for record in high_conf)),
            "rejection_reasons": dict(Counter(error for item in rejected for error in item["errors"])),
        },
    )
    print(f"nuextract_stage2={out_dir / 'schema_records.jsonl'} records={len(records)} high_conf={len(high_conf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
