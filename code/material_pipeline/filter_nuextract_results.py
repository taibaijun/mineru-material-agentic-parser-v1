from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import ensure_dir, read_jsonl, write_json, write_jsonl


UNIT_PATTERNS = {
    "ionic_conductivity": re.compile(
        r"(?:[µμu]?S|mS|S)\s*(?:/|\\cdot|·|\*)?\s*c\s*m\s*(?:\$?\s*\^?\s*\{?\s*[-−⁻]?\s*1\s*\}?\s*\$?|[-−⁻]1|⁻¹)?",
        re.I,
    ),
    "discharge_capacity": re.compile(r"mAh\s*(?:/|g\s*\$?\s*\^?\s*\{?\s*[-−⁻]?\s*1\s*\}?\s*\$?|g[-−⁻]?1|\s*g\b)", re.I),
    "process_temperature": re.compile(r"(?:°\s*C|℃|deg\s*C|\bC\b|\bK\b)", re.I),
    "thermal_transition": re.compile(r"(?:°\s*C|℃|deg\s*C|\bC\b|\bK\b)", re.I),
    "adhesion_strength": re.compile(r"(?:\bMPa\b|\bkPa\b|N\s*/\s*mm|N\s*cm\s*\^?\s*-?1|N\s*/\s*m)", re.I),
}

QUANTITY_PATTERNS = {
    "ionic_conductivity": re.compile(r"(ionic|Li\+?|conductiv|电导|离子)", re.I),
    "discharge_capacity": re.compile(r"(capacity|specific capacity|discharge|比容量|放电)", re.I),
    "process_temperature": re.compile(
        r"(sinter|anneal|calcina|cur|dry|depos|polymeri|heat(?:ed|ing)?|温度|烧结|退火|干燥|固化|煅烧)",
        re.I,
    ),
    "thermal_transition": re.compile(
        r"(Tg|T_g|Tm|T_m|Tc|T_c|glass transition|melting|crystalli[sz]ation|heat distortion|decomposition|"
        r"玻璃化|熔点|结晶|热变形|分解)",
        re.I,
    ),
    "adhesion_strength": re.compile(r"(adhesion|bond|peel|lap|shear|tensile|strength|粘|剥离|剪切|强度)", re.I),
}

REJECT_PATTERNS = {
    "ionic_conductivity": re.compile(r"(capacity|efficiency|voltage|thickness|cycle)", re.I),
    "discharge_capacity": re.compile(r"(efficiency|voltage|retention|conductiv|thickness)", re.I),
    "process_temperature": re.compile(
        r"(ionic conductivity|capacity|efficiency|time to ignite|burning behavior|modulus|Tg|T_g|Tm|T_m|Tc|T_c)",
        re.I,
    ),
    "thermal_transition": re.compile(r"(modulus|stress|strength|capacity|conductiv|MPa|kPa|S/cm|mAh/g)", re.I),
    "adhesion_strength": re.compile(r"(conductiv|capacity|efficiency)", re.I),
}


def text_field(parsed: dict[str, Any], *names: str) -> str:
    values: list[str] = []
    for name in names:
        value = parsed.get(name)
        if isinstance(value, str):
            values.append(value)
    return " ".join(values)


def contains_relaxed(needle: str | None, haystack: str) -> bool:
    if not needle:
        return False
    needle_norm = relaxed_text(needle)
    haystack_norm = relaxed_text(haystack)
    if needle_norm in haystack_norm:
        return True
    raw_number = re.search(r"[-+]?\d+(?:\.\d+)?", needle_norm)
    return bool(raw_number and raw_number.group(0) in haystack_norm)


def relaxed_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\\times", "x").replace("\times", "x").replace("×", "x")
    text = text.replace("\t", "x")
    text = text.replace("⁻", "-").replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("¹", "1").replace("²", "2").replace("³", "3")
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def near_raw_value(parsed: dict[str, Any], evidence: str, radius: int = 140) -> str:
    raw = parsed.get("raw_value")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    candidates = [raw]
    candidates.append(raw.replace("×", "\\times"))
    candidates.append(raw.replace("×", "x"))
    number_match = re.search(r"\d+(?:\.\d+)?", raw)
    if number_match:
        candidates.append(number_match.group(0))
    lowered = evidence.lower()
    for candidate in candidates:
        idx = lowered.find(candidate.lower())
        if idx >= 0:
            return evidence[max(0, idx - radius) : min(len(evidence), idx + len(candidate) + radius)]
    return ""


def validate_row(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    parsed = row.get("parsed")
    prop = row.get("property_hint")
    evidence = str(row.get("evidence") or "")
    if not isinstance(parsed, dict):
        return False, {"reason": "missing_parsed"}
    if parsed.get("is_relevant") != "yes":
        return False, {"reason": "not_relevant"}
    if parsed.get("property") != prop:
        return False, {"reason": "property_mismatch"}

    unit_text = text_field(parsed, "unit", "raw_value")
    quantity_text = text_field(parsed, "quantity_name", "evidence_quote", "condition")
    raw_near = near_raw_value(parsed, evidence)
    support_text = " ".join([unit_text, quantity_text, raw_near, evidence[:800]])
    unit_ok = bool(UNIT_PATTERNS[prop].search(unit_text))
    quantity_ok = bool(QUANTITY_PATTERNS[prop].search(quantity_text) or QUANTITY_PATTERNS[prop].search(raw_near))
    if prop == "process_temperature":
        quantity_ok = bool(QUANTITY_PATTERNS[prop].search(raw_near) or QUANTITY_PATTERNS[prop].search(text_field(parsed, "condition", "method", "evidence_quote")))
        axis_like = re.search(r"(temperature\s*\([^)]*\)|^temperature$|modulus|GPa|time to ignite)", text_field(parsed, "quantity_name", "evidence_quote"), re.I)
        if axis_like and not QUANTITY_PATTERNS[prop].search(text_field(parsed, "condition", "method", "evidence_quote")):
            quantity_ok = False
    if prop == "thermal_transition":
        quantity_ok = bool(QUANTITY_PATTERNS[prop].search(text_field(parsed, "quantity_name", "evidence_quote")))
        raw_value = str(parsed.get("raw_value") or "")
        if re.search(r"\b(?:between|and)\b", raw_value, re.I) and not re.search(r"\b(?:Tg|T_g|Tm|T_m|Tc|T_c|glass transition|melting point)\b", raw_value, re.I):
            quantity_ok = False
        if re.search(r"\bT_sw\b|switching|thermomechanical experiments", text_field(parsed, "quantity_name", "condition", "method", "evidence_quote"), re.I):
            quantity_ok = False
    reject_hit = bool(REJECT_PATTERNS[prop].search(text_field(parsed, "quantity_name", "unit", "raw_value", "evidence_quote")))
    raw_supported = contains_relaxed(parsed.get("raw_value"), evidence) or contains_relaxed(parsed.get("evidence_quote"), evidence)
    unit_supported = contains_relaxed(parsed.get("unit"), evidence) or bool(UNIT_PATTERNS[prop].search(evidence))

    checks = {
        "unit_ok": unit_ok,
        "quantity_ok": quantity_ok,
        "reject_hit": reject_hit,
        "raw_or_quote_supported": raw_supported,
        "unit_supported": unit_supported,
    }
    ok = unit_ok and quantity_ok and not reject_hit and raw_supported and unit_supported
    if not ok:
        reasons = [name for name, value in checks.items() if (name == "reject_hit" and value) or (name != "reject_hit" and not value)]
        checks["reason"] = ",".join(reasons)
    checks["support_text_sample"] = support_text[:300]
    return ok, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter NuExtract JSONL trials with property-specific validators.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_prop = Counter()
    accepted_by_prop = Counter()
    reason_counts = Counter()

    for row in rows:
        prop = str(row.get("property_hint"))
        by_prop[prop] += 1
        ok, strict_checks = validate_row(row)
        row = {**row, "strict_checks": strict_checks, "strict_accepted": ok}
        if ok:
            accepted.append(row)
            accepted_by_prop[prop] += 1
        else:
            rejected.append(row)
            reason_counts[strict_checks.get("reason", "unknown")] += 1

    ensure_dir(args.output.parent)
    write_jsonl(args.output, accepted)
    if args.rejected_output:
        write_jsonl(args.rejected_output, rejected)
    write_json(
        args.summary,
        {
            "input": str(args.input),
            "output": str(args.output),
            "total": len(rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "by_property": dict(sorted(by_prop.items())),
            "accepted_by_property": dict(sorted(accepted_by_prop.items())),
            "rejection_reasons": dict(reason_counts.most_common()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
