from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl


AUDIT_VERSION = "semantic_audit_material_v1"
SCRIPT_DIR = Path(__file__).resolve().parent

VERDICTS = ("ACCEPT", "PARTIAL", "REJECT", "NEEDS_SCHEMA_DECISION")

SCHEMA_BOUNDARY: dict[str, Any] = {
    "schema_decision": (
        "Use the current five-property material schema as the delivery boundary. "
        "For thermal_transition, Tg/Tm/Tc/melting/crystallization are in scope. "
        "Decomposition/TGA/weight-loss/heat-distortion/LCST/UCST records are routed to "
        "NEEDS_SCHEMA_DECISION until the schema is explicitly widened with subtypes."
    ),
    "properties": {
        "adhesion_strength": {
            "accept": "Adhesion, peel, bond, lap-shear, interface, or adhesive-joint strength.",
            "reject": "Generic modulus, flexural/compressive strength, or tensile strength without adhesive/interface context.",
        },
        "ionic_conductivity": {
            "accept": "Ionic/Li+ conductivity for electrolyte or solid-state battery materials.",
            "reject": "Voltage, capacity, efficiency, thickness, cycle count, or generic electrical data.",
        },
        "discharge_capacity": {
            "accept": "Battery discharge/specific capacity with material/sample and important test conditions.",
            "reject": "Voltage, efficiency, retention, cycle number, plotted coordinate, or adjacent table column.",
        },
        "thermal_transition": {
            "accept": "Tg/Tm/Tc/melting/crystallization transition temperatures.",
            "needs_schema_decision": "Decomposition, degradation, TGA weight-loss, heat-distortion, LCST, or UCST.",
            "reject": "Delta/change-only values, test program temperatures, cycle settings, and process temperatures.",
        },
        "process_temperature": {
            "accept": "Preparation, curing, sintering, annealing, drying, calcination, deposition, polymerization, or synthesis temperatures.",
            "reject": "Heating rate, test/measurement temperature, DSC/TGA program, plot axis, operation temperature, or cycle setting.",
        },
    },
}

UNIT_PATTERNS = {
    "adhesion_strength": re.compile(r"(?:\bMPa\b|\bkPa\b|N\s*/\s*mm|N\s*/\s*m|N\s*cm\s*[-^]?\s*1)", re.I),
    "ionic_conductivity": re.compile(r"(?:[uµμ]?S|mS|S)\s*(?:/|cm|c\s*m|\\cdot|[.]|\s)+\s*(?:cm|c\s*m)?\s*(?:[-^]?\s*1|[-^]\{?1\}?|⁻¹)?", re.I),
    "discharge_capacity": re.compile(r"mAh\s*(?:/|g|g\s*\^?\s*[-]?\s*1|g[-]1|g⁻¹)", re.I),
    "thermal_transition": re.compile(r"(?:deg\s*C|degrees?\s*C|°\s*C|℃|\bC\b|\bK\b)", re.I),
    "process_temperature": re.compile(r"(?:deg\s*C|degrees?\s*C|°\s*C|℃|\bC\b|\bK\b)", re.I),
}

PREP_PROCESS_RE = re.compile(
    r"(sinter|anneal|calcina|cur(?:e|ed|ing)|dry|dried|depos|baking|heated|hot-?press|"
    r"hydrothermal|solvothermal|prepared|preparation|synthesi|polymeri|stirring|"
    r"烧结|退火|干燥|固化|煅烧|沉积|制备|水热|溶剂热|搅拌)",
    re.I,
)
TEST_CONTEXT_RE = re.compile(
    r"(test(?:ed|ing)?|measur(?:ed|ement|ing)|DSC|TGA|thermogravimetric|DMA|thermomechanical|"
    r"cycle|cycling|operation|operating|surface temperature|temperature axis|plot|测试|测量|循环|温度轴)",
    re.I,
)
HEATING_RATE_RE = re.compile(
    r"(?:°\s*C|℃|\bC\b|\bK\b)\s*(?:/|per)\s*min|(?:K|C)\s*min\s*[-^]?\s*1|"
    r"heating rate|scan rate|ramp(?:ing)? rate|升温速率",
    re.I,
)
THERMAL_NEEDS_DECISION_RE = re.compile(
    r"(decompos|degrad|TGA|thermogravimetric|weight\s*loss|T5\b|T10\b|T50\b|"
    r"heat\s*distortion|HDT\b|LCST|UCST|分解|降解|热重|失重|热变形)",
    re.I,
)
THERMAL_TRANSITION_RE = re.compile(
    r"(Tg\b|T_g|glass\s*transition|Tm\b|T_m|melting|melting\s*point|Tc\b|T_c|crystalli[sz]ation|"
    r"玻璃化|熔点|结晶)",
    re.I,
)
DELTA_ONLY_RE = re.compile(
    r"(increase[sd]?|decrease[sd]?|raise[sd]?|lower[sd]?|shift(?:ed)?|change[sd]?|improv(?:ed|ement)|"
    r"reduc(?:ed|tion))\s+(?:\w+\s+){0,6}(?:by|of|for each)\b|"
    r"(?:higher|lower)\s+by\s+[-+]?\d",
    re.I,
)
BATTERY_CONDITION_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\s*C\b|C-rate|current density|mA\s*g|cycle|cycles|after\s+\d+|"
    r"\b\d+(?:\.\d+)?\s*V\b|voltage window|vs\.?|cathode|anode|electrode|coin cell|"
    r"room temperature|RT\b|°\s*C|℃)",
    re.I,
)
VOLTAGE_OR_EFFICIENCY_RE = re.compile(r"(voltage|mV\b|\bV\b|efficien|retention|coulombic|plateau)", re.I)
CAPACITY_CELL_BINDING_RE = re.compile(
    r"(capacity|capacities)\s+of\s+(?:the\s+)?(?:battery|cell)|battery\s+(?:delivered|exhibited|showed|with|had).*capacity|"
    r"charge(?:-| )discharge\s+capacit",
    re.I,
)
ELECTROLYTE_MATERIAL_RE = re.compile(
    r"(electrolyte|separator|LLZO|LLZTO|LATP|LAGP|garnet|NASICON|PEO|polymer electrolyte|solid electrolyte)",
    re.I,
)
CELL_COMPONENT_CONTEXT_RE = re.compile(
    r"(positive electrode|negative electrode|cathode|anode|solid electrolyte|electrolyte|separator|coin cell|full cell|battery consisting)",
    re.I,
)
ADHESION_CONTEXT_RE = re.compile(r"(adhesion|adhesive|bond|peel|lap|joint|interface|interfacial|粘|剥离|胶|界面)", re.I)
GENERIC_STRENGTH_RE = re.compile(r"(modulus|flexural|compressive|tensile|cohesive|mechanical strength|elastic)", re.I)
WEAK_MATERIAL_RE = re.compile(r"^(?:reported material|reported process/material|material|sample|group|specimen|row|column|none|null|unknown)\b", re.I)


def local_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    text = str(path)
    match = re.match(r"^[\\/]+mnt[\\/](?P<drive>[A-Za-z])[\\/](?P<rest>.*)$", text)
    if not match:
        return path
    rest = match.group("rest").replace("/", "\\")
    return Path(f"{match.group('drive').upper()}:\\{rest}")


def stable_record_id(record: dict[str, Any]) -> str:
    existing = record.get("record_id")
    if existing:
        return str(existing)
    payload = json.dumps(
        {
            "doc_id": record.get("doc_id"),
            "page_range": record.get("page_range"),
            "material": record.get("material"),
            "property": record.get("property"),
            "value": record.get("value"),
            "value_max": record.get("value_max"),
            "unit": record.get("unit"),
            "evidence": record.get("evidence"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\\times", "x").replace("×", "x")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("⁻", "-").replace("¹", "1")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def condition_text(record: dict[str, Any]) -> str:
    condition = record.get("condition")
    if isinstance(condition, dict):
        return " ".join(str(v) for v in condition.values() if v is not None)
    if isinstance(condition, str):
        return condition
    return ""


def support_text(record: dict[str, Any]) -> str:
    pieces = [
        record.get("material"),
        record.get("property"),
        record.get("value_text"),
        record.get("value"),
        record.get("value_max"),
        record.get("unit"),
        condition_text(record),
        record.get("evidence"),
    ]
    return " ".join(str(p) for p in pieces if p is not None)


def numeric_value(record: dict[str, Any]) -> float | None:
    value = record.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def duplicate_cluster_key(record: dict[str, Any]) -> str:
    value = numeric_value(record)
    value_part = "" if value is None else f"{value:.10g}"
    payload = [
        str(record.get("doc_id") or ""),
        norm_text(record.get("material")),
        str(record.get("property") or ""),
        value_part,
        norm_text(record.get("unit")),
    ]
    return "|".join(payload)


def add_flag(flags: list[dict[str, str]], severity: str, code: str, detail: str) -> None:
    flags.append({"severity": severity, "code": code, "detail": detail})


def unit_supported(prop: str, record: dict[str, Any], text: str) -> bool:
    pattern = UNIT_PATTERNS.get(prop)
    if not pattern:
        return False
    unit = str(record.get("unit") or "")
    return bool(pattern.search(unit) or pattern.search(text))


def value_supported(record: dict[str, Any], text: str) -> bool:
    value_text = str(record.get("value_text") or "").strip()
    if value_text and norm_text(value_text) in norm_text(text):
        return True
    value = numeric_value(record)
    if value is None:
        return False
    candidates = {f"{value:g}", f"{value:.1f}", f"{value:.2f}", str(int(value)) if value.is_integer() else ""}
    return any(candidate and re.search(rf"(?<!\d){re.escape(candidate)}(?!\d)", text) for candidate in candidates)


def audit_record(record: dict[str, Any], duplicate_count: int = 1) -> dict[str, Any]:
    prop = str(record.get("property") or "")
    text = support_text(record)
    flags: list[dict[str, str]] = []
    checks: dict[str, bool] = {
        "schema_scope_ok": True,
        "property_ok": True,
        "property_subtype_ok": True,
        "material_ok": True,
        "value_ok": True,
        "unit_ok": True,
        "condition_ok": True,
        "evidence_support_ok": True,
    }

    if prop not in SCHEMA_BOUNDARY["properties"]:
        checks["property_ok"] = False
        add_flag(flags, "hard_reject", "unknown_property", f"Property {prop!r} is outside the current schema.")
    if not str(record.get("evidence") or "").strip():
        checks["evidence_support_ok"] = False
        add_flag(flags, "hard_reject", "missing_evidence", "Record has no evidence text.")
    if not unit_supported(prop, record, text):
        checks["unit_ok"] = False
        add_flag(flags, "hard_reject", "unit_not_supported", "Unit is missing or does not match the property family.")
    if not value_supported(record, text):
        checks["value_ok"] = False
        add_flag(flags, "partial", "value_not_verbatim_supported", "Numeric value is not clearly found in the evidence/context text.")
    if WEAK_MATERIAL_RE.search(str(record.get("material") or "")):
        checks["material_ok"] = False
        add_flag(flags, "partial", "weak_material_binding", "Material is generic or weakly bound.")
    if duplicate_count > 1:
        add_flag(flags, "risk", "near_duplicate_cluster", f"Same doc/material/property/value/unit appears {duplicate_count} times.")

    value = numeric_value(record)
    lower = norm_text(text)

    if prop == "process_temperature":
        if HEATING_RATE_RE.search(text):
            checks["property_ok"] = False
            checks["value_ok"] = False
            add_flag(flags, "hard_reject", "process_temperature_is_rate", "Temperature-like value appears to be a heating/scan/ramp rate.")
        if TEST_CONTEXT_RE.search(text) and not PREP_PROCESS_RE.search(text):
            checks["property_ok"] = False
            add_flag(flags, "hard_reject", "process_temperature_test_context", "Temperature appears to be test/measurement/axis/operation context, not preparation.")
        if "heating temperature" in lower and not PREP_PROCESS_RE.search(text):
            checks["condition_ok"] = False
            add_flag(flags, "partial", "broad_heating_temperature_context", "Heating temperature needs process context confirmation.")

    elif prop == "thermal_transition":
        if THERMAL_NEEDS_DECISION_RE.search(text):
            checks["schema_scope_ok"] = False
            checks["property_subtype_ok"] = False
            add_flag(flags, "schema_decision", "thermal_stability_boundary", "Thermal stability/decomposition style value needs schema decision and subtype.")
        if DELTA_ONLY_RE.search(text) and not re.search(r"(?:=|:)\s*[-+]?\d", text):
            checks["value_ok"] = False
            add_flag(flags, "hard_reject", "thermal_delta_only", "Evidence appears to state a change/delta, not an actual transition value.")
        if PREP_PROCESS_RE.search(text) and not THERMAL_TRANSITION_RE.search(text) and not THERMAL_NEEDS_DECISION_RE.search(text):
            checks["property_ok"] = False
            add_flag(flags, "hard_reject", "thermal_transition_process_temperature", "Thermal record looks like a process temperature.")

    elif prop == "discharge_capacity":
        if VOLTAGE_OR_EFFICIENCY_RE.search(text) and not re.search(r"(capacity|specific capacity|discharge|mAh)", text, re.I):
            checks["property_ok"] = False
            add_flag(flags, "hard_reject", "capacity_wrong_quantity_context", "Evidence is dominated by voltage/efficiency/retention context.")
        if (
            CAPACITY_CELL_BINDING_RE.search(text)
            and CELL_COMPONENT_CONTEXT_RE.search(text)
            and ELECTROLYTE_MATERIAL_RE.search(str(record.get("material") or ""))
        ):
            checks["material_ok"] = False
            add_flag(
                flags,
                "partial",
                "capacity_cell_metric_bound_to_electrolyte",
                "Capacity appears to describe a full battery/cell while the record material is an electrolyte or cell component.",
            )
        if value is not None and value <= 5 and re.search(r"(table header|row:|target column|target value)", text, re.I):
            checks["value_ok"] = False
            add_flag(flags, "hard_reject", "capacity_suspicious_low_table_value", "Very low table value is likely voltage or adjacent-column leakage.")
        if not BATTERY_CONDITION_RE.search(text):
            checks["condition_ok"] = False
            add_flag(flags, "partial", "capacity_missing_test_condition", "Capacity lacks clear rate/cycle/voltage/electrode/temperature condition.")
        if re.search(r"(table header|target column|row context)", text, re.I):
            add_flag(flags, "risk", "capacity_table_binding_needs_review", "Table-derived capacity requires row/column binding review.")

    elif prop == "ionic_conductivity":
        if re.search(r"(capacity|mAh|voltage|efficien|retention|thickness|cycle)", text, re.I):
            checks["property_ok"] = False
            add_flag(flags, "hard_reject", "conductivity_wrong_quantity_context", "Evidence has battery/capacity/voltage or non-conductivity context.")
        if value is not None and value > 10:
            checks["value_ok"] = False
            add_flag(flags, "partial", "conductivity_value_extreme", "Ionic conductivity value is unusually high and needs review.")
        if re.search(r"(stored|storage|aged|after|at\s+\d+|°\s*C|℃|wt\s*%|filler)", text, re.I) and not condition_text(record).strip():
            checks["condition_ok"] = False
            add_flag(flags, "partial", "conductivity_condition_likely_missing", "Evidence suggests temperature/storage/filler condition but condition is empty.")

    elif prop == "adhesion_strength":
        if GENERIC_STRENGTH_RE.search(text) and not ADHESION_CONTEXT_RE.search(text):
            checks["property_ok"] = False
            add_flag(flags, "hard_reject", "adhesion_generic_mechanical_strength", "Mechanical strength lacks adhesive/interface context.")
        if re.search(r"\bshear strength\b", text, re.I) and not ADHESION_CONTEXT_RE.search(text):
            checks["condition_ok"] = False
            add_flag(flags, "partial", "shear_strength_context_unclear", "Shear strength needs adhesive/interface context confirmation.")

    severity_order = {flag["severity"] for flag in flags}
    if "hard_reject" in severity_order:
        verdict = "REJECT"
    elif "schema_decision" in severity_order:
        verdict = "NEEDS_SCHEMA_DECISION"
    elif "partial" in severity_order:
        verdict = "PARTIAL"
    else:
        verdict = "ACCEPT"

    return {
        "audit_version": AUDIT_VERSION,
        "record_id": stable_record_id(record),
        "doc_id": record.get("doc_id"),
        "property": prop,
        "duplicate_cluster_key": duplicate_cluster_key(record),
        "duplicate_count": duplicate_count,
        "rule_checks": checks,
        "flags": flags,
        "rule_verdict": verdict,
        "risk_score": sum({"hard_reject": 5, "schema_decision": 4, "partial": 3, "risk": 1}.get(flag["severity"], 0) for flag in flags),
        "record": record,
    }


def load_by_key(rows: list[dict[str, Any]], *keys: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value:
                out[str(value)] = row
    return out


def direct_find_context(text: str, fragments: list[str], radius: int) -> dict[str, Any]:
    lower = text.lower()
    for raw in fragments:
        fragment = " ".join(str(raw or "").split())
        if len(fragment) < 16:
            continue
        for candidate in [fragment, fragment[:240], fragment[:120]]:
            if len(candidate) < 16:
                continue
            idx = lower.find(candidate.lower())
            if idx >= 0:
                start = max(0, idx - radius)
                end = min(len(text), idx + len(candidate) + radius)
                return {
                    "status": "found",
                    "char_start": idx,
                    "char_end": idx + len(candidate),
                    "context": text[start:end],
                }
    return {"status": "not_found", "context": ""}


def context_fragments(record: dict[str, Any], trial: dict[str, Any] | None, candidate: dict[str, Any] | None) -> list[str]:
    fragments: list[str] = []
    for obj in [record, trial, candidate]:
        if not isinstance(obj, dict):
            continue
        for key in ["evidence", "target_value", "signal", "section_title"]:
            value = obj.get(key)
            if isinstance(value, str):
                fragments.append(value.replace("Section:", "").strip())
        parsed = obj.get("parsed")
        if isinstance(parsed, dict):
            for key in ["evidence_quote", "raw_value", "material"]:
                value = parsed.get(key)
                if isinstance(value, str):
                    fragments.append(value)
    condition = record.get("condition")
    if isinstance(condition, dict):
        for key in ["evidence_quote", "quantity_name"]:
            value = condition.get(key)
            if isinstance(value, str):
                fragments.append(value)
    material = str(record.get("material") or "")
    value_text = str(record.get("value_text") or record.get("value") or "")
    if material and value_text:
        fragments.append(f"{material} {value_text}")
    expanded: list[str] = []
    for fragment in fragments:
        expanded.append(fragment)
        expanded.extend(part.strip() for part in re.split(r"\||Row:|Target value:|Target column:|Row context:", fragment) if part.strip())
    return expanded


def load_combined_context(
    combined_dir: Path,
    record: dict[str, Any],
    trial: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    cache: dict[Path, str],
    radius: int,
) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    path = combined_dir / doc_id / f"{doc_id}_combined.md"
    if not path.exists():
        return {"status": "missing_combined", "combined_path": str(path), "context": ""}
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8", errors="ignore")
    found = direct_find_context(cache[path], context_fragments(record, trial, candidate), radius)
    found["combined_path"] = str(path)
    return found


def make_judge_packet(audit_row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    record = audit_row["record"]
    payload = {
        "record": record,
        "rule_precheck": {
            "verdict": audit_row["rule_verdict"],
            "checks": audit_row["rule_checks"],
            "flags": audit_row["flags"],
        },
        "source_item_key": record.get("source_item_key"),
        "schema_trial": context.get("schema_trial"),
        "recall_candidate": context.get("recall_candidate"),
        "combined_context": context.get("combined_context"),
        "schema_boundary": SCHEMA_BOUNDARY,
    }
    system_prompt = (
        "You are a strict semantic judge for a materials information extraction task. "
        "Return only one JSON object."
    )
    user_prompt = (
        "Judge whether the record is supported by the evidence and context. "
        "Check schema_scope_ok, property_ok, property_subtype_ok, material_ok, value_ok, "
        "unit_ok, condition_ok, and evidence_support_ok. "
        "verdict must be one of ACCEPT, PARTIAL, REJECT, NEEDS_SCHEMA_DECISION. "
        "Do not accept only because JSON is well-formed or a number is near a unit. "
        "For tables, verify row/column binding. "
        "Thermal decomposition/TGA records should be NEEDS_SCHEMA_DECISION unless the schema is widened."
    )
    return {
        "packet_id": audit_row["record_id"],
        "audit_version": AUDIT_VERSION,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\n\nPayload:\n" + json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        "payload": payload,
    }


def choose_samples(
    audit_rows: list[dict[str, Any]],
    random_per_property: int,
    risk_per_property: int,
    schema_decision_limit: int,
    duplicate_cluster_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}

    def mark(row: dict[str, Any], reason: str) -> None:
        rid = row["record_id"]
        if rid not in selected:
            selected[rid] = {**row, "sample_reasons": []}
        selected[rid]["sample_reasons"].append(reason)

    by_prop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        by_prop[str(row.get("property") or "")].append(row)

    for prop, rows in sorted(by_prop.items()):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for row in shuffled[: max(0, random_per_property)]:
            mark(row, f"random_property:{prop}")

        risky = [row for row in rows if int(row.get("risk_score") or 0) > 0]
        risky.sort(key=lambda row: (-int(row.get("risk_score") or 0), str(row.get("record_id") or "")))
        for row in risky[: max(0, risk_per_property)]:
            mark(row, f"risk_property:{prop}")

    needs = [row for row in audit_rows if row.get("rule_verdict") == "NEEDS_SCHEMA_DECISION"]
    needs.sort(key=lambda row: (-int(row.get("risk_score") or 0), str(row.get("record_id") or "")))
    for row in needs[: max(0, schema_decision_limit)]:
        mark(row, "needs_schema_decision")

    seen_clusters: set[str] = set()
    duplicate_rows = [row for row in audit_rows if int(row.get("duplicate_count") or 0) > 1]
    duplicate_rows.sort(key=lambda row: (str(row.get("duplicate_cluster_key") or ""), str(row.get("record_id") or "")))
    for row in duplicate_rows:
        cluster = str(row.get("duplicate_cluster_key") or "")
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        mark(row, "duplicate_cluster")
        if len(seen_clusters) >= duplicate_cluster_limit:
            break

    out = list(selected.values())
    out.sort(
        key=lambda row: (
            str(row.get("property") or ""),
            -int(row.get("risk_score") or 0),
            str(row.get("doc_id") or ""),
            str(row.get("record_id") or ""),
        )
    )
    return out


def summarize(audit_rows: list[dict[str, Any]], samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_property = Counter(str(row.get("property") or "") for row in audit_rows)
    verdict_by_property: dict[str, dict[str, int]] = {}
    flag_counts = Counter()
    for row in audit_rows:
        prop = str(row.get("property") or "")
        verdict_by_property.setdefault(prop, Counter())
        verdict_by_property[prop][str(row.get("rule_verdict") or "")] += 1
        for flag in row.get("flags") or []:
            flag_counts[str(flag.get("code") or "unknown")] += 1
    return {
        "audit_version": AUDIT_VERSION,
        "records_input": str(args.records),
        "schema": str(args.schema),
        "stage2_schema": str(args.stage2_schema),
        "total_records": len(audit_rows),
        "sample_records": len(samples),
        "by_property": dict(sorted(by_property.items())),
        "rule_verdict_by_property": {prop: dict(counts) for prop, counts in sorted(verdict_by_property.items())},
        "flag_counts": dict(flag_counts.most_common()),
        "schema_boundary": SCHEMA_BOUNDARY,
    }


def write_report(path: Path, summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    lines = ["# Material Semantic Audit", "", "## Summary", ""]
    lines.append(f"- total_records: {summary['total_records']}")
    lines.append(f"- sample_records: {summary['sample_records']}")
    lines.append("")
    lines.append("## Rule Verdict By Property")
    lines.append("")
    for prop, counts in (summary.get("rule_verdict_by_property") or {}).items():
        rendered = ", ".join(f"{verdict}={counts.get(verdict, 0)}" for verdict in VERDICTS)
        lines.append(f"- {prop}: {rendered}")
    lines.append("")
    lines.append("## Top Flags")
    lines.append("")
    for code, count in list((summary.get("flag_counts") or {}).items())[:20]:
        lines.append(f"- {code}: {count}")
    lines.append("")
    lines.append("## Sample Examples")
    lines.append("")
    for row in samples[:20]:
        record = row.get("record") or {}
        flags = ",".join(flag.get("code", "") for flag in row.get("flags") or []) or "none"
        lines.append(
            f"- {row.get('rule_verdict')} doc {record.get('doc_id')} p{record.get('page_range')} "
            f"{record.get('property')} | {record.get('material')} | {record.get('value')} {record.get('unit')} | {flags}"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build semantic audit samples and AI judge packets for material records.")
    parser.add_argument("--records", type=Path, default=DEFAULT_WORK_ROOT / "stage2_nuextract_recall_v3_full" / "schema_records.high_confidence.jsonl")
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR / "schemas" / "default_material_schema.json")
    parser.add_argument("--stage2-schema", type=Path, default=SCRIPT_DIR / "schemas" / "stage2_material_fact_schema.json")
    parser.add_argument("--trials", type=Path, default=DEFAULT_WORK_ROOT / "model_experiments" / "stage2_nuextract_8b_recall_v3_full" / "schema_trials.jsonl")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_ROOT / "candidates" / "focused_evidence_candidates.recall_v3.jsonl")
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_WORK_ROOT / "combined")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "semantic_audit_material_v1")
    parser.add_argument("--random-per-property", type=int, default=30)
    parser.add_argument("--risk-per-property", type=int, default=50)
    parser.add_argument("--schema-decision-limit", type=int, default=120)
    parser.add_argument("--duplicate-cluster-limit", type=int, default=120)
    parser.add_argument("--context-radius", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    args.records = local_path(args.records)
    args.schema = local_path(args.schema)
    args.stage2_schema = local_path(args.stage2_schema)
    args.trials = local_path(args.trials)
    args.candidates = local_path(args.candidates)
    args.combined_dir = local_path(args.combined_dir)
    args.output_dir = local_path(args.output_dir)

    records = read_jsonl(args.records)
    duplicate_counts = Counter(duplicate_cluster_key(record) for record in records)
    audit_rows = [audit_record(record, duplicate_counts[duplicate_cluster_key(record)]) for record in records]
    samples = choose_samples(
        audit_rows,
        random_per_property=args.random_per_property,
        risk_per_property=args.risk_per_property,
        schema_decision_limit=args.schema_decision_limit,
        duplicate_cluster_limit=args.duplicate_cluster_limit,
        seed=args.seed,
    )

    trials = load_by_key(read_jsonl(args.trials), "item_key")
    candidates = load_by_key(read_jsonl(args.candidates), "candidate_key")
    combined_cache: dict[Path, str] = {}

    enriched_samples: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    for row in samples:
        record = row["record"]
        source_key = str(record.get("source_item_key") or "")
        trial = trials.get(source_key)
        candidate = candidates.get(source_key)
        combined_context = load_combined_context(args.combined_dir, record, trial, candidate, combined_cache, args.context_radius)
        context = {
            "schema_trial": trial,
            "recall_candidate": candidate,
            "combined_context": combined_context,
        }
        enriched = {**row, **context}
        enriched_samples.append(enriched)
        packets.append(make_judge_packet(row, context))

    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "schema_boundary_decision.json", SCHEMA_BOUNDARY)
    write_jsonl(out_dir / "rule_audit_all.jsonl", audit_rows)
    write_jsonl(out_dir / "audit_samples.jsonl", enriched_samples)
    write_jsonl(out_dir / "ai_judge_packets.jsonl", packets)
    by_prop_dir = ensure_dir(out_dir / "by_property")
    for prop in sorted({str(row.get("property") or "") for row in enriched_samples}):
        write_jsonl(by_prop_dir / f"{prop}.audit_samples.jsonl", [row for row in enriched_samples if row.get("property") == prop])
    summary = summarize(audit_rows, enriched_samples, args)
    write_json(out_dir / "semantic_audit_summary.json", summary)
    write_report(out_dir / "semantic_audit_report.md", summary, enriched_samples)
    print(f"semantic_audit={out_dir} records={len(records)} samples={len(enriched_samples)} packets={len(packets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
