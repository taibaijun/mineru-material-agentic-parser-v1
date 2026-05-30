from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl
from filter_nuextract_results import validate_row
from normalize_nuextract_records import convert_row


EXPERIMENT_VERSION = "deepseek_v4_flash_json_probe_v6_quality_gate"

PROPERTY_LABELS = [
    "ionic_conductivity",
    "discharge_capacity",
    "process_temperature",
    "thermal_transition",
    "adhesion_strength",
]

PROPERTY_INSTRUCTIONS = {
    "ionic_conductivity": (
        "Target only ionic/Li+/electrolyte conductivity measurements. "
        "Valid units include S/cm, S cm^-1, mS/cm, or mS cm^-1. "
        "Reject voltage, capacity, thickness, temperature-only, and cycle-count values."
    ),
    "discharge_capacity": (
        "Target only battery specific capacity or discharge capacity measurements. "
        "Valid units include mAh/g or mAh g^-1. "
        "Reject efficiency percentages, voltage plateaus, cycle numbers, and retention percentages. "
        "If the evidence is just a plotted curve point without material/sample support, set is_relevant to no."
    ),
    "process_temperature": (
        "Target only synthesis, curing, sintering, annealing, drying, calcination, deposition, or preparation process temperatures. "
        "Valid units include degC, C, °C, ℃, or K. "
        "Reject measurement axes, test temperatures, DSC/TGA transition values, operation temperatures, time-to-ignite, and values whose unit is not temperature."
    ),
    "thermal_transition": (
        "Target only thermal transition temperatures such as Tg, glass transition, Tm, melting point, Tc, crystallization, heat distortion, or decomposition temperature. "
        "Valid units include degC, C, °C, ℃, or K. "
        "Reject modulus, stress, strength, capacity, conductivity, and process temperatures."
    ),
    "adhesion_strength": (
        "Target only adhesion, bond, peel, shear, tensile, or lap-shear strength measurements for adhesives or interfaces. "
        "Valid units include MPa, kPa, N/mm, N cm^-1, or similar force/area or peel units. "
        "Reject elastic modulus, flexural strength, temperature, capacity, and conductivity unless the evidence explicitly says adhesion/bond strength."
    ),
}

TEMPLATE: dict[str, Any] = {
    "is_relevant": "yes or no",
    "material": "verbatim string or null",
    "property": "one of: ionic_conductivity, discharge_capacity, process_temperature, thermal_transition, adhesion_strength",
    "quantity_name": "verbatim string or null",
    "raw_value": "target numeric value with unit as written, or null",
    "value": "number or null",
    "unit": "verbatim unit or null",
    "condition": "verbatim condition string or null",
    "method": "verbatim method string or null",
    "evidence_quote": "short verbatim supporting quote or table fragment, or null",
    "confidence_reason": "brief reason",
}

PROCESS_KEYWORD_RE = re.compile(
    r"(sinter|anneal|calcina|cur(?:e|ed|ing)|dry|dried|depos|baking|heated|hot-?press|hydrothermal|solvothermal|"
    r"prepared|synthesi|polymeri|stirring|烧结|退火|干燥|固化|煅烧|沉积|制备|水热|溶剂热|搅拌)",
    re.I,
)
TRANSFORMED_AXIS_RE = re.compile(r"(\bln\s*\(|\blog\s*\(|log\s+conductiv|1000\s*/\s*T|1000\s*T\s*[-⁻]?\s*1)", re.I)
THERMAL_DELTA_RE = re.compile(
    r"(increase[sd]?|decrease[sd]?|raise[sd]?|lower[sd]?|shift[sed]?|change[sd]?|var(?:y|ied))\s+"
    r"(?:\w+\s+){0,5}(?:by|per|for each)\b|per\s+(?:C\s+unit|carbon|monomer)",
    re.I,
)
REAL_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
GENERIC_MATERIALS = {"", "none", "null", "reported material", "reported process/material", "material", "sample", "target"}
WEAK_MATERIAL_RE = re.compile(r"^(?:group|sample|specimen|entry|row|column)\s*[#:]?\s*[A-Za-z0-9-]+$", re.I)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def item_key(row: dict[str, Any]) -> str:
    payload = {
        "doc_id": row.get("doc_id"),
        "page_range": row.get("page_range"),
        "property_hint": row.get("property_hint"),
        "evidence": row.get("evidence"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def local_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    text = str(path)
    match = re.match(r"^[\\/]+mnt[\\/](?P<drive>[A-Za-z])[\\/](?P<rest>.*)$", text)
    if not match:
        return path
    rest = match.group("rest").replace("/", "\\")
    return Path(f"{match.group('drive').upper()}:\\{rest}")


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        key = row.get("item_key")
        if key:
            completed.add(str(key))
    return completed


def normalize_parsed(obj: Any, property_hint: str) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    parsed = dict(obj)
    relevant = parsed.get("is_relevant")
    if isinstance(relevant, bool):
        parsed["is_relevant"] = "yes" if relevant else "no"
    elif isinstance(relevant, str):
        parsed["is_relevant"] = "yes" if relevant.strip().lower() in {"yes", "true", "relevant"} else "no"
    else:
        parsed["is_relevant"] = "no"

    prop = parsed.get("property")
    if isinstance(prop, list):
        prop = prop[0] if prop else None
    if isinstance(prop, str) and prop in PROPERTY_LABELS:
        parsed["property"] = prop
    else:
        parsed["property"] = property_hint if parsed["is_relevant"] == "yes" else prop

    for field in ["material", "quantity_name", "raw_value", "unit", "condition", "method", "evidence_quote", "confidence_reason"]:
        if field not in parsed:
            parsed[field] = None
        elif parsed[field] is not None and not isinstance(parsed[field], str):
            parsed[field] = str(parsed[field])
    return parsed


def parse_model_content(content: str, property_hint: str) -> tuple[dict[str, Any] | None, str | None]:
    text = content.strip()
    if not text:
        return None, "empty_output"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_failed: {exc}"
    parsed = normalize_parsed(obj, property_hint)
    if parsed is None:
        return None, "json_not_object"
    return parsed, None


def field_text(parsed: dict[str, Any], *names: str) -> str:
    return " ".join(str(parsed.get(name) or "") for name in names)


def has_specific_material(parsed: dict[str, Any]) -> bool:
    material = re.sub(r"\s+", " ", str(parsed.get("material") or "")).strip(" ,.;:()[]").lower()
    if material in GENERIC_MATERIALS:
        return False
    if len(material) < 2:
        return False
    if WEAK_MATERIAL_RE.search(material):
        return False
    return True


def quality_validate_row(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A stricter gate for AI-repaired records before they enter the normalizer/submission path."""
    if row.get("strict_accepted") is not True:
        return False, {"reason": "strict_not_accepted"}
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return False, {"reason": "missing_parsed"}
    prop = str(row.get("property_hint") or parsed.get("property") or "")
    evidence = str(row.get("evidence") or "")
    support = field_text(parsed, "quantity_name", "raw_value", "condition", "method", "evidence_quote")
    raw_value = str(parsed.get("raw_value") or "")

    record, errors = convert_row(row)
    checks: dict[str, Any] = {
        "normalizer_ok": bool(record and not errors),
        "normalizer_errors": errors,
        "has_specific_material": has_specific_material(parsed),
        "has_numeric_raw_or_value": bool(REAL_NUMBER_RE.search(raw_value) or isinstance(parsed.get("value"), (int, float))),
        "transformed_axis": bool(TRANSFORMED_AXIS_RE.search(f"{support} {evidence}")),
        "figure_caption": "Figure Captions" in evidence,
    }

    reasons: list[str] = []
    if not checks["normalizer_ok"]:
        reasons.extend(errors or ["normalizer_failed"])
    if not checks["has_numeric_raw_or_value"]:
        reasons.append("missing_numeric_value")
    if prop in {"ionic_conductivity", "discharge_capacity", "adhesion_strength", "thermal_transition"} and not checks["has_specific_material"]:
        reasons.append("missing_specific_material")
    if prop == "process_temperature" and not (checks["has_specific_material"] or PROCESS_KEYWORD_RE.search(support)):
        reasons.append("missing_process_context")
    if checks["transformed_axis"]:
        reasons.append("transformed_axis_value")
    if checks["figure_caption"] and not checks["has_specific_material"]:
        reasons.append("figure_caption_without_material")
    if prop == "discharge_capacity" and str(row.get("source_type") or "").endswith("_cell") and "Row label:" not in evidence:
        reasons.append("capacity_cell_without_row_context")
    if prop == "process_temperature" and not PROCESS_KEYWORD_RE.search(support):
        reasons.append("process_keyword_missing_near_value")
    if prop == "thermal_transition" and THERMAL_DELTA_RE.search(f"{support} {evidence}"):
        reasons.append("thermal_delta_not_transition_value")

    if reasons:
        checks["reason"] = ",".join(dict.fromkeys(reasons))
        return False, checks
    checks["reason"] = "accepted"
    return True, checks


def make_messages(row: dict[str, Any], max_evidence_chars: int) -> list[dict[str, str]]:
    prop = str(row.get("property_hint") or "")
    evidence = compact_text(str(row.get("evidence") or ""), max_evidence_chars)
    system = (
        "You are a precise materials-science information extraction engine. "
        "Return only one valid json object. Use null for missing fields. "
        "Do not infer values, materials, units, or conditions that are not explicitly supported by the evidence. "
        "The evidence_quote must be copied from the evidence as a short supporting phrase or table fragment."
    )
    user = (
        "Extract at most one target measurement record from the evidence.\n"
        f"Expected property hint: {prop}\n"
        f"Property-specific rules: {PROPERTY_INSTRUCTIONS.get(prop, '')}\n"
        f"Document id: {row.get('doc_id')}. Page range: {row.get('page_range')}.\n\n"
        "If the evidence does not support the expected property, set is_relevant to no and set value/unit/material to null.\n"
        "A yes record must contain a real numeric target measurement. If the evidence only has a column header, unit label, method name, "
        "or material name without the numeric measurement value, set is_relevant to no.\n"
        "Reject transformed plot or axis values such as ln(sigma), log(sigma), log conductivity, 1000/T, x/y curve coordinates, "
        "or negative log values unless the evidence also states the actual untransformed target measurement value.\n"
        "For thermal transitions, reject delta/change statements such as 'Tm increases by 1-3 °C per carbon unit'; "
        "only extract the actual transition temperature of a material.\n"
        "For process_temperature, put the process word such as sintering, curing, annealing, drying, calcination, deposition, "
        "or stirring in condition/method/evidence_quote together with the temperature value.\n"
        "When the evidence uses a symbol or abbreviated table header, quantity_name must include the canonical property words plus the symbol, "
        "for example 'ionic conductivity (sigma_total)', 'discharge capacity', 'process temperature (sintering)', "
        "'thermal transition temperature (Tg)', or 'adhesion strength'.\n"
        "Return json in exactly this shape:\n"
        f"{json.dumps(TEMPLATE, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
    thinking: str,
    retries: int,
) -> tuple[str, dict[str, Any], str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    last_error: str | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = str(obj["choices"][0]["message"].get("content") or "")
            return content, obj.get("usage") or {}, None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"http_{exc.code}: {detail[:500]}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    return "", {}, last_error or "unknown_api_error"


def load_baseline_status(
    schema_trials: Path | None,
    accepted_path: Path | None,
    rejected_path: Path | None,
) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    if schema_trials and schema_trials.exists():
        for row in read_jsonl(schema_trials):
            key = row.get("item_key")
            if key:
                baseline[str(key)] = {"trial": row, "strict_accepted": None, "strict_reason": None}
    if accepted_path and accepted_path.exists():
        for row in read_jsonl(accepted_path):
            key = row.get("item_key")
            if key:
                baseline.setdefault(str(key), {})["strict_accepted"] = True
                baseline[str(key)]["strict_reason"] = "accepted"
                baseline[str(key)]["trial"] = row
    if rejected_path and rejected_path.exists():
        for row in read_jsonl(rejected_path):
            key = row.get("item_key")
            if key:
                checks = row.get("strict_checks") or {}
                baseline.setdefault(str(key), {})["strict_accepted"] = False
                baseline[str(key)]["strict_reason"] = checks.get("reason") or row.get("parse_error") or "rejected"
                baseline[str(key)]["trial"] = row
    return baseline


def source_priority(row: dict[str, Any]) -> int:
    source = str(row.get("source_type") or "")
    if source.endswith("_cell"):
        return 0
    if source.endswith("_row"):
        return 1
    return 2


def choose_probe_rows(
    candidates: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    per_property_accepted: int,
    per_property_rejected: int,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_prop_status: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        prop = str(row.get("property_hint") or "")
        if prop not in PROPERTY_LABELS:
            continue
        key = item_key(row)
        status = baseline.get(key, {}).get("strict_accepted")
        bucket = "accepted" if status is True else "rejected" if status is False else "unknown"
        row = {**row, "item_key": key}
        rows_by_prop_status[(prop, bucket)].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for prop in PROPERTY_LABELS:
        accepted = rows_by_prop_status.get((prop, "accepted"), [])
        accepted.sort(key=lambda r: (-float(r.get("score") or 0.0), source_priority(r), str(r.get("doc_id")), str(r.get("page_range"))))
        selected.extend(accepted[:per_property_accepted])

        rejected = rows_by_prop_status.get((prop, "rejected"), [])
        reason_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rejected:
            reason = str(baseline.get(row["item_key"], {}).get("strict_reason") or "rejected")
            reason_groups[reason].append(row)
        prop_rejected: list[dict[str, Any]] = []
        for reason in sorted(reason_groups, key=lambda r: (0 if "quantity_ok" in r or "unit_ok" in r else 1, r)):
            group = reason_groups[reason]
            rng.shuffle(group)
            prop_rejected.extend(group[: max(1, per_property_rejected // 2)])
            if len(prop_rejected) >= per_property_rejected:
                break
        if len(prop_rejected) < per_property_rejected:
            pool = [row for group in reason_groups.values() for row in group if row not in prop_rejected]
            pool.sort(key=lambda r: (-float(r.get("score") or 0.0), source_priority(r), str(r.get("doc_id")), str(r.get("page_range"))))
            prop_rejected.extend(pool[: per_property_rejected - len(prop_rejected)])
        selected.extend(prop_rejected[:per_property_rejected])

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in selected:
        key = row["item_key"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:limit]


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_prop = Counter(str(row.get("property_hint")) for row in rows)
    accepted = [row for row in rows if row.get("strict_accepted") is True]
    rejected = [row for row in rows if row.get("strict_accepted") is False]
    quality_accepted = [row for row in rows if row.get("quality_accepted") is True]
    quality_rejected = [row for row in rows if row.get("quality_accepted") is False]
    baseline_accept = [row for row in rows if row.get("baseline_strict_accepted") is True]
    baseline_reject = [row for row in rows if row.get("baseline_strict_accepted") is False]
    rescued = [row for row in rows if row.get("baseline_strict_accepted") is False and row.get("strict_accepted") is True]
    lost = [row for row in rows if row.get("baseline_strict_accepted") is True and row.get("strict_accepted") is False]
    quality_rescued = [row for row in rows if row.get("baseline_strict_accepted") is False and row.get("quality_accepted") is True]
    quality_lost = [row for row in rows if row.get("baseline_strict_accepted") is True and row.get("quality_accepted") is False]

    normalized_ok = 0
    normalized_reasons = Counter()
    for row in accepted:
        record, errors = convert_row(row)
        if record and not errors:
            normalized_ok += 1
        else:
            normalized_reasons.update(errors or ["unknown"])

    return {
        "time": now_iso(),
        "version": EXPERIMENT_VERSION,
        "total": len(rows),
        "json_ok": sum(1 for row in rows if row.get("parse_error") is None and row.get("parsed")),
        "api_or_parse_failures": sum(1 for row in rows if row.get("parse_error")),
        "strict_accepted": len(accepted),
        "strict_rejected": len(rejected),
        "strict_accept_rate": round(len(accepted) / len(rows), 4) if rows else 0.0,
        "quality_accepted": len(quality_accepted),
        "quality_rejected": len(quality_rejected),
        "quality_accept_rate": round(len(quality_accepted) / len(rows), 4) if rows else 0.0,
        "baseline_accepted_in_sample": len(baseline_accept),
        "baseline_rejected_in_sample": len(baseline_reject),
        "agreement_with_baseline": sum(
            1
            for row in rows
            if row.get("baseline_strict_accepted") is not None
            and row.get("baseline_strict_accepted") == row.get("strict_accepted")
        ),
        "rescued_baseline_rejections": len(rescued),
        "lost_baseline_acceptances": len(lost),
        "quality_rescued_baseline_rejections": len(quality_rescued),
        "quality_lost_baseline_acceptances": len(quality_lost),
        "normalized_ok_from_deepseek_accepted": normalized_ok,
        "normalization_rejection_reasons": dict(normalized_reasons),
        "by_property": dict(sorted(by_prop.items())),
        "strict_accepted_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in accepted).items())),
        "quality_accepted_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in quality_accepted).items())),
        "rescued_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in rescued).items())),
        "lost_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in lost).items())),
        "quality_rescued_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in quality_rescued).items())),
        "quality_lost_by_property": dict(sorted(Counter(str(row.get("property_hint")) for row in quality_lost).items())),
        "rejection_reasons": dict(Counter(str((row.get("strict_checks") or {}).get("reason", "unknown")) for row in rejected).most_common()),
        "quality_rejection_reasons": dict(Counter(str((row.get("quality_checks") or {}).get("reason", "unknown")) for row in quality_rejected).most_common()),
    }


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# DeepSeek Extraction Probe")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in [
        "total",
        "json_ok",
        "strict_accepted",
        "strict_rejected",
        "strict_accept_rate",
        "quality_accepted",
        "quality_rejected",
        "quality_accept_rate",
        "baseline_accepted_in_sample",
        "baseline_rejected_in_sample",
        "agreement_with_baseline",
        "rescued_baseline_rejections",
        "lost_baseline_acceptances",
        "quality_rescued_baseline_rejections",
        "quality_lost_baseline_acceptances",
        "normalized_ok_from_deepseek_accepted",
    ]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## By Property")
    lines.append("")
    for prop, count in (summary.get("by_property") or {}).items():
        accepted = (summary.get("strict_accepted_by_property") or {}).get(prop, 0)
        quality = (summary.get("quality_accepted_by_property") or {}).get(prop, 0)
        rescued = (summary.get("rescued_by_property") or {}).get(prop, 0)
        lost = (summary.get("lost_by_property") or {}).get(prop, 0)
        quality_rescued = (summary.get("quality_rescued_by_property") or {}).get(prop, 0)
        quality_lost = (summary.get("quality_lost_by_property") or {}).get(prop, 0)
        lines.append(
            f"- {prop}: strict={accepted}/{count}, quality={quality}/{count}, "
            f"rescued={rescued}, lost={lost}, quality_rescued={quality_rescued}, quality_lost={quality_lost}"
        )
    lines.append("")
    lines.append("## Rejection Reasons")
    lines.append("")
    for reason, count in (summary.get("rejection_reasons") or {}).items():
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("## Quality Gate Rejection Reasons")
    lines.append("")
    for reason, count in (summary.get("quality_rejection_reasons") or {}).items():
        lines.append(f"- {reason}: {count}")

    def add_examples(title: str, items: list[dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        if not items:
            lines.append("- none")
            return
        for row in items[:8]:
            parsed = row.get("parsed") or {}
            checks = row.get("strict_checks") or {}
            lines.append(
                f"- {row.get('property_hint')} doc {row.get('doc_id')} p{row.get('page_range')}: "
                f"baseline={row.get('baseline_strict_accepted')}({row.get('baseline_strict_reason')}) "
                f"deepseek={row.get('strict_accepted')}({checks.get('reason', 'accepted')}) "
                f"quality={row.get('quality_accepted')}({(row.get('quality_checks') or {}).get('reason', 'accepted')}); "
                f"{parsed.get('material')} | {parsed.get('raw_value')} {parsed.get('unit')} | "
                f"{compact_text(parsed.get('evidence_quote') or row.get('evidence') or '', 180)}"
            )

    add_examples(
        "Rescued Baseline Rejections",
        [row for row in rows if row.get("baseline_strict_accepted") is False and row.get("strict_accepted") is True],
    )
    add_examples(
        "Quality Rescued Baseline Rejections",
        [row for row in rows if row.get("baseline_strict_accepted") is False and row.get("quality_accepted") is True],
    )
    add_examples(
        "Lost Baseline Acceptances",
        [row for row in rows if row.get("baseline_strict_accepted") is True and row.get("strict_accepted") is False],
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def regrade_result_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        row["strict_accepted"] = False
        row["strict_checks"] = {"reason": row.get("parse_error") or "missing_parsed"}
        row["checks"] = row["strict_checks"]
        row["quality_accepted"] = False
        row["quality_checks"] = {"reason": "strict_not_accepted"}
        return row
    strict_accepted, strict_checks = validate_row(row)
    quality_accepted, quality_checks = quality_validate_row(
        {**row, "strict_accepted": strict_accepted, "strict_checks": strict_checks}
    )
    row["strict_accepted"] = strict_accepted
    row["strict_checks"] = strict_checks
    row["checks"] = strict_checks
    row["quality_accepted"] = quality_accepted
    row["quality_checks"] = quality_checks
    return row


def process_one_row(
    *,
    row: dict[str, Any],
    index: int,
    api_key: str,
    args: argparse.Namespace,
    baseline_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = row["item_key"]
    prop = str(row.get("property_hint") or "")
    started = time.time()
    messages = make_messages(row, args.max_evidence_chars)
    output_text, usage, api_error = call_deepseek(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
        thinking=args.thinking,
        retries=args.retries,
    )
    parsed: dict[str, Any] | None = None
    parse_error = api_error
    strict_accepted = False
    strict_checks: dict[str, Any] = {"reason": "api_or_parse_error"} if api_error else {}
    quality_accepted = False
    quality_checks: dict[str, Any] = {"reason": "strict_not_accepted"}
    if api_error is None:
        parsed, parse_error = parse_model_content(output_text, prop)
        strict_accepted, strict_checks = validate_row({**row, "parsed": parsed}) if parsed else (False, {"reason": parse_error})
        quality_accepted, quality_checks = quality_validate_row(
            {**row, "parsed": parsed, "strict_accepted": strict_accepted, "strict_checks": strict_checks}
        ) if parsed else (False, {"reason": parse_error})

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "time": now_iso(),
        "model": args.model,
        "item_key": key,
        "doc_id": row.get("doc_id"),
        "page_range": row.get("page_range"),
        "property_hint": prop,
        "candidate_score": row.get("score"),
        "candidate_units": row.get("units"),
        "candidate_values": row.get("values"),
        "source_type": row.get("source_type"),
        "evidence": row.get("evidence"),
        "output_text": output_text,
        "parsed": parsed,
        "parse_error": parse_error,
        "checks": strict_checks,
        "strict_checks": strict_checks,
        "strict_accepted": strict_accepted,
        "quality_checks": quality_checks,
        "quality_accepted": quality_accepted,
        "baseline_strict_accepted": baseline_info.get("strict_accepted"),
        "baseline_strict_reason": baseline_info.get("strict_reason"),
        "baseline_parsed": (baseline_info.get("trial") or {}).get("parsed"),
        "usage": usage,
        "elapsed_sec": round(time.time() - started, 3),
    }
    log_row = {
        "event": "item_done",
        "time": now_iso(),
        "index": index,
        "item_key": key,
        "property_hint": prop,
        "strict_accepted": strict_accepted,
        "quality_accepted": quality_accepted,
        "baseline_strict_accepted": baseline_info.get("strict_accepted"),
        "parse_error": parse_error,
        "elapsed_sec": result["elapsed_sec"],
    }
    return result, log_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe DeepSeek JSON extraction on recall v3 material evidence candidates.")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--candidates-path", type=Path, default=None)
    parser.add_argument("--baseline-schema-trials", type=Path, default=None)
    parser.add_argument("--baseline-accepted", type=Path, default=None)
    parser.add_argument("--baseline-rejected", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="disabled")
    parser.add_argument("--per-property-accepted", type=int, default=4)
    parser.add_argument("--per-property-rejected", type=int, default=4)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--max-evidence-chars", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} before running.")

    work_root = local_path(args.work_root)
    candidates_path = local_path(args.candidates_path) if args.candidates_path else (work_root / "candidates" / "focused_evidence_candidates.recall_v3.jsonl")
    exp_root = work_root / "model_experiments" / "stage2_nuextract_8b_recall_v3_full"
    baseline_schema = local_path(args.baseline_schema_trials) if args.baseline_schema_trials else (exp_root / "schema_trials.jsonl")
    baseline_accepted = local_path(args.baseline_accepted) if args.baseline_accepted else (exp_root / "accepted_strict.jsonl")
    baseline_rejected = local_path(args.baseline_rejected) if args.baseline_rejected else (exp_root / "rejected_strict.jsonl")
    output_dir = local_path(args.output_dir) if args.output_dir else (work_root / "model_experiments" / "stage2_deepseek_v4_flash_recall_v3_probe")
    output_path = output_dir / "schema_trials.jsonl"
    summary_path = output_dir / "summary.json"
    accepted_path = output_dir / "accepted_strict.jsonl"
    rejected_path = output_dir / "rejected_strict.jsonl"
    accepted_quality_path = output_dir / "accepted_quality.jsonl"
    rejected_quality_path = output_dir / "rejected_quality.jsonl"
    report_path = output_dir / "deepseek_probe_report.md"
    log_path = output_dir / "run.log"
    ensure_dir(output_dir)

    candidates = read_jsonl(candidates_path)
    baseline = load_baseline_status(baseline_schema, baseline_accepted, baseline_rejected)
    selected = choose_probe_rows(
        candidates,
        baseline,
        per_property_accepted=args.per_property_accepted,
        per_property_rejected=args.per_property_rejected,
        limit=args.limit,
        seed=args.seed,
    )
    completed = load_completed(output_path)
    pending = [row for row in selected if row["item_key"] not in completed]

    append_jsonl(
        log_path,
        {
            "event": "start",
            "time": now_iso(),
            "version": EXPERIMENT_VERSION,
            "model": args.model,
            "candidates": len(candidates),
            "selected": len(selected),
            "completed": len(completed),
            "pending": len(pending),
        },
    )

    write_lock = threading.Lock()

    def write_result(result: dict[str, Any], log_row: dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(output_path, result)
            append_jsonl(log_path, log_row)

    if args.workers <= 1:
        for index, row in enumerate(pending, start=1):
            result, log_row = process_one_row(
                row=row,
                index=index,
                api_key=api_key,
                args=args,
                baseline_info=baseline.get(row["item_key"], {}),
            )
            write_result(result, log_row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_one_row,
                    row=row,
                    index=index,
                    api_key=api_key,
                    args=args,
                    baseline_info=baseline.get(row["item_key"], {}),
                )
                for index, row in enumerate(pending, start=1)
            ]
            for future in as_completed(futures):
                result, log_row = future.result()
                write_result(result, log_row)

    rows = read_jsonl(output_path)
    selected_keys = {row["item_key"] for row in selected}
    rows = [regrade_result_row(row) for row in rows if row.get("item_key") in selected_keys]
    write_jsonl(output_path, rows)
    accepted = [row for row in rows if row.get("strict_accepted") is True]
    rejected = [row for row in rows if row.get("strict_accepted") is not True]
    accepted_quality = [row for row in rows if row.get("quality_accepted") is True]
    rejected_quality = [row for row in rows if row.get("quality_accepted") is not True]
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)
    write_jsonl(accepted_quality_path, accepted_quality)
    write_jsonl(rejected_quality_path, rejected_quality)
    summary = summarize_results(rows)
    summary.update(
        {
            "model": args.model,
            "base_url": args.base_url,
            "candidates_path": str(candidates_path),
            "baseline_schema_trials": str(baseline_schema),
            "baseline_accepted": str(baseline_accepted),
            "baseline_rejected": str(baseline_rejected),
            "output_path": str(output_path),
            "accepted_path": str(accepted_path),
            "rejected_path": str(rejected_path),
            "accepted_quality_path": str(accepted_quality_path),
            "rejected_quality_path": str(rejected_quality_path),
            "report_path": str(report_path),
        }
    )
    write_json(summary_path, summary)
    write_report(report_path, rows, summary)
    append_jsonl(log_path, {"event": "finished", **summary})
    print(
        f"deepseek_probe={output_dir} rows={len(rows)} strict_accepted={len(accepted)} "
        f"quality_accepted={len(accepted_quality)} quality_rescued={summary['quality_rescued_baseline_rejections']} "
        f"quality_lost={summary['quality_lost_baseline_acceptances']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
