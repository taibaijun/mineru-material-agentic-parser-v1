from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, write_json, write_jsonl
from run_ai_certified_extraction import (
    call_deepseek_json,
    candidate_key,
    combined_context,
    load_env_file,
    problem_brief,
    schema_text,
)
from run_deepseek_extraction_experiment import append_jsonl, local_path, now_iso


V5_VERSION = "agentic_material_corpus_v5_semantic_binder"
ALLOWED_PROPERTIES = {
    "adhesion_strength",
    "discharge_capacity",
    "ionic_conductivity",
    "process_temperature",
    "thermal_transition",
}
ALLOWED_UNIT_TOKENS = {
    "adhesion_strength": ["mpa", "kpa", "n/mm", "n/m"],
    "discharge_capacity": ["mah/g", "mah g^-1", "mah g^{-1}", "mahg^-1", "mahg^{-1}"],
    "ionic_conductivity": ["s/cm", "s cm^-1", "s cm^{-1}", "ms/cm", "ms cm^-1", "ms cm^{-1}"],
    "process_temperature": ["°c", "℃", "k"],
    "thermal_transition": ["°c", "℃", "k"],
}


def compact_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def stable_id(payload: dict[str, Any]) -> str:
    import hashlib

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def read_jsonl_lenient(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, [{"path": str(path), "line": 0, "error": "missing_file"}]
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append({"path": str(path), "line": line_no, "error": str(exc), "raw": raw[:1000]})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                errors.append({"path": str(path), "line": line_no, "error": "json_not_object", "raw": raw[:1000]})
    return rows, errors


def parse_json_object(text: str) -> tuple[dict[str, Any], str | None]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"json_parse_failed:{exc}"
    if not isinstance(obj, dict):
        return {}, "json_not_object"
    return obj, None


def read_combined(combined_dir: Path, doc_id: str) -> tuple[str, Path]:
    path = combined_dir / doc_id / f"{doc_id}_combined.md"
    if not path.exists():
        return "", path
    return path.read_text(encoding="utf-8", errors="ignore"), path


def line_context_for_index(text: str, char_index: int, line_radius: int) -> str:
    before = text[:char_index]
    line_no = before.count("\n")
    lines = text.splitlines()
    start = max(0, line_no - line_radius)
    end = min(len(lines), line_no + line_radius + 1)
    return "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(start, end))


def find_context_by_quote(text: str, quote: str, line_radius: int) -> dict[str, Any] | None:
    quote = str(quote or "").strip()
    if not quote:
        return None
    idx = text.find(quote)
    if idx >= 0:
        return {
            "source": "combined_exact_quote",
            "quote_alignment": "exact",
            "char_start": idx,
            "context": line_context_for_index(text, idx, line_radius),
        }
    compact_quote = compact_ws(quote)
    if len(compact_quote) < 12:
        return None
    compact_text_body = compact_ws(text)
    compact_idx = compact_text_body.find(compact_quote)
    if compact_idx >= 0:
        return {
            "source": "combined_compact_quote",
            "quote_alignment": "whitespace_compact_no_char_map",
            "char_start_compact": compact_idx,
            "context": compact_text_body[max(0, compact_idx - 1800): compact_idx + len(compact_quote) + 1800],
        }
    return None


def score_line_for_record(line: str, record: dict[str, Any]) -> int:
    line_l = compact_ws(line).lower()
    if not line_l:
        return 0
    score = 0
    material = compact_ws(record.get("material")).lower()
    value_text = compact_ws(record.get("value_text")).lower()
    value = compact_ws(record.get("value")).lower()
    value_max = compact_ws(record.get("value_max")).lower()
    unit = compact_ws(record.get("unit")).lower()
    prop = compact_ws(record.get("property")).lower()
    for token, weight in [
        (material, 5),
        (value_text, 5),
        (value, 4),
        (value_max, 3),
        (unit, 2),
        (prop.replace("_", " "), 2),
    ]:
        if token and token != "none" and token in line_l:
            score += weight
    return score


def fallback_context_from_record(text: str, record: dict[str, Any], line_radius: int) -> dict[str, Any] | None:
    best_index = -1
    best_score = 0
    lines = text.splitlines()
    for index, line in enumerate(lines):
        score = score_line_for_record(line, record)
        if score > best_score:
            best_score = score
            best_index = index
    if best_index < 0 or best_score < 4:
        return None
    start = max(0, best_index - line_radius)
    end = min(len(lines), best_index + line_radius + 1)
    return {
        "source": "combined_line_search",
        "quote_alignment": "line_score_location_only",
        "line_score": best_score,
        "context": "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(start, end)),
    }


def normalize_candidate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = candidate_key(row)
        out[key] = row
    return out


def candidate_contexts_for_record(
    record: dict[str, Any],
    candidate_by_key: dict[str, dict[str, Any]],
    combined_dir: Path,
    context_radius: int,
    max_candidate_contexts: int,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    keys = [str(key) for key in record.get("candidate_keys") or []]
    for key in keys[:max_candidate_contexts]:
        row = candidate_by_key.get(key)
        if not row:
            continue
        ctx = combined_context(row, combined_dir, radius=context_radius)
        contexts.append(
            {
                "candidate_key": key,
                "property_hint": row.get("property_hint"),
                "source_type": row.get("source_type"),
                "page_range": row.get("page_range"),
                "target_column": row.get("target_column"),
                "target_value": row.get("target_value"),
                "candidate_evidence": compact_text(str(row.get("evidence") or ""), 1600),
                "combined_context_status": ctx.get("status"),
                "combined_context": compact_text(str(ctx.get("context") or ""), context_radius * 2),
            }
        )
    return contexts


def source_packet_for_record(
    record: dict[str, Any],
    *,
    combined_dir: Path,
    candidate_by_key: dict[str, dict[str, Any]],
    line_radius: int,
    context_radius: int,
    max_candidate_contexts: int,
) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or record.get("doc_no") or "")
    text, path = read_combined(combined_dir, doc_id)
    context = None
    current_quote_check = {}
    if text:
        current_quote_check = quote_in_combined(record, combined_dir)
        context = find_context_by_quote(text, str(record.get("evidence_quote") or record.get("evidence") or ""), line_radius)
        if context is None:
            context = fallback_context_from_record(text, record, line_radius)
    return {
        "record_id": record.get("record_id"),
        "doc_id": doc_id,
        "combined_path": str(path),
        "combined_available": bool(text),
        "current_quote_check": current_quote_check or {"status": "missing_combined"},
        "direct_context": context or {"source": "not_found", "context": ""},
        "candidate_contexts": candidate_contexts_for_record(
            record,
            candidate_by_key,
            combined_dir,
            context_radius,
            max_candidate_contexts,
        ),
    }


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "doc_id": record.get("doc_id"),
        "page_range": record.get("page_range"),
        "material": record.get("material"),
        "property": record.get("property"),
        "property_subtype": record.get("property_subtype"),
        "value": record.get("value"),
        "value_max": record.get("value_max"),
        "value_text": record.get("value_text"),
        "unit": record.get("unit"),
        "condition": record.get("condition"),
        "evidence_quote": record.get("evidence_quote") or record.get("evidence"),
        "binding_explanation": compact_text(str(record.get("binding_explanation") or ""), 700),
        "confidence": record.get("confidence"),
        "source_doc_type": record.get("source_doc_type"),
    }


def semantic_binder_messages(
    records: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are a strict material-science semantic binder for a competition dataset. "
        "You must read the provided source contexts, not pattern-match. "
        "Return exactly one valid JSON object and no markdown."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Proposed extraction records:
{json.dumps([compact_record(row) for row in records], ensure_ascii=False, indent=2)}

Source packets. The direct_context and candidate_contexts are excerpts from combined.md around the proposed evidence/candidate:
{json.dumps(packets, ensure_ascii=False, indent=2)}

For every record, decide whether it is scientifically correct and submission-safe.

You must explicitly check these bindings:
1. Source property label: what property name/table header/paragraph phrase in the source owns the value?
2. Target property: does that source property actually match the requested schema property?
   - ionic_conductivity is NOT electronic conductivity, electrical conductivity, resistance, impedance, voltage, capacity, or efficiency.
   - discharge_capacity must be discharge/specific capacity for a material/electrode/cell, not voltage/retention/rate.
   - process_temperature must be an intentional fabrication/preparation condition, not measurement, cycling, operation, DSC/TGA program, decomposition, or plot axis.
   - thermal_transition must be Tg/Tm/Tc/melting/crystallization/phase transition, not a process temperature or decomposition-only value.
   - adhesion_strength must describe adhesive/bond/interface/peel/joint strength. If the measured entity is an interface, the material should name that interface or coating, not only one passive substrate. For glue-on-substrate tests, bind adhesive/coating/substrate roles; do not report only the passive substrate as the material.
3. Material holder: what physical entity actually owns the measured value? Include coating/interface/adhesive/substrate roles when relevant.
4. Evidence quality: can the evidence quote be copied from combined.md and include enough row/column or sentence context to bind material-value-property?
5. Source type: primary research, patent, review, book chapter, or unknown. Review/book-chapter facts may be correct but should be marked REVIEW unless the surrounding context gives direct table-level data clearly enough for high-confidence extraction.

Important evidence rule:
- Inspect current_quote_check for each record. If the current evidence_quote is not exact or whitespace_compact in combined.md, do not return ACCEPT as-is.
- If the fact is scientifically correct, return REVISE and replace evidence_quote with an exact substring from direct_context/candidate_context combined.md context.
- For table rows in combined.md, it is OK to quote an exact HTML <tr>...</tr> row or a compact exact table fragment, as long as material, source property column, and value are all bound.

Verdicts:
- ACCEPT: record is correct as written and source-safe.
- REVISE: the fact is real but the record must be corrected. Provide corrected_record with all required fields.
- REVIEW: plausible but too broad, review-derived, book-chapter-derived, ambiguous, or not high-confidence enough for final.
- REJECT: wrong property, wrong material/value binding, unsupported, malformed, or cannot be repaired from source.

Protocol constraints:
- If target_property_match is false, the item cannot be final accepted. Return REJECT unless corrected_record changes it to a property inside the schema and the source truly supports that schema property.
- For adhesion_strength, roles cannot be only a passive substrate/support. The record must bind the adhesive/coating/interface/joint/composite that actually owns the strength value.
- For any final-safe ACCEPT/REVISE item, material_binding.problem must be null and evidence_assessment.problem must be null. If there is still a caveat, ambiguity, or unresolved issue, return REVIEW or REJECT instead of accepting with a caveat.

For REVISE, corrected_record must keep the same record_id only if the meaning is unchanged; otherwise omit record_id and code will assign one.
For corrected evidence, quote exact source text from combined.md only, not synthetic "Row:" text.

Return JSON:
{{
  "items": [
    {{
      "record_id": "...",
      "verdict": "ACCEPT|REVISE|REVIEW|REJECT",
      "reason": "short scientific reason",
      "source_doc_type": "primary_research|patent|review|book_chapter|unknown",
      "source_property_label": "exact source label/header/phrase or null",
      "target_property_match": true,
      "material_binding": {{
        "measured_entity": "...",
        "roles": {{"adhesive": "...", "coating": "...", "substrate": "...", "electrolyte": "...", "electrode": "..."}},
        "problem": null
      }},
      "evidence_assessment": {{
        "quote_in_combined": true,
        "row_column_binding_clear": true,
        "problem": null
      }},
      "corrected_record": null
    }}
  ],
  "global_notes": ["..."]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_ai_json(args: argparse.Namespace, api_key: str, messages: list[dict[str, str]], note: str) -> tuple[dict[str, Any], dict[str, Any], str | None, str]:
    started = time.time()
    text, usage, error = call_deepseek_json(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort if args.thinking != "disabled" else "",
        retries=args.retries,
    )
    obj, parse_error = ({}, error) if error else parse_json_object(text)
    usage = {**usage, "elapsed_sec": round(time.time() - started, 3), "temperature_note": note}
    return obj, usage, parse_error, text


def normalize_record(record: dict[str, Any], source_version: str) -> dict[str, Any]:
    out = dict(record)
    doc_id = str(out.get("doc_id") or out.get("doc_no") or "")
    out["doc_id"] = doc_id
    out.pop("doc_no", None)
    out["material"] = compact_ws(out.get("material"))
    out["property"] = compact_ws(out.get("property"))
    out["unit"] = compact_ws(out.get("unit"))
    out["value_text"] = compact_ws(out.get("value_text") or out.get("value"))
    out["value_text"] = clean_value_text(out["value_text"], out["unit"])
    out["evidence_quote"] = str(out.get("evidence_quote") or out.get("evidence") or "").strip()
    out["evidence"] = out["evidence_quote"]
    if not isinstance(out.get("condition"), dict):
        out["condition"] = {}
    try:
        out["confidence"] = round(float(out.get("confidence") or 0.0), 3)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    if not out.get("record_id"):
        out["record_id"] = stable_id(
            {
                "doc_id": out.get("doc_id"),
                "material": out.get("material"),
                "property": out.get("property"),
                "value": out.get("value"),
                "value_max": out.get("value_max"),
                "unit": out.get("unit"),
                "evidence_quote": out.get("evidence_quote"),
            }
        )
    out["agentic_version"] = V5_VERSION
    out["agentic_source"] = source_version
    return out


def clean_value_text(value_text: Any, unit: Any) -> str:
    text = compact_ws(value_text)
    unit_text = compact_ws(unit)
    if not text or not unit_text:
        return text
    variants = {
        unit_text,
        unit_text.replace("°C", "℃"),
        unit_text.replace("℃", "°C"),
        unit_text.replace("μ", "u"),
        unit_text.replace("u", "μ"),
    }
    for variant in sorted((v for v in variants if v), key=len, reverse=True):
        text = re.sub(rf"\s*{re.escape(variant)}\s*", " ", text, flags=re.IGNORECASE)
    return compact_ws(text)


def schema_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["doc_id", "material", "property", "value", "unit", "evidence_quote", "record_id"]
    for field in required:
        value = record.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"missing_{field}")
    if str(record.get("property") or "") not in ALLOWED_PROPERTIES:
        errors.append("property_outside_schema")
    if not unit_matches_property(str(record.get("property") or ""), record.get("unit")):
        errors.append("unit_outside_property_schema")
    return errors


def unit_matches_property(prop: str, unit: Any) -> bool:
    allowed = ALLOWED_UNIT_TOKENS.get(str(prop or ""))
    if not allowed:
        return True
    text = compact_ws(unit).lower().replace("·", " ").replace("−", "-")
    text = text.replace(" ", "")
    return any(token.replace(" ", "") in text for token in allowed)


def quote_in_combined(record: dict[str, Any], combined_dir: Path) -> dict[str, Any]:
    doc_id = str(record.get("doc_id") or "")
    text, path = read_combined(combined_dir, doc_id)
    quote = str(record.get("evidence_quote") or record.get("evidence") or "").strip()
    if not text:
        return {"status": "missing_combined", "path": str(path)}
    if not quote:
        return {"status": "missing_quote", "path": str(path)}
    idx = text.find(quote)
    if idx >= 0:
        return {"status": "exact", "path": str(path), "char_start": idx}
    table_row = find_table_row_match(text, quote)
    if table_row is not None:
        return {
            "status": "table_row_exact",
            "path": str(path),
            "char_start": table_row["char_start"],
            "matched_quote": table_row["quote"],
        }
    compact_quote = compact_ws(quote)
    if len(compact_quote) >= 12 and compact_ws(text).find(compact_quote) >= 0:
        return {"status": "whitespace_compact", "path": str(path)}
    return {"status": "not_found", "path": str(path), "quote": compact_quote[:400]}


def find_table_row_match(text: str, quote: str) -> dict[str, Any] | None:
    table_quote = table_signature(quote)
    if len(table_quote) < 30 or not any(ch.isdigit() for ch in table_quote):
        return None
    for match in re.finditer(r"<tr\b[^>]*>.*?</tr>", text, flags=re.IGNORECASE | re.DOTALL):
        row = match.group(0).strip()
        row_signature = table_signature(row)
        if table_signatures_match(table_quote, row_signature):
            return {"char_start": match.start(), "quote": row}
    running_index = 0
    for line in text.splitlines(keepends=True):
        clean_line = line.strip()
        if "|" in clean_line:
            row_signature = table_signature(clean_line)
            if table_signatures_match(table_quote, row_signature):
                return {"char_start": running_index, "quote": clean_line}
        running_index += len(line)
    return None


def table_signature(value: Any) -> str:
    text = html.unescape(str(value or "")).lower()
    text = re.sub(r"\brow\s*:\s*", " ", text)
    text = re.sub(r"</t[dh]\s*>", " | ", text)
    text = re.sub(r"<t[dh][^>]*>", " | ", text)
    text = re.sub(r"</?tr[^>]*>", " | ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("$", " ")
    text = re.sub(r"[|,;:()\[\]{}]", " ", text)
    text = re.sub(r"[_^=+\\/\-]", " ", text)
    return compact_ws(text)


def table_signatures_match(query_signature: str, row_signature: str) -> bool:
    if query_signature in row_signature:
        return True
    if row_signature in query_signature and len(row_signature) >= max(30, int(len(query_signature) * 0.7)):
        return True
    return False


def adhesion_binding_incomplete(record: dict[str, Any]) -> bool:
    if str(record.get("property") or "") != "adhesion_strength":
        return False
    binding = record.get("material_binding") if isinstance(record.get("material_binding"), dict) else {}
    if compact_ws(binding.get("problem")):
        return True
    roles = binding.get("roles") if isinstance(binding.get("roles"), dict) else {}
    unknown_values = {"unknown", "unspecified", "not specified", "none", "null", "n/a", "na"}
    role_values = {
        compact_ws(key).lower(): compact_ws(value)
        for key, value in roles.items()
        if compact_ws(value).lower() not in unknown_values
    }
    non_passive_roles = {
        key: value
        for key, value in role_values.items()
        if key not in {"substrate", "support", "reference", "reference_material", "passive_substrate"}
    }
    if non_passive_roles:
        return False
    condition = record.get("condition") if isinstance(record.get("condition"), dict) else {}
    if condition and not role_values:
        return False
    material = compact_ws(record.get("material")).lower()
    measured = compact_ws(binding.get("measured_entity")).lower()
    holder = f"{material} {measured}"
    explicit_holder_words = ["interface", "joint", "composite", "adhesive", "coating", "bond", "peel", "/", "-to-"]
    if any(word in holder for word in explicit_holder_words):
        return False
    return True


def validate_for_final(record: dict[str, Any], combined_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    errors = schema_errors(record)
    quote = quote_in_combined(record, combined_dir)
    if quote.get("status") not in {"exact", "whitespace_compact", "table_row_exact"}:
        errors.append(f"quote_{quote.get('status')}")
    if record.get("target_property_match") is False:
        errors.append("target_property_mismatch")
    errors.extend(property_label_guard_errors(record))
    binding = record.get("material_binding") if isinstance(record.get("material_binding"), dict) else {}
    if compact_ws(binding.get("problem")):
        errors.append("material_binding_problem")
    assessment = record.get("evidence_assessment") if isinstance(record.get("evidence_assessment"), dict) else {}
    if assessment.get("row_column_binding_clear") is False:
        errors.append("evidence_binding_unclear")
    if adhesion_binding_incomplete(record):
        errors.append("adhesion_binding_incomplete")
    return not errors, errors, quote


def property_label_guard_errors(record: dict[str, Any]) -> list[str]:
    prop = str(record.get("property") or "")
    label = compact_ws(record.get("source_property_label")).lower()
    reason = compact_ws(record.get("v5_semantic_reason")).lower()
    evidence = compact_ws(record.get("evidence_quote") or record.get("evidence")).lower()
    condition = record.get("condition") if isinstance(record.get("condition"), dict) else {}
    condition_text = compact_ws(json.dumps(condition, ensure_ascii=False)).lower()
    haystack = " ".join([label, reason, evidence, condition_text])
    errors: list[str] = []
    if prop == "process_temperature":
        process_words = ["curing", "cure", "sinter", "calcination", "calcined", "anneal", "heating step", "bake", "preparation", "fabrication", "polymerization", "煅烧", "烧结", "固化", "制备"]
        non_process_words = ["operating temperature", "operation temperature", "working temperature", "measurement", "test condition", "dsc", "tga", "cycling", "boiling point"]
        if any(word in haystack for word in non_process_words) and not any(word in haystack for word in process_words):
            errors.append("process_temperature_not_fabrication")
    if prop == "adhesion_strength":
        adhesion_words = ["adhesion", "adhesive", "bond", "lap shear", "shear strength", "peel", "interface", "interfacial", "joint", "粘", "黏", "剥离"]
        label_is_tensile_only = "tensile strength" in label and not any(word in label for word in adhesion_words)
        condition_is_tensile_only = "tensile strength" in condition_text and not any(word in label for word in adhesion_words)
        tensile_only = label_is_tensile_only or condition_is_tensile_only
        if tensile_only:
            errors.append("adhesion_property_label_mismatch")
    return errors


def apply_judgment(
    record: dict[str, Any],
    item: dict[str, Any] | None,
    *,
    combined_dir: Path,
    min_accept_confidence: float,
    exclude_review_sources: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    base = normalize_record(record, "v5_input_record")
    if not item:
        return None, {"type": "missing_ai_judgment", "record": base, "reason": "AI did not return an item for this record."}

    verdict = str(item.get("verdict") or "").upper()
    source_doc_type = str(item.get("source_doc_type") or "unknown")
    review_source = source_doc_type in {"review", "book_chapter"} and exclude_review_sources

    if verdict == "ACCEPT":
        candidate = base
        candidate["v5_semantic_verdict"] = "ACCEPT"
        candidate["v5_semantic_reason"] = item.get("reason")
    elif verdict == "REVISE":
        corrected = item.get("corrected_record")
        if not isinstance(corrected, dict):
            return None, {"type": "v5_revise_missing_record", "record": base, "judgment": item}
        merged = {**base, **corrected}
        if not corrected.get("record_id"):
            merged.pop("record_id", None)
        candidate = normalize_record(merged, "v5_semantic_revised")
        candidate["v5_semantic_verdict"] = "REVISE"
        candidate["v5_semantic_reason"] = item.get("reason")
    elif verdict in {"REVIEW", "REJECT"}:
        return None, {"type": f"v5_{verdict.lower()}", "record": base, "judgment": item}
    else:
        return None, {"type": "v5_bad_verdict", "record": base, "judgment": item}

    candidate["source_doc_type"] = source_doc_type
    candidate["source_property_label"] = item.get("source_property_label")
    candidate["target_property_match"] = bool(item.get("target_property_match"))
    candidate["material_binding"] = item.get("material_binding")
    candidate["evidence_assessment"] = item.get("evidence_assessment")

    try:
        confidence = float(candidate.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_accept_confidence:
        return None, {"type": "v5_low_confidence", "record": candidate, "judgment": item}
    if review_source:
        return None, {"type": "v5_review_source", "record": candidate, "judgment": item}
    ok, errors, quote = validate_for_final(candidate, combined_dir)
    if not ok:
        return None, {"type": "v5_local_gate", "record": candidate, "judgment": item, "errors": errors, "quote": quote}
    if quote.get("matched_quote"):
        candidate["evidence_quote"] = str(quote["matched_quote"])
        candidate["evidence"] = candidate["evidence_quote"]
    return candidate, {"type": "v5_accept", "record_id": candidate.get("record_id"), "judgment": item, "quote": quote}


def doc_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    return (int(text), text) if text.isdigit() else (10**9, text)


def group_records(records: list[dict[str, Any]], records_per_call: int) -> list[list[dict[str, Any]]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_doc[str(record.get("doc_id") or record.get("doc_no") or "")].append(record)
    groups: list[list[dict[str, Any]]] = []
    for doc_id in sorted(by_doc, key=doc_sort_key):
        rows = sorted(by_doc[doc_id], key=lambda row: (str(row.get("property") or ""), str(row.get("record_id") or "")))
        for index in range(0, len(rows), records_per_call):
            groups.append(rows[index:index + records_per_call])
    return groups


def duplicate_key(record: dict[str, Any]) -> str:
    payload = {
        "doc_id": str(record.get("doc_id") or ""),
        "material": compact_ws(record.get("material")).lower(),
        "property": compact_ws(record.get("property")).lower(),
        "property_subtype": compact_ws(record.get("property_subtype")).lower(),
        "value": compact_ws(record.get("value")),
        "value_max": compact_ws(record.get("value_max")),
        "unit": compact_ws(record.get("unit")).lower(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def judge_group(
    group_index: int,
    records: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    problem_text: str,
    candidate_by_key: dict[str, dict[str, Any]],
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    packets = [
        source_packet_for_record(
            record,
            combined_dir=args.combined_dir,
            candidate_by_key=candidate_by_key,
            line_radius=args.line_radius,
            context_radius=args.context_radius,
            max_candidate_contexts=args.max_candidate_contexts,
        )
        for record in records
    ]
    obj, usage, error, raw = call_ai_json(
        args,
        api_key,
        semantic_binder_messages(records, packets, schema, problem_text),
        note=f"v5_semantic_binder_group_{group_index}",
    )
    if error:
        obj = {"items": [], "global_notes": [error], "raw": raw[:3000]}
    return {
        "group_index": group_index,
        "records": records,
        "packets": packets,
        "judgment": obj,
        "usage": usage,
        "error": error,
        "elapsed_sec": round(time.time() - started, 3),
    }


def summarize(
    *,
    input_records: list[dict[str, Any]],
    input_errors: list[dict[str, Any]],
    group_results: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    review: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    verdict_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    for result in group_results:
        for item in (result.get("judgment") or {}).get("items") or []:
            if isinstance(item, dict):
                verdict_counts[str(item.get("verdict") or "missing")] += 1
                source_counts[str(item.get("source_doc_type") or "unknown")] += 1
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]:
            try:
                usage_totals[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return {
        "version": V5_VERSION,
        "time": now_iso(),
        "model": args.model,
        "input_records": len(input_records),
        "input_json_errors": len(input_errors),
        "groups": len(group_results),
        "final_records": len(final_records),
        "review_records": len(review),
        "final_by_property": dict(Counter(str(row.get("property") or "unknown") for row in final_records)),
        "review_by_type": dict(Counter(str(row.get("type") or "unknown") for row in review if isinstance(row, dict))),
        "ai_verdict_counts": dict(verdict_counts),
        "ai_source_type_counts": dict(source_counts),
        "exclude_review_sources": args.exclude_review_sources and not args.keep_review_sources,
        "min_accept_confidence": args.min_accept_confidence,
        "api_usage_totals": dict(usage_totals),
        "semantic_gate": "DeepSeek semantic binder must verify source property label, target property match, measured entity/material roles, and exact combined.md evidence before final acceptance.",
    }


def write_report(path: Path, summary: dict[str, Any], final_records: list[dict[str, Any]], review: list[dict[str, Any]]) -> None:
    lines = ["# V5 Semantic Binder Report", "", "## Conclusion", ""]
    lines.append(f"- final records: {summary.get('final_records')}")
    lines.append(f"- review/rejected records: {summary.get('review_records')}")
    lines.append(f"- input JSON errors: {summary.get('input_json_errors')}")
    lines.append("")
    lines.append("## Final By Property")
    lines.append("")
    for prop, count in (summary.get("final_by_property") or {}).items():
        lines.append(f"- {prop}: {count}")
    lines.append("")
    lines.append("## Review By Type")
    lines.append("")
    for kind, count in (summary.get("review_by_type") or {}).items():
        lines.append(f"- {kind}: {count}")
    lines.append("")
    lines.append("## Representative Rejections")
    lines.append("")
    for row in review[:25]:
        record = row.get("record") if isinstance(row, dict) else None
        judgment = row.get("judgment") if isinstance(row, dict) else None
        if not isinstance(record, dict):
            continue
        reason = ""
        if isinstance(judgment, dict):
            reason = str(judgment.get("reason") or "")
        lines.append(
            f"- {row.get('type')} doc {record.get('doc_id')} {record.get('property')} | "
            f"{record.get('material')} | {record.get('value')} {record.get('unit')} | {compact_text(reason, 200)}"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def validate_jsonl_files(out_dir: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for rel in ["dataset.jsonl", "records/submission_candidates.jsonl", "review/all_review.jsonl", "semantic/audit_results.jsonl"]:
        path = out_dir / rel
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append({"path": rel, "line": line_no, "error": str(exc)})
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="V5 strict semantic material binder and cleaner.")
    parser.add_argument("--records", type=Path, default=DEFAULT_WORK_ROOT / "agentic_material_corpus_v4_flash20_finalaudit" / "records" / "submission_candidates.jsonl")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_ROOT / "candidates_full768_20260523" / "focused_evidence_candidates.recall_v3.jsonl")
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_WORK_ROOT / "combined")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parent / "schemas" / "default_material_schema.json")
    parser.add_argument("--problem", type=Path, default=Path(__file__).resolve().parents[1] / "璧涢璇存槑" / "鏉愭枡璧涢.md")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "agentic_material_corpus_v5_semantic_binder")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--records-per-call", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--line-radius", type=int, default=8)
    parser.add_argument("--context-radius", type=int, default=2200)
    parser.add_argument("--max-candidate-contexts", type=int, default=2)
    parser.add_argument("--min-accept-confidence", type=float, default=0.80)
    parser.add_argument("--exclude-review-sources", action="store_true", help="Route review/book-chapter records to review even when AI says ACCEPT.")
    parser.add_argument("--keep-review-sources", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    args.records = local_path(args.records)
    args.candidates = local_path(args.candidates)
    args.combined_dir = local_path(args.combined_dir)
    args.schema = local_path(args.schema)
    args.problem = local_path(args.problem)
    args.output_dir = local_path(args.output_dir)
    args.env_file = [local_path(path) for path in args.env_file]

    for env_path in args.env_file:
        load_env_file(env_path)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --env-file.")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    problem_text = args.problem.read_text(encoding="utf-8", errors="ignore")
    records, input_errors = read_jsonl_lenient(args.records)
    candidates, candidate_errors = read_jsonl_lenient(args.candidates)
    input_errors.extend(candidate_errors)
    if args.doc_id:
        wanted = {str(doc_id) for doc_id in args.doc_id}
        records = [row for row in records if str(row.get("doc_id") or row.get("doc_no") or "") in wanted]
    if args.limit_records > 0:
        records = records[: args.limit_records]

    candidate_by_key = normalize_candidate_rows(candidates)
    groups = group_records(records, max(1, args.records_per_call))
    out_dir = ensure_dir(args.output_dir)
    for name in ["records", "review", "semantic", "packets"]:
        ensure_dir(out_dir / name)
    log_path = out_dir / "run.log"
    append_jsonl(log_path, {"event": "start", "time": now_iso(), "records": len(records), "groups": len(groups), "model": args.model})

    group_results: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                judge_group,
                index,
                group,
                schema=schema,
                problem_text=problem_text,
                candidate_by_key=candidate_by_key,
                api_key=api_key,
                args=args,
            ): index
            for index, group in enumerate(groups, start=1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "group_index": index,
                    "records": groups[index - 1],
                    "packets": [],
                    "judgment": {"items": [], "global_notes": [f"{type(exc).__name__}: {exc}"]},
                    "usage": {},
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": 0,
                }
            with lock:
                group_results.append(result)
                write_json(out_dir / "semantic" / f"group_{index:04d}.json", result)
                write_jsonl(out_dir / "packets" / f"group_{index:04d}.packets.jsonl", result.get("packets") or [])
                append_jsonl(log_path, {"event": "group_done", "time": now_iso(), "group_index": index, "records": len(result.get("records") or []), "error": result.get("error")})

    group_results.sort(key=lambda row: int(row.get("group_index") or 0))
    final_records: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_final: set[str] = set()
    exclude_review_sources = bool(args.exclude_review_sources) and not bool(args.keep_review_sources)
    for result in group_results:
        items = {
            str(item.get("record_id") or ""): item
            for item in (result.get("judgment") or {}).get("items") or []
            if isinstance(item, dict)
        }
        for record in result.get("records") or []:
            rid = str(record.get("record_id") or "")
            accepted, audit = apply_judgment(
                record,
                items.get(rid),
                combined_dir=args.combined_dir,
                min_accept_confidence=args.min_accept_confidence,
                exclude_review_sources=exclude_review_sources,
            )
            audit_rows.append(audit)
            if accepted:
                key = duplicate_key(accepted)
                if key in seen_final:
                    review.append({"type": "v5_duplicate_final", "record": accepted, "judgment": audit.get("judgment")})
                else:
                    seen_final.add(key)
                    final_records.append(accepted)
            elif audit.get("type") != "v5_accept":
                review.append(audit)

    final_records.sort(key=lambda row: (doc_sort_key(row.get("doc_id")), str(row.get("property") or ""), str(row.get("record_id") or "")))
    review.sort(key=lambda row: (str(row.get("type") or ""), str(((row.get("record") or {}) if isinstance(row, dict) else {}).get("record_id") or "")))

    write_jsonl(out_dir / "records" / "submission_candidates.jsonl", final_records)
    write_jsonl(out_dir / "dataset.jsonl", final_records)
    write_jsonl(out_dir / "review" / "all_review.jsonl", review)
    write_jsonl(out_dir / "semantic" / "audit_results.jsonl", audit_rows)
    if input_errors:
        write_jsonl(out_dir / "review" / "input_json_errors.jsonl", input_errors)

    summary = summarize(
        input_records=records,
        input_errors=input_errors,
        group_results=group_results,
        final_records=final_records,
        review=review,
        args=args,
    )
    jsonl_errors = validate_jsonl_files(out_dir)
    summary["output_jsonl_errors"] = jsonl_errors
    summary["local_check_pass"] = not jsonl_errors and not input_errors
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "semantic_binder_report.md", summary, final_records, review)
    append_jsonl(log_path, {"event": "finished", **summary})
    print(f"agentic_material_corpus_v5={out_dir} input={len(records)} final={len(final_records)} review={len(review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
