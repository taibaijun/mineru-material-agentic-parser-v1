from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl
from run_ai_certified_extraction import (
    call_deepseek_json,
    candidate_key,
    combined_context,
    load_env_file,
    problem_brief,
    schema_text,
)
from run_deepseek_extraction_experiment import append_jsonl, local_path, now_iso


AGENT_VERSION = "agentic_material_ai_reader_v3"
ALLOWED_PROPERTIES = {
    "adhesion_strength",
    "discharge_capacity",
    "ionic_conductivity",
    "process_temperature",
    "thermal_transition",
}


def stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_json(text: str) -> tuple[dict[str, Any], str | None]:
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


def compact_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def find_quote(quote: str, combined_text: str, packet_text: str = "") -> dict[str, Any]:
    quote_clean = compact_ws(quote)
    if not quote_clean:
        return {"status": "missing_quote"}
    sources = [
        ("combined", combined_text),
        ("packet", packet_text),
    ]
    for source_name, source_text in sources:
        if not source_text:
            continue
        idx = source_text.find(quote)
        if idx >= 0:
            return {"status": "exact", "source": source_name, "char_start": idx, "quote": quote}
        compact_source = compact_ws(source_text)
        idx_compact = compact_source.find(quote_clean)
        if idx_compact >= 0:
            return {"status": "whitespace_compact", "source": source_name, "char_start_compact": idx_compact, "quote": quote_clean}
    return {"status": "not_found", "quote": quote_clean}


def schema_gate_fact(fact: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prop = str(fact.get("property") or "")
    if prop not in ALLOWED_PROPERTIES:
        errors.append("property_outside_v3_schema")
    elif prop not in (schema.get("properties") or {}):
        errors.append("property_missing_from_schema_file")
    for field in ["material", "property", "value_text", "unit", "evidence_quote"]:
        value = fact.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"missing_{field}")
    try:
        confidence = float(fact.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.0 or confidence > 1.0:
        errors.append("confidence_out_of_range")
    return errors


def read_combined(combined_dir: Path, doc_id: str) -> str:
    path = combined_dir / doc_id / f"{doc_id}_combined.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def doc_title_excerpt(text: str, limit: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
        if len("\n".join(lines)) >= limit:
            break
    return "\n".join(lines)[:limit]


def source_richness_rank(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
    source_type = str(candidate.get("source_type") or "")
    prop = str(candidate.get("property_hint") or "")
    base_priority = {
        "text_event_window": 0,
        "table_markdown_row": 1,
        "table_html_row": 1,
        "table_markdown_cell": 3,
        "table_html_cell": 3,
    }.get(source_type, 4)
    evidence = str(candidate.get("evidence") or "")
    has_row_binding = any(token in evidence for token in ["Row context", "Row label", "Table header", "Target column"])
    if source_type.endswith("_cell") and has_row_binding:
        base_priority -= 1
    if prop in {"discharge_capacity", "ionic_conductivity"} and source_type.endswith("_cell"):
        base_priority += 1
    rich_len = min(len(evidence), 2400)
    return (
        base_priority,
        -rich_len,
        -float(candidate.get("score") or 0.0),
        str(candidate_key(candidate)),
    )


def property_round_order(props: list[str]) -> list[str]:
    priority = {
        "ionic_conductivity": 0,
        "discharge_capacity": 1,
        "process_temperature": 2,
        "thermal_transition": 3,
        "adhesion_strength": 4,
    }
    return sorted(props, key=lambda prop: (priority.get(prop, 99), prop))


def select_doc_candidates(
    candidates: list[dict[str, Any]],
    *,
    doc_id: str,
    per_property: int,
    max_packets: int,
    min_score: float,
    combined_dir: Path,
    context_radius: int,
) -> list[dict[str, Any]]:
    selected = [
        row for row in candidates
        if str(row.get("doc_id") or "") == doc_id and float(row.get("score") or 0.0) >= min_score
    ]
    by_prop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_prop[str(row.get("property_hint") or "unknown")].append(row)
    packets: list[dict[str, Any]] = []
    for prop in sorted(by_prop):
        rows = sorted(by_prop[prop], key=source_richness_rank)[:per_property]
        for row in rows:
            ctx = combined_context(row, combined_dir, radius=context_radius).get("context", "")
            packets.append(
                {
                    "candidate_key": candidate_key(row),
                    "doc_id": str(row.get("doc_id") or ""),
                    "page_range": str(row.get("page_range") or ""),
                    "property_hint": str(row.get("property_hint") or ""),
                    "score": row.get("score"),
                    "source_type": row.get("source_type"),
                    "section_title": row.get("section_title"),
                    "target_column": row.get("target_column"),
                    "target_value": row.get("target_value"),
                    "material_hints": row.get("material_hints"),
                    "values": row.get("values"),
                    "units": row.get("units"),
                    "evidence": compact_text(str(row.get("evidence") or ""), 1800),
                    "source_context": compact_text(ctx, 2600),
                }
            )
    if max_packets > 0:
        balanced_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for packet in sorted(
            packets,
            key=lambda item: (
                str(item.get("property_hint") or ""),
                -float(item.get("score") or 0.0),
                str(item.get("candidate_key") or ""),
            ),
        ):
            balanced_groups[str(packet.get("property_hint") or "unknown")].append(packet)
        balanced: list[dict[str, Any]] = []
        props = property_round_order(list(balanced_groups))
        cursor = 0
        while len(balanced) < max_packets and props:
            progressed = False
            for prop in props:
                rows = balanced_groups[prop]
                if cursor < len(rows):
                    balanced.append(rows[cursor])
                    progressed = True
                    if len(balanced) >= max_packets:
                        break
            if not progressed:
                break
            cursor += 1
        packets = balanced
    return packets


def grouped_packets(packets: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        groups[str(packet.get("property_hint") or "unknown")].append(packet)
    batches: list[list[dict[str, Any]]] = []
    for prop in sorted(groups):
        rows = groups[prop]
        for idx in range(0, len(rows), batch_size):
            batches.append(rows[idx:idx + batch_size])
    return batches


def ai_call(
    *,
    args: argparse.Namespace,
    api_key: str,
    messages: list[dict[str, str]],
    temperature_note: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None, str]:
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
    obj, parse_error = ({}, error) if error else parse_json(text)
    usage = {**usage, "elapsed_sec": round(time.time() - started, 3), "temperature_note": temperature_note}
    return obj, usage, parse_error, text


def plan_messages(doc_id: str, title_excerpt: str, packets: list[dict[str, Any]], schema: dict[str, Any], problem_text: str) -> list[dict[str, str]]:
    counts = Counter(str(packet.get("property_hint") or "unknown") for packet in packets)
    examples = [
        {
            "candidate_key": packet.get("candidate_key"),
            "property_hint": packet.get("property_hint"),
            "page_range": packet.get("page_range"),
            "evidence": packet.get("evidence"),
        }
        for packet in packets[:20]
    ]
    system = (
        "You are a material-science paper reading agent. "
        "You build a reading plan before extracting structured facts. "
        "Use semantic judgment, not keyword matching. Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Document id: {doc_id}

Title / contents / opening excerpt:
{title_excerpt}

Candidate property distribution:
{json.dumps(dict(counts), ensure_ascii=False, indent=2)}

Candidate examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Decide how to read this document for material facts. If it is a review, table-only survey, or low-confidence source, say how extraction should be conservative.

Return JSON:
{{
  "doc_id": "{doc_id}",
  "doc_type": "research|review|patent|unknown",
  "material_domains": ["..."],
  "reading_plan": [
    {{"target": "abstract/results/methods/table/caption", "purpose": "...", "risk": "..."}}
  ],
  "property_focus": {{
    "adhesion_strength": "read|ignore|cautious",
    "discharge_capacity": "read|ignore|cautious",
    "ionic_conductivity": "read|ignore|cautious",
    "process_temperature": "read|ignore|cautious",
    "thermal_transition": "read|ignore|cautious"
  }},
  "global_risks": ["..."]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_messages(
    doc_id: str,
    plan: dict[str, Any],
    packet_batch: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are an agentic material-fact extractor. "
        "Read the evidence packets as source excerpts. "
        "Extract only facts that are explicitly supported. "
        "Do not use regex-style surface matching; reason about material-property-value-condition binding. "
        "Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Document id: {doc_id}

Reading plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Evidence packets:
{json.dumps(packet_batch, ensure_ascii=False, indent=2)}

Extract material facts only when the source context supports all bindings:
- material identity
- property name and subtype
- value and unit
- conditions such as temperature, time, atmosphere, cycle, rate, voltage window, substrate, method
- minimal evidence_quote copied from source text

Reject noisy table cells where row/column/material binding is unclear.
For table-derived facts, evidence_quote must include the material/row label and the target value in the same quote. Do not quote only the numeric cell.

Return JSON:
{{
  "facts": [
    {{
      "candidate_keys": ["..."],
      "doc_id": "{doc_id}",
      "page_range": "...",
      "material": "...",
      "property": "adhesion_strength|discharge_capacity|ionic_conductivity|process_temperature|thermal_transition",
      "property_subtype": null,
      "value": 0,
      "value_max": null,
      "value_text": "...",
      "unit": "...",
      "condition": {{}},
      "evidence_quote": "minimal exact quote from one evidence packet",
      "binding_explanation": "why material/value/unit/condition belong together",
      "confidence": 0.0
    }}
  ],
  "rejected_candidates": [
    {{"candidate_key": "...", "reason": "semantic reason, not keyword reason"}}
  ]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def critic_messages(
    doc_id: str,
    plan: dict[str, Any],
    facts: list[dict[str, Any]],
    quote_results: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are a strict material-data critic. "
        "Your job is to remove weak, hallucinated, or poorly bound facts. "
        "Use semantic reading. Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Document id: {doc_id}

Reading plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Proposed facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Quote/schema alignment results from code:
{json.dumps(quote_results, ensure_ascii=False, indent=2)}

Audit every proposed fact. Accept only if it is explicitly source-supported and useful for the material schema.
For table-derived facts, reject evidence_quote that only contains the target value/cell and omits the row material or column binding.

Return JSON:
{{
  "critic_items": [
    {{
      "fact_index": 0,
      "verdict": "ACCEPT|REVISE|REJECT",
      "reason": "...",
      "required_change": "..."
    }}
  ],
  "global_notes": ["..."]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def revise_messages(
    doc_id: str,
    facts: list[dict[str, Any]],
    critic: dict[str, Any],
    packets: list[dict[str, Any]],
    schema: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You revise material facts once. "
        "Use only the supplied evidence packets. Drop facts that cannot be repaired. "
        "Return exactly one JSON object."
    )
    user = f"""
Schema:
{schema_text(schema)}

Document id: {doc_id}

Original facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Critic:
{json.dumps(critic, ensure_ascii=False, indent=2)}

Evidence packets:
{json.dumps(packets, ensure_ascii=False, indent=2)}

For table-derived facts, evidence_quote must include the material/row label and target value together. Drop facts if you can only quote the isolated numeric cell.

Return JSON:
{{
  "facts": [
    {{
      "candidate_keys": ["..."],
      "doc_id": "{doc_id}",
      "page_range": "...",
      "material": "...",
      "property": "...",
      "property_subtype": null,
      "value": 0,
      "value_max": null,
      "value_text": "...",
      "unit": "...",
      "condition": {{}},
      "evidence_quote": "minimal exact source quote",
      "binding_explanation": "...",
      "confidence": 0.0
    }}
  ],
  "dropped": [
    {{"fact_index": 0, "reason": "..."}}
  ]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_fact(fact: dict[str, Any], doc_id: str, source: str) -> dict[str, Any]:
    out = dict(fact)
    out["doc_id"] = str(out.get("doc_id") or doc_id)
    out["material"] = str(out.get("material") or "").strip()
    out["property"] = str(out.get("property") or "").strip()
    out["page_range"] = str(out.get("page_range") or "").strip()
    out["unit"] = str(out.get("unit") or "").strip()
    out["value_text"] = str(out.get("value_text") or "").strip()
    out["evidence_quote"] = str(out.get("evidence_quote") or "").strip()
    out["evidence"] = out["evidence_quote"]
    out["condition"] = out.get("condition") if isinstance(out.get("condition"), dict) else {}
    try:
        out["confidence"] = round(float(out.get("confidence") or 0.0), 3)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out["agentic_version"] = AGENT_VERSION
    out["agentic_source"] = source
    out["record_id"] = stable_id(
        {
            "doc_id": out.get("doc_id"),
            "page_range": out.get("page_range"),
            "material": out.get("material"),
            "property": out.get("property"),
            "value_text": out.get("value_text"),
            "unit": out.get("unit"),
            "evidence_quote": out.get("evidence_quote"),
        }
    )
    return out


def packet_text_by_key(packets: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for packet in packets:
        key = str(packet.get("candidate_key") or "")
        out[key] = "\n".join(
            [
                str(packet.get("evidence") or ""),
                str(packet.get("source_context") or ""),
            ]
        )
    return out


def align_facts(facts: list[dict[str, Any]], combined_text: str, packets: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    packet_map = packet_text_by_key(packets)
    results: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        packet_text = "\n".join(packet_map.get(str(key), "") for key in fact.get("candidate_keys") or [])
        quote_result = find_quote(str(fact.get("evidence_quote") or ""), combined_text, packet_text)
        schema_errors = schema_gate_fact(fact, schema)
        results.append(
            {
                "fact_index": index,
                "record_id": fact.get("record_id"),
                "quote_result": quote_result,
                "schema_errors": schema_errors,
                "code_gate": "PASS" if quote_result.get("status") in {"exact", "whitespace_compact"} and not schema_errors else "REVIEW",
            }
        )
    return results


def final_filter(
    facts: list[dict[str, Any]],
    critic: dict[str, Any],
    alignment: list[dict[str, Any]],
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    critic_by_index = {
        int(item.get("fact_index")): item
        for item in critic.get("critic_items") or []
        if isinstance(item, dict) and str(item.get("fact_index", "")).lstrip("-").isdigit()
    }
    align_by_index = {int(row["fact_index"]): row for row in alignment}
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, fact in enumerate(facts):
        item = critic_by_index.get(index, {})
        align = align_by_index.get(index, {})
        confidence = float(fact.get("confidence") or 0.0)
        reasons: list[str] = []
        if item.get("verdict") != "ACCEPT":
            reasons.append(f"critic_{item.get('verdict') or 'missing'}")
        if align.get("code_gate") != "PASS":
            reasons.append("quote_or_schema_review")
        if confidence < min_confidence:
            reasons.append("confidence_below_threshold")
        if fact.get("record_id") in seen:
            reasons.append("duplicate_record_id")
        if reasons:
            review.append({"fact": fact, "critic": item, "alignment": align, "reasons": reasons})
            continue
        seen.add(str(fact.get("record_id") or ""))
        accepted.append(fact)
    return accepted, review


def process_doc(
    doc_id: str,
    candidates: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    combined_text = read_combined(args.combined_dir, doc_id)
    packets = select_doc_candidates(
        candidates,
        doc_id=doc_id,
        per_property=args.per_property,
        max_packets=args.max_packets_per_doc,
        min_score=args.min_score,
        combined_dir=args.combined_dir,
        context_radius=args.context_radius,
    )
    state: dict[str, Any] = {
        "agentic_version": AGENT_VERSION,
        "doc_id": doc_id,
        "started": now_iso(),
        "candidate_count": len(packets),
        "candidate_by_property": dict(Counter(str(packet.get("property_hint") or "unknown") for packet in packets)),
        "events": [],
        "usage": {},
    }
    if not combined_text:
        state["status"] = "failed"
        state["error"] = "missing_combined_markdown"
        return {"state": state, "records": [], "review": []}
    if not packets:
        state["status"] = "skipped"
        state["error"] = "no_candidate_packets"
        return {"state": state, "records": [], "review": []}

    plan, usage, error, raw = ai_call(
        args=args,
        api_key=api_key,
        messages=plan_messages(doc_id, doc_title_excerpt(combined_text, args.title_excerpt_limit), packets, schema, problem_text),
        temperature_note="plan_temperature_0.0",
    )
    state["usage"]["plan"] = usage
    state["plan"] = plan
    if error:
        state["status"] = "failed"
        state["error"] = error
        state["raw_plan_output"] = raw[:4000]
        return {"state": state, "records": [], "review": []}

    proposed: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    extract_usages: list[dict[str, Any]] = []
    raw_extract_errors: list[dict[str, Any]] = []
    source_model = str(getattr(args, "model", "deepseek-v4-pro")).replace("-", "_").replace(".", "_")
    for batch_index, batch in enumerate(grouped_packets(packets, args.batch_size), start=1):
        obj, batch_usage, batch_error, batch_raw = ai_call(
            args=args,
            api_key=api_key,
            messages=extract_messages(doc_id, plan, batch, schema, problem_text),
            temperature_note="extract_temperature_0.0",
        )
        batch_usage["batch_index"] = batch_index
        extract_usages.append(batch_usage)
        if batch_error:
            raw_extract_errors.append({"batch_index": batch_index, "error": batch_error, "raw": batch_raw[:4000]})
            continue
        for fact in obj.get("facts") or []:
            if isinstance(fact, dict):
                proposed.append(normalize_fact(fact, doc_id, f"{source_model}_ai_reader_extract"))
        for item in obj.get("rejected_candidates") or []:
            if isinstance(item, dict):
                rejected_candidates.append(item)
    state["usage"]["extract"] = extract_usages
    state["extract_errors"] = raw_extract_errors
    state["proposed_facts"] = proposed
    state["rejected_candidates"] = rejected_candidates

    initial_alignment = align_facts(proposed, combined_text, packets, schema)
    critic, critic_usage, critic_error, critic_raw = ai_call(
        args=args,
        api_key=api_key,
        messages=critic_messages(doc_id, plan, proposed, initial_alignment, schema, problem_text),
        temperature_note="critic_temperature_0.0",
    )
    state["usage"]["critic"] = critic_usage
    state["critic"] = critic
    if critic_error:
        state["status"] = "failed"
        state["error"] = critic_error
        state["raw_critic_output"] = critic_raw[:4000]
        return {"state": state, "records": [], "review": []}

    needs_revise = any(
        item.get("verdict") == "REVISE"
        for item in critic.get("critic_items") or []
        if isinstance(item, dict)
    ) or any(row.get("code_gate") != "PASS" for row in initial_alignment)
    final_facts = proposed
    if needs_revise:
        revised, revise_usage, revise_error, revise_raw = ai_call(
            args=args,
            api_key=api_key,
            messages=revise_messages(doc_id, proposed, critic, packets, schema),
            temperature_note="revise_temperature_0.0",
        )
        state["usage"]["revise"] = revise_usage
        if revise_error:
            state["revise_error"] = revise_error
            state["raw_revise_output"] = revise_raw[:4000]
        else:
            final_facts = [
                normalize_fact(fact, doc_id, f"{source_model}_ai_reader_revised")
                for fact in revised.get("facts") or []
                if isinstance(fact, dict)
            ]
            state["revision"] = revised

    final_alignment = align_facts(final_facts, combined_text, packets, schema)
    final_critic = critic
    if final_facts is not proposed:
        final_critic, final_critic_usage, final_critic_error, final_critic_raw = ai_call(
            args=args,
            api_key=api_key,
            messages=critic_messages(doc_id, plan, final_facts, final_alignment, schema, problem_text),
            temperature_note="final_critic_temperature_0.0",
        )
        state["usage"]["final_critic"] = final_critic_usage
        if final_critic_error:
            state["final_critic_error"] = final_critic_error
            state["raw_final_critic_output"] = final_critic_raw[:4000]
            final_critic = {"critic_items": []}
    state["final_critic"] = final_critic

    accepted, review = final_filter(final_facts, final_critic, final_alignment, args.min_accept_confidence)
    for item in rejected_candidates:
        review.append(
            {
                "type": "ai_rejected_candidate",
                "candidate_key": item.get("candidate_key"),
                "reason": item.get("reason"),
                "stage": "extract_facts",
            }
        )
    state["final_alignment"] = final_alignment
    state["accepted_count"] = len(accepted)
    state["review_count"] = len(review)
    state["accepted_by_property"] = dict(Counter(str(record.get("property") or "unknown") for record in accepted))
    state["status"] = "completed"
    state["finished"] = now_iso()
    return {"state": state, "records": accepted, "review": review}


def summarize(states: list[dict[str, Any]], records: list[dict[str, Any]], reviews: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    usage_totals: Counter[str] = Counter()
    for state in states:
        usage = state.get("usage") if isinstance(state.get("usage"), dict) else {}
        for phase, item in usage.items():
            items = item if isinstance(item, list) else [item]
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                for key in ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]:
                    try:
                        usage_totals[f"{phase}_{key}"] += int(entry.get(key) or 0)
                    except (TypeError, ValueError):
                        pass
    return {
        "agentic_version": AGENT_VERSION,
        "time": now_iso(),
        "model": args.model,
        "output_dir": str(args.output_dir),
        "doc_count": len(states),
        "status_counts": dict(Counter(str(state.get("status") or "unknown") for state in states)),
        "accepted_records": len(records),
        "review_records": len(reviews),
        "accepted_by_property": dict(Counter(str(record.get("property") or "unknown") for record in records)),
        "records_by_doc": dict(Counter(str(record.get("doc_id") or "unknown") for record in records)),
        "min_accept_confidence": args.min_accept_confidence,
        "per_property": args.per_property,
        "batch_size": args.batch_size,
        "semantic_gate": "DeepSeek-v4-pro plan/extract/critic/revise; code only validates JSON/schema fields and quote alignment.",
        "api_usage_totals": dict(usage_totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic AI-first material paper reader. No regex semantic gate.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_ROOT / "candidates_full768_20260523" / "focused_evidence_candidates.recall_v3.jsonl")
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_WORK_ROOT / "combined")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parent / "schemas" / "default_material_schema.json")
    parser.add_argument("--problem", type=Path, default=Path(__file__).resolve().parents[1] / "赛题说明" / "材料赛题.md")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "agentic_material_v3_ai_reader")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--limit-docs", type=int, default=3)
    parser.add_argument("--per-property", type=int, default=10)
    parser.add_argument("--max-packets-per-doc", type=int, default=0, help="Hard cap evidence packets per document; 0 means no cap.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-accept-confidence", type=float, default=0.82)
    parser.add_argument("--context-radius", type=int, default=1800)
    parser.add_argument("--title-excerpt-limit", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Write selected packets and exit without API calls.")
    args = parser.parse_args()

    args.candidates = local_path(args.candidates)
    args.combined_dir = local_path(args.combined_dir)
    args.schema = local_path(args.schema)
    args.problem = local_path(args.problem)
    args.output_dir = local_path(args.output_dir)
    args.env_file = [local_path(path) for path in args.env_file]

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    problem_text = args.problem.read_text(encoding="utf-8", errors="ignore")
    candidates = read_jsonl(args.candidates)
    doc_ids = sorted({str(row.get("doc_id") or "") for row in candidates if row.get("doc_id")}, key=lambda value: int(value) if value.isdigit() else 10**9)
    if args.doc_id:
        wanted = {str(value) for value in args.doc_id}
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    elif args.limit_docs > 0:
        doc_ids = doc_ids[:args.limit_docs]

    out_dir = ensure_dir(args.output_dir)
    ensure_dir(out_dir / "states")
    ensure_dir(out_dir / "records")
    ensure_dir(out_dir / "review")
    ensure_dir(out_dir / "packets")
    log_path = out_dir / "run.log"
    append_jsonl(log_path, {"event": "start", "time": now_iso(), "docs": doc_ids, "model": args.model, "dry_run": args.dry_run})

    if args.dry_run:
        for doc_id in doc_ids:
            packets = select_doc_candidates(
                candidates,
                doc_id=doc_id,
                per_property=args.per_property,
                max_packets=args.max_packets_per_doc,
                min_score=args.min_score,
                combined_dir=args.combined_dir,
                context_radius=args.context_radius,
            )
            write_jsonl(out_dir / "packets" / f"{doc_id}.packets.jsonl", packets)
        write_json(out_dir / "summary.json", {"agentic_version": AGENT_VERSION, "dry_run": True, "docs": doc_ids})
        print(f"dry_run_packets={out_dir / 'packets'} docs={len(doc_ids)}")
        return 0

    for env_path in args.env_file:
        load_env_file(env_path)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --env-file.")

    states: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_doc, doc_id, candidates, schema, problem_text, api_key, args): doc_id
            for doc_id in doc_ids
        }
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "state": {
                        "agentic_version": AGENT_VERSION,
                        "doc_id": doc_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished": now_iso(),
                    },
                    "records": [],
                    "review": [],
                }
            state = result["state"]
            doc_records = result["records"]
            doc_review = result["review"]
            with lock:
                states.append(state)
                records.extend(doc_records)
                reviews.extend(doc_review)
                write_json(out_dir / "states" / f"{doc_id}.json", state)
                write_jsonl(out_dir / "records" / f"{doc_id}.records.jsonl", doc_records)
                write_jsonl(out_dir / "review" / f"{doc_id}.review.jsonl", doc_review)
                append_jsonl(
                    log_path,
                    {
                        "event": "doc_done",
                        "time": now_iso(),
                        "doc_id": doc_id,
                        "status": state.get("status"),
                        "accepted": len(doc_records),
                        "review": len(doc_review),
                    },
                )

    records.sort(key=lambda row: (int(row["doc_id"]) if str(row.get("doc_id", "")).isdigit() else 10**9, str(row.get("property") or ""), str(row.get("record_id") or "")))
    write_jsonl(out_dir / "records" / "submission_candidates.jsonl", records)
    write_jsonl(out_dir / "dataset.jsonl", records)
    write_jsonl(out_dir / "review" / "all_review.jsonl", reviews)
    summary = summarize(states, records, reviews, args)
    write_json(out_dir / "summary.json", summary)
    append_jsonl(log_path, {"event": "finished", **summary})
    print(f"agentic_material_v3={out_dir} docs={len(states)} accepted={len(records)} review={len(reviews)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
