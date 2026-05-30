from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl
from run_agentic_material_reader_v3 import (
    AGENT_VERSION as DOC_AGENT_VERSION,
    ai_call,
    doc_title_excerpt,
    process_doc,
    read_combined,
    stable_id,
)
from run_ai_certified_extraction import load_env_file, problem_brief, schema_text
from run_deepseek_extraction_experiment import append_jsonl, local_path, now_iso


CORPUS_VERSION = "agentic_material_corpus_v4"


def sort_doc_ids(doc_ids: set[str]) -> list[str]:
    return sorted(doc_ids, key=lambda value: int(value) if value.isdigit() else 10**9)


def compact_record(record: dict[str, Any], limit: int = 520) -> dict[str, Any]:
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
        "confidence": record.get("confidence"),
        "evidence_quote": compact_text(str(record.get("evidence_quote") or record.get("evidence") or ""), limit),
        "binding_explanation": compact_text(str(record.get("binding_explanation") or ""), limit),
    }


def memory_messages(
    doc_id: str,
    combined_text: str,
    state: dict[str, Any],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    title = doc_title_excerpt(combined_text, 5000)
    payload = {
        "doc_id": doc_id,
        "doc_type": (state.get("plan") or {}).get("doc_type"),
        "material_domains": (state.get("plan") or {}).get("material_domains"),
        "global_risks": (state.get("plan") or {}).get("global_risks"),
        "accepted_records": [compact_record(record) for record in records],
        "rejected_candidate_examples": (state.get("rejected_candidates") or [])[:12],
    }
    system = (
        "You are a material-science document memory builder. "
        "Create a compact memory for cross-document material linking. "
        "Use semantic material identity, formulas, aliases, dopants, sample labels, and evidence. "
        "Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Document title / opening excerpt:
{title}

Document extraction state:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Build a document-level material memory. Include only source-grounded materials; distinguish broad families from specific compositions.

Return JSON:
{{
  "doc_id": "{doc_id}",
  "doc_type": "research|review|patent|unknown",
  "materials": [
    {{
      "local_material_id": "doc{doc_id}_m1",
      "name": "...",
      "canonical_hint": "...",
      "aliases": ["..."],
      "formula": "...",
      "material_family": "...",
      "specificity": "specific_composition|family|sample_label|component",
      "linked_record_ids": ["..."],
      "evidence_quotes": ["..."],
      "notes": "..."
    }}
  ],
  "doc_level_notes": ["..."]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def corpus_link_messages(
    memories: list[dict[str, Any]],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    record_payload = [compact_record(record, 280) for record in records]
    memory_payload = [
        {
            "doc_id": memory.get("doc_id"),
            "doc_type": memory.get("doc_type"),
            "materials": memory.get("materials") or [],
            "doc_level_notes": memory.get("doc_level_notes") or [],
        }
        for memory in memories
    ]
    system = (
        "You are a corpus-level material entity linker. "
        "Cluster records that refer to the same material or intentionally same material family. "
        "Do not over-merge different dopants, stoichiometries, concentrations, or sample conditions. "
        "Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Document material memories:
{json.dumps(memory_payload, ensure_ascii=False, indent=2)}

Accepted fact records:
{json.dumps(record_payload, ensure_ascii=False, indent=2)}

Create cross-document material clusters only when two or more records should be linked, or when a record needs a non-trivial canonical material/alias decision. Do not enumerate ordinary singleton records; code will create fallback singleton clusters for omitted records.

Return JSON:
{{
  "clusters": [
    {{
      "cluster_id": "mc_001",
      "canonical_material": "...",
      "aliases": ["..."],
      "material_family": "...",
      "specificity": "specific_composition|family|sample_label|component",
      "member_record_ids": ["..."],
      "doc_ids": ["..."],
      "merge_confidence": 0.0,
      "link_reason": "semantic reason for linking or keeping singleton"
    }}
  ],
  "unlinked_record_ids": [],
  "linking_notes": ["..."]
}}

Set unlinked_record_ids to [] unless there is a specific non-singleton linking problem. Do not list ordinary omitted singleton record ids.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def corpus_critic_messages(
    clusters: dict[str, Any],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    record_payload = [compact_record(record, 360) for record in records]
    system = (
        "You are a strict material-science dataset critic. "
        "Audit both corpus-level material linking and record-level property semantics. "
        "Use scientific meaning and local evidence, not regex matching. Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Material clusters:
{json.dumps(clusters, ensure_ascii=False, indent=2)}

Records:
{json.dumps(record_payload, ensure_ascii=False, indent=2)}

Audit cluster links and records.

Semantic acceptance rules:
- process_temperature must be an intentional preparation/processing condition: drying, curing, sintering, calcination, annealing, deposition, polymerization/reaction, solvent removal, or other fabrication step. Mark REVIEW or REJECT when the temperature is a measurement/test/operating temperature, DSC/TGA program, degradation onset, thermal decomposition, weight-loss temperature, T5/T10/Tmax, graph axis, or battery cycling condition.
- thermal_transition should be a material transition such as Tg, Tm, Tc, melting, crystallization, phase transition, or explicitly named transition. Mark REVIEW when it is actually decomposition/TGA/weight-loss/heat-distortion/LCST/UCST and the schema does not clearly support that subtype.
- discharge_capacity must be tied to a material/electrode/cell and a capacity value, not just a generic requirement or unrelated electrochemical condition.
- ionic_conductivity must be tied to an electrolyte/material and conductivity value; keep measurement temperature in condition, not as process_temperature.
- adhesion_strength must be a measured adhesive/interface strength value; broad biomedical requirement statements or animal tissue facts should be REVIEW unless the material entity is a true engineered material in the task scope.
- A fallback singleton cluster is normal. Do not mark REVIEW only because a record has no cross-document match or merge_confidence 0.5. Review it only when the record itself is weak, too generic, or semantically mismatched.

Use REJECT for clear wrong-property / unsupported material-property binding. Use REVIEW for plausible but schema-ambiguous or too-generic records.

Important output rule:
- Return only REVIEW or REJECT cluster_items / record_items.
- Do not enumerate ACCEPT items; code treats omitted items as ACCEPT.

Return JSON:
{{
  "cluster_items": [
    {{"cluster_id": "mc_001", "verdict": "ACCEPT|REVIEW|REJECT", "reason": "..."}}
  ],
  "record_items": [
    {{"record_id": "...", "verdict": "ACCEPT|REVIEW|REJECT", "reason": "..."}}
  ],
  "global_notes": ["..."]
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def final_semantic_audit_messages(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
) -> list[dict[str, str]]:
    record_payload = []
    for record in records:
        item = compact_record(record, 520)
        item.update(
            {
                "canonical_material": record.get("canonical_material"),
                "material_family": record.get("material_family"),
                "cross_doc_link_reason": compact_text(str(record.get("cross_doc_link_reason") or ""), 320),
                "corpus_record_reason": compact_text(str(record.get("corpus_record_reason") or ""), 320),
                "corpus_cluster_reason": compact_text(str(record.get("corpus_cluster_reason") or ""), 320),
            }
        )
        record_payload.append(item)
    system = (
        "You are the final semantic quality gate for a material extraction dataset. "
        "The code has already checked JSON schema and exact evidence quote alignment. "
        "Your only job is scientific/property-semantic auditing. Return exactly one JSON object."
    )
    user = f"""
Competition task:
{problem_brief(problem_text)}

Schema:
{schema_text(schema)}

Currently accepted records:
{json.dumps(record_payload, ensure_ascii=False, indent=2)}

Return only records that should NOT remain in the high-confidence dataset.

Critical semantic rules:
- process_temperature is only for intentional preparation/fabrication/application conditions: dried, cured, sintered, calcined, annealed, baked, deposited, polymerized/reacted, stirred/heated during synthesis, solvent removal, or coating treatment.
- Mark process_temperature as REVIEW or REJECT when it is a test/measurement/operating condition, shape-memory/thermomechanical/cyclic test temperature, deformation or cooling temperature (T_high/T_low), DSC/TGA program, degradation/decomposition/weight-loss temperature, T5/T10/Tmax, graph axis, boiling point, melting point, phase-transition temperature, or battery cycling temperature.
- thermal_transition should be an actual material transition (Tg/Tm/Tc/melting/crystallization/phase/switching transition). If the evidence is actually a process step, test setting, or decomposition/weight-loss value, mark REVIEW.
- discharge_capacity, ionic_conductivity, and adhesion_strength must be measured or reported for a concrete material/sample/electrode/interface. Generic requirements, broad background facts, or biological tissue facts should be REVIEW.
- Do not penalize a record merely for being a singleton or not cross-linked. Penalize only weak or wrong scientific binding.
- If a value is a range and the record stores the lower bound in value plus the upper bound in value_max, that is acceptable.

Return JSON:
{{
  "record_items": [
    {{"record_id": "...", "verdict": "REVIEW|REJECT", "reason": "..."}}
  ],
  "global_notes": ["..."]
}}

Do not include ACCEPT records.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_clusters(link_obj: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    record_ids = {str(record.get("record_id") or "") for record in records}
    seen: set[str] = set()
    clusters: list[dict[str, Any]] = []
    for index, raw in enumerate(link_obj.get("clusters") or [], start=1):
        if not isinstance(raw, dict):
            continue
        members = [str(rid) for rid in raw.get("member_record_ids") or [] if str(rid) in record_ids]
        if not members:
            continue
        cluster_id = str(raw.get("cluster_id") or f"mc_{index:03d}")
        seen.update(members)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "canonical_material": str(raw.get("canonical_material") or ""),
                "aliases": raw.get("aliases") if isinstance(raw.get("aliases"), list) else [],
                "material_family": str(raw.get("material_family") or ""),
                "specificity": str(raw.get("specificity") or ""),
                "member_record_ids": members,
                "doc_ids": [str(doc_id) for doc_id in raw.get("doc_ids") or []],
                "merge_confidence": raw.get("merge_confidence"),
                "link_reason": str(raw.get("link_reason") or ""),
            }
        )
    missing = sorted(record_ids - seen)
    next_index = len(clusters) + 1
    by_record = {str(record.get("record_id") or ""): record for record in records}
    for rid in missing:
        record = by_record[rid]
        clusters.append(
            {
                "cluster_id": f"mc_{next_index:03d}",
                "canonical_material": str(record.get("material") or ""),
                "aliases": [],
                "material_family": "",
                "specificity": "singleton_fallback",
                "member_record_ids": [rid],
                "doc_ids": [str(record.get("doc_id") or "")],
                "merge_confidence": 0.5,
                "link_reason": "Fallback singleton because corpus linker did not return this record.",
            }
        )
        next_index += 1
    return {
        "clusters": clusters,
        "unlinked_record_ids": link_obj.get("unlinked_record_ids") or [],
        "linking_notes": link_obj.get("linking_notes") or [],
        "fallback_singletons": missing,
    }


def chunk_records_by_doc(records: list[dict[str, Any]], max_records: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda row: (
            int(str(row.get("doc_id"))) if str(row.get("doc_id", "")).isdigit() else 10**9,
            str(row.get("material") or "").lower(),
            str(row.get("property") or ""),
            str(row.get("record_id") or ""),
        ),
    ):
        if current and len(current) >= max_records:
            chunks.append(current)
            current = []
        current.append(record)
    if current:
        chunks.append(current)
    return chunks


def memory_subset(memories: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_ids = {str(record.get("doc_id") or "") for record in records}
    return [memory for memory in memories if str(memory.get("doc_id") or "") in doc_ids]


def run_link_chunks(
    *,
    memories: list[dict[str, Any]],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_clusters: list[dict[str, Any]] = []
    fallback_singletons: list[str] = []
    chunks_meta: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, chunk in enumerate(chunk_records_by_doc(records, args.corpus_link_chunk_size), start=1):
        link_obj, usage, error, raw = ai_call(
            args=args,
            api_key=api_key,
            messages=corpus_link_messages(memory_subset(memories, chunk), chunk, schema, problem_text),
            temperature_note=f"corpus_link_chunk_{index}_temperature_0.0",
        )
        if error:
            link_obj = {"clusters": [], "unlinked_record_ids": [], "linking_notes": [error], "raw": raw[:3000]}
        normalized = normalize_clusters(link_obj, chunk)
        for cluster in normalized.get("clusters") or []:
            out = dict(cluster)
            out["cluster_id"] = f"mc_{len(all_clusters) + 1:03d}"
            out["corpus_link_chunk"] = index
            all_clusters.append(out)
        fallback_singletons.extend(normalized.get("fallback_singletons") or [])
        notes.extend(normalized.get("linking_notes") or [])
        chunks_meta.append(
            {
                "chunk_index": index,
                "records": len(chunk),
                "clusters": len(normalized.get("clusters") or []),
                "fallback_singletons": len(normalized.get("fallback_singletons") or []),
                "usage": usage,
                "error": error,
            }
        )
    return {
        "clusters": all_clusters,
        "unlinked_record_ids": [],
        "linking_notes": notes,
        "fallback_singletons": sorted(set(fallback_singletons)),
        "chunks": chunks_meta,
    }


def run_critic_chunks(
    *,
    clusters: dict[str, Any],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record_by_id = {str(record.get("record_id") or ""): record for record in records}
    cluster_items: list[dict[str, Any]] = []
    record_items: list[dict[str, Any]] = []
    notes: list[str] = []
    chunks_meta: list[dict[str, Any]] = []
    cluster_list = clusters.get("clusters") or []
    for index in range(0, len(cluster_list), args.corpus_critic_chunk_size):
        chunk_clusters = cluster_list[index:index + args.corpus_critic_chunk_size]
        member_ids = {
            str(rid)
            for cluster in chunk_clusters
            for rid in cluster.get("member_record_ids") or []
        }
        chunk_records = [record_by_id[rid] for rid in member_ids if rid in record_by_id]
        chunk_obj = {
            **clusters,
            "clusters": chunk_clusters,
            "fallback_singletons": [],
        }
        critic, usage, error, raw = ai_call(
            args=args,
            api_key=api_key,
            messages=corpus_critic_messages(chunk_obj, chunk_records, schema, problem_text),
            temperature_note=f"corpus_critic_chunk_{len(chunks_meta) + 1}_temperature_0.0",
        )
        if error:
            critic = {"cluster_items": [], "record_items": [], "global_notes": [error], "raw": raw[:3000]}
        cluster_items.extend([item for item in critic.get("cluster_items") or [] if isinstance(item, dict)])
        record_items.extend([item for item in critic.get("record_items") or [] if isinstance(item, dict)])
        notes.extend([str(note) for note in critic.get("global_notes") or []])
        chunks_meta.append(
            {
                "chunk_index": len(chunks_meta) + 1,
                "clusters": len(chunk_clusters),
                "records": len(chunk_records),
                "usage": usage,
                "error": error,
            }
        )
    return {
        "cluster_items": cluster_items,
        "record_items": record_items,
        "global_notes": notes,
        "chunks": chunks_meta,
    }


def run_final_audit_chunks(
    *,
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    problem_text: str,
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record_items: list[dict[str, Any]] = []
    notes: list[str] = []
    chunks_meta: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunk_records_by_doc(records, args.final_audit_chunk_size), start=1):
        audit, usage, error, raw = ai_call(
            args=args,
            api_key=api_key,
            messages=final_semantic_audit_messages(chunk, schema, problem_text),
            temperature_note=f"final_semantic_audit_chunk_{index}_temperature_0.0",
        )
        if error:
            audit = {"record_items": [], "global_notes": [error], "raw": raw[:3000]}
        record_items.extend([item for item in audit.get("record_items") or [] if isinstance(item, dict)])
        notes.extend([str(note) for note in audit.get("global_notes") or []])
        chunks_meta.append(
            {
                "chunk_index": index,
                "records": len(chunk),
                "usage": usage,
                "error": error,
            }
        )
    return {
        "record_items": record_items,
        "global_notes": notes,
        "chunks": chunks_meta,
    }


def attach_clusters(
    records: list[dict[str, Any]],
    clusters: dict[str, Any],
    critic: dict[str, Any],
    keep_corpus_review: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_by_record: dict[str, dict[str, Any]] = {}
    for cluster in clusters.get("clusters") or []:
        for rid in cluster.get("member_record_ids") or []:
            cluster_by_record[str(rid)] = cluster
    cluster_critic = {str(item.get("cluster_id")): item for item in critic.get("cluster_items") or [] if isinstance(item, dict)}
    record_critic = {str(item.get("record_id")): item for item in critic.get("record_items") or [] if isinstance(item, dict)}

    final: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for record in records:
        rid = str(record.get("record_id") or "")
        cluster = cluster_by_record.get(rid)
        rc = record_critic.get(rid, {"verdict": "ACCEPT", "reason": "No corpus-level objection returned."})
        cc = cluster_critic.get(str((cluster or {}).get("cluster_id") or ""), {"verdict": "ACCEPT", "reason": "No cluster-level objection returned."})
        enriched = {
            **record,
            "corpus_version": CORPUS_VERSION,
            "material_cluster_id": (cluster or {}).get("cluster_id"),
            "canonical_material": (cluster or {}).get("canonical_material") or record.get("material"),
            "material_aliases": (cluster or {}).get("aliases") or [],
            "material_family": (cluster or {}).get("material_family") or "",
            "cross_doc_merge_confidence": (cluster or {}).get("merge_confidence"),
            "cross_doc_link_reason": (cluster or {}).get("link_reason"),
            "corpus_record_verdict": rc.get("verdict"),
            "corpus_record_reason": rc.get("reason"),
            "corpus_cluster_verdict": cc.get("verdict"),
            "corpus_cluster_reason": cc.get("reason"),
        }
        rc_verdict = str(rc.get("verdict") or "ACCEPT").upper()
        cc_verdict = str(cc.get("verdict") or "ACCEPT").upper()
        if not cluster:
            review.append({"type": "missing_cluster", "record": enriched, "reason": "No material cluster was assigned."})
        elif rc_verdict == "REJECT" or cc_verdict == "REJECT":
            review.append({"type": "corpus_reject", "record": enriched, "record_critic": rc, "cluster_critic": cc})
        elif not keep_corpus_review and (rc_verdict == "REVIEW" or cc_verdict == "REVIEW"):
            review.append({"type": "corpus_review", "record": enriched, "record_critic": rc, "cluster_critic": cc})
        else:
            final.append(enriched)
    return final, review


def apply_final_audit(
    records: list[dict[str, Any]],
    audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_by_record = {str(item.get("record_id")): item for item in audit.get("record_items") or [] if isinstance(item, dict)}
    final: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for record in records:
        rid = str(record.get("record_id") or "")
        item = audit_by_record.get(rid)
        verdict = str((item or {}).get("verdict") or "ACCEPT").upper()
        if verdict in {"REVIEW", "REJECT"}:
            enriched = {
                **record,
                "final_audit_verdict": verdict,
                "final_audit_reason": (item or {}).get("reason"),
            }
            review.append(
                {
                    "type": "final_semantic_reject" if verdict == "REJECT" else "final_semantic_review",
                    "record": enriched,
                    "final_audit": item,
                }
            )
        else:
            final.append({**record, "final_audit_verdict": "ACCEPT", "final_audit_reason": "No final semantic objection returned."})
    return final, review


def local_check(
    out_dir: Path,
    states: list[dict[str, Any]],
    records: list[dict[str, Any]],
    review: list[dict[str, Any]],
) -> dict[str, Any]:
    required = [
        "doc_id",
        "page_range",
        "material",
        "property",
        "value",
        "unit",
        "evidence_quote",
        "confidence",
        "record_id",
        "material_cluster_id",
        "canonical_material",
    ]
    missing: list[dict[str, Any]] = []
    for record in records:
        bad = [
            field for field in required
            if record.get(field) is None or (isinstance(record.get(field), str) and not record.get(field).strip())
        ]
        if bad:
            missing.append({"record_id": record.get("record_id"), "missing": bad})

    align_by_record: dict[str, dict[str, Any]] = {}
    for state in states:
        for row in state.get("final_alignment") or []:
            align_by_record[str(row.get("record_id") or "")] = row

    quote_checks = []
    for record in records:
        align = align_by_record.get(str(record.get("record_id") or ""))
        quote_checks.append(
            {
                "record_id": record.get("record_id"),
                "code_gate": (align or {}).get("code_gate"),
                "quote_status": ((align or {}).get("quote_result") or {}).get("status"),
            }
        )

    ids = [str(record.get("record_id") or "") for record in records]
    jsonl_parse_errors: list[dict[str, Any]] = []
    for rel in [
        "records/submission_candidates.jsonl",
        "dataset.jsonl",
        "review/corpus_review.jsonl",
        "review/all_review.jsonl",
    ]:
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
                    jsonl_parse_errors.append({"path": rel, "line": line_no, "error": str(exc)})
    report = {
        "run_dir": str(out_dir),
        "doc_status_counts": dict(Counter(str(state.get("status") or "unknown") for state in states)),
        "doc_count": len(states),
        "accepted_records": len(records),
        "review_records": len(review),
        "accepted_by_property": dict(Counter(str(record.get("property") or "unknown") for record in records)),
        "clusters": len({str(record.get("material_cluster_id") or "") for record in records}),
        "missing_required": missing,
        "duplicate_record_ids": len(ids) - len(set(ids)),
        "quote_checks": quote_checks,
        "all_quotes_code_pass": all(item.get("code_gate") == "PASS" for item in quote_checks),
        "jsonl_parse_errors": jsonl_parse_errors,
        "jsonl_parse_pass": not jsonl_parse_errors,
        "records_with_corpus_verdict": sum(1 for record in records if record.get("corpus_record_verdict")),
        "accepted_corpus_review_records": sum(
            1
            for record in records
            if str(record.get("corpus_record_verdict") or "").upper() == "REVIEW"
            or str(record.get("corpus_cluster_verdict") or "").upper() == "REVIEW"
        ),
        "accepted_final_audit_review_records": sum(
            1
            for record in records
            if str(record.get("final_audit_verdict") or "").upper() == "REVIEW"
        ),
        "local_check_pass": (
            not missing
            and len(ids) == len(set(ids))
            and all(item.get("code_gate") == "PASS" for item in quote_checks)
            and not jsonl_parse_errors
            and not any(
                str(record.get("corpus_record_verdict") or "").upper() == "REVIEW"
                or str(record.get("corpus_cluster_verdict") or "").upper() == "REVIEW"
                for record in records
            )
            and not any(str(record.get("final_audit_verdict") or "").upper() == "REVIEW" for record in records)
        ),
    }
    write_json(out_dir / "local_check_report.json", report)
    lines = ["# Local Check Report", "", "## Verdict", ""]
    lines.append("PASS" if report["local_check_pass"] else "REVIEW")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- docs: {report['doc_count']}",
            f"- accepted records: {report['accepted_records']}",
            f"- review records: {report['review_records']}",
            f"- clusters: {report['clusters']}",
            f"- duplicate record ids: {report['duplicate_record_ids']}",
            f"- missing required records: {len(missing)}",
            f"- all quote gates pass: {report['all_quotes_code_pass']}",
            f"- JSONL parse pass: {report['jsonl_parse_pass']}",
            f"- accepted corpus REVIEW records: {report['accepted_corpus_review_records']}",
            f"- accepted final audit REVIEW records: {report['accepted_final_audit_review_records']}",
            "",
            "## Accepted By Property",
            "",
        ]
    )
    for prop, count in report["accepted_by_property"].items():
        lines.append(f"- {prop}: {count}")
    (out_dir / "local_check_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def summarize(
    states: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    doc_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    review: list[dict[str, Any]],
    clusters: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
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
                        usage_totals[f"doc_{phase}_{key}"] += int(entry.get(key) or 0)
                    except (TypeError, ValueError):
                        pass
    return {
        "corpus_version": CORPUS_VERSION,
        "doc_agent_version": DOC_AGENT_VERSION,
        "time": now_iso(),
        "model": args.model,
        "output_dir": str(args.output_dir),
        "doc_count": len(states),
        "doc_status_counts": dict(Counter(str(state.get("status") or "unknown") for state in states)),
        "doc_level_accepted_records": len(doc_records),
        "final_accepted_records": len(final_records),
        "review_records": len(review),
        "review_by_type": dict(Counter(str(item.get("type") or "unknown") for item in review if isinstance(item, dict))),
        "accepted_by_property": dict(Counter(str(record.get("property") or "unknown") for record in final_records)),
        "records_by_doc": dict(Counter(str(record.get("doc_id") or "unknown") for record in final_records)),
        "memory_count": len(memories),
        "cluster_count": len(clusters.get("clusters") or []),
        "fallback_singletons": len(clusters.get("fallback_singletons") or []),
        "limit_docs": args.limit_docs,
        "max_packets_per_doc": args.max_packets_per_doc,
        "per_property": args.per_property,
        "batch_size": args.batch_size,
        "keep_corpus_review": args.keep_corpus_review,
        "semantic_gate": "DeepSeek-v4-flash doc reader + doc memory + corpus linker + corpus critic; code only validates schema/quote/provenance structure.",
        "api_usage_totals": dict(usage_totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V4 corpus-level AI material extraction agent.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_ROOT / "candidates_full768_20260523" / "focused_evidence_candidates.recall_v3.jsonl")
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_WORK_ROOT / "combined")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parent / "schemas" / "default_material_schema.json")
    parser.add_argument("--problem", type=Path, default=Path(__file__).resolve().parents[1] / "赛题说明" / "材料赛题.md")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "agentic_material_corpus_v4_flash20")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="disabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--limit-docs", type=int, default=20)
    parser.add_argument("--per-property", type=int, default=8)
    parser.add_argument("--max-packets-per-doc", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-accept-confidence", type=float, default=0.80)
    parser.add_argument("--context-radius", type=int, default=1800)
    parser.add_argument("--title-excerpt-limit", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-doc-dir", type=Path, default=None, help="Reuse states, memories, and doc-level records from a previous V4 run; rerun only corpus linking/critic.")
    parser.add_argument("--corpus-link-chunk-size", type=int, default=35)
    parser.add_argument("--corpus-critic-chunk-size", type=int, default=35)
    parser.add_argument("--final-audit-chunk-size", type=int, default=25)
    parser.add_argument("--keep-corpus-review", action="store_true", help="Keep corpus-level REVIEW records in final output instead of routing them to review.")
    parser.add_argument("--skip-final-audit", action="store_true", help="Skip the final AI semantic audit stage.")
    args = parser.parse_args()

    args.candidates = local_path(args.candidates)
    args.combined_dir = local_path(args.combined_dir)
    args.schema = local_path(args.schema)
    args.problem = local_path(args.problem)
    args.output_dir = local_path(args.output_dir)
    args.reuse_doc_dir = local_path(args.reuse_doc_dir) if args.reuse_doc_dir else None
    args.env_file = [local_path(path) for path in args.env_file]

    for env_path in args.env_file:
        load_env_file(env_path)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --env-file.")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    problem_text = args.problem.read_text(encoding="utf-8", errors="ignore")
    candidates = read_jsonl(args.candidates)
    doc_ids = sort_doc_ids({str(row.get("doc_id") or "") for row in candidates if row.get("doc_id")})
    if args.doc_id:
        wanted = {str(value) for value in args.doc_id}
        doc_ids = [doc_id for doc_id in doc_ids if doc_id in wanted]
    elif args.limit_docs > 0:
        doc_ids = doc_ids[:args.limit_docs]

    out_dir = ensure_dir(args.output_dir)
    for name in ["states", "records", "review", "memories", "corpus"]:
        ensure_dir(out_dir / name)
    log_path = out_dir / "run.log"
    append_jsonl(log_path, {"event": "start", "time": now_iso(), "docs": doc_ids, "model": args.model, "corpus_version": CORPUS_VERSION})

    if args.reuse_doc_dir:
        states = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.reuse_doc_dir / "states").glob("*.json"))]
        doc_records = read_jsonl(args.reuse_doc_dir / "records" / "doc_level_records.jsonl")
        doc_review = read_jsonl(args.reuse_doc_dir / "review" / "doc_level_review.jsonl")
        memories = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((args.reuse_doc_dir / "memories").glob("*.material_memory.json"))
        ]
        for state in states:
            write_json(out_dir / "states" / f"{state.get('doc_id')}.json", state)
        for memory in memories:
            write_json(out_dir / "memories" / f"{memory.get('doc_id')}.material_memory.json", memory)
        write_jsonl(out_dir / "records" / "doc_level_records.jsonl", doc_records)
        write_jsonl(out_dir / "review" / "doc_level_review.jsonl", doc_review)
        append_jsonl(log_path, {"event": "reuse_doc_level", "time": now_iso(), "source": str(args.reuse_doc_dir), "states": len(states), "doc_records": len(doc_records)})
    else:
        states: list[dict[str, Any]] = []
        doc_records: list[dict[str, Any]] = []
        doc_review: list[dict[str, Any]] = []
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
                            "agentic_version": DOC_AGENT_VERSION,
                            "doc_id": doc_id,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "finished": now_iso(),
                        },
                        "records": [],
                        "review": [],
                    }
                state = result["state"]
                records = result["records"]
                review = result["review"]
                with lock:
                    states.append(state)
                    doc_records.extend(records)
                    doc_review.extend(review)
                    write_json(out_dir / "states" / f"{doc_id}.json", state)
                    write_jsonl(out_dir / "records" / f"{doc_id}.doc_records.jsonl", records)
                    write_jsonl(out_dir / "review" / f"{doc_id}.doc_review.jsonl", review)
                    append_jsonl(log_path, {"event": "doc_done", "time": now_iso(), "doc_id": doc_id, "status": state.get("status"), "accepted": len(records), "review": len(review)})

        states.sort(key=lambda state: int(str(state.get("doc_id"))) if str(state.get("doc_id", "")).isdigit() else 10**9)
        doc_records.sort(key=lambda record: (int(str(record.get("doc_id"))) if str(record.get("doc_id", "")).isdigit() else 10**9, str(record.get("property") or ""), str(record.get("record_id") or "")))
        write_jsonl(out_dir / "records" / "doc_level_records.jsonl", doc_records)
        write_jsonl(out_dir / "review" / "doc_level_review.jsonl", doc_review)

        memories: list[dict[str, Any]] = []
        for state in states:
            doc_id = str(state.get("doc_id") or "")
            combined_text = read_combined(args.combined_dir, doc_id)
            records = [record for record in doc_records if str(record.get("doc_id") or "") == doc_id]
            if not combined_text:
                continue
            memory, usage, error, raw = ai_call(
                args=args,
                api_key=api_key,
                messages=memory_messages(doc_id, combined_text, state, records, schema, problem_text),
                temperature_note="memory_temperature_0.0",
            )
            state.setdefault("usage", {})["memory"] = usage
            if error:
                memory = {"doc_id": doc_id, "error": error, "raw": raw[:3000], "materials": []}
            memories.append(memory)
            write_json(out_dir / "memories" / f"{doc_id}.material_memory.json", memory)

    clusters = run_link_chunks(
        memories=memories,
        records=doc_records,
        schema=schema,
        problem_text=problem_text,
        api_key=api_key,
        args=args,
    )
    write_json(out_dir / "corpus" / "material_clusters.json", clusters)

    critic = run_critic_chunks(
        clusters=clusters,
        records=doc_records,
        schema=schema,
        problem_text=problem_text,
        api_key=api_key,
        args=args,
    )
    write_json(out_dir / "corpus" / "corpus_critic.json", critic)

    final_records, corpus_review = attach_clusters(doc_records, clusters, critic, keep_corpus_review=args.keep_corpus_review)
    if args.skip_final_audit:
        final_audit = {"record_items": [], "global_notes": ["skipped"], "chunks": []}
        final_audit_review: list[dict[str, Any]] = []
    else:
        final_audit = run_final_audit_chunks(
            records=final_records,
            schema=schema,
            problem_text=problem_text,
            api_key=api_key,
            args=args,
        )
        write_json(out_dir / "corpus" / "final_semantic_audit.json", final_audit)
        final_records, final_audit_review = apply_final_audit(final_records, final_audit)
    all_review = doc_review + corpus_review + final_audit_review
    final_records.sort(key=lambda record: (str(record.get("material_cluster_id") or ""), int(str(record.get("doc_id"))) if str(record.get("doc_id", "")).isdigit() else 10**9, str(record.get("property") or ""), str(record.get("record_id") or "")))
    write_jsonl(out_dir / "records" / "submission_candidates.jsonl", final_records)
    write_jsonl(out_dir / "dataset.jsonl", final_records)
    write_jsonl(out_dir / "review" / "corpus_review.jsonl", corpus_review)
    write_jsonl(out_dir / "review" / "all_review.jsonl", all_review)

    summary = summarize(states, memories, doc_records, final_records, all_review, clusters, args)
    summary["corpus_link_chunks"] = clusters.get("chunks") or []
    summary["corpus_critic_chunks"] = critic.get("chunks") or []
    summary["final_semantic_audit_chunks"] = final_audit.get("chunks") or []
    summary["final_semantic_audit_record_items"] = len(final_audit.get("record_items") or [])
    write_json(out_dir / "summary.json", summary)
    check_report = local_check(out_dir, states, final_records, all_review)
    append_jsonl(log_path, {"event": "finished", **summary, "local_check_pass": check_report.get("local_check_pass")})
    print(f"agentic_material_corpus_v4={out_dir} docs={len(states)} doc_records={len(doc_records)} final={len(final_records)} review={len(all_review)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
