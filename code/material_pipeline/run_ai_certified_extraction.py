from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import DEFAULT_WORK_ROOT, compact_text, ensure_dir, read_jsonl, write_json, write_jsonl
from run_deepseek_extraction_experiment import append_jsonl, local_path, now_iso


CERT_VERSION = "ai_certified_material_extraction_v1"
VERDICTS = {"ACCEPT", "PARTIAL", "REJECT", "NEEDS_SCHEMA_DECISION"}
CHECK_FIELDS = [
    "schema_scope_ok",
    "property_ok",
    "property_subtype_ok",
    "material_ok",
    "value_ok",
    "unit_ok",
    "condition_ok",
    "evidence_support_ok",
]


def call_deepseek_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_sec: int,
    thinking: str,
    reasoning_effort: str,
    retries: int,
) -> tuple[str, dict[str, Any], str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def candidate_key(row: dict[str, Any]) -> str:
    existing = row.get("candidate_key")
    if existing:
        return str(existing)
    payload = json.dumps(
        {
            "doc_id": row.get("doc_id"),
            "page_range": row.get("page_range"),
            "property_hint": row.get("property_hint"),
            "evidence": row.get("evidence"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def record_id(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "candidate_key": candidate_key(candidate),
            "doc_id": record.get("doc_id"),
            "page_range": record.get("page_range"),
            "material": record.get("material"),
            "property": record.get("property"),
            "value": record.get("value"),
            "value_max": record.get("value_max"),
            "unit": record.get("unit"),
            "condition": record.get("condition"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_completed(path: Path) -> set[str]:
    return {str(row.get("candidate_key") or "") for row in read_jsonl(path) if row.get("candidate_key")}


def schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)


def problem_brief(problem_text: str) -> str:
    return compact_text(problem_text, 1800)


def context_fragments(candidate: dict[str, Any]) -> list[str]:
    evidence = str(candidate.get("evidence") or "")
    fragments: list[str] = []
    target_column = candidate.get("target_column")
    target_value = candidate.get("target_value")
    if isinstance(target_column, str) and isinstance(target_value, str) and target_column.strip() and target_value.strip():
        fragments.append(f"{target_column.strip()} {target_value.strip()}")
    for key in ["target_column", "target_value", "signal"]:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    fragments.extend(
        part.strip()
        for part in re.split(r"\||Row:|Row label:|Row context:|Target column:|Target value:|Material/table context:", evidence)
        if part.strip() and not part.strip().startswith("Section:")
    )
    fragments.append(evidence.replace("Section:", "").strip())
    return [fragment for fragment in fragments if len(fragment) >= 8]


def combined_context(candidate: dict[str, Any], combined_dir: Path, radius: int) -> dict[str, Any]:
    doc_id = str(candidate.get("doc_id") or "")
    path = combined_dir / doc_id / f"{doc_id}_combined.md"
    if not path.exists():
        return {"status": "missing", "path": str(path), "context": ""}
    text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    best_idx = -1
    best_fragment = ""
    for fragment in context_fragments(candidate):
        for probe in [fragment, fragment[:240], fragment[:120], fragment[:60]]:
            probe = " ".join(probe.split())
            if len(probe) < 8:
                continue
            idx = lowered.find(probe.lower())
            if idx >= 0:
                best_idx = idx
                best_fragment = probe
                break
        if best_idx >= 0:
            break
    if best_idx < 0:
        return {"status": "not_found", "path": str(path), "context": ""}
    start = max(0, best_idx - radius)
    end = min(len(text), best_idx + len(best_fragment) + radius)
    return {"status": "found", "path": str(path), "char_start": best_idx, "context": text[start:end]}


def extraction_messages(candidate: dict[str, Any], schema: dict[str, Any], problem_text: str) -> list[dict[str, str]]:
    prop = str(candidate.get("property_hint") or "")
    evidence = compact_text(str(candidate.get("evidence") or ""), 2600)
    context = compact_text(str(candidate.get("_combined_context") or ""), 3600)
    system = (
        "你是材料赛题的信息抽取专家。你必须只根据给定 Evidence 抽取。"
        "如果证据不足、行列绑定不清、材料绑定不清、属性不属于 schema，必须拒绝。"
        "返回且只返回一个 JSON object。"
    )
    user = f"""
赛题要求摘要：
{problem_brief(problem_text)}

当前 schema：
{schema_text(schema)}

候选目标 property_hint: {prop}

Evidence:
{evidence}

Combined 原文上下文（用于核对表格/段落，不要只相信候选 evidence 的简写标签）：
{context}

请判断 Evidence 是否能支持一个且仅一个材料属性事实。

必须逐字段判断：
- schema_scope_ok
- property_ok
- property_subtype_ok
- material_ok
- value_ok
- unit_ok
- condition_ok
- evidence_support_ok

硬性要求：
1. material、property、value、unit、condition 必须属于同一实体/同一表格行列逻辑/同一句语义。
2. 表格必须核对目标列、行标签、材料列/列头，不能把邻近列、坐标、循环数、电压、效率当属性值。
3. discharge_capacity 必须是电池放电/比容量；如果材料只是电解质/隔膜/基体，而证据说的是整电池容量，不能 ACCEPT。
4. process_temperature 只接受制备/加工/固化/烧结/退火/干燥/煅烧/沉积/聚合温度；拒绝测试温度、升温速率、图坐标、热分析程序温度。
5. thermal_transition 当前只明确接受 Tg/Tm/Tc/melting/crystallization；decomposition/TGA/weight-loss/heat-distortion/LCST/UCST 标 NEEDS_SCHEMA_DECISION，除非 schema 明确 subtype。
6. adhesion_strength 必须是粘接/剥离/界面/胶黏接头语境。
7. ionic_conductivity 必须是离子/Li+电导率，材料与数值绑定清楚。
8. 如果 evidence 中有明确条件，例如 for 12 h、in Ar、95%Ar/5%H2、电压窗口、循环数、倍率、测试温度，condition 必须保留；缺关键条件不能 ACCEPT。

返回 JSON 格式：
{{
  "verdict": "ACCEPT|PARTIAL|REJECT|NEEDS_SCHEMA_DECISION",
  "schema_scope_ok": true,
  "property_ok": true,
  "property_subtype_ok": true,
  "material_ok": true,
  "value_ok": true,
  "unit_ok": true,
  "condition_ok": true,
  "evidence_support_ok": true,
  "reason_code": "short_code",
  "reason": "中文说明，指出证据如何支持或为何不支持",
  "fact": {{
    "material": "证据中的材料实体，必须原文可追溯",
    "property": "{prop}",
    "property_subtype": null,
    "value": 0,
    "value_max": null,
    "value_text": "原文数值",
    "unit": "标准单位",
    "condition": {{}},
    "evidence_quote": "最小充分证据片段"
  }},
  "confidence": 0.0
}}

如果 verdict 不是 ACCEPT，fact 可以为 null。
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verifier_messages(candidate: dict[str, Any], extraction: dict[str, Any], schema: dict[str, Any], problem_text: str) -> list[dict[str, str]]:
    evidence = compact_text(str(candidate.get("evidence") or ""), 2800)
    context = compact_text(str(candidate.get("_combined_context") or ""), 3800)
    system = (
        "你是材料赛题的语义审计裁判。你的任务是挑错。"
        "如果材料、属性、数值、单位、条件、证据任一项不能严格对应，不能 ACCEPT。"
        "返回且只返回一个 JSON object。"
    )
    user = f"""
赛题要求摘要：
{problem_brief(problem_text)}

当前 schema：
{schema_text(schema)}

候选 Evidence:
{evidence}

Combined 原文上下文：
{context}

待验证抽取结果：
{json.dumps(extraction, ensure_ascii=False, indent=2)}

请独立验证该抽取结果是否正确。不要相信上一轮 verdict。

逐字段输出：
{CHECK_FIELDS}

verdict 只能是 ACCEPT / PARTIAL / REJECT / NEEDS_SCHEMA_DECISION。

ACCEPT 必须非常严格：
- 所有 *_ok 字段都为 true。
- 如果 evidence 写了时间、气氛、倍率、循环数、测试温度、电压窗口等条件，待验证结果的 condition 必须保留。
- 如果 evidence 是表格，必须能从原文上下文看出材料列/行标签/目标值的绑定。
- 任何一项缺失或不确定都不能 ACCEPT。

返回 JSON：
{{
  "verdict": "ACCEPT|PARTIAL|REJECT|NEEDS_SCHEMA_DECISION",
  "schema_scope_ok": true,
  "property_ok": true,
  "property_subtype_ok": true,
  "material_ok": true,
  "value_ok": true,
  "unit_ok": true,
  "condition_ok": true,
  "evidence_support_ok": true,
  "reason_code": "short_code",
  "reason": "中文说明",
  "confidence": 0.0
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_failed:{exc}"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    verdict = str(obj.get("verdict") or "").strip().upper()
    if verdict not in VERDICTS:
        return None, f"bad_verdict:{verdict}"
    obj["verdict"] = verdict
    for field in CHECK_FIELDS:
        obj[field] = bool(obj.get(field))
    try:
        obj["confidence"] = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        obj["confidence"] = 0.0
    obj["reason_code"] = str(obj.get("reason_code") or "unspecified")
    obj["reason_code"] = obj["reason_code"].strip().lower()
    obj["reason"] = str(obj.get("reason") or "")
    return obj, None


def fact_to_record(fact: dict[str, Any], candidate: dict[str, Any], confidence: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "doc_id": str(candidate.get("doc_id") or ""),
        "page_range": str(candidate.get("page_range") or ""),
        "material": str(fact.get("material") or ""),
        "property": str(fact.get("property") or candidate.get("property_hint") or ""),
        "value": fact.get("value"),
        "unit": str(fact.get("unit") or ""),
        "value_text": str(fact.get("value_text") or ""),
        "condition": fact.get("condition") if isinstance(fact.get("condition"), dict) else {},
        "evidence": str(fact.get("evidence_quote") or candidate.get("evidence") or ""),
        "confidence": round(confidence, 3),
        "source_type": "ai_certified_candidate_evidence",
        "extraction_method": CERT_VERSION,
        "source_candidate_key": candidate_key(candidate),
        "source_property_hint": candidate.get("property_hint"),
    }
    if fact.get("property_subtype"):
        record["property_subtype"] = fact.get("property_subtype")
    if fact.get("value_max") is not None:
        record["value_max"] = fact.get("value_max")
    record["record_id"] = record_id(record, candidate)
    return record


def certify_one(candidate: dict[str, Any], index: int, api_key: str, schema: dict[str, Any], problem_text: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ckey = candidate_key(candidate)
    extract_text, extract_usage, extract_error = call_deepseek_json(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=extraction_messages(candidate, schema, problem_text),
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        retries=args.retries,
    )
    extraction, extract_parse_error = (None, extract_error) if extract_error else parse_json_object(extract_text)
    verifier: dict[str, Any] | None = None
    verify_text = ""
    verify_usage: dict[str, Any] = {}
    verify_parse_error: str | None = None
    final_verdict = "REJECT"
    record: dict[str, Any] | None = None

    if extraction and extraction.get("verdict") == "ACCEPT" and isinstance(extraction.get("fact"), dict):
        verify_text, verify_usage, verify_error = call_deepseek_json(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            messages=verifier_messages(candidate, extraction, schema, problem_text),
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            retries=args.retries,
        )
        verifier, verify_parse_error = (None, verify_error) if verify_error else parse_json_object(verify_text)
        extraction_confidence = float(extraction.get("confidence") or 0.0)
        verifier_confidence = float((verifier or {}).get("confidence") or 0.0)
        extraction_checks_ok = all(bool(extraction.get(field)) for field in CHECK_FIELDS)
        verifier_checks_ok = bool(verifier) and all(bool(verifier.get(field)) for field in CHECK_FIELDS)
        if (
            verifier
            and verifier.get("verdict") == "ACCEPT"
            and extraction_confidence >= args.min_accept_confidence
            and verifier_confidence >= args.min_accept_confidence
            and extraction_checks_ok
            and verifier_checks_ok
        ):
            final_verdict = "ACCEPT"
            confidence = min(extraction_confidence, verifier_confidence)
            record = fact_to_record(extraction["fact"], candidate, confidence)
        elif verifier:
            final_verdict = str(verifier.get("verdict") or "REJECT")
    elif extraction:
        final_verdict = str(extraction.get("verdict") or "REJECT")

    return {
        "cert_version": CERT_VERSION,
        "time": now_iso(),
        "index": index,
        "candidate_key": ckey,
        "doc_id": candidate.get("doc_id"),
        "page_range": candidate.get("page_range"),
        "property_hint": candidate.get("property_hint"),
        "candidate_score": candidate.get("score"),
        "final_verdict": final_verdict,
        "record": record,
        "extraction": extraction,
        "verifier": verifier,
        "extract_parse_error": extract_parse_error,
        "verify_parse_error": verify_parse_error,
        "extract_output_text": extract_text,
        "verify_output_text": verify_text,
        "usage": {"extract": extract_usage, "verify": verify_usage},
        "elapsed_sec": round(time.time() - started, 3),
        "candidate": candidate,
    }


def select_candidates(
    rows: list[dict[str, Any]],
    limit: int,
    per_property: int | None,
    min_score: float,
    doc_id: str | None,
) -> list[dict[str, Any]]:
    if doc_id:
        rows = [row for row in rows if str(row.get("doc_id") or "") == doc_id]
    filtered = [row for row in rows if float(row.get("score") or 0.0) >= min_score and str(row.get("evidence") or "").strip()]
    filtered.sort(
        key=lambda row: (
            str(row.get("property_hint") or ""),
            -float(row.get("score") or 0.0),
            str(row.get("doc_id") or ""),
            str(row.get("page_range") or ""),
            candidate_key(row),
        )
    )
    if per_property is not None and per_property > 0:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in filtered:
            prop = str(row.get("property_hint") or "")
            if len(grouped[prop]) < per_property:
                grouped[prop].append(row)
        selected: list[dict[str, Any]] = []
        for prop in sorted(grouped):
            selected.extend(grouped[prop])
        return selected[:limit] if limit > 0 else selected
    return filtered[:limit] if limit > 0 else filtered


def summarize(rows: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "time": now_iso(),
        "cert_version": CERT_VERSION,
        "candidates_path": str(args.candidates),
        "output_dir": str(args.output_dir),
        "total_trials": len(rows),
        "accepted_records": len(records),
        "final_verdict_counts": dict(Counter(str(row.get("final_verdict") or "UNKNOWN") for row in rows)),
        "accepted_by_property": dict(Counter(str(record.get("property") or "") for record in records)),
        "rejection_reason_codes": dict(Counter(str(((row.get("verifier") or row.get("extraction") or {}).get("reason_code")) or row.get("extract_parse_error") or row.get("verify_parse_error") or "unknown") for row in rows if row.get("final_verdict") != "ACCEPT").most_common()),
        "model": args.model,
        "min_score": args.min_score,
        "min_accept_confidence": args.min_accept_confidence,
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = ["# AI Certified Material Extraction", "", "## Summary", ""]
    for key in ["total_trials", "accepted_records", "model", "min_score", "min_accept_confidence"]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## Final Verdict Counts")
    lines.append("")
    for verdict, count in (summary.get("final_verdict_counts") or {}).items():
        lines.append(f"- {verdict}: {count}")
    lines.append("")
    lines.append("## Accepted By Property")
    lines.append("")
    for prop, count in (summary.get("accepted_by_property") or {}).items():
        lines.append(f"- {prop}: {count}")
    lines.append("")
    lines.append("## Non-Accept Examples")
    lines.append("")
    for row in [r for r in rows if r.get("final_verdict") != "ACCEPT"][:30]:
        judge = row.get("verifier") or row.get("extraction") or {}
        candidate = row.get("candidate") or {}
        lines.append(
            f"- {row.get('final_verdict')} doc {candidate.get('doc_id')} p{candidate.get('page_range')} "
            f"{candidate.get('property_hint')} | {judge.get('reason_code') or row.get('extract_parse_error') or row.get('verify_parse_error')}: "
            f"{compact_text(judge.get('reason') or '', 180)}"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-only certified extraction from material recall candidates.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_WORK_ROOT / "candidates_full768_20260523" / "focused_evidence_candidates.recall_v3.jsonl")
    parser.add_argument("--schema", type=Path, default=Path(__file__).resolve().parent / "schemas" / "default_material_schema.json")
    parser.add_argument("--problem", type=Path, default=Path(__file__).resolve().parents[1] / "赛题说明" / "材料赛题.md")
    parser.add_argument("--combined-dir", type=Path, default=DEFAULT_WORK_ROOT / "combined")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_ROOT / "ai_certified_material_v1")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Maximum candidates to process; 0 means all selected candidates.")
    parser.add_argument("--per-property", type=int, default=0, help="Maximum candidates per property; 0 means no per-property cap.")
    parser.add_argument("--min-score", type=float, default=0.58)
    parser.add_argument("--min-accept-confidence", type=float, default=0.78)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    args.candidates = local_path(args.candidates)
    args.schema = local_path(args.schema)
    args.problem = local_path(args.problem)
    args.combined_dir = local_path(args.combined_dir)
    args.output_dir = local_path(args.output_dir)
    args.env_file = [local_path(path) for path in args.env_file]

    for env_path in args.env_file:
        load_env_file(env_path)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} or pass --env-file containing it.")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    problem_text = args.problem.read_text(encoding="utf-8", errors="ignore")
    candidates = select_candidates(read_jsonl(args.candidates), args.limit, args.per_property, args.min_score, args.doc_id)
    candidates = [
        {
            **candidate,
            "_combined_context": combined_context(candidate, args.combined_dir, radius=2200).get("context", ""),
        }
        for candidate in candidates
    ]

    out_dir = ensure_dir(args.output_dir)
    trials_path = out_dir / "ai_certification_trials.jsonl"
    accepted_path = out_dir / "certified_records.accepted.jsonl"
    rejected_path = out_dir / "certified_trials.non_accept.jsonl"
    summary_path = out_dir / "certification_summary.json"
    report_path = out_dir / "certification_report.md"
    log_path = out_dir / "run.log"

    completed = load_completed(trials_path)
    pending = [candidate for candidate in candidates if candidate_key(candidate) not in completed]
    append_jsonl(log_path, {"event": "start", "time": now_iso(), "selected": len(candidates), "completed": len(completed), "pending": len(pending), "model": args.model})

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(certify_one, candidate, index, api_key, schema, problem_text, args) for index, candidate in enumerate(pending, start=1)]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                append_jsonl(trials_path, row)
                append_jsonl(log_path, {"event": "item_done", "time": now_iso(), "candidate_key": row.get("candidate_key"), "final_verdict": row.get("final_verdict"), "elapsed_sec": row.get("elapsed_sec")})

    wanted = {candidate_key(candidate) for candidate in candidates}
    rows = [row for row in read_jsonl(trials_path) if str(row.get("candidate_key") or "") in wanted]
    write_jsonl(trials_path, rows)
    accepted_records = [row["record"] for row in rows if row.get("final_verdict") == "ACCEPT" and isinstance(row.get("record"), dict)]
    write_jsonl(accepted_path, accepted_records)
    write_jsonl(rejected_path, [row for row in rows if row.get("final_verdict") != "ACCEPT"])
    summary = summarize(rows, accepted_records, args)
    write_json(summary_path, summary)
    write_report(report_path, summary, rows)
    append_jsonl(log_path, {"event": "finished", **summary})
    print(f"ai_certified={out_dir} trials={len(rows)} accepted={len(accepted_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
