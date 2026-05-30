from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import compact_text, ensure_dir, read_jsonl, write_json, write_jsonl
from run_deepseek_extraction_experiment import append_jsonl, call_deepseek, local_path, now_iso


JUDGE_VERSION = "semantic_audit_deepseek_judge_v1"
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


def packet_id(packet: dict[str, Any]) -> str:
    return str(packet.get("packet_id") or (packet.get("payload") or {}).get("record", {}).get("record_id") or "")


def load_completed(path: Path) -> set[str]:
    return {str(row.get("packet_id") or "") for row in read_jsonl(path) if row.get("packet_id")}


def compact_packet(packet: dict[str, Any], max_context_chars: int) -> list[dict[str, str]]:
    payload = packet.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    combined = payload.get("combined_context")
    if isinstance(combined, dict) and isinstance(combined.get("context"), str):
        combined["context"] = compact_text(combined["context"], max_context_chars)
    trial = payload.get("schema_trial")
    if isinstance(trial, dict) and isinstance(trial.get("output_text"), str):
        trial["output_text"] = compact_text(trial["output_text"], 800)
    system = (
        "You are a strict semantic judge for materials information extraction. "
        "Return only one valid JSON object, with no markdown."
    )
    user = (
        "Judge whether the normalized extraction record is truly supported by the evidence and surrounding context.\n"
        "Output exactly these fields: schema_scope_ok, property_ok, property_subtype_ok, material_ok, "
        "value_ok, unit_ok, condition_ok, evidence_support_ok, verdict, reason_code, reason, confidence.\n"
        "Each *_ok field must be boolean. verdict must be one of ACCEPT, PARTIAL, REJECT, NEEDS_SCHEMA_DECISION.\n"
        "Be conservative. Reject wrong table rows/columns, wrong material binding, wrong unit/value, plot coordinates, "
        "heating rates treated as temperatures, test temperatures treated as process temperatures, and weak evidence.\n"
        "For thermal decomposition/TGA/weight-loss/heat-distortion/LCST/UCST, use NEEDS_SCHEMA_DECISION unless an explicit schema subtype is present.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judgment(text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_parse_failed: {exc}"
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
    obj["reason"] = str(obj.get("reason") or "")
    return obj, None


def judge_one(packet: dict[str, Any], index: int, api_key: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    content, usage, api_error = call_deepseek(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        messages=compact_packet(packet, args.max_context_chars),
        max_tokens=args.max_tokens,
        temperature=0.0,
        timeout_sec=args.timeout_sec,
        thinking=args.thinking,
        retries=args.retries,
    )
    judgment, parse_error = (None, api_error) if api_error else parse_judgment(content)
    payload = packet.get("payload") or {}
    record = payload.get("record") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        record = {}
    return {
        "judge_version": JUDGE_VERSION,
        "time": now_iso(),
        "index": index,
        "packet_id": packet_id(packet),
        "record_id": record.get("record_id") or packet_id(packet),
        "doc_id": record.get("doc_id"),
        "property": record.get("property"),
        "verdict": judgment,
        "parse_error": parse_error,
        "output_text": content,
        "usage": usage,
        "elapsed_sec": round(time.time() - started, 3),
        "record": record,
    }


def external_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    verdict = row.get("verdict") or {}
    return {
        "audit_version": JUDGE_VERSION,
        "record_id": row.get("record_id"),
        "verdict": verdict.get("verdict") if isinstance(verdict, dict) else None,
        "reason_code": verdict.get("reason_code") if isinstance(verdict, dict) else row.get("parse_error"),
        "reason": verdict.get("reason") if isinstance(verdict, dict) else row.get("parse_error"),
        "confidence": verdict.get("confidence") if isinstance(verdict, dict) else 0.0,
        "checks": {field: verdict.get(field) for field in CHECK_FIELDS} if isinstance(verdict, dict) else {},
        "record": row.get("record"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if isinstance(row.get("verdict"), dict)]
    invalid = [row for row in rows if not isinstance(row.get("verdict"), dict)]
    verdict_counts = Counter((row.get("verdict") or {}).get("verdict") for row in valid)
    by_property: dict[str, Counter] = {}
    for row in valid:
        prop = str(row.get("property") or "")
        by_property.setdefault(prop, Counter())
        by_property[prop][str((row.get("verdict") or {}).get("verdict"))] += 1
    return {
        "time": now_iso(),
        "judge_version": JUDGE_VERSION,
        "total": len(rows),
        "valid": len(valid),
        "invalid": len(invalid),
        "verdict_counts": dict(verdict_counts),
        "verdict_by_property": {prop: dict(counts) for prop, counts in sorted(by_property.items())},
        "parse_errors": dict(Counter(str(row.get("parse_error") or "unknown") for row in invalid).most_common()),
        "reason_codes": dict(Counter(str((row.get("verdict") or {}).get("reason_code") or "unknown") for row in valid).most_common()),
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = ["# Semantic Audit Judge", "", "## Summary", ""]
    for key in ["total", "valid", "invalid"]:
        lines.append(f"- {key}: {summary.get(key)}")
    lines.append("")
    lines.append("## Verdict Counts")
    lines.append("")
    for verdict, count in (summary.get("verdict_counts") or {}).items():
        lines.append(f"- {verdict}: {count}")
    lines.append("")
    lines.append("## Rejected / Partial Examples")
    lines.append("")
    for row in [r for r in rows if (r.get("verdict") or {}).get("verdict") in {"REJECT", "PARTIAL", "NEEDS_SCHEMA_DECISION"}][:20]:
        verdict = row.get("verdict") or {}
        record = row.get("record") or {}
        lines.append(
            f"- {verdict.get('verdict')} doc {record.get('doc_id')} p{record.get('page_range')} "
            f"{record.get('property')} | {record.get('material')} | {record.get('value')} {record.get('unit')} | "
            f"{verdict.get('reason_code')}: {compact_text(verdict.get('reason') or '', 180)}"
        )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an AI semantic judge on material audit packets.")
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="disabled")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=2800)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} before running.")

    packets_path = local_path(args.packets)
    output_dir = ensure_dir(local_path(args.output_dir))
    trials_path = output_dir / "judge_trials.jsonl"
    audit_results_path = output_dir / "judge_audit_results.jsonl"
    summary_path = output_dir / "judge_summary.json"
    report_path = output_dir / "judge_report.md"
    log_path = output_dir / "run.log"

    packets = read_jsonl(packets_path)
    if args.limit > 0:
        packets = packets[: args.limit]
    completed = load_completed(trials_path)
    pending = [packet for packet in packets if packet_id(packet) not in completed]
    append_jsonl(log_path, {"event": "start", "time": now_iso(), "packets": len(packets), "completed": len(completed), "pending": len(pending)})

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(judge_one, packet, index, api_key, args) for index, packet in enumerate(pending, start=1)]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                append_jsonl(trials_path, row)
                append_jsonl(log_path, {"event": "item_done", "time": now_iso(), "packet_id": row.get("packet_id"), "verdict": (row.get("verdict") or {}).get("verdict"), "parse_error": row.get("parse_error"), "elapsed_sec": row.get("elapsed_sec")})

    wanted = {packet_id(packet) for packet in packets}
    rows = [row for row in read_jsonl(trials_path) if str(row.get("packet_id") or "") in wanted]
    write_jsonl(trials_path, rows)
    write_jsonl(audit_results_path, [external_audit_row(row) for row in rows if isinstance(row.get("verdict"), dict)])
    summary = summarize(rows)
    summary.update({"packets": str(packets_path), "output_dir": str(output_dir), "audit_results": str(audit_results_path)})
    write_json(summary_path, summary)
    write_report(report_path, summary, rows)
    append_jsonl(log_path, {"event": "finished", **summary})
    print(f"semantic_judge={output_dir} total={summary['total']} valid={summary['valid']} invalid={summary['invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
