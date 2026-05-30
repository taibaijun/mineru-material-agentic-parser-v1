from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PDF_ROOT = Path("/mnt/d/2026MinerU赛事附件/材料赛题/材料文献/赛题数据")
DEFAULT_WORK_ROOT = Path("/mnt/d/中兴内核优化比赛/material_pipeline_run")
DEFAULT_MINERU_BIN = Path("/home/ubuntu/miniforge3/envs/cudabase/bin/mineru")
DEFAULT_MODELS_DIR = Path("/mnt/c/Users/Administrator/mineru")
DEFAULT_CONFIG_PATH = Path("/tmp/mineru_local_config.json")


def numeric_pdf_key(path: Path) -> tuple[int, str]:
    if path.stem.isdigit():
        return int(path.stem), path.name
    return 10**12, path.name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_text(text: str, limit: int | None = None) -> str:
    text = " ".join(text.split())
    return text[:limit] if limit is not None else text


def infer_language_hint(text: str) -> tuple[str, float]:
    """Return MinerU language hint and a rough confidence from sampled text."""
    if not text:
        return "en", 0.0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = max(cjk + latin, 1)
    cjk_ratio = cjk / total
    if cjk >= 30 and cjk_ratio >= 0.20:
        return "ch", round(cjk_ratio, 3)
    return "en", round(1.0 - cjk_ratio, 3)


def create_mineru_config(config_path: Path, models_dir: Path) -> None:
    """Write UTF-8-no-BOM MinerU local model config."""
    ensure_dir(config_path.parent)
    config = {"models-dir": {"vlm": str(models_dir)}}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def find_mineru_markdown(output_dir: Path, doc_id: str) -> Path | None:
    candidates = [
        output_dir / doc_id / "vlm" / f"{doc_id}.md",
        output_dir / doc_id / "auto" / f"{doc_id}.md",
    ]
    candidates.extend(sorted(output_dir.glob(f"*/vlm/{doc_id}.md")))
    candidates.extend(sorted(output_dir.glob("*/*/*.md")))
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def rel_link(from_file: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_file.parent).replace("\\", "/")
