# -*- coding: utf-8 -*-
"""
Build Hybrid Evaluation Dataset
===============================

Creates evaluation/test_cases.json from:
- PhoST spoken translation data (local evaluation/data first, then Hugging Face)
- ViMedCSS medical code-switching data from tensorxt/ViMedCSS

Output schema is designed for LLM Judge localization benchmark:
- english_original: original English/source sentence when available
- input_raw: raw input to localize/evaluate
- ai_translation: pipeline or baseline translation/localization
- expected_localization: reference answer
- cs_terms_list: code-switching terms when available
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
DEFAULT_OUTPUT = EVAL_DIR / "test_cases.json"
DEFAULT_VIMEDCSS_DATASET = "tensorxt/ViMedCSS"
DEFAULT_PHOST_DATASET = "vinai/PhoST"


def _first_existing(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else list(data.values())
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return []


def _find_local_phost_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    patterns = [
        "*phost*.json", "*phost*.jsonl", "*phost*.csv",
        "*.json", "*.jsonl", "*.csv",
        "*.en", "*.vi",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.rglob(pattern))
    # stable unique order
    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _load_parallel_phost_pair(path: Path, limit: int) -> list[dict[str, Any]]:
    """Load PhoST parallel text files, e.g. evaluation/data/1188.en + 1188.vi."""
    if path.suffix.lower() not in {".en", ".vi"}:
        return []

    stem = path.with_suffix("")
    en_path = stem.with_suffix(".en")
    vi_path = stem.with_suffix(".vi")
    if not en_path.exists() or not vi_path.exists():
        return []

    en_lines = en_path.read_text(encoding="utf-8").splitlines()
    vi_lines = vi_path.read_text(encoding="utf-8").splitlines()
    count = min(len(en_lines), len(vi_lines), limit)
    cases: list[dict[str, Any]] = []
    for idx in range(count):
        en = en_lines[idx].strip()
        vi = vi_lines[idx].strip()
        if not en or not vi:
            continue
        cases.append({
            "id": f"phost_{len(cases)+1:03d}",
            "source": f"PhoST/local:{en_path.name}+{vi_path.name}",
            "domain": "speech_flow",
            "type": "spoken_translation",
            "input_raw": en,
            "english_original": en,
            "ai_translation": vi,
            "expected_localization": vi,
            "gold": vi,
            "predicted": vi,
        })
    return cases


def _rows_to_phost_cases(rows: list[dict[str, Any]], limit: int, source_name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        en = _first_existing(row, ["en", "english", "source", "src", "source_text", "text_en", "input", "input_raw"])
        vi = _first_existing(row, ["vi", "vietnamese", "target", "tgt", "target_text", "text_vi", "gold", "expected_localization"])
        if not en or not vi:
            continue
        cases.append({
            "id": f"phost_{len(cases)+1:03d}",
            "source": source_name,
            "domain": "speech_flow",
            "type": "spoken_translation",
            "input_raw": en,
            "english_original": en,
            "ai_translation": vi,
            "expected_localization": vi,
            "gold": vi,
            "predicted": vi,
        })
        if len(cases) >= limit:
            break
    return cases


def load_phost_cases(limit: int = 35, local_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    # Prefer user-provided local PhoST data in evaluation/data.
    for file in _find_local_phost_files(local_dir):
        if file.suffix.lower() in {".en", ".vi"}:
            cases = _load_parallel_phost_pair(file, limit)
            if cases:
                print(f"Loaded {len(cases)} PhoST cases from local parallel files near: {file}")
                return cases
            continue

        rows = _read_table(file)
        cases = _rows_to_phost_cases(rows, limit, f"PhoST/local:{file.name}")
        if cases:
            print(f"Loaded {len(cases)} PhoST cases from local file: {file}")
            return cases

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install datasets") from exc

    candidates = [
        (DEFAULT_PHOST_DATASET, "en-vi", "test"),
        (DEFAULT_PHOST_DATASET, None, "test"),
        (DEFAULT_PHOST_DATASET, "en-vi", "validation"),
        (DEFAULT_PHOST_DATASET, None, "validation"),
        (DEFAULT_PHOST_DATASET, "en-vi", "train"),
        (DEFAULT_PHOST_DATASET, None, "train"),
    ]
    last_error: Exception | None = None
    for dataset_id, config, split in candidates:
        try:
            ds = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
            cases = _rows_to_phost_cases(list(ds), limit, f"PhoST/{split}")
            if cases:
                return cases
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Could not load PhoST. Last error: {last_error}")


def _parse_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def load_vimedcss_rows(dataset_id: str, local_file: Path | None, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if local_file and local_file.exists():
        rows = _read_table(local_file)
        source_name = f"ViMedCSS/local:{local_file.name}"
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, split="train", streaming=True)
            rows = []
            for idx, row in enumerate(ds):
                rows.append(row)
                if len(rows) >= limit:
                    break
            source_name = f"{dataset_id}/streaming"
        except Exception as exc:
            print(f"[WARN] Could not load ViMedCSS dataset '{dataset_id}': {exc}")
            rows = []
            source_name = dataset_id

    cases: list[dict[str, Any]] = []
    for row in rows:
        segment = _first_existing(row, ["segment_text", "text", "sentence", "utterance"])
        if not segment:
            continue
        terms = _parse_terms(row.get("cs_terms_list"))
        case_id = _first_existing(row, ["segment_id", "id"]) or f"vimedcss_{len(cases)+1:03d}"
        cases.append({
            "id": str(case_id),
            "source": source_name,
            "domain": row.get("topic") or "medical_code_switching",
            "type": "medical_code_switching",
            "input_raw": segment,
            "english_original": segment,
            "cs_terms_list": terms,
            "cs_terms_count": row.get("cs_terms_count", len(terms)),
            "ai_translation": segment,
            "expected_localization": segment,
            "gold": segment,
            "predicted": segment,
            "original_video_link": row.get("original_video_link", ""),
            "start_time": row.get("start_time", ""),
            "end_time": row.get("end_time", ""),
        })
        if len(cases) >= limit:
            break
    return cases


def fallback_stress_cases() -> list[dict[str, Any]]:
    raw = [
        ("medical", "We use mRNA vaccines for COVID-19 prevention.", "Chúng tôi sử dụng vắc-xin em a ren a để phòng ngừa cô vít mười chín."),
        ("medical", "Vitamin B12 deficiency affects red blood cells.", "Thiếu vi ta min bê mười hai ảnh hưởng đến hồng cầu."),
        ("medical", "H1N1 infection requires early monitoring.", "Nhiễm cúm hát một en một cần được theo dõi sớm."),
        ("finance", "P2P lending uses T+2 settlement.", "Cho vay pi tu pi sử dụng thanh toán tê cộng hai."),
        ("finance", "EBITDA margin improved this quarter.", "Biên lợi nhuận i bít đa đã cải thiện trong quý này."),
        ("aviation", "The A320 aircraft is ready.", "Máy bay a ba hai không đã sẵn sàng."),
        ("logistics", "The 4PL provider handles FCL cargo.", "Nhà cung cấp four pê e lờ xử lý hàng hóa ép xê lờ."),
        ("it", "Implement OAuth2 with PostgreSQL.", "Triển khai ô o hai với pốt gờ rê ét qui eo."),
        ("it", "Configure Nginx to serve HTML5.", "Cấu hình en gin ích để phục vụ ét ch em e lăm."),
        ("it", "Deploy BM25 and GraphQL on GitHub.", "Triển khai bi em hai lăm và gờ ráp qui eo trên gít húp."),
        ("it", "Use FFmpeg to convert MP4 to WebP.", "Sử dụng ép ép em pếch để chuyển đổi em pi pho sang wép pi."),
        ("business", "B2B sales teams track CRM data.", "Đội bán hàng bi tu bi theo dõi dữ liệu xi a em."),
        ("it", "LLM agents run on CUDA.", "Các tác nhân eo eo em chạy trên cu đa."),
        ("it", "Connect via SSH protocol.", "Kết nối qua giao thức ét ét hát."),
        ("it", "Scale the Kubernetes cluster.", "Mở rộng cụm cu bơ nê tịt."),
    ]
    return [
        {
            "id": f"stress_{idx+1:03d}",
            "source": "curated_stress",
            "domain": domain,
            "type": "alphanumeric_oov",
            "input_raw": src,
            "english_original": src,
            "ai_translation": gold,
            "expected_localization": gold,
            "gold": gold,
            "predicted": gold,
        }
        for idx, (domain, src, gold) in enumerate(raw)
    ]


def build_dataset(phost_limit: int, stress_limit: int, vimedcss_dataset: str, vimedcss_file: Path | None, output: Path) -> list[dict[str, Any]]:
    cases = load_phost_cases(phost_limit)
    stress = load_vimedcss_rows(vimedcss_dataset, vimedcss_file, stress_limit)
    if not stress:
        print("[WARN] ViMedCSS rows unavailable. Falling back to curated multi-domain stress cases.")
        stress = fallback_stress_cases()[:stress_limit]
    cases.extend(stress[:stress_limit])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hybrid localization benchmark dataset.")
    parser.add_argument("--phost-limit", type=int, default=35)
    parser.add_argument("--stress-limit", type=int, default=15)
    parser.add_argument("--vimedcss-dataset", default=os.environ.get("VIMEDCSS_DATASET_ID", DEFAULT_VIMEDCSS_DATASET))
    parser.add_argument("--vimedcss-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = build_dataset(args.phost_limit, args.stress_limit, args.vimedcss_dataset, args.vimedcss_file, args.output)
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"PhoST target: {args.phost_limit}; ViMedCSS/stress target: {args.stress_limit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
