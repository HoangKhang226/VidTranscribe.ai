# -*- coding: utf-8 -*-
"""
Group-level Audio/Video Sync Benchmark
=====================================

Đo sync đúng theo tầng TTS grouping của pipeline. Script ưu tiên đọc metadata
`src/output/audio/temp_segments/tts_groups.json` do step5_tts.py sinh khi chạy
pipeline với `--keep-temp-segments`.

Metric:
    overflow_ms = max(0, audio_duration_ms - allowed_window_ms)
    MAE = mean(overflow_ms)

Target CV/production baseline:
    MAE < 150ms
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from pydub import AudioSegment

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_DIR  # noqa: E402


def load_group_metadata(groups_json: Path) -> list[dict[str, Any]]:
    if not groups_json.exists():
        raise FileNotFoundError(
            f"Không tìm thấy group metadata: {groups_json}\n"
            "Hãy chạy full pipeline với: evaluation/benchmark_perf.py --mode end_to_end --keep-temp-segments"
        )
    data = json.loads(groups_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tts_groups.json phải là list group metadata")
    return data


def run_group_sync_benchmark(
    groups_json: Path = Path(AUDIO_DIR) / "temp_segments" / "tts_groups.json",
    output_csv: Path = EVAL_DIR / "sync_group_results.csv",
    output_json: Path = EVAL_DIR / "sync_group_results.json",
    target_mae_ms: float = 150.0,
) -> dict[str, Any]:
    groups = load_group_metadata(groups_json)

    rows: list[dict[str, Any]] = []
    total_overflow = 0
    max_overflow = 0
    missing_audio = 0
    valid_count = 0

    print("=" * 72)
    print("GROUP-LEVEL AUDIO/VIDEO SYNC BENCHMARK")
    print("=" * 72)
    print(f"Group metadata   : {groups_json}")
    print(f"Total groups     : {len(groups)}")

    for group in groups:
        group_index = int(group.get("group_index", -1))
        audio_file = Path(str(group.get("audio_file", "")))
        if not audio_file.is_absolute():
            audio_file = (groups_json.parent / audio_file).resolve()

        start_ms = int(group.get("start_ms", 0))
        end_ms = int(group.get("end_ms", 0))
        allowed_ms = int(group.get("allowed_window_ms") or max(0, end_ms - start_ms))

        if not audio_file.exists():
            missing_audio += 1
            rows.append({
                "group_index": group_index,
                "subtitle_indices": json.dumps(group.get("subtitle_indices", []), ensure_ascii=False),
                "audio_file": str(audio_file),
                "allowed_ms": allowed_ms,
                "audio_duration_ms": "",
                "overflow_ms": "",
                "status": "MISSING_AUDIO",
            })
            continue

        audio_duration_ms = len(AudioSegment.from_file(audio_file))
        overflow_ms = max(0, audio_duration_ms - allowed_ms)
        total_overflow += overflow_ms
        max_overflow = max(max_overflow, overflow_ms)
        valid_count += 1

        rows.append({
            "group_index": group_index,
            "subtitle_indices": json.dumps(group.get("subtitle_indices", []), ensure_ascii=False),
            "audio_file": str(audio_file),
            "allowed_ms": allowed_ms,
            "audio_duration_ms": audio_duration_ms,
            "overflow_ms": overflow_ms,
            "status": "PASS" if overflow_ms <= target_mae_ms else "OVERFLOW",
        })

    mae = total_overflow / valid_count if valid_count else 0.0
    result = {
        "groups_json": str(groups_json),
        "total_groups": len(groups),
        "valid_groups": valid_count,
        "missing_audio": missing_audio,
        "mae_overflow_ms": mae,
        "max_overflow_ms": max_overflow,
        "target_mae_ms": target_mae_ms,
        "target_pass": mae < target_mae_ms,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group_index"])
        writer.writeheader()
        writer.writerows(rows)

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Valid groups     : {valid_count}")
    print(f"Missing audio    : {missing_audio}")
    print(f"MAE overflow     : {mae:.2f} ms")
    print(f"Max overflow     : {max_overflow} ms")
    print(f"Target           : {'PASS' if mae < target_mae_ms else 'WARN'} (MAE < {target_mae_ms:.0f}ms)")
    print(f"CSV output       : {output_csv}")
    print(f"JSON output      : {output_json}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run group-level sync benchmark.")
    parser.add_argument("--groups-json", type=Path, default=Path(AUDIO_DIR) / "temp_segments" / "tts_groups.json")
    parser.add_argument("--output-csv", type=Path, default=EVAL_DIR / "sync_group_results.csv")
    parser.add_argument("--output-json", type=Path, default=EVAL_DIR / "sync_group_results.json")
    parser.add_argument("--target-mae-ms", type=float, default=150.0)
    args = parser.parse_args()

    run_group_sync_benchmark(args.groups_json, args.output_csv, args.output_json, args.target_mae_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
