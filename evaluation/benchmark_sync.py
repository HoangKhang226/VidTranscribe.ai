# -*- coding: utf-8 -*-
"""
Audio-Video Sync Offset Benchmark
=================================

Đánh giá độ khớp thời gian giữa audio TTS tiếng Việt và timestamp SRT.

Metric chính:
    MAE = mean(max(0, audio_duration_ms - allowed_window_ms))

Nếu audio ngắn hơn hoặc bằng cửa sổ SRT, overflow = 0ms.
Nếu audio dài hơn cửa sổ SRT, phần vượt bị tính là sync offset.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydub import AudioSegment

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_DIR, SUBTITLES_DIR  # noqa: E402
from src.utils.srt_utils import SRTEntry, parse_srt  # noqa: E402


def candidate_segment_names(index: int) -> list[str]:
    """Support current and legacy segment naming conventions."""
    return [
        f"segment_{index}.mp3",
        f"temp_segment_{index}.mp3",
        f"temp_group_{index}.mp3",
        f"{index}.mp3",
    ]


def find_audio_segment(segment_dir: Path, index: int) -> Path | None:
    for name in candidate_segment_names(index):
        path = segment_dir / name
        if path.exists():
            return path

    # Fallback: any file ending with _{index}.mp3 or containing the index as a token.
    token_pattern = re.compile(rf"(^|[_-]){index}($|[_-])")
    for path in segment_dir.glob("*.mp3"):
        stem = path.stem
        if token_pattern.search(stem):
            return path
    return None


def allowed_window_ms(entry: SRTEntry) -> int:
    """Use the actual SRTEntry fields in this project."""
    return max(0, int(entry.end_ms) - int(entry.start_ms))


def run_sync_benchmark(
    srt_path: Path = Path(SUBTITLES_DIR) / "subtitles_vi.srt",
    segment_dir: Path = Path(AUDIO_DIR) / "temp_segments",
    output_csv: Path = EVAL_DIR / "sync_results.csv",
    output_json: Path | None = EVAL_DIR / "sync_results.json",
    target_mae_ms: float = 150.0,
) -> dict[str, Any]:
    if not srt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy SRT: {srt_path}")
    if not segment_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục audio segments: {segment_dir}")

    subtitles = parse_srt(str(srt_path))
    if not subtitles:
        raise ValueError(f"Không parse được subtitle nào từ: {srt_path}")

    rows: list[dict[str, Any]] = []
    missing_segments = 0
    total_overflow = 0
    max_overflow = 0
    valid_count = 0

    print("=" * 72)
    print("AUDIO-VIDEO SYNC OFFSET BENCHMARK")
    print("=" * 72)
    print(f"SRT file         : {srt_path}")
    print(f"Segment dir      : {segment_dir}")
    print(f"Subtitle entries : {len(subtitles)}")

    for entry in subtitles:
        audio_path = find_audio_segment(segment_dir, entry.index)
        allowed_ms = allowed_window_ms(entry)

        if audio_path is None:
            missing_segments += 1
            rows.append(
                {
                    "index": entry.index,
                    "audio_file": "",
                    "allowed_ms": allowed_ms,
                    "audio_duration_ms": "",
                    "overflow_ms": "",
                    "status": "MISSING",
                }
            )
            continue

        audio = AudioSegment.from_file(audio_path)
        audio_duration_ms = len(audio)
        overflow_ms = max(0, audio_duration_ms - allowed_ms)

        total_overflow += overflow_ms
        max_overflow = max(max_overflow, overflow_ms)
        valid_count += 1

        rows.append(
            {
                "index": entry.index,
                "audio_file": str(audio_path),
                "allowed_ms": allowed_ms,
                "audio_duration_ms": audio_duration_ms,
                "overflow_ms": overflow_ms,
                "status": "PASS" if overflow_ms <= target_mae_ms else "OVERFLOW",
            }
        )

    mae = total_overflow / valid_count if valid_count else 0.0
    results: dict[str, Any] = {
        "srt_path": str(srt_path),
        "segment_dir": str(segment_dir),
        "subtitle_count": len(subtitles),
        "valid_segments": valid_count,
        "missing_segments": missing_segments,
        "mae_ms": mae,
        "max_overflow_ms": max_overflow,
        "target_mae_ms": target_mae_ms,
        "target_pass": mae < target_mae_ms,
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "audio_file", "allowed_ms", "audio_duration_ms", "overflow_ms", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 72)
    print(f"Valid segments   : {valid_count}")
    print(f"Missing segments : {missing_segments}")
    print(f"MAE overflow     : {mae:.2f} ms")
    print(f"Max overflow     : {max_overflow} ms")
    print(f"Target           : {'PASS' if mae < target_mae_ms else 'WARN'} (MAE < {target_mae_ms:.0f}ms)")
    print(f"CSV output       : {output_csv}")
    if output_json:
        print(f"JSON output      : {output_json}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run audio-video sync offset benchmark.")
    parser.add_argument("--srt", type=Path, default=Path(SUBTITLES_DIR) / "subtitles_vi.srt")
    parser.add_argument("--segments", type=Path, default=Path(AUDIO_DIR) / "temp_segments")
    parser.add_argument("--output-csv", type=Path, default=EVAL_DIR / "sync_results.csv")
    parser.add_argument("--output-json", type=Path, default=EVAL_DIR / "sync_results.json")
    parser.add_argument("--target-mae-ms", type=float, default=150.0)
    args = parser.parse_args()

    run_sync_benchmark(
        srt_path=args.srt,
        segment_dir=args.segments,
        output_csv=args.output_csv,
        output_json=args.output_json,
        target_mae_ms=args.target_mae_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
