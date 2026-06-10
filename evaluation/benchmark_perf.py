# -*- coding: utf-8 -*-
"""
End-to-End Performance & VRAM Benchmark
=======================================

Đo hiệu năng pipeline trên video mẫu:
- Real-time Factor, RTF = video_duration / processing_time
- PyTorch Peak VRAM bằng torch.cuda.max_memory_allocated()
- Optional NVIDIA total VRAM delta bằng nvidia-smi

Lưu ý quan trọng:
- PyTorch Peak VRAM KHÔNG bao gồm VRAM của Ollama/GGML.
- Nếu cần đo tổng VRAM cả hệ thống/Ollama, bật --include-nvidia-smi.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydub import AudioSegment  # noqa: E402
from src.pipeline.orchestrator import PipelineOrchestrator  # noqa: E402

try:
    import torch
except Exception:  # pragma: no cover - benchmark should still explain missing torch
    torch = None  # type: ignore[assignment]


def get_media_duration_seconds(path: Path) -> float:
    """Return media duration using pydub/ffprobe backend."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy video benchmark: {path}")
    media = AudioSegment.from_file(path)
    return len(media) / 1000.0


def get_nvidia_total_vram_used_mb() -> int | None:
    """Return total used VRAM across NVIDIA GPUs via nvidia-smi, if available."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return sum(values) if values else None
    except Exception:
        return None


def reset_torch_peak_vram() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def get_torch_peak_vram_gb() -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def run_performance_benchmark(
    video_path: Path = EVAL_DIR / "Download.mp4",
    model_name: str = "gemma4:e4b",
    mode: str = "end_to_end",
    hardsub: bool = False,
    include_nvidia_smi: bool = False,
    keep_temp_segments: bool = False,
    output_json: Path | None = EVAL_DIR / "perf_results.json",
) -> dict[str, Any]:
    video_duration_sec = get_media_duration_seconds(video_path)

    print("=" * 72)
    print("END-TO-END PERFORMANCE & VRAM BENCHMARK")
    print("=" * 72)
    print(f"Video            : {video_path}")
    print(f"Video duration   : {video_duration_sec:.2f}s")
    print(f"Model            : {model_name}")
    print(f"Mode             : {mode}")
    print(f"Hardsub          : {hardsub}")

    base_torch_vram = reset_torch_peak_vram()
    base_nvidia_vram = get_nvidia_total_vram_used_mb() if include_nvidia_smi else None

    if base_torch_vram is None:
        print("PyTorch CUDA      : unavailable; PyTorch Peak VRAM will be N/A")
    else:
        print(f"PyTorch base VRAM : {base_torch_vram / (1024**2):.2f} MB")

    if include_nvidia_smi:
        if base_nvidia_vram is None:
            print("nvidia-smi        : unavailable")
        else:
            print(f"NVIDIA base VRAM  : {base_nvidia_vram} MB")

    start_time = time.perf_counter()
    orchestrator = PipelineOrchestrator()
    final_output = orchestrator.run_pipeline(
        source=str(video_path),
        hardsub=hardsub,
        ollama_model=model_name,
        mode=mode,
        keep_temp_segments=keep_temp_segments,
    )
    processing_time_sec = time.perf_counter() - start_time

    rtf = video_duration_sec / processing_time_sec if processing_time_sec > 0 else 0.0
    torch_peak_gb = get_torch_peak_vram_gb()
    end_nvidia_vram = get_nvidia_total_vram_used_mb() if include_nvidia_smi else None
    nvidia_delta_mb = (
        end_nvidia_vram - base_nvidia_vram
        if end_nvidia_vram is not None and base_nvidia_vram is not None
        else None
    )

    results: dict[str, Any] = {
        "video": str(video_path),
        "video_duration_sec": video_duration_sec,
        "processing_time_sec": processing_time_sec,
        "real_time_factor": rtf,
        "torch_peak_vram_gb": torch_peak_gb,
        "nvidia_base_vram_mb": base_nvidia_vram,
        "nvidia_end_vram_mb": end_nvidia_vram,
        "nvidia_delta_vram_mb": nvidia_delta_mb,
        "model": model_name,
        "mode": mode,
        "hardsub": hardsub,
        "final_output": final_output,
    }

    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)
    print(f"Processing time  : {processing_time_sec:.2f}s")
    print(f"RTF              : {rtf:.2f}x")
    print(f"Target RTF       : {'PASS' if rtf > 1.5 else 'WARN'} (> 1.5x)")

    if torch_peak_gb is None:
        print("PyTorch Peak VRAM : N/A")
    else:
        print(f"PyTorch Peak VRAM : {torch_peak_gb:.2f} GB")
        print(f"Target PyTorch VRAM: {'PASS' if torch_peak_gb < 4.5 else 'WARN'} (< 4.5 GB)")

    if include_nvidia_smi:
        print(f"NVIDIA VRAM delta : {nvidia_delta_mb if nvidia_delta_mb is not None else 'N/A'} MB")
        print("Note             : NVIDIA delta is system-level and may include Ollama/other processes.")

    print(f"Final output     : {final_output}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON output      : {output_json}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end performance benchmark.")
    parser.add_argument("--video", type=Path, default=EVAL_DIR / "Download.mp4")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--mode", choices=["end_to_end", "translate_only"], default="end_to_end")
    parser.add_argument("--hardsub", action="store_true")
    parser.add_argument("--include-nvidia-smi", action="store_true")
    parser.add_argument("--keep-temp-segments", action="store_true")
    parser.add_argument("--output-json", type=Path, default=EVAL_DIR / "perf_results.json")
    args = parser.parse_args()

    run_performance_benchmark(
        video_path=args.video,
        model_name=args.model,
        mode=args.mode,
        hardsub=args.hardsub,
        include_nvidia_smi=args.include_nvidia_smi,
        keep_temp_segments=args.keep_temp_segments,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
