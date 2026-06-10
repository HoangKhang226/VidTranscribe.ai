# -*- coding: utf-8 -*-
"""
G2P Accuracy Benchmark
======================

Đánh giá độ chính xác của module G2P/phonetic localization bằng cách so sánh
kết quả `transliterate_word()` với ground-truth do con người gán nhãn.

Metric chính:
    Word-level Accuracy = PASS / total * 100

Một term được tính PASS nếu SequenceMatcher similarity >= threshold.
Mặc định threshold = 0.80 để phản ánh mức "đủ giống để TTS đọc chấp nhận được".
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.g2p_helper import transliterate_word  # noqa: E402


def similarity_ratio(predicted: str, expected: str) -> float:
    """Return character-level similarity ratio in [0, 1]."""
    return difflib.SequenceMatcher(None, predicted.strip(), expected.strip()).ratio()


def load_terms(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy test terms: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ground_truth(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ground truth: {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ground_truth.json phải là object dạng {term: phonetic}")
    return {str(k): str(v) for k, v in data.items()}


def run_benchmark(
    terms_file: Path = EVAL_DIR / "test_terms.txt",
    ground_truth_file: Path = EVAL_DIR / "ground_truth.json",
    output_csv: Path = EVAL_DIR / "g2p_results.csv",
    threshold: float = 0.80,
) -> dict[str, Any]:
    terms = load_terms(terms_file)
    ground_truth = load_ground_truth(ground_truth_file)

    rows: list[dict[str, Any]] = []
    pass_count = 0
    evaluated_count = 0

    print("=" * 72)
    print("G2P ACCURACY BENCHMARK")
    print("=" * 72)
    print(f"Terms file       : {terms_file}")
    print(f"Ground truth     : {ground_truth_file}")
    print(f"Similarity cutoff: {threshold:.2f}")
    print(f"Total input terms: {len(terms)}")

    for term in terms:
        predicted = transliterate_word(term)
        expected = ground_truth.get(term)

        if expected is None:
            status = "N/A"
            score = None
        else:
            score = similarity_ratio(predicted, expected)
            status = "PASS" if score >= threshold else "FAIL"
            evaluated_count += 1
            if status == "PASS":
                pass_count += 1

        rows.append(
            {
                "term": term,
                "predicted_phonetic": predicted,
                "expected_phonetic": expected or "",
                "similarity": "" if score is None else f"{score:.4f}",
                "status": status,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["term", "predicted_phonetic", "expected_phonetic", "similarity", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = (pass_count / evaluated_count * 100.0) if evaluated_count else 0.0
    failed_rows = [row for row in rows if row["status"] == "FAIL"]

    print("-" * 72)
    print(f"Evaluated terms  : {evaluated_count}")
    print(f"PASS             : {pass_count}")
    print(f"FAIL             : {len(failed_rows)}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print(f"CSV output       : {output_csv}")
    print("-" * 72)

    if failed_rows:
        print("Top failures:")
        for row in failed_rows[:10]:
            print(
                f"  - {row['term']}: predicted='{row['predicted_phonetic']}', "
                f"expected='{row['expected_phonetic']}', sim={row['similarity']}"
            )

    if accuracy >= 95.0:
        print("[PASS] Target đạt: Accuracy >= 95.0%")
    else:
        print("[WARN] Target chưa đạt: Accuracy < 95.0%")

    return {
        "total_terms": len(terms),
        "evaluated_terms": evaluated_count,
        "pass": pass_count,
        "fail": len(failed_rows),
        "accuracy": accuracy,
        "output_csv": str(output_csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run G2P accuracy benchmark.")
    parser.add_argument("--terms", type=Path, default=EVAL_DIR / "test_terms.txt")
    parser.add_argument("--ground-truth", type=Path, default=EVAL_DIR / "ground_truth.json")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "g2p_results.csv")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    run_benchmark(args.terms, args.ground_truth, args.output, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
