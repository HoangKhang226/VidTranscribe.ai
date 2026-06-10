# -*- coding: utf-8 -*-
"""
Text Normalization & Localization Benchmark
===========================================

Đánh giá chất lượng sửa tiếng Việt bồi / code-switching trước khi đưa vào TTS.
Benchmark này so sánh output của pipeline với golden target bằng:
- normalized token error rate (WER-like)
- optional LLM-as-a-judge score nếu có file chấm tay

Input mặc định:
    evaluation/test_cases.json

Output:
    evaluation/localization_results.csv
    evaluation/localization_results.json

Định dạng test_cases.json hỗ trợ:
[
  {
    "id": "case_001",
    "input": "I am... studying AI agents trên Ollama.",
    "gold": "Tôi đang nghiên cứu các thực thể trí tuệ nhân tạo trên nền tảng Ô-la-ma.",
    "predicted": "Tôi đang nghiên cứu các tác nhân AI trên Ollama."
  }
]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CaseResult:
    case_id: str
    input_text: str
    gold_text: str
    predicted_text: str
    token_error_rate: float
    exact_match: bool


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s\u00a0]+", " ", text)
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        curr = [i]
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def token_error_rate(predicted: str, gold: str) -> float:
    pred_tokens = normalize_text(predicted).split()
    gold_tokens = normalize_text(gold).split()
    if not gold_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein(pred_tokens, gold_tokens) / max(1, len(gold_tokens))


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy test cases: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("test_cases.json phải là list các case")
    return data


def run_localization_benchmark(
    cases_file: Path = EVAL_DIR / "test_cases.json",
    output_csv: Path = EVAL_DIR / "localization_results.csv",
    output_json: Path = EVAL_DIR / "localization_results.json",
    target_ter: float = 0.15,
    target_judge_score: float = 4.5,
) -> dict[str, Any]:
    cases = load_cases(cases_file)

    results: list[CaseResult] = []
    for idx, case in enumerate(cases):
        case_id = str(case.get("id", f"case_{idx+1:03d}"))
        input_text = str(case.get("input", ""))
        gold_text = str(case.get("gold", ""))
        predicted_text = str(case.get("predicted", ""))
        ter = token_error_rate(predicted_text, gold_text)
        results.append(CaseResult(case_id, input_text, gold_text, predicted_text, ter, normalize_text(predicted_text) == normalize_text(gold_text)))

    avg_ter = sum(r.token_error_rate for r in results) / len(results) if results else 0.0
    exact_match_rate = sum(1 for r in results if r.exact_match) / len(results) * 100 if results else 0.0

    rows = [
        {
            "id": r.case_id,
            "input": r.input_text,
            "gold": r.gold_text,
            "predicted": r.predicted_text,
            "token_error_rate": f"{r.token_error_rate:.4f}",
            "exact_match": r.exact_match,
        }
        for r in results
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cases_file": str(cases_file),
        "total_cases": len(results),
        "avg_token_error_rate": avg_ter,
        "exact_match_rate_percent": exact_match_rate,
        "target_ter": target_ter,
        "target_pass": avg_ter <= target_ter,
        "target_judge_score": target_judge_score,
        "judge_score_note": "Add a manual judge score field if you want GPT/Claude-style evaluation.",
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("TEXT NORMALIZATION & LOCALIZATION BENCHMARK")
    print("=" * 72)
    print(f"Cases            : {cases_file}")
    print(f"Total cases       : {len(results)}")
    print(f"Avg TER           : {avg_ter:.4f}")
    print(f"Exact match rate  : {exact_match_rate:.2f}%")
    print(f"Target TER        : {'PASS' if avg_ter <= target_ter else 'WARN'} (<= {target_ter:.2f})")
    print(f"CSV output        : {output_csv}")
    print(f"JSON output       : {output_json}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run text normalization/localization benchmark.")
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "test_cases.json")
    parser.add_argument("--output-csv", type=Path, default=EVAL_DIR / "localization_results.csv")
    parser.add_argument("--output-json", type=Path, default=EVAL_DIR / "localization_results.json")
    parser.add_argument("--target-ter", type=float, default=0.15)
    parser.add_argument("--target-judge-score", type=float, default=4.5)
    args = parser.parse_args()

    run_localization_benchmark(args.cases, args.output_csv, args.output_json, args.target_ter, args.target_judge_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
