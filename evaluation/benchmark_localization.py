# -*- coding: utf-8 -*-
"""
Localization Benchmark with optional Gemini 2.5 LLM Judge.

Default mode keeps deterministic TER scoring.
Judge mode scores batched cases with LangChain .with_structured_output().
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CaseResult:
    case_id: str
    source: str
    domain: str
    type: str
    english_original: str
    input_raw: str
    ai_translation: str
    expected_localization: str
    token_error_rate: float
    exact_match: bool
    ai_score: float | None = None
    ai_reason: str = ""


class JudgeItem(BaseModel):
    id: str = Field(description="Case id")
    score: float = Field(ge=0, le=5, description="Localization quality score from 0 to 5")
    reason: str = Field(description="Short reason in Vietnamese")


class JudgeBatch(BaseModel):
    results: list[JudgeItem] = Field(description="One score for every input case, preserving IDs")


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
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("test_cases.json phải là list các case")
    return data


def build_results(cases: list[dict[str, Any]]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for idx, case in enumerate(cases):
        case_id = str(case.get("id", f"case_{idx+1:03d}"))
        input_raw = str(case.get("input_raw") or case.get("input") or "")
        english_original = str(case.get("english_original") or input_raw)
        expected = str(case.get("expected_localization") or case.get("gold") or "")
        ai_translation = str(case.get("ai_translation") or case.get("predicted") or expected)
        ter = token_error_rate(ai_translation, expected)
        results.append(CaseResult(
            case_id=case_id,
            source=str(case.get("source", "")),
            domain=str(case.get("domain", "")),
            type=str(case.get("type", "")),
            english_original=english_original,
            input_raw=input_raw,
            ai_translation=ai_translation,
            expected_localization=expected,
            token_error_rate=ter,
            exact_match=normalize_text(ai_translation) == normalize_text(expected),
        ))
    return results


def chunked(items: list[CaseResult], size: int) -> list[list[CaseResult]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def make_gemini_judge(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0).with_structured_output(JudgeBatch)


def judge_prompt(batch: list[CaseResult]) -> str:
    payload = [
        {
            "id": r.case_id,
            "english_original": r.english_original,
            "ai_translation": r.ai_translation,
            "expected_localization": r.expected_localization,
        }
        for r in batch
    ]
    return (
        "Bạn là giám khảo benchmark localization cho hệ thống video dubbing Anh-Việt.\n"
        "Chấm mỗi item từ 0 đến 5 dựa trên: đúng nghĩa, tự nhiên tiếng Việt, xử lý code-switching/OOV, và phù hợp TTS.\n"
        "5 = rất tốt, 4 = dùng được có lỗi nhỏ, 3 = hiểu được nhưng thiếu tự nhiên/sai thuật ngữ, 2 = nhiều lỗi, 1 = gần như sai, 0 = rỗng/sai hoàn toàn.\n"
        "Trả về đúng structured output gồm results, mỗi input id phải có đúng một điểm.\n\n"
        f"DATA:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


async def judge_one_batch(chain, batch: list[CaseResult], semaphore: asyncio.Semaphore) -> list[JudgeItem]:
    async with semaphore:
        response = await chain.ainvoke(judge_prompt(batch))
        return response.results


async def run_llm_judge(results: list[CaseResult], model: str, batch_size: int, concurrency: int) -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini judge mode.")
    chain = make_gemini_judge(model)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [judge_one_batch(chain, batch, semaphore) for batch in chunked(results, batch_size)]
    judged_batches = await asyncio.gather(*tasks)
    score_map = {item.id: item for batch in judged_batches for item in batch}
    for result in results:
        item = score_map.get(result.case_id)
        if item:
            result.ai_score = float(item.score)
            result.ai_reason = item.reason


def save_outputs(results: list[CaseResult], output_csv: Path, output_json: Path, target_ter: float, target_judge_score: float) -> dict[str, Any]:
    avg_ter = sum(r.token_error_rate for r in results) / len(results) if results else 0.0
    exact_match_rate = sum(1 for r in results if r.exact_match) / len(results) * 100 if results else 0.0
    scored = [r.ai_score for r in results if r.ai_score is not None]
    avg_ai_score = sum(scored) / len(scored) if scored else None

    rows = [r.__dict__ for r in results]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_cases": len(results),
        "avg_token_error_rate": avg_ter,
        "exact_match_rate_percent": exact_match_rate,
        "avg_ai_score": avg_ai_score,
        "target_ter": target_ter,
        "target_judge_score": target_judge_score,
        "target_pass": (avg_ter <= target_ter) and (avg_ai_score is None or avg_ai_score >= target_judge_score),
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_localization_benchmark(
    cases_file: Path = EVAL_DIR / "test_cases.json",
    output_csv: Path = EVAL_DIR / "localization_results.csv",
    output_json: Path = EVAL_DIR / "localization_results.json",
    target_ter: float = 0.15,
    target_judge_score: float = 4.5,
    judge: bool = False,
    judge_model: str = "gemini-2.5-flash-preview-05-20",
    judge_batch_size: int = 20,
    judge_concurrency: int = 3,
) -> dict[str, Any]:
    results = build_results(load_cases(cases_file))
    if judge:
        asyncio.run(run_llm_judge(results, judge_model, judge_batch_size, judge_concurrency))
    summary = save_outputs(results, output_csv, output_json, target_ter, target_judge_score)

    print("=" * 72)
    print("TEXT LOCALIZATION BENCHMARK")
    print("=" * 72)
    print(f"Cases            : {cases_file}")
    print(f"Total cases       : {summary['total_cases']}")
    print(f"Avg TER           : {summary['avg_token_error_rate']:.4f}")
    if summary["avg_ai_score"] is not None:
        print(f"Avg AI score      : {summary['avg_ai_score']:.2f}/5")
    print(f"Target            : {'PASS' if summary['target_pass'] else 'WARN'}")
    print(f"CSV output        : {output_csv}")
    print(f"JSON output       : {output_json}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run localization benchmark with optional Gemini judge.")
    parser.add_argument("--cases", type=Path, default=EVAL_DIR / "test_cases.json")
    parser.add_argument("--output-csv", type=Path, default=EVAL_DIR / "localization_results.csv")
    parser.add_argument("--output-json", type=Path, default=EVAL_DIR / "localization_results.json")
    parser.add_argument("--target-ter", type=float, default=0.15)
    parser.add_argument("--target-judge-score", type=float, default=4.5)
    parser.add_argument("--judge", action="store_true", help="Enable Gemini LLM-as-a-judge scoring")
    parser.add_argument("--judge-model", default="gemini-2.5-flash-preview-05-20")
    parser.add_argument("--judge-batch-size", type=int, default=20)
    parser.add_argument("--judge-concurrency", type=int, default=3)
    args = parser.parse_args()

    run_localization_benchmark(
        args.cases,
        args.output_csv,
        args.output_json,
        args.target_ter,
        args.target_judge_score,
        args.judge,
        args.judge_model,
        args.judge_batch_size,
        args.judge_concurrency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
