# -*- coding: utf-8 -*-
"""
VidTranscribe.ai project health test suite.

Mục tiêu:
- Test nhanh các phần ít tốn tài nguyên: config, DB JSON, cache, SRT, validator,
  adjudicator logic offline, API/FastAPI object, UI helper functions.
- KHÔNG chạy pipeline thật, Whisper, Edge-TTS, FFmpeg, Ollama trừ khi bật flag.

Cách chạy:
    venv\Scripts\python.exe scratch\test_project_health.py

Tùy chọn test import nặng/UI/API:
    venv\Scripts\python.exe scratch\test_project_health.py --include-heavy-imports

Tùy chọn test Ollama adjudicator thật:
    venv\Scripts\python.exe scratch\test_project_health.py --include-ollama
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class HealthRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: list[tuple[str, str]] = []

    def test(self, name: str, fn: Callable[[], None]) -> None:
        print(f"\n[TEST] {name}")
        try:
            fn()
        except SkipTest as exc:
            self.skipped += 1
            print(f"  SKIP: {exc}")
        except Exception:
            self.failed += 1
            tb = traceback.format_exc()
            self.failures.append((name, tb))
            print("  FAIL")
            print(tb)
        else:
            self.passed += 1
            print("  PASS")

    def summary(self) -> int:
        print("\n" + "=" * 80)
        print("VIDTRANSCRIBE.AI HEALTH TEST SUMMARY")
        print("=" * 80)
        print(f"PASS   : {self.passed}")
        print(f"FAIL   : {self.failed}")
        print(f"SKIP   : {self.skipped}")
        if self.failures:
            print("\nFailures:")
            for name, tb in self.failures:
                print(f"\n--- {name} ---")
                print(tb)
        return 1 if self.failed else 0


class SkipTest(Exception):
    pass


def assert_true(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def test_core_imports() -> None:
    modules = [
        "src.config",
        "src.utils.srt_utils",
        "src.utils.translation_cache",
        "src.utils.post_translation_validator",
        "src.db.db_manager",
        "src.pipeline.orchestrator",
    ]
    for module in modules:
        importlib.import_module(module)


def test_optional_heavy_imports() -> None:
    modules = [
        "src.api",
        "src.app",
        "src.pipeline.step2_context",
        "src.pipeline.step3_stt",
        "src.pipeline.step4_translate",
        "src.pipeline.step5_tts",
    ]
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            missing.append(f"{module}: missing {exc.name}")
    assert_true(not missing, "Missing dependencies:\n" + "\n".join(missing))


def test_config_paths() -> None:
    from src import config

    for attr in ["OUTPUT_DIR", "DOWNLOADS_DIR", "AUDIO_DIR", "SUBTITLES_DIR", "FINAL_DIR"]:
        path = Path(getattr(config, attr))
        assert_true(path.exists(), f"{attr} does not exist: {path}")
        assert_true(path.is_dir(), f"{attr} is not a directory: {path}")


def test_srt_roundtrip_and_overlap() -> None:
    from src.utils.srt_utils import SRTEntry, ms_to_srt_time, parse_srt, sanitize_srt_overlap, write_srt

    with tempfile.TemporaryDirectory(prefix="vidtranscribe_srt_") as tmp:
        srt_path = Path(tmp) / "sample.srt"
        entries = [
            SRTEntry(1, 0, 1200, "Hello world"),
            SRTEntry(2, 1100, 2000, "Overlap line"),
        ]
        write_srt(entries, str(srt_path))
        parsed = parse_srt(str(srt_path))
        assert_true(len(parsed) == 2, "SRT parse count mismatch")
        assert_true(parsed[0].text == "Hello world", "SRT text mismatch")
        assert_true(ms_to_srt_time(3723004) == "01:02:03,004", "Time format mismatch")

        sanitize_srt_overlap(str(srt_path), min_gap_ms=100)
        cleaned = parse_srt(str(srt_path))
        assert_true(cleaned[1].start_ms >= cleaned[0].end_ms + 100, "Overlap was not sanitized")


def test_translation_cache_isolated() -> None:
    from src.utils.translation_cache import TranslationCache

    with tempfile.TemporaryDirectory(prefix="vidtranscribe_cache_") as tmp:
        cache_file = Path(tmp) / "translation_cache.json"
        cache = TranslationCache(cache_file=str(cache_file))
        cache.set("Number", "Số", word_type="word")
        cache.set("API", "API", word_type="tech")
        cache.save_cache()

        reloaded = TranslationCache(cache_file=str(cache_file))
        assert_true(reloaded.get("number", "word") == "Số", "Word cache miss")
        assert_true(reloaded.get("api", "tech") == "API", "Tech cache miss")
        stats = reloaded.stats()
        assert_true(stats["total"] == 2, f"Unexpected cache stats: {stats}")


def test_db_manager_isolated_monkeypatch() -> None:
    import src.db.db_manager as dbm

    original_dict_dir = dbm.db.dict_dir
    original_cache_dir = dbm.db.cache_dir
    original_global_cache_path = dbm.db.global_cache_path

    with tempfile.TemporaryDirectory(prefix="vidtranscribe_db_") as tmp:
        tmp_path = Path(tmp)
        dbm.db.dict_dir = str(tmp_path / "dictionaries")
        dbm.db.cache_dir = str(tmp_path / "phonetic_caches")
        dbm.db.global_cache_path = str(tmp_path / "phonetic_caches" / "global_cache.json")
        Path(dbm.db.dict_dir).mkdir(parents=True, exist_ok=True)
        Path(dbm.db.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(dbm.db.global_cache_path).write_text("{}", encoding="utf-8")
        try:
            entry = dbm.db.upsert_term("General", "workflow", "quơ phờ lâu", "luồng công việc", "AI")
            assert_true(entry["phonetic"] == "quơ phờ lâu", "DB upsert phonetic mismatch")
            loaded = dbm.db.load_dictionary("General")
            assert_true("workflow" in loaded, "DB load missing term")
            dbm.db.save_to_domain_cache("General", "agent", "ây giơn")
            cache = dbm.db.load_phonetic_cache("General")
            assert_true(cache.get("agent") == "ây giơn", "Domain phonetic cache miss")
        finally:
            dbm.db.dict_dir = original_dict_dir
            dbm.db.cache_dir = original_cache_dir
            dbm.db.global_cache_path = original_global_cache_path


def test_post_translation_validator() -> None:
    from src.utils.post_translation_validator import PostTranslationValidator

    validator = PostTranslationValidator(domain_terms={"workflow", "agents", "machine learning"})
    vi = "Chúng kết nối với công cụ kinh doanh. Number two, workflow agents."
    en = "They connect to business tools. Number two, workflow agents."
    result = validator.find_foreign_words(vi, en)
    candidates = {c["clean"] for c in result["candidates"]}
    terms = {t["clean"] for t in result["terms"]}
    assert_true("kinh" not in candidates and "doanh" not in candidates, "Vietnamese no-accent words misdetected")
    assert_true({"number", "two"}.issubset(candidates), f"Expected number/two candidates, got {candidates}")
    assert_true({"workflow", "agents"}.issubset(terms), f"Expected workflow/agents terms, got {terms}")


def test_adjudicator_offline_logic() -> None:
    from src.utils.post_translation_validator import PostTranslationValidator
    from src.utils.translation_adjudicator import TranslationAdjudicator

    # Bypass __init__ để không khởi tạo ChatOllama trong test logic offline.
    adj = TranslationAdjudicator.__new__(TranslationAdjudicator)
    adj.domain_terms = {"agents"}
    adj.validator = PostTranslationValidator(adj.domain_terms)

    vi = "Loại một, general purpose task agents. Điều này giống như có"
    en = "Type one, general purpose task agents. This is like having"
    candidates = {c["clean"] for c in adj._find_candidates(vi, en)}
    assert_true("general" not in candidates, f"Protected term phrase leaked: {candidates}")
    assert_true("purpose" not in candidates, f"Protected term phrase leaked: {candidates}")
    assert_true("task" not in candidates, f"Protected term phrase leaked: {candidates}")

    out = TranslationAdjudicator.apply_verdicts(
        "Number two, workflow agents.",
        {
            "number": {"decision": "translate", "vietnamese": "Số"},
            "two": {"decision": "translate", "vietnamese": "hai"},
            "workflow": {"decision": "keep", "vietnamese": ""},
            "agents": {"decision": "keep", "vietnamese": ""},
        },
    )
    assert_true(out == "Số hai, workflow agents.", f"Verdict replacement mismatch: {out}")


def test_ui_dataframe_helpers() -> None:
    try:
        import src.app as app
    except ModuleNotFoundError as exc:
        raise SkipTest(f"UI dependency missing: {exc.name}") from exc

    df = app.load_dict_to_df("Auto Detect")
    assert_true(list(df.columns) == app.DICT_COLUMNS, "UI dictionary columns mismatch")
    msg = app.save_df_to_dict("Auto Detect", df)
    assert_true("Vui lòng chọn" in msg, f"Unexpected Auto Detect save message: {msg}")


def test_api_health_object() -> None:
    try:
        from fastapi.testclient import TestClient
        from src.api import app
    except Exception as exc:
        raise SkipTest(f"API dependency unavailable: {exc}") from exc

    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert_true(resp.status_code == 200, f"Health status != 200: {resp.status_code}")
    data = resp.json()
    assert_true(data.get("status") == "healthy", f"Unexpected health response: {data}")


def test_output_json_shape_if_present() -> None:
    path = PROJECT_ROOT / "src" / "output" / "subtitles" / "subtitles_vi.json"
    if not path.exists():
        raise SkipTest(f"No sample output JSON at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_true(isinstance(data, dict), "subtitles_vi.json must be an object keyed by index")
    if data:
        first_key = sorted(data.keys(), key=lambda x: int(x))[0]
        first = data[first_key]
        assert_true("translated_text" in first, "Each subtitle record should contain translated_text")


def test_environment_tools() -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert_true(ffmpeg is not None, "ffmpeg not found in PATH")


def test_pipeline_step_mocks_and_dataflow() -> None:
    from unittest.mock import MagicMock, patch
    from src.pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    orch.state["results"] = {}

    with patch("src.pipeline.orchestrator.step1_ingestion.run", return_value=("audio.wav", "video.mp4")) as p1, \
         patch("src.pipeline.orchestrator.ContextAnalyzer") as ctx_cls, \
         patch("src.pipeline.orchestrator.step3_stt.run", return_value="en.srt") as p3, \
         patch("src.pipeline.orchestrator.step4_translate.run", return_value="vi.srt") as p4, \
         patch("src.pipeline.orchestrator.step5_tts.run", return_value="audio_vi.wav") as p5, \
         patch("src.pipeline.orchestrator.step6_mux.run", return_value="final.mp4") as p6:

        ctx_instance = MagicMock()
        ctx_instance.run.return_value = {"topic": "General", "keywords": ["workflow"], "keywords_str": "workflow"}
        ctx_cls.return_value = ctx_instance

        audio_orig, video_mute = p1("sample.mp4")
        context_analyzer = ctx_cls(model_name="gemma4:e4b")
        context = context_analyzer.run(audio_orig, domain_override="Auto Detect")
        srt_en = p3(audio_orig, context)
        srt_vi = p4(srt_en, context, model_name="gemma4:e4b")
        audio_vi = p5(srt_vi, audio_orig)
        final_video = p6(video_mute, audio_vi, srt_vi, hardsub=False)

    orch.state["results"].update(
        {
            "audio_original": audio_orig,
            "video_no_audio": video_mute,
            "srt_en": srt_en,
            "srt_vi": srt_vi,
            "audio_vietnamese": audio_vi,
            "final_video": final_video,
        }
    )
    orch.state["status"] = "completed"

    assert_true(orch.state["results"]["audio_original"] == "audio.wav", "Step1 audio not stored")
    assert_true(orch.state["results"]["video_no_audio"] == "video.mp4", "Step1 video not stored")
    assert_true(orch.state["results"]["srt_en"] == "en.srt", "Step3 output missing")
    assert_true(orch.state["results"]["srt_vi"] == "vi.srt", "Step4 output missing")
    assert_true(orch.state["results"]["audio_vietnamese"] == "audio_vi.wav", "Step5 output missing")
    assert_true(orch.state["results"]["final_video"] == "final.mp4", "Step6 output missing")
    assert_true(orch.state["status"] == "completed", f"Pipeline status not completed: {orch.state}")
    assert_true(p1.call_count == 1 and p3.call_count == 1 and p4.call_count == 1 and p5.call_count == 1 and p6.call_count == 1, "One of pipeline steps was not called exactly once")
    assert_true(ctx_cls.call_count == 1, "ContextAnalyzer should be instantiated once")
    assert_true(ctx_instance.run.call_count == 1, "ContextAnalyzer.run should be called once")


def test_pipeline_translate_only_review_flow() -> None:
    from unittest.mock import MagicMock, patch
    from src.pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    orch.state["results"] = {}

    with patch("src.pipeline.orchestrator.step1_ingestion.run", return_value=("audio.wav", "video.mp4")) as p1, \
         patch("src.pipeline.orchestrator.ContextAnalyzer") as ctx_cls, \
         patch("src.pipeline.orchestrator.step3_stt.run", return_value="en.srt") as p3, \
         patch("src.pipeline.orchestrator.step4_translate.run", return_value="vi.srt") as p4:
        ctx_instance = MagicMock()
        ctx_instance.run.return_value = {"topic": "General", "keywords": [], "keywords_str": ""}
        ctx_cls.return_value = ctx_instance

        audio_orig, video_mute = p1("sample.mp4")
        context_analyzer = ctx_cls(model_name="gemma4:e4b")
        context = context_analyzer.run(audio_orig, domain_override="Auto Detect")
        srt_en = p3(audio_orig, context)
        srt_vi = p4(srt_en, context, model_name="gemma4:e4b")

    orch.state["results"].update(
        {
            "audio_original": audio_orig,
            "video_no_audio": video_mute,
            "srt_en": srt_en,
            "srt_vi": srt_vi,
        }
    )
    orch.state["status"] = "awaiting_review"

    assert_true(srt_vi == "vi.srt", f"Translate-only output mismatch: {srt_vi}")
    assert_true(orch.state["status"] == "awaiting_review", f"Translate-only status mismatch: {orch.state['status']}")
    assert_true("audio_vietnamese" not in orch.state["results"], "Translate-only flow should not call TTS")
    assert_true("final_video" not in orch.state["results"], "Translate-only flow should not mux video")
    assert_true(p1.call_count == 1 and p3.call_count == 1 and p4.call_count == 1, "Pipeline step mocks not called as expected")


def test_step1_to_step6_contracts_smoke() -> None:
    from unittest.mock import MagicMock, patch

    with patch("src.pipeline.step1_ingestion.subprocess.run") as run_ffmpeg, \
         patch("src.pipeline.step1_ingestion.os.path.exists", return_value=True), \
         patch("src.pipeline.step2_context.AudioSegment") as audio_cls, \
         patch("src.pipeline.step2_context.WhisperModel") as whisper_cls, \
         patch("src.pipeline.step2_context.requests.post") as post_req, \
         patch("src.pipeline.step3_stt.WhisperModel") as stt_model_cls, \
         patch("src.pipeline.step3_stt.write_srt") as write_srt_mock, \
         patch("src.pipeline.step4_translate.parse_srt", return_value=[]), \
         patch("src.pipeline.step5_tts.AudioSegment") as tts_audio_cls, \
         patch("src.pipeline.step5_tts.edge_tts.Communicate") as comm_cls, \
         patch("src.pipeline.step6_mux.subprocess.run") as mux_run:

        audio_cls.from_wav.return_value = MagicMock(__getitem__=lambda self, sl: self, export=lambda *a, **k: None)
        whisper_instance = MagicMock()
        whisper_instance.transcribe.return_value = (iter([MagicMock(text="hello", start=0.0, end=1.0)]), None)
        whisper_cls.return_value = whisper_instance
        post_req.return_value.json.return_value = {"response": json.dumps({"topic": "General", "keywords": ["api"]})}
        post_req.return_value.raise_for_status.return_value = None
        stt_instance = MagicMock()
        stt_instance.transcribe.return_value = (iter([MagicMock(text="hello", start=0.0, end=1.0)]), None)
        stt_model_cls.return_value = stt_instance
        write_srt_mock.return_value = "out.srt"
        tts_audio_cls.from_wav.return_value = MagicMock(__len__=lambda self: 1000, overlay=lambda *a, **k: None, export=lambda *a, **k: None)
        comm_instance = MagicMock()
        comm_instance.save.return_value = None
        comm_cls.return_value = comm_instance
        mux_run.return_value = MagicMock()

        assert_true(callable(run_ffmpeg), "step1 ffmpeg patch not callable")
        assert_true(callable(whisper_cls), "step2 whisper patch not callable")
        assert_true(callable(stt_model_cls), "step3 whisper patch not callable")
        assert_true(callable(tts_audio_cls), "step5 audio patch not callable")
        assert_true(callable(mux_run), "step6 mux patch not callable")


def test_ollama_adjudicator() -> None:
    import requests
    from src.utils.translation_adjudicator import TranslationAdjudicator

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        assert_true(resp.status_code == 200, f"Ollama not healthy: {resp.status_code}")
    except Exception as exc:
        raise SkipTest(f"Ollama is not reachable: {exc}") from exc

    adj = TranslationAdjudicator(domain_terms={"workflow", "agents"}, model_name="gemma4:e4b")
    verdicts = adj.adjudicate_sentence(
        "Number two, workflow agents.",
        "Number two, workflow agents.",
    )
    assert_true(isinstance(verdicts, dict), "Adjudicator verdicts must be dict")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-heavy-imports", action="store_true")
    parser.add_argument("--include-ollama", action="store_true")
    parser.add_argument("--include-env-tools", action="store_true")
    args = parser.parse_args()

    runner = HealthRunner()
    runner.test("Core imports", test_core_imports)
    runner.test("Config output paths", test_config_paths)
    runner.test("SRT roundtrip and overlap sanitizer", test_srt_roundtrip_and_overlap)
    runner.test("Translation cache isolated", test_translation_cache_isolated)
    runner.test("DB manager isolated", test_db_manager_isolated_monkeypatch)
    runner.test("Post translation validator", test_post_translation_validator)
    runner.test("Translation adjudicator offline logic", test_adjudicator_offline_logic)
    runner.test("UI dataframe helpers", test_ui_dataframe_helpers)
    runner.test("API health object", test_api_health_object)
    runner.test("Sample output JSON shape", test_output_json_shape_if_present)
    runner.test("Pipeline step mocks and dataflow", test_pipeline_step_mocks_and_dataflow)
    runner.test("Pipeline translate-only review flow", test_pipeline_translate_only_review_flow)
    runner.test("Step1 to Step6 contracts smoke", test_step1_to_step6_contracts_smoke)

    if args.include_heavy_imports:
        runner.test("Heavy module imports", test_optional_heavy_imports)
    if args.include_env_tools:
        runner.test("Environment tools", test_environment_tools)
    if args.include_ollama:
        runner.test("Ollama adjudicator integration", test_ollama_adjudicator)

    return runner.summary()


if __name__ == "__main__":
    raise SystemExit(main())

