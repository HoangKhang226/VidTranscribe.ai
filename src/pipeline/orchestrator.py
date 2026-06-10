import time
import os
from src.utils.logger import logger
from src.utils.memory import clean_memory, log_memory_usage
from src.pipeline import step1_ingestion
from src.pipeline import step2_context
from src.pipeline import step3_stt
from src.pipeline import step4_translate
from src.pipeline import step5_tts
from src.pipeline import step6_mux
from src.pipeline.step2_context import ContextAnalyzer

class PipelineOrchestrator:
    def __init__(self):
        self.state = {
            "status": "idle",
            "progress": 0.0,
            "current_step": "",
            "error": None,
            "start_time": 0.0,
            "elapsed_time": 0.0,
            "results": {}
        }

    def update_progress(self, step_name: str, progress: float):
        """Cập nhật tiến độ của Pipeline."""
        self.state["current_step"] = step_name
        self.state["progress"] = progress
        logger.info(f"Progress: {progress*100:.0f}% | Current Step: {step_name}")

    def run_pipeline(self, source: str, hardsub: bool = True, ollama_model: str = None, domain_override: str = "Auto Detect", mode: str = "end_to_end", keep_temp_segments: bool = False) -> str:
        """
        Chạy pipeline lồng tiếng phụ đề AI.

        mode:
          - "end_to_end":    chạy đầy đủ tới khi xuất video cuối.
          - "translate_only": dừng sau bước Dịch (Bước 5), trả về phụ đề tiếng Việt
                              để người dùng xem/sửa trực tiếp trước khi lồng tiếng.
        keep_temp_segments:
          - True: giữ lại src/output/audio/temp_segments để chạy evaluation/benchmark_sync.py.
        Đảm bảo dọn dẹp RAM/VRAM sau mỗi bước.
        """
        self.state["status"] = "processing"
        self.state["progress"] = 0.0
        self.state["error"] = None
        self.state["start_time"] = time.time()
        self.state["mode"] = mode
        
        logger.info(f"BẮT ĐẦU CHẠY PIPELINE (mode={mode})...")
        log_memory_usage("Orchestrator - Khởi tạo")
        
        try:
            # Bước 1: Ingestion
            self.update_progress("Bước 1: Ingestion (Tải/Phân tách Video & Audio)", 0.1)
            audio_orig, video_mute = step1_ingestion.run(source)
            self.state["results"]["audio_original"] = audio_orig
            self.state["results"]["video_no_audio"] = video_mute
            self.state["results"]["source_video"] = source
            clean_memory()
            
            # Bước 2 & 3: Dynamic Context Layer
            self.update_progress("Bước 2 & 3: Dynamic Context Layer (Nhận diện chủ đề & Từ khóa)", 0.25)
            context_analyzer = ContextAnalyzer(model_name=ollama_model) if ollama_model else ContextAnalyzer()
            context = context_analyzer.run(audio_orig, domain_override=domain_override)
            self.state["results"]["context"] = context
            clean_memory()
            
            # Bước 4: Speech-To-Text (Whisper)
            self.update_progress("Bước 4: Speech-To-Text (Nhận diện giọng nói chính xác)", 0.45)
            srt_en = step3_stt.run(audio_orig, context)
            self.state["results"]["srt_en"] = srt_en
            clean_memory()
            
            # Bước 5: Translation (Ollama)
            self.update_progress("Bước 5: Dịch thuật ngữ cảnh gối đầu (Ollama)", 0.45)
            
            def translation_progress(current, total):
                percent = current / total
                # Scale progress linearly from 0.45 to 0.75
                p = 0.45 + (0.30 * percent)
                self.update_progress(f"Đang dịch câu {current}/{total}...", p)
                
            if ollama_model:
                srt_vi = step4_translate.run(srt_en, context, model_name=ollama_model, progress_callback=translation_progress)
            else:
                srt_vi = step4_translate.run(srt_en, context, progress_callback=translation_progress)
            
            self.state["results"]["srt_vi"] = srt_vi
            clean_memory()
            
            # === CHẾ ĐỘ CHỈ DỊCH: Dừng lại để người dùng xem & sửa phụ đề ===
            if mode == "translate_only":
                self.state["status"] = "awaiting_review"
                self.state["progress"] = 0.75
                self.state["elapsed_time"] = time.time() - self.state["start_time"]
                self.update_progress("Đã dịch xong. Chờ người dùng xem & chỉnh sửa phụ đề...", 0.75)
                logger.info("PIPELINE DỪNG Ở CHẾ ĐỘ CHỈ DỊCH. Phụ đề sẵn sàng để xem/sửa.")
                return srt_vi
            
            # Bước 6 & 7: Lồng tiếng + Muxing
            return self._finalize(video_mute, audio_orig, srt_vi, hardsub, keep_temp_segments=keep_temp_segments)
            
        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            logger.error(f"PIPELINE BỊ LỖI VÀ DỪNG LẠI: {e}")
            clean_memory()
            raise e

    def _finalize(self, video_mute: str, audio_orig: str, srt_vi: str, hardsub: bool, keep_temp_segments: bool = False) -> str:
        """Chạy Bước 6 (Lồng tiếng) + Bước 7 (Muxing) để xuất video cuối."""
        # Bước 6: Speech Synthesis (Edge-TTS)
        self.update_progress("Bước 6: Lồng tiếng Code-Switching (Edge-TTS & Phase Vocoder)", 0.8)
        audio_vi = step5_tts.run(srt_vi, audio_orig, keep_temp_segments=keep_temp_segments)
        self.state["results"]["audio_vietnamese"] = audio_vi
        clean_memory()
        
        # Bước 7: Muxing (FFmpeg)
        self.update_progress("Bước 7: Hợp nhất Video hoàn chỉnh (FFmpeg Muxing)", 0.95)
        final_video = step6_mux.run(video_mute, audio_vi, srt_vi, hardsub=hardsub)
        self.state["results"]["final_video"] = final_video
        clean_memory()
        
        # Hoàn tất
        self.state["status"] = "completed"
        self.state["progress"] = 1.0
        self.state["elapsed_time"] = time.time() - self.state["start_time"]
        logger.info(f"PIPELINE HOÀN TẤT THÀNH CÔNG SAU {self.state['elapsed_time']:.2f} GIÂY!")
        return final_video

    def resume_from_subtitles(self, hardsub: bool = True) -> str:
        """
        Tiếp tục pipeline từ chế độ 'awaiting_review' (sau khi người dùng đã sửa phụ đề).
        Dùng lại video_no_audio, audio_original và file phụ đề tiếng Việt đã chỉnh sửa.
        """
        results = self.state.get("results", {})
        video_mute = results.get("video_no_audio")
        audio_orig = results.get("audio_original")
        srt_vi = results.get("srt_vi")
        if not (video_mute and audio_orig and srt_vi):
            raise ValueError("Không đủ dữ liệu để tiếp tục lồng tiếng (thiếu video/audio/phụ đề).")
        
        self.state["status"] = "processing"
        self.state["error"] = None
        if not self.state.get("start_time"):
            self.state["start_time"] = time.time()
        logger.info("TIẾP TỤC LỒNG TIẾNG TỪ PHỤ ĐỀ ĐÃ CHỈNH SỬA...")
        try:
            return self._finalize(video_mute, audio_orig, srt_vi, hardsub)
        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            logger.error(f"LỖI KHI TIẾP TỤC LỒNG TIẾNG: {e}")
            clean_memory()
            raise e
            
    def get_status(self) -> dict:
        """Lấy trạng thái tiến trình phục vụ API."""
        if self.state["status"] == "processing":
            self.state["elapsed_time"] = time.time() - self.state["start_time"]
        return self.state
