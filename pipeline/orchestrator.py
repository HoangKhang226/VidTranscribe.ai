import time
import os
from utils.logger import logger
from utils.memory import clean_memory, log_memory_usage
from pipeline import step1_ingestion
from pipeline import step2_context
from pipeline import step3_stt
from pipeline import step4_translate
from pipeline import step5_tts
from pipeline import step6_mux

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
        logger.info(f"🔄 Progress: {progress*100:.0f}% | Current Step: {step_name}")

    def run_pipeline(self, source: str, is_url: bool = False, hardsub: bool = True, ollama_model: str = None) -> str:
        """
        Chạy tuần tự 7 bước lồng tiếng phụ đề AI.
        Đảm bảo dọn dẹp RAM/VRAM sau mỗi bước.
        """
        self.state["status"] = "processing"
        self.state["progress"] = 0.0
        self.state["error"] = None
        self.state["start_time"] = time.time()
        
        logger.info("🚀 BẮT ĐẦU CHẠY PIPELINE LỒNG TIẾNG VÀ DỊCH THUẬT AI...")
        log_memory_usage("Orchestrator - Khởi tạo")
        
        try:
            # Bước 1: Ingestion
            self.update_progress("Bước 1: Ingestion (Tải/Phân tách Video & Audio)", 0.1)
            audio_orig, video_mute = step1_ingestion.run(source, is_url)
            self.state["results"]["audio_original"] = audio_orig
            self.state["results"]["video_no_audio"] = video_mute
            clean_memory()
            
            # Bước 2 & 3: Dynamic Context Layer
            self.update_progress("Bước 2 & 3: Dynamic Context Layer (Nhận diện chủ đề & Từ khóa)", 0.25)
            context = step2_context.run(audio_orig, model_name=ollama_model) if ollama_model else step2_context.run(audio_orig)
            self.state["results"]["context"] = context
            clean_memory()
            
            # Bước 4: Speech-To-Text (Whisper)
            self.update_progress("Bước 4: Speech-To-Text (Nhận diện giọng nói chính xác)", 0.45)
            srt_en = step3_stt.run(audio_orig, context)
            self.state["results"]["srt_en"] = srt_en
            clean_memory()
            
            # Bước 5: Translation (Ollama)
            self.update_progress("Bước 5: Dịch thuật ngữ cảnh gối đầu (Ollama Qwen2.5)", 0.65)
            srt_vi = step4_translate.run(srt_en, context, model_name=ollama_model) if ollama_model else step4_translate.run(srt_en, context)
            self.state["results"]["srt_vi"] = srt_vi
            clean_memory()
            
            # Bước 6: Speech Synthesis (Edge-TTS)
            self.update_progress("Bước 6: Lồng tiếng Code-Switching (Edge-TTS & Phase Vocoder)", 0.8)
            audio_vi = step5_tts.run(srt_vi, audio_orig)
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
            logger.info(f"🏆 PIPELINE HOÀN TẤT THÀNH CÔNG SAU {self.state['elapsed_time']:.2f} GIÂY!")
            return final_video
            
        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            logger.error(f"❌ PIPELINE BỊ LỖI VÀ DỪNG LẠI: {e}")
            clean_memory()
            raise e
            
    def get_status(self) -> dict:
        """Lấy trạng thái tiến trình phục vụ API."""
        if self.state["status"] == "processing":
            self.state["elapsed_time"] = time.time() - self.state["start_time"]
        return self.state
