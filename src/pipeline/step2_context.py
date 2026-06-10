import os
import json
import requests
from pydub import AudioSegment
from faster_whisper import WhisperModel
from src.utils.logger import logger
from src.utils.memory import clean_memory, log_memory_usage
from src.config import (
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    LLM_NUM_CTX,
    LLM_THINK,
    LLM_NUM_GPU,
    CONTEXT_WHISPER_DEVICE,
    CONTEXT_WHISPER_COMPUTE_TYPE,
)

class ContextAnalyzer:
    """Module nhận diện chủ đề và ngữ cảnh động (Dynamic Context) từ audio."""
    
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME):
        self.model_name = model_name

    def get_rough_transcript(self, audio_path: str, duration_sec: int = 30) -> str:
        """Cắt 30s đầu audio và chạy Whisper-Tiny để lấy văn bản thô cực nhanh."""
        logger.info(f"Đang trích xuất {duration_sec}s đầu để phân tích ngữ cảnh...")
        
        # 1. Cắt âm thanh
        audio = AudioSegment.from_wav(audio_path)
        cut_ms = duration_sec * 1000
        temp_cut_path = audio_path.replace(".wav", f"_temp_{duration_sec}s.wav")
        audio[:cut_ms].export(temp_cut_path, format="wav")
        
        # 2. Tải Whisper-Tiny. Trên Colab ưu tiên CUDA để tận dụng VRAM 15GB.
        logger.info(f"Đang nạp Whisper-Tiny ({CONTEXT_WHISPER_DEVICE}, {CONTEXT_WHISPER_COMPUTE_TYPE})...")
        log_memory_usage("Dynamic Context - Trước khi nạp Tiny")
        
        model = WhisperModel("tiny", device=CONTEXT_WHISPER_DEVICE, compute_type=CONTEXT_WHISPER_COMPUTE_TYPE)
        
        segments, _ = model.transcribe(temp_cut_path, beam_size=1)
        text_segments = [seg.text for seg in segments]
        raw_text = " ".join(text_segments).strip()
        
        logger.info(f"Raw Text (30s đầu): {raw_text}")
        
        # 3. Dọn dẹp Whisper model khỏi bộ nhớ ngay lập tức
        del model
        if os.path.exists(temp_cut_path):
            os.remove(temp_cut_path)
        clean_memory()
        
        return raw_text

    def query_ollama_for_context(self, raw_text: str) -> dict:
        """Gửi raw text qua Ollama để nhận diện chủ đề và keywords."""
        logger.info(f"Đang gửi yêu cầu đến Ollama {self.model_name} để trích xuất ngữ cảnh...")
        
        system_prompt = (
            "Bạn là một chuyên gia phân tích nội dung video ĐA NGÀNH (công nghệ, y tế, tài chính, "
            "giáo dục, pháp lý, khoa học, marketing, du lịch, ẩm thực, thể thao, nghệ thuật...). "
            "Nhiệm vụ của bạn là đọc văn bản thô tiếng Anh (có thể bị sai chính tả) từ 30 giây đầu "
            "của video, sau đó:\n"
            "1) Đoán chủ đề chính xác bằng tiếng Việt (ngắn gọn, ví dụ: 'Y học lâm sàng', 'Đầu tư chứng khoán', "
            "'Lập trình AI', 'Nấu ăn Á-Âu', 'Lịch sử thế giới',...).\n"
            "2) Liệt kê các thuật ngữ/từ khóa ĐẶC THÙ NGÀNH bằng tiếng Anh cần GIỮ NGUYÊN khi dịch "
            "(thuật ngữ chuyên môn, tên sản phẩm, danh từ riêng, từ viết tắt...).\n"
            "BẠN PHẢI LUÔN TRẢ VỀ định dạng JSON chính xác theo cấu trúc:\n"
            "{\n"
            '  "topic": "Tên chủ đề ngắn bằng tiếng Việt",\n'
            '  "keywords": ["từ khóa 1", "từ khóa 2", "từ khóa 3", ...]\n'
            "}\n"
            "Tuyệt đối không giải thích dông dài, không viết thêm markdown hoặc ghi chú."
        )
        
        user_prompt = f"Phân tích đoạn text sau:\n\n{raw_text}"
        
        options = {
            "temperature": 0.2,
            "num_ctx": LLM_NUM_CTX,
            "think": LLM_THINK,
        }
        if LLM_NUM_GPU is not None:
            options["num_gpu"] = LLM_NUM_GPU

        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }
        
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            raw_response = res_json.get("response", "{}").strip()
            
            # Parse kết quả JSON trả về
            context_data = json.loads(raw_response)
            
            topic = context_data.get("topic", "Chung")
            keywords = context_data.get("keywords", [])
            
            # Xử lý làm sạch keywords: chuyển về chữ thường
            keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
            
            logger.info(f"Kết quả nhận diện chủ đề: {topic}")
            logger.info(f"Danh sách từ khóa chuyên ngành: {keywords}")
            
            return {
                "topic": topic,
                "keywords": keywords,
                "keywords_str": ", ".join(keywords)
            }
        except Exception as e:
            logger.error(f"Lỗi khi gọi Ollama: {e}")
            # Fallback an toàn
            return {
                "topic": "Chung",
                "keywords": [],
                "keywords_str": ""
            }

    def run(self, audio_path: str, domain_override: str = "Auto Detect") -> dict:
        """Chạy toàn bộ pipeline nhận diện ngữ cảnh động, hoặc dùng domain chỉ định sẵn."""
        logger.info("=== BƯỚC 2 & 3: DYNAMIC CONTEXT LAYER ===")
        
        # 1. Nếu người dùng chọn sẵn ngành (Manual Override)
        if domain_override and domain_override.lower() != "auto detect":
            logger.info(f"Bỏ qua Auto Detect. Dùng ngành đã chỉ định: {domain_override}")
            return {
                "topic": domain_override,
                "keywords": [], # Sẽ được quét động ở Step 4 (Chunk-Level RAG)
                "keywords_str": ""
            }
        
        # 2. Nếu Auto Detect
        raw_text = self.get_rough_transcript(audio_path, duration_sec=30)
        if not raw_text:
            logger.warning("Không nhận dạng được âm thanh nào trong 30s đầu. Dùng ngữ cảnh mặc định.")
            return {
                "topic": "General",
                "keywords": [],
                "keywords_str": ""
            }
            
        context = self.query_ollama_for_context(raw_text)
        return context

# Wrapper để tương thích ngược nếu còn module nào import thẳng
def run(audio_path: str, model_name: str = OLLAMA_MODEL_NAME, domain_override: str = "Auto Detect") -> dict:
    analyzer = ContextAnalyzer(model_name=model_name)
    return analyzer.run(audio_path, domain_override)
