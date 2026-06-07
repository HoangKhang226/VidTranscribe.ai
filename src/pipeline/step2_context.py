import os
import json
import requests
from pydub import AudioSegment
from faster_whisper import WhisperModel
from src.utils.logger import logger
from src.utils.memory import clean_memory, log_memory_usage
from src.config import OLLAMA_API_URL, OLLAMA_MODEL_NAME

def get_rough_transcript(audio_path: str, duration_sec: int = 30) -> str:
    """Cắt 30s đầu audio và chạy Whisper-Tiny để lấy văn bản thô cực nhanh."""
    logger.info(f"Đang trích xuất {duration_sec}s đầu để phân tích ngữ cảnh...")
    
    # 1. Cắt âm thanh
    audio = AudioSegment.from_wav(audio_path)
    cut_ms = duration_sec * 1000
    temp_cut_path = audio_path.replace(".wav", f"_temp_{duration_sec}s.wav")
    audio[:cut_ms].export(temp_cut_path, format="wav")
    
    # 2. Tải Whisper-Tiny chạy trên CPU để không chiếm VRAM GPU
    logger.info("Đang nạp Whisper-Tiny trên CPU...")
    log_memory_usage("Dynamic Context - Trước khi nạp Tiny")
    
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
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

def query_ollama_for_context(raw_text: str, model_name: str = OLLAMA_MODEL_NAME) -> dict:
    """Gửi raw text qua Ollama để nhận diện chủ đề và keywords."""
    logger.info(f"Đang gửi yêu cầu đến Ollama {model_name} để trích xuất ngữ cảnh...")
    
    system_prompt = (
        "Bạn là một chuyên gia phân tích video. Nhiệm vụ của bạn là đọc văn bản thô tiếng Anh (có thể bị sai chính tả) "
        "từ 30 giây đầu của video, sau đó đoán chủ đề và liệt kê các thuật ngữ/từ khóa chuyên ngành kỹ thuật quan trọng bằng tiếng Anh "
        "cần phải GIỮ NGUYÊN (không dịch sang tiếng Việt).\n"
        "BẠN PHẢI LUÔN TRẢ VỀ định dạng JSON chính xác theo cấu trúc:\n"
        "{\n"
        '  "topic": "Tên chủ đề ngắn bằng tiếng Việt",\n'
        '  "keywords": ["từ khóa 1", "từ khóa 2", "từ khóa 3", ...]\n'
        "}\n"
        "Tuyệt đối không giải thích dông dài, không viết thêm markdown hoặc ghi chú."
    )
    
    user_prompt = f"Phân tích đoạn text sau:\n\n{raw_text}"
    
    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2
        }
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

def run(audio_path: str, model_name: str = OLLAMA_MODEL_NAME) -> dict:
    """Chạy toàn bộ pipeline nhận diện ngữ cảnh động."""
    logger.info("=== BƯỚC 2 & 3: DYNAMIC CONTEXT LAYER ===")
    
    raw_text = get_rough_transcript(audio_path, duration_sec=30)
    if not raw_text:
        logger.warning("Không nhận dạng được âm thanh nào trong 30s đầu. Dùng ngữ cảnh mặc định.")
        return {
            "topic": "Chung",
            "keywords": [],
            "keywords_str": ""
        }
        
    context = query_ollama_for_context(raw_text, model_name=model_name)
    return context
