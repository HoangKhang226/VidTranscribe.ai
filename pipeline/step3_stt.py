import os
from faster_whisper import WhisperModel
from utils.logger import logger
from utils.memory import clean_memory, log_memory_usage
from utils.srt_utils import SRTEntry, write_srt
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, SUBTITLES_DIR

def run(audio_path: str, context: dict) -> str:
    """
    Chạy Speech-To-Text chính thức bằng Whisper.
    Nạp danh sách từ khóa vào initial_prompt để định hướng nhận dạng đúng thuật ngữ.
    """
    logger.info("=== BƯỚC 4: SPEECH-TO-TEXT (WHISPER) ===")
    log_memory_usage("STT - Trước khi nạp Whisper-Medium")
    
    # 1. Chuẩn bị initial prompt từ keywords chuyên ngành
    initial_prompt = ""
    if context.get("keywords"):
        initial_prompt = "The terminology used in this video includes: " + ", ".join(context["keywords"]) + "."
        logger.info(f"Dùng Initial Prompt hướng dẫn Whisper: '{initial_prompt}'")
    
    # 2. Khởi tạo mô hình
    logger.info(f"Đang nạp mô hình Whisper-{WHISPER_MODEL_SIZE} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=4
    )
    
    # 3. Tiến hành nhận diện âm thanh
    logger.info("Đang bóc băng toàn bộ âm thanh video...")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=False, # Không cần thiết cho phụ đề thông thường, giúp tăng tốc độ
        initial_prompt=initial_prompt
    )
    
    # 4. Chuyển đổi kết quả thành danh sách SRTEntry
    entries = []
    for index, seg in enumerate(segments, start=1):
        # Chuyển đổi giây sang mili giây
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        text = seg.text.strip()
        
        # Bỏ qua các đoạn không có tiếng
        if text:
            entries.append(SRTEntry(index, start_ms, end_ms, text))
            
    logger.info(f"Bóc băng hoàn tất. Tổng số câu thoại: {len(entries)}")
    
    # 5. Ghi file SRT tiếng Anh thô
    srt_output_path = os.path.join(SUBTITLES_DIR, "subtitles_en.srt")
    write_srt(entries, srt_output_path)
    
    # 6. Giải phóng VRAM/RAM ngay lập tức
    del model
    clean_memory()
    
    return srt_output_path
