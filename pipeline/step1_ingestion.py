import os
import subprocess
import shutil
from utils.logger import logger
from utils.memory import log_memory_usage
from config import DOWNLOADS_DIR, AUDIO_DIR, FINAL_DIR

def extract_audio_and_video(video_path: str) -> tuple[str, str]:
    """
    Tách luồng video và audio:
    - Audio: 16kHz, Mono WAV (phù hợp với Whisper)
    - Video: Muted MP4 (không chứa tiếng) để ghép tiếng Việt về sau
    """
    logger.info(f"Bắt đầu tách luồng từ video gốc: {video_path}")
    log_memory_usage("Ingestion - Khởi chạy FFmpeg")
    
    # Định nghĩa các đường dẫn đầu ra
    audio_output = os.path.join(AUDIO_DIR, "audio_original.wav")
    video_output = os.path.join(DOWNLOADS_DIR, "video_no_audio.mp4")
    
    # 1. Trích xuất Audio (16kHz, 1 channel/mono, 16-bit PCM WAV)
    audio_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_output
    ]
    
    # 2. Trích xuất Video không tiếng (Copy Stream cực nhanh, không re-encode)
    video_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-an",
        "-vcodec", "copy",
        video_output
    ]
    
    try:
        logger.info("Đang trích xuất audio (16kHz mono)...")
        subprocess.run(audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("Đang tạo video tắt tiếng (video_no_audio)...")
        subprocess.run(video_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("Tách luồng audio và video thành công.")
        return audio_output, video_output
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy FFmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"FFmpeg error: {e}")

def run(source: str) -> tuple[str, str]:
    """
    Điểm chạy chính của Bước 1 (Ingestion).
    Chỉ chấp nhận file video local trực tiếp.
    Trả về: (audio_original_path, video_no_audio_path)
    """
    logger.info("=== BƯỚC 1: INGESTION ===")
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"File video nguồn không tồn tại: {source}")
    video_input = source
    logger.info(f"Sử dụng file video: {video_input}")
        
    # Tiến hành tách luồng
    audio_path, video_path = extract_audio_and_video(video_input)
    return audio_path, video_path
