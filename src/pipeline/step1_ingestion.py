import os
import subprocess
import shutil
from src.utils.logger import logger
from src.utils.memory import log_memory_usage
from src.config import DOWNLOADS_DIR, AUDIO_DIR, FINAL_DIR

def _decode_tail(result: subprocess.CompletedProcess[bytes], limit: int = 4000) -> str:
    return result.stderr.decode("utf-8", errors="ignore")[-limit:]


def normalize_input_media(video_path: str) -> str:
    """
    Chuẩn hóa container/audio trước khi pipeline xử lý.

    Một số video tải từ mạng dùng HE-AACv2 hoặc stream AAC không chuẩn khiến FFmpeg
    spam lỗi decode ở các bước sau. Bước này ép audio về AAC-LC phổ thông trong một
    file trung gian ổn định hơn, còn video thì copy stream để tránh re-encode tốn thời gian.
    """
    normalized_output = os.path.join(DOWNLOADS_DIR, "input_normalized.mp4")
    normalize_cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-fflags", "+discardcorrupt",
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-profile:a", "aac_low",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        "-movflags", "+faststart",
        normalized_output,
    ]

    logger.info("Đang chuẩn hóa input media sang AAC-LC...")
    result = subprocess.run(
        normalize_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    normalized_ok = os.path.exists(normalized_output) and os.path.getsize(normalized_output) > 0
    if normalized_ok:
        if result.returncode != 0:
            logger.warning("FFmpeg báo lỗi khi chuẩn hóa nhưng vẫn tạo được file normalized; tiếp tục pipeline.")
        return normalized_output

    raise RuntimeError(f"Không chuẩn hóa được input media: {_decode_tail(result)}")


def extract_audio_and_video(video_path: str) -> tuple[str, str]:
    """
    Tách luồng video và audio:
    - Audio: 16kHz, Mono WAV (phù hợp với Whisper)
    - Video: Muted MP4 (không chứa tiếng) để ghép tiếng Việt về sau
    """
    logger.info(f"Bắt đầu tách luồng từ video gốc: {video_path}")
    log_memory_usage("Ingestion - Khởi chạy FFmpeg")
    normalized_input = normalize_input_media(video_path)
    
    # Định nghĩa các đường dẫn đầu ra
    audio_output = os.path.join(AUDIO_DIR, "audio_original.wav")
    video_output = os.path.join(DOWNLOADS_DIR, "video_no_audio.mp4")
    
    # 1. Trích xuất Audio (16kHz, 1 channel/mono, 16-bit PCM WAV)
    audio_cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", normalized_input,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_output
    ]
    
    # 2. Trích xuất Video không tiếng (Copy Stream cực nhanh, không re-encode)
    video_cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", normalized_input,
        "-an",
        "-vcodec", "copy",
        video_output
    ]
    
    logger.info("Đang trích xuất audio (16kHz mono)...")
    audio_result = subprocess.run(
        audio_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    logger.info("Đang tạo video tắt tiếng (video_no_audio)...")
    video_result = subprocess.run(
        video_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    audio_ok = os.path.exists(audio_output) and os.path.getsize(audio_output) > 0
    video_ok = os.path.exists(video_output) and os.path.getsize(video_output) > 0
    if audio_ok and video_ok:
        if audio_result.returncode != 0 or video_result.returncode != 0:
            logger.warning("FFmpeg báo lỗi decode nhưng vẫn tạo được output hợp lệ; tiếp tục pipeline.")
        logger.info("Tách luồng audio và video thành công.")
        return audio_output, video_output

    error_parts = []
    if not audio_ok:
        error_parts.append("audio output missing")
    if not video_ok:
        error_parts.append("video output missing")
    if audio_result.returncode != 0:
        error_parts.append(_decode_tail(audio_result))
    if video_result.returncode != 0:
        error_parts.append(_decode_tail(video_result))
    raise RuntimeError(f"FFmpeg error: {' | '.join(error_parts)}")

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
