import os
import subprocess
from utils.logger import logger
from utils.srt_utils import sanitize_srt_overlap
from config import FINAL_DIR, BASE_DIR

def get_relative_path(absolute_path: str) -> str:
    """Chuyển đổi đường dẫn tuyệt đối sang tương đối từ BASE_DIR để tương thích với FFmpeg trên Windows."""
    try:
        rel_path = os.path.relpath(absolute_path, BASE_DIR)
        # Sử dụng dấu gạch chéo xuôi cho FFmpeg
        return rel_path.replace("\\", "/")
    except Exception:
        return absolute_path

def run(video_path: str, audio_vi_path: str, srt_vi_path: str, hardsub: bool = False) -> str:
    """
    Hợp nhất Video + Audio Việt thành file output hoàn chỉnh (không kèm phụ đề).
    """
    logger.info("=== BƯỚC 7: HỢP NHẤT VIDEO (MUXING) ===")
    
    output_path = os.path.join(FINAL_DIR, "final_output.mp4")
    
    # Chuyển đổi sang đường dẫn tương đối để tránh lỗi bộ lọc FFmpeg trên Windows
    rel_video = get_relative_path(video_path)
    rel_audio = get_relative_path(audio_vi_path)
    rel_output = get_relative_path(output_path)
    
    logger.info(f"File video không tiếng: {rel_video}")
    logger.info(f"File audio tiếng Việt: {rel_audio}")
    
    # Định nghĩa câu lệnh FFmpeg chỉ ghép tiếng Việt vào video (không hiện phụ đề)
    logger.info("Đang lồng tiếng Việt vào video (không kèm phụ đề)...")
    cmd = [
        "ffmpeg", "-y",
        "-i", rel_video,
        "-i", rel_audio,
        "-c:v", "copy",  # Sao chép luồng video trực tiếp, không re-encode (cực nhanh và giữ nguyên chất lượng gốc)
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        rel_output
    ]
        
    try:
        # Chạy lệnh FFmpeg ở thư mục BASE_DIR để đường dẫn tương đối hoạt động chính xác
        logger.info(f"Đang thực thi lệnh FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Xuất video hoàn chỉnh thành công tại: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore")
        logger.error(f"Lỗi ghép video bằng FFmpeg: {error_msg}")
        raise RuntimeError(f"FFmpeg Muxing error: {error_msg}")
