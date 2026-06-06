import os
import subprocess
import shutil
import yt_dlp
from utils.logger import logger
from utils.memory import log_memory_usage
from config import DOWNLOADS_DIR, AUDIO_DIR, FINAL_DIR

def download_youtube_video(url: str, output_dir: str) -> str:
    """Tải video YouTube ở độ phân giải tốt nhất có cả hình lẫn tiếng."""
    logger.info(f"📥 Bắt đầu tải video từ YouTube: {url}")
    
    # Định nghĩa đường dẫn file cookies.txt ở thư mục gốc dự án
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cookies_txt_path = os.path.join(project_root, "cookies.txt")
    
    # Chuẩn bị danh sách các tùy chọn cấu hình để fallback nếu bị chặn
    ydl_opts_list = []
    
    # Nếu tồn tại file cookies.txt ở thư mục gốc, ưu tiên dùng file này trước
    if os.path.exists(cookies_txt_path):
        ydl_opts_list.append({
            "name": "Sử dụng file cookies.txt ở thư mục gốc",
            "opts": {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(output_dir, 'youtube_download.%(ext)s'),
                'quiet': False,
                'no_warnings': True,
                'noplaylist': True,
                'cookiefile': cookies_txt_path,
            }
        })
        
    # Thử 1: Tải không dùng cookies (nhanh nhất)
    ydl_opts_list.append({
        "name": "Không dùng cookies",
        "opts": {
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, 'youtube_download.%(ext)s'),
            'quiet': False,
            'no_warnings': True,
            'noplaylist': True,
        }
    })
    
    # Thử 2: Ép dùng iOS/Android Client (Vượt bot check cực kỳ mạnh mẽ)
    ydl_opts_list.append({
        "name": "Client iOS/Android (Không dùng cookies)",
        "opts": {
            'format': 'best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, 'youtube_download.%(ext)s'),
            'quiet': False,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['ios', 'android', 'web_creator']}},
            'noplaylist': True,
        }
    })
    
    # Thêm từng trình duyệt riêng biệt làm các lượt thử dự phòng tiếp theo
    for browser in ['chrome', 'edge', 'opera', 'firefox']:
        ydl_opts_list.append({
            "name": f"Cookies từ {browser.capitalize()}",
            "opts": {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(output_dir, 'youtube_download.%(ext)s'),
                'quiet': False,
                'no_warnings': True,
                'cookiesfrombrowser': (browser,),
                'noplaylist': True,
            }
        })
    
    last_err = None
    for i, item in enumerate(ydl_opts_list):
        opts_name = item["name"]
        opts = item["opts"]
        try:
            if i > 0:
                logger.info(f"🍪 Phát hiện bot block, đang thử: {opts_name}...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                logger.info(f"✅ Tải thành công YouTube video: {filename}")
                return filename
        except Exception as e:
            last_err = e
            logger.warning(f"⚠️ Thử tải bằng cách '{opts_name}' thất bại: {e}")


            
    logger.error("❌ Thất bại hoàn toàn khi tải video từ YouTube.")
    raise last_err


def extract_audio_and_video(video_path: str) -> tuple[str, str]:
    """
    Tách luồng video và audio:
    - Audio: 16kHz, Mono WAV (phù hợp với Whisper)
    - Video: Muted MP4 (không chứa tiếng) để ghép tiếng Việt về sau
    """
    logger.info(f"🎞️ Bắt đầu tách luồng từ video gốc: {video_path}")
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
        logger.info("🔊 Đang trích xuất audio (16kHz mono)...")
        subprocess.run(audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("📹 Đang tạo video tắt tiếng (video_no_audio)...")
        subprocess.run(video_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info("✅ Tách luồng audio và video thành công.")
        return audio_output, video_output
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Lỗi khi chạy FFmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"FFmpeg error: {e}")

def run(source: str, is_url: bool = False) -> tuple[str, str]:
    """
    Điểm chạy chính của Bước 1 (Ingestion).
    Trả về: (audio_original_path, video_no_audio_path)
    """
    logger.info("=== BƯỚC 1: INGESTION ===")
    
    if is_url:
        # Tải từ link YouTube trước
        downloaded_path = download_youtube_video(source, DOWNLOADS_DIR)
        video_input = downloaded_path
    else:
        # Dùng file upload local trực tiếp
        if not os.path.exists(source):
            raise FileNotFoundError(f"File video nguồn không tồn tại: {source}")
        video_input = source
        logger.info(f"📂 Sử dụng file video tải lên: {video_input}")
        
    # Tiến hành tách luồng
    audio_path, video_path = extract_audio_and_video(video_input)
    return audio_path, video_path
