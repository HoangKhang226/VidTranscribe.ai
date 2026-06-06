import os
import sys

# Trên Windows, tự động nạp lại PATH từ Registry (HKCU & HKLM) để cập nhật các biến môi trường mới
# (như FFmpeg do winget vừa cài đặt) mà không cần người dùng tắt đi bật lại terminal.
if sys.platform == "win32":
    try:
        import winreg
        paths = []
        # Đọc User PATH từ Registry
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                user_path, _ = winreg.QueryValueEx(key, "Path")
                if user_path:
                    paths.extend(user_path.split(";"))
        except Exception:
            pass
        # Đọc System PATH từ Registry
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                sys_path, _ = winreg.QueryValueEx(key, "Path")
                if sys_path:
                    paths.extend(sys_path.split(";"))
        except Exception:
            pass
        # Ghép với PATH hiện tại và làm sạch
        current_paths = os.environ.get("PATH", "").split(";")
        all_paths = current_paths + paths
        unique_paths = []
        for p in all_paths:
            p_clean = os.path.expandvars(p.strip())
            if p_clean and p_clean not in unique_paths:
                unique_paths.append(p_clean)
        os.environ["PATH"] = ";".join(unique_paths)
    except Exception as e:
        sys.stderr.write(f"Không thể tự động cập nhật PATH từ Registry: {e}\n")

# Đường dẫn thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Các thư mục lưu trữ output
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DOWNLOADS_DIR = os.path.join(OUTPUT_DIR, "downloads")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
SUBTITLES_DIR = os.path.join(OUTPUT_DIR, "subtitles")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")

# Đảm bảo các thư mục tồn tại
for path in [OUTPUT_DIR, DOWNLOADS_DIR, AUDIO_DIR, SUBTITLES_DIR, FINAL_DIR]:
    os.makedirs(path, exist_ok=True)

# Cấu hình Mô hình
WHISPER_MODEL_SIZE = "medium"  # "tiny", "base", "small", "medium", "large-v3-turbo"
WHISPER_DEVICE = "cpu"        # "cpu" hoặc "cuda" (RTX 2050 4GB)
WHISPER_COMPUTE_TYPE = "int8" # Lượng hóa để tiết kiệm RAM/VRAM

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"

EDGE_TTS_VOICE = "vi-VN-NamMinhNeural"

# API & Web UI config
API_HOST = "127.0.0.1"
API_PORT = 8000
GRADIO_PORT = 7860
