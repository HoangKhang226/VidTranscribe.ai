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
# Mặc định để chạy ổn trên máy local/CPU. Khi chạy Colab GPU, comment block LOCAL
# và bỏ comment block COLAB bên dưới.

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

# === LOCAL / CPU DEFAULT ===
WHISPER_MODEL_SIZE = "medium"  # "tiny", "base", "small", "medium", "large-v3-turbo"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_CPU_THREADS = 4
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = False
CONTEXT_WHISPER_DEVICE = "cpu"
CONTEXT_WHISPER_COMPUTE_TYPE = "int8"

# === COLAB / GPU 15GB PRESET ===
# Bỏ comment block này khi chạy Colab để tận dụng VRAM/GPU.
# WHISPER_MODEL_SIZE = "medium"
# WHISPER_DEVICE = "cuda"
# WHISPER_COMPUTE_TYPE = "float16"
# WHISPER_CPU_THREADS = 4
# WHISPER_BEAM_SIZE = 3
# WHISPER_VAD_FILTER = True
# CONTEXT_WHISPER_DEVICE = "cuda"
# CONTEXT_WHISPER_COMPUTE_TYPE = "float16"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"

# === Cấu hình VRAM cho LLM (Ollama) ===
# None để Ollama tự offload tối đa theo VRAM hiện có. Trên Colab 15GB thường sẽ dùng GPU tốt hơn.
LLM_NUM_CTX = int(os.environ.get("VIDTRANSCRIBE_LLM_NUM_CTX", "2048"))
LLM_NUM_CTX_LARGE = int(os.environ.get("VIDTRANSCRIBE_LLM_NUM_CTX_LARGE", "4096"))
LLM_THINK = _env_bool("VIDTRANSCRIBE_LLM_THINK", False)
_llm_num_gpu = os.environ.get("VIDTRANSCRIBE_LLM_NUM_GPU")
LLM_NUM_GPU = int(_llm_num_gpu) if _llm_num_gpu not in (None, "", "auto") else None

EDGE_TTS_VOICE = "vi-VN-NamMinhNeural"

# API & Web UI config
API_HOST = "127.0.0.1"
API_PORT = 8000
GRADIO_PORT = 7860
