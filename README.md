# 📡 VidTranscribe.ai: Hệ Sinh Thái Lồng Tiếng & Dịch Phụ Đề Video Tự Động Đa Ngành

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FFD21E?style=for-the-badge&logo=gradio&logoColor=black)](https://gradio.app/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

## 🌟 Tầm Nhìn Dự Án

**VidTranscribe.ai** là một hệ thống tự động hóa hoàn toàn quy trình dịch thuật, căn chỉnh phụ đề và lồng tiếng (dubbing) video chất lượng cao từ tiếng Anh sang tiếng Việt. Dự án giúp chuyển đổi các video tiếng Anh thành phiên bản tiếng Việt có kèm phụ đề căn chỉnh khớp thời gian và âm thanh lồng tiếng tự nhiên.

Điểm vượt trội của **VidTranscribe.ai** là thiết kế **hoàn toàn đa ngành (domain-agnostic)**. Hệ thống có khả năng xử lý chính xác video của mọi lĩnh vực (Y sinh, Tài chính, Công nghệ, Doanh nghiệp, Hóa học, Vật lý...) nhờ công cụ phiên âm ngôn ngữ học tùy biến: tự động nhận diện từ viết tắt, tách từ ghép CamelCase và phiên âm sang tiếng Việt bồi để bộ đọc phát âm chuẩn xác mà không cần dựa vào các bộ từ điển cứng.

---

## 🚀 Tính Năng Nổi Bật

- **🏗️ Pipeline 6 Bước Modul Hóa**: Tách biệt rõ ràng các nhiệm vụ trong luồng xử lý:
  1. **Data Ingestion (Thu Thập)**: Tự động tải video từ YouTube (`yt-dlp`) hoặc nhận file local, bóc tách luồng âm thanh gốc chất lượng cao.
  2. **Context Analysis (Phân Tích)**: Phát hiện hoạt động giọng nói (VAD) và tính toán khoảng lặng để tối ưu hóa việc căn chỉnh thời gian.
  3. **Speech-To-Text (STT)**: Chuyển âm thanh thành văn bản ngoại tuyến (offline) bằng `Faster-Whisper` (hỗ trợ cả CPU và GPU).
  4. **Bilingual Translation (Dịch Song Ngữ)**: Sử dụng LLM chạy cục bộ thông qua Ollama (Qwen2.5) với cơ chế chống rò rỉ chữ Trung Quốc và tự động dịch đơn lẻ dự phòng khi lỗi batch.
  5. **Domain-Agnostic Speech Synthesis (TTS)**: Bộ công cụ chuyển đổi chữ cái tiếng Anh sang tiếng Việt bồi tự động:
     - **Tách từ CamelCase**: Nhận diện từ ghép (Ví dụ: `pH` $\rightarrow$ `pi ếch`, `mRNA` $\rightarrow$ `em a en ê`, `ChatGPT` $\rightarrow$ `chat gi pi ti`).
     - **Đánh vần từ viết tắt (Acronyms)**: Đọc từng chữ cái theo phát âm chuẩn (Ví dụ: `CRM` $\rightarrow$ `xi a em`, `DNA` $\rightarrow$ `đi en ê`, `SQL` $\rightarrow$ `ét qui eo`).
     - **Phân tách âm tiết MOP**: Áp dụng nguyên lý *Maximal Onset Principle* để tự động tạo cách đọc tiếng Việt bồi cho các từ tiếng Anh thông thường.
     - **Tích hợp Edge-TTS**: Căn chỉnh tốc độ nói tự động theo độ dài câu thoại gốc và cơ chế thử lại (exponential backoff) khi bị giới hạn lượt gọi.
  6. **Video Muxing (Trộn Video)**: Tự động trộn phụ đề (hardsub hoặc softsub) và đè âm thanh lồng tiếng khớp mốc thời gian bằng `FFmpeg`.
- **⚡ Kiến Trúc Giao Diện Kép**:
  - **FastAPI Backend**: Cung cấp API bất đồng bộ (Async) mạnh mẽ, chạy ngầm các tác vụ nặng và hỗ trợ kiểm tra tiến trình.
  - **Gradio Web UI**: Giao diện Web tối giản với thiết kế Dark Mode Glassmorphism cao cấp, hỗ trợ thao tác chỉ với một cú click chuột.
- **🛡️ Cơ Chế Giải Phóng Bộ Nhớ**: Tự động giải phóng RAM/VRAM và dọn dẹp cache PyTorch ngay sau khi hoàn thành các bước nặng để tránh tràn bộ nhớ.

---

## 🛠️ Công Nghệ Sử Dụng

| Danh mục                | Công cụ / Thư viện                              |
| :---------------------- | :---------------------------------------------- |
| **Ngôn ngữ chính**      | Python 3.12                                     |
| **Tải video & Xử lý**   | `yt-dlp`, FFmpeg                                |
| **Nhận diện giọng nói** | `faster-whisper` (medium/large-v3-turbo)        |
| **Dịch thuật (LLM)**    | Ollama (`qwen2.5:3b-instruct-q4_K_M` hoặc `7b`)   |
| **Xử lý phiên âm bồi**  | `g2p-en` + Bộ phân tách âm tiết MOP tự xây dựng |
| **Lồng tiếng (TTS)**    | Microsoft Edge-TTS (`vi-VN-NamMinhNeural`)      |
| **Giao diện & API**     | FastAPI, Gradio                                 |
| **Xử lý âm thanh**      | `pydub`, `librosa`, `soundfile`                 |

---

## 🏗️ Cấu Trúc Thư Mục Dự Án

```text
.
├── api.py                 # FastAPI Async API Server (Backend)
├── app.py                 # Gradio Web UI (Giao diện người dùng Glassmorphism)
├── config.py              # File cấu hình tập trung (Đường dẫn, Cổng, Model)
├── main.py                # Điểm khởi chạy chính (CLI / API / Web)
├── requirements.txt       # Danh sách thư viện phụ thuộc
├── pipeline/              # Thư mục chứa các bước của Pipeline
│   ├── orchestrator.py    # Quản lý luồng chạy giữa các bước
│   ├── step1_ingestion.py # Tải video và trích xuất âm thanh gốc
│   ├── step2_context.py   # Phân tích giọng nói và thời lượng thoại
│   ├── step3_stt.py       # Nhận dạng giọng nói sang văn bản (Whisper)
│   ├── step4_translate.py # Dịch phụ đề song ngữ & trích xuất từ chuyên ngành
│   ├── step5_tts.py       # Tạo tiếng nói bồi & căn chỉnh tốc độ khớp hình
│   └── step6_mux.py       # Trộn âm thanh, phụ đề cứng vào video bằng FFmpeg
└── utils/                 # Các module tiện ích dùng chung
    ├── g2p_helper.py      # Bộ phiên âm từ tiếng Anh sang tiếng Việt bồi
    ├── logger.py          # Bộ ghi log tập trung ra console và file
    ├── memory.py          # Quản lý giải phóng bộ nhớ RAM và VRAM
    └── srt_utils.py       # Xử lý tệp phụ đề SRT và đồng bộ thời gian
```

---

## ⚡ Hướng Dẫn Nhanh

### 🔌 Yêu Cầu Cài Đặt Trước

1. **Ollama**: Tải và cài đặt [Ollama](https://ollama.com/), sau đó tải mô hình dịch thuật bằng lệnh:
   ```bash
   ollama pull qwen2.5:3b-instruct-q4_K_M
   ```
2. **FFmpeg**: Cài đặt `ffmpeg` và đảm bảo đã thêm vào biến môi trường `PATH` của hệ thống.
   - Windows: `winget install Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`

### ⚙️ Thiết Lập Môi Trường Local

```bash
# 1. Tải mã nguồn về máy
git clone https://github.com/HoangKhang226/VidTranscribe.ai.git
cd VidTranscribe.ai

# 2. Khởi tạo môi trường ảo Python
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
# Kích hoạt trên Linux/macOS:
source venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 🚀 Khởi Chạy Hệ Thống

Bạn có thể chạy dự án ở 4 chế độ khác nhau thông qua file `main.py`:

```bash
# Chế độ 1: Chạy toàn bộ hệ thống (Cả Backend API và Giao diện Web UI)
python main.py --mode all

# Chế độ 2: Chạy trực tiếp qua dòng lệnh CLI (cho URL YouTube hoặc file video local)
python main.py --mode cli --source "https://www.youtube.com/watch?v=example" --model "qwen2.5:3b-instruct-q4_K_M"

# Chế độ 3: Chỉ khởi chạy giao diện Web UI Gradio
python main.py --mode web

# Chế độ 4: Chỉ khởi chạy FastAPI API Backend
python main.py --mode api
```

Sau khi khởi chạy:
- **Giao diện Web UI** hoạt động tại: `http://127.0.0.1:7860`
- **Tài liệu API Swagger** hoạt động tại: `http://127.0.0.1:8000/docs`

---

## 🎨 Giao Diện Điều Khiển Gradio

Giao diện Web UI cung cấp đầy đủ công cụ:
- **Tải lên video trực tiếp** hoặc **Dán link YouTube**.
- **Lựa chọn mô hình dịch thuật** linh hoạt thông qua dropdown.
- **Theo dõi log tiến trình chạy** theo thời gian thực ngay trên màn hình.
- **Trình phát video tích hợp** giúp xem và đối chiếu trực tiếp video gốc với video đã lồng tiếng sau khi xử lý xong.

---

## 🔒 Điều Chỉnh Phiên Âm Thủ Công

Trong quá trình xử lý, tất cả các từ tiếng Anh được hệ thống phiên âm sẽ được lưu trữ tạm thời tại file:
`output/subtitles/phonetic_cache.json`

Nếu bạn muốn tùy chỉnh cách đọc của một từ chuyên ngành theo ý thích cá nhân (ví dụ: muốn từ `"marketing"` đọc là `"ma két tinh"` thay vì `"ma cờ tinh"` mặc định), bạn có thể **mở file này ra và sửa trực tiếp**. Trong những lần chạy tiếp theo, hệ thống sẽ ưu tiên đọc các từ đã chỉnh sửa trong cache này mà không cần tự động tạo lại.

---
_Dự án được xây dựng và tối ưu hóa theo các tiêu chuẩn công nghệ AI hiện đại._
