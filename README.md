# 🎙️ VidTranscribe.ai - Agentic Video Dubbing Pipeline

**VidTranscribe.ai** là một pipeline lồng tiếng và dịch thuật video hoàn toàn tự động, được thiết kế với kiến trúc **Agentic AI** chuyên sâu để giải quyết những bài toán khó nhất trong việc lồng tiếng đa ngôn ngữ: Giữ nguyên thuật ngữ chuyên ngành đa lĩnh vực, chống "dịch thô" (Literal Translation), và duy trì giọng đọc tự nhiên khớp hoàn hảo với chuyển động video.

---

## Tính Năng Nổi Bật (Advanced Features)

### 1. Dynamic Two-Pass Translation (Dịch Thuật Đa Ngành Động)

Không giống các hệ thống dịch hardcode, VidTranscribe.ai áp dụng luồng chạy 2 bước (Two-Pass):

- **Pass 1 (Context Analyzer):** LLM quét lướt toàn bộ kịch bản để nhận diện ngữ cảnh và trích xuất tự động các Thuật ngữ chuyên ngành, Tên riêng (Proper Nouns) phù hợp với lĩnh vực của video (VD: Y tế, Tài chính, IT).
- **Pass 2 (Translator Injector):** Các thuật ngữ này được bơm động (inject) vào Hệ quy tắc cốt lõi (Core Rules). Hệ thống sẽ bắt LLM bảo toàn thuật ngữ (VD: _Front-end, Copilot, MRI_) và dịch thoát ý tự nhiên (VD: _"You don't."_ -> _"Chưa chắc đâu."_ thay vì _"Bạn không làm."_).

### 2. Robust Fallback System (Cơ Chế Chống Hallucination)

Với các LLM local cỡ nhỏ (như Qwen 2.5 7B), hiện tượng "ảo giác" (nhả chữ tiếng Trung, rò rỉ prompt) rất dễ xảy ra. Pipeline giải quyết triệt để bằng hệ thống Fallback 3 tầng:

- Tầng 1: Dịch theo Batch JSON (Tốc độ cao).
- Tầng 2: Nếu lỗi/rò rỉ -> Chuyển sang Dịch Paragraph thô.
- Tầng 3: Nếu vẫn lỗi -> Rã nhỏ kịch bản và Dịch từng câu lẻ (Kèm Context) để ép model tuân thủ.

### 3. G2P Engine (Phiên Âm Thuật Ngữ Động)

Thay vì để các mô hình TTS tiếng Việt (như Edge-TTS vi-VN) đọc sai bét các từ tiếng Anh hoặc từ chối sinh âm thanh, hệ thống tích hợp bộ mã hóa G2P (Grapheme-to-Phoneme):

- Tự động nhận diện từ vựng tiếng Anh trong câu tiếng Việt.
- Chuyển đổi thành phiên âm bồi chuẩn tả (VD: _"Agent"_ -> _"Ây dần"_, _"seamlessly"_ -> _"xim lét xli"_).
- Chỉ phiên âm đúng các thuật ngữ cần thiết, **không bao giờ over-phonetize** (phiên âm lố) các câu tiếng Việt bình thường.

### 4. "Cheating Gaps" Audio Sync (Đồng Bộ Âm Thanh Thông Minh)

Khi thời lượng dịch tiếng Việt dài hơn tiếng Anh gốc, thay vì "ép tốc độ" (Speedup x1.5, x1.8) làm biến dạng giọng đọc thành tiếng sóc chuột (chipmunk), hệ thống áp dụng thuật toán **Cheating Gaps**:

- Tự động tính toán các khoảng lặng (silence gaps) trước và sau mỗi câu phụ đề.
- "Vay mượn" thời gian từ các khoảng trống này (tối đa 1.5s) để mở rộng khung giờ nói.
- Giảm thiểu tối đa việc phải ép tốc độ đọc, giữ cho âm thanh phát ra tự nhiên nhất mà vẫn **chuẩn khớp 100% với khung hình**.

---

## Pipeline Workflow

Hệ thống hoạt động theo 6 bước tuần tự, được kiểm soát bộ nhớ nghiêm ngặt để có thể chạy mượt mà trên phần cứng giới hạn:

1. `step1_extract.py`: Tách âm thanh từ file Video gốc.
2. `step2_stt.py`: Nhận dạng giọng nói (Speech-to-Text) tạo phụ đề tiếng Anh gốc.
3. `step3_sync.py`: Đồng bộ và chia nhỏ đoạn (Paragraph Batching).
4. `step4_translate.py`: Dịch thuật Agentic song ngữ (Two-pass & Fallback).
5. `step5_tts.py`: Phân tích G2P và Tổng hợp giọng nói có ăn gian khoảng trống (Cheating Gaps).
6. `step6_merge.py`: Merge (Mux) âm thanh đã lồng tiếng ngược lại vào video gốc với FFMPEG.

---

## Công Nghệ Sử Dụng (Tech Stack)

- **Core LLM:** Qwen 2.5 7B (qua Ollama).
- **Orchestration:** LangChain, Pydantic.
- **Audio Processing:** Pydub, Edge-TTS.
- **UI/UX:** Gradio (Giao diện web trực quan).

---

## Tầm Nhìn Dự Án

**VidTranscribe.ai** là một hệ thống tự động hóa hoàn toàn quy trình dịch thuật, căn chỉnh phụ đề và lồng tiếng video chất lượng cao từ tiếng Anh sang tiếng Việt. Dự án giúp chuyển đổi các video tiếng Anh thành phiên bản tiếng Việt có kèm phụ đề căn chỉnh khớp thời gian và âm thanh lồng tiếng tự nhiên.

Điểm vượt trội của **VidTranscribe.ai** là thiết kế **hoàn toàn đa ngành (domain-agnostic)**. Hệ thống có khả năng xử lý chính xác video của mọi lĩnh vực (Y sinh, Tài chính, Công nghệ, Doanh nghiệp, Hóa học, Vật lý...) nhờ công cụ phiên âm ngôn ngữ học tùy biến: tự động nhận diện từ viết tắt, tách từ ghép CamelCase và phiên âm sang tiếng Việt bồi để bộ đọc phát âm chuẩn xác mà không cần dựa vào các bộ từ điển cứng.

---

## Tính Năng Nổi Bật

- **Pipeline 6 Bước Modul Hóa**: Tách biệt rõ ràng các nhiệm vụ trong luồng xử lý:
  1. **Data Ingestion (Thu Thập)**: Nhận tệp tin video local, bóc tách luồng âm thanh gốc chất lượng cao.
  2. **Context Analysis (Phân Tích)**: Phát hiện hoạt động giọng nói (VAD) và tính toán khoảng lặng để tối ưu hóa việc căn chỉnh thời gian.
  3. **Speech-To-Text (STT)**: Chuyển âm thanh thành văn bản ngoại tuyến (offline) bằng `Faster-Whisper` (hỗ trợ cả CPU và GPU).
  4. **Bilingual Translation (Dịch Song Ngữ)**: Sử dụng LLM chạy cục bộ thông qua Ollama (Qwen2.5) với cơ chế chống rò rỉ chữ Trung Quốc và tự động dịch đơn lẻ dự phòng khi lỗi batch.
  5. **Domain-Agnostic Speech Synthesis (TTS)**: Bộ công cụ chuyển đổi chữ cái tiếng Anh sang tiếng Việt bồi tự động:
     - **Tách từ CamelCase**: Nhận diện từ ghép (Ví dụ: `pH` -> `pi ếch`, `mRNA` -> `em a en ê`, `ChatGPT` -> `chat gi pi ti`).
     - **Đánh vần từ viết tắt (Acronyms)**: Đọc từng chữ cái theo phát âm chuẩn (Ví dụ: `CRM` -> `xi a em`, `DNA` -> `đi en ây`, `SQL` -> `ét qui eo`).
     - **Phân tách âm tiết MOP**: Áp dụng nguyên lý _Maximal Onset Principle_ để tự động tạo cách đọc tiếng Việt bồi cho các từ tiếng Anh thông thường.
     - **Tích hợp Edge-TTS**: Căn chỉnh tốc độ nói tự động theo độ dài câu thoại gốc và cơ chế thử lại (exponential backoff) khi bị giới hạn lượt gọi.
  6. **Video Muxing (Trộn Video)**: Tự động trộn phụ đề (hardsub hoặc softsub) và đè âm thanh lồng tiếng khớp mốc thời gian bằng `FFmpeg`.
- **Kiến Trúc Giao Diện Kép**:
  - **FastAPI Backend**: Cung cấp API bất đồng bộ (Async) mạnh mẽ, chạy ngầm các tác vụ nặng và hỗ trợ kiểm tra tiến trình.
  - **Gradio Web UI**: Giao diện Web tối giản với thiết kế Dark Mode Glassmorphism cao cấp, hỗ trợ thao tác chỉ với một cú click chuột.
- **Cơ Chế Giải Phóng Bộ Nhớ**: Tự động giải phóng RAM/VRAM và dọn dẹp cache PyTorch ngay sau khi hoàn thành các bước nặng để tránh tràn bộ nhớ.

---

## Công Nghệ Sử Dụng

| Danh mục                | Công cụ / Thư viện                              |
| :---------------------- | :---------------------------------------------- |
| **Ngôn ngữ chính**      | Python 3.12                                     |
| **Xử lý video**         | FFmpeg                                          |
| **Nhận diện giọng nói** | `faster-whisper` (medium/large-v3-turbo)        |
| **Dịch thuật (LLM)**    | Ollama (`qwen2.5:3b-instruct-q4_K_M` hoặc `7b`) |
| **Xử lý phiên âm bồi**  | `g2p-en` + Bộ phân tách âm tiết MOP tự xây dựng |
| **Lồng tiếng (TTS)**    | Microsoft Edge-TTS (`vi-VN-NamMinhNeural`)      |
| **Giao diện & API**     | FastAPI, Gradio                                 |
| **Xử lý âm thanh**      | `pydub`, `librosa`, `soundfile`                 |

---

## Cấu Trúc Thư Mục Dự Án

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

## Hướng Dẫn Nhanh

### Yêu Cầu Cài Đặt Trước

1. **Ollama**: Tải và cài đặt [Ollama](https://ollama.com/), sau đó tải mô hình dịch thuật bằng lệnh:
   ```bash
   ollama pull qwen2.5:3b-instruct-q4_K_M
   ```
2. **FFmpeg**: Cài đặt `ffmpeg` và đảm bảo đã thêm vào biến môi trường `PATH` của hệ thống.
   - Windows: `winget install Gyan.FFmpeg`
   - Linux: `sudo apt install ffmpeg`

### Thiết Lập Môi Trường Local

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

### Khởi Chạy Hệ Thống

Bạn có thể chạy dự án ở 4 chế độ khác nhau thông qua file `main.py`:

```bash
# Chế độ 1: Chạy toàn bộ hệ thống (Cả Backend API và Giao diện Web UI)
python main.py --mode all

# Chế độ 2: Chạy trực tiếp qua dòng lệnh CLI (cho file video local)
python main.py --mode cli --source "path/to/video.mp4" --model "qwen2.5:3b-instruct-q4_K_M"
```

# Chế độ 3: Chỉ khởi chạy giao diện Web UI Gradio

python main.py --mode web

# Chế độ 4: Chỉ khởi chạy FastAPI API Backend

python main.py --mode api

```

Sau khi khởi chạy:
- **Giao diện Web UI** hoạt động tại: `http://127.0.0.1:7860`
- **Tài liệu API Swagger** hoạt động tại: `http://127.0.0.1:8000/docs`

---

## Giao Diện Điều Khiển Gradio

Giao diện Web UI cung cấp đầy đủ công cụ:
- **Tải lên video trực tiếp** từ máy tính (.mp4, .mkv, .mov).
- **Lựa chọn mô hình dịch thuật** linh hoạt thông qua dropdown.
- **Theo dõi log tiến trình chạy** theo thời gian thực ngay trên màn hình.
- **Trình phát video tích hợp** giúp xem và đối chiếu trực tiếp video gốc với video đã lồng tiếng sau khi xử lý xong.

---

## Điều Chỉnh Phiên Âm Thủ Công

Trong quá trình xử lý, tất cả các từ tiếng Anh được hệ thống phiên âm sẽ được lưu trữ tạm thời tại file:
`output/subtitles/phonetic_cache.json`

Nếu bạn muốn tùy chỉnh cách đọc của một từ chuyên ngành theo ý thích cá nhân (ví dụ: muốn từ `"marketing"` đọc là `"ma két tinh"` thay vì `"ma cờ tinh"` mặc định), bạn có thể **mở file này ra và sửa trực tiếp**. Trong những lần chạy tiếp theo, hệ thống sẽ ưu tiên đọc các từ đã chỉnh sửa trong cache này mà không cần tự động tạo lại.

---
_Dự án được xây dựng và tối ưu hóa theo các tiêu chuẩn công nghệ AI hiện đại._
```
