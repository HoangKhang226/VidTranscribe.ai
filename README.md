# VidTranscribe.ai: Agentic Video Dubbing & Translation Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

## Giới thiệu

VidTranscribe.ai là một **pipeline lồng tiếng & dịch thuật video tự động theo hướng Agentic AI**, chạy hoàn toàn cục bộ (local-first). Hệ thống nhận video tiếng Anh, tự nhận diện chủ đề/ngành, dịch sang tiếng Việt với độ chính xác thuật ngữ cao, phiên âm các từ tiếng Anh sang "tiếng Việt bồi" để bộ đọc phát âm đúng, rồi lồng tiếng và ghép lại thành video hoàn chỉnh.

Dự án được thiết kế **đa ngành** (multi-domain): y tế, tài chính, pháp lý, công nghệ, giáo dục, ẩm thực, thể thao... Không có luật/từ vựng nào bị hardcode cho riêng một lĩnh vực.

---

## Tính năng chính

- **Pipeline 6 bước module hóa**: tách biệt rõ ràng giữa xử lý audio, nhận dạng, dịch, lồng tiếng và ghép video.
- **Hai chế độ xử lý**:
  - **End-to-End**: chạy đầy đủ và xuất luôn video lồng tiếng cuối cùng.
  - **Chỉ dịch (Review)**: dừng sau bước dịch, hiển thị video gốc kèm phụ đề tiếng Việt để người dùng **xem & chỉnh sửa trực tiếp**, sau đó bấm **Tiếp tục lồng tiếng** để hoàn tất.
- **Dịch đa lượt theo ngữ cảnh**:
  - **Context Analyzer**: tự trích xuất thuật ngữ đặc thù ngành từ 30 giây đầu của transcript (đa ngành).
  - **Sliding-Window Translation**: dịch theo lô có ngữ cảnh gối đầu để LLM hiểu mạch hội thoại.
- **Khắc phục lỗi dịch sót (Incomplete Translation Recovery)** — 3 lớp phòng thủ:
  1. Tự kiểm tra trong lô (đủ index, tỉ lệ độ dài, chống rò rỉ ký tự lạ, fallback dịch từng câu).
  2. `PostTranslationValidator`: phát hiện từ tiếng Anh còn sót dựa trên đối chiếu câu gốc (đa ngành, không cần danh sách từ tiếng Việt hardcode).
  3. `TranslationAdjudicator` (LLM): phán xử từng từ còn sót → **dịch bổ sung** (từ phổ thông bị quên) hoặc **giữ nguyên** (thuật ngữ mới do mô hình suy luận), kèm bảo vệ cụm thuật ngữ nhiều từ. Chi tiết: [`docs/incomplete_translation_recovery.md`](docs/incomplete_translation_recovery.md).
- **Từ điển ngành có metadata**: mỗi thuật ngữ lưu kèm _tiếng Việt bồi, giải thích, ngày bổ sung, người bổ sung (AI/Human), ngày cập nhật cuối_. AI tự ghi nhận thuật ngữ mới; người dùng thêm/sửa/xóa tùy ý qua giao diện.
- **G2P (Grapheme-to-Phoneme) đa ngành**: phiên âm chính xác các từ/viết tắt tiếng Anh sang tiếng Việt bồi (vd "Agent" → "Ây giơn") mà không phiên âm nhầm ngôn ngữ giao tiếp thông thường.
- **Đồng bộ audio "mượn khoảng lặng"**: thuật toán tính khoảng im lặng trước/sau câu để kéo dài cửa sổ đọc TTS, giảm thiểu việc tăng tốc audio quá đà, giữ giọng tự nhiên và khớp hình.

## Kết quả benchmark mới nhất

| Benchmark | Result | Status |
|---|---:|---|
| Localization LLM Judge | 3.91/5 trên 100 cases | WARN |
| Performance & Hardware | RTF 0.2543x, VRAM delta 4444 MB | PASS |
| Group-level Sync | MAE overflow 43.75 ms | PASS |

> Kết quả trên được lấy từ Colab benchmark gần nhất. Localization vẫn còn vài case điểm thấp, nhưng pipeline end-to-end, performance và sync đều đã chạy ổn định trên GPU.

---

## Công nghệ & Kiến trúc

| Hạng mục               | Công cụ                              |
| :--------------------- | :----------------------------------- |
| **Ngôn ngữ**           | Python 3.12                          |
| **LLM cục bộ**         | Gemma 4 E4B (qua Ollama)             |
| **Điều phối**          | LangChain, Pydantic                  |
| **Speech-to-Text**     | Faster-Whisper                       |
| **Text-to-Speech**     | Edge-TTS (Microsoft)                 |
| **Audio/Video Engine** | FFmpeg, Pydub                        |
| **Web Frameworks**     | FastAPI (Backend), Gradio (Frontend) |

---

## Vòng đời pipeline (6 bước)

Mỗi bước là một module độc lập trong `src/pipeline/`, được điều phối bởi `orchestrator.py`.

1. **step1_ingestion.py** — Tách audio khỏi video gốc bằng FFmpeg, chuẩn hóa định dạng cho STT; tạo bản video không tiếng để ghép lại sau.
2. **step2_context.py** — _Dynamic Context Layer_: chạy Whisper-tiny trên 30s đầu, dùng LLM nhận diện chủ đề/ngành và trích xuất thuật ngữ cần giữ nguyên (đa ngành). Hỗ trợ chỉ định ngành thủ công (Domain Override).
3. **step3_stt.py** — Speech-to-Text đầy đủ bằng Faster-Whisper, dùng thuật ngữ ở bước 2 làm `initial_prompt` để nhận dạng đúng; xuất `subtitles_en.srt`.
4. **step4_translate.py** — Dịch Agentic: sliding-window theo lô, RAG từ điển ngành, validator + adjudicator khắc phục lỗi dịch sót, G2P phiên âm; xuất `subtitles_vi.srt` + metadata `subtitles_vi.json`.
5. **step5_tts.py** — Tổng hợp giọng nói Edge-TTS, gộp câu để đọc tự nhiên, "mượn khoảng lặng" + time-stretch để khớp timeline.
6. **step6_mux.py** — Ghép (mux) audio tiếng Việt vào video bằng FFmpeg, xuất video cuối.

> Ghi chú: ở **chế độ Chỉ dịch**, pipeline dừng sau bước 4. Sau khi người dùng sửa phụ đề, `orchestrator.resume_from_subtitles()` chạy tiếp bước 5 & 6.

---

## Bắt đầu nhanh

```bash
# 1. Clone & vào thư mục
git clone https://github.com/HoangKhang226/VidTranscribe.ai.git
cd VidTranscribe.ai

# 2. Tạo môi trường
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 3. Tải LLM cục bộ qua Ollama
ollama pull gemma4:e4b

# 4. Chạy toàn hệ thống (Backend API + Gradio UI)
python main.py --mode all

# Hoặc chạy trực tiếp bằng CLI:
python main.py --mode cli --source "video.mp4" --model "gemma4:e4b"

# Chế độ chỉ dịch (dừng để xem/sửa phụ đề) qua CLI:
python main.py --mode cli --source "video.mp4" --pipeline-mode translate_only
```

Yêu cầu thêm: **FFmpeg** phải có trong PATH; **Ollama** đang chạy ở `http://localhost:11434`.

---

## Cấu trúc dự án

```text
.
├── main.py                 # Điểm khởi chạy (CLI / API / Web / all)
├── docs/
│   └── incomplete_translation_recovery.md   # Thiết kế khắc phục lỗi dịch sót
├── output/                 # Dữ liệu đầu ra (audio, subtitles, final video, logs)
└── src/
    ├── api.py              # FastAPI Backend
    ├── app.py              # Gradio Frontend (2 chế độ + Knowledge Base)
    ├── config.py           # Cấu hình tập trung
    ├── db/
    │   ├── db_manager.py   # Từ điển ngành (schema metadata) + phonetic cache
    │   └── dictionaries/   # Từ điển theo ngành (JSON)
    ├── pipeline/           # 6 bước pipeline + orchestrator
    ├── routers/            # Router FastAPI (pipeline, dictionary)
    └── utils/
        ├── g2p_helper.py                 # Engine G2P đa ngành
        ├── post_translation_validator.py # Phát hiện từ tiếng Anh còn sót
        ├── translation_adjudicator.py    # Trọng tài hậu dịch (LLM)
        ├── translation_cache.py
        └── srt_utils.py
```

---

## Đóng góp

Dự án được phát triển và duy trì bởi **[HoangKhang226](https://github.com/HoangKhang226)**.
