# Kiến trúc Pipeline & các Module

Thư mục này chứa logic cốt lõi của pipeline lồng tiếng VidTranscribe.ai. Kiến trúc được chia thành các bước module hóa, điều phối bởi một controller trung tâm. Toàn bộ thiết kế **đa ngành** — không hardcode luật/từ vựng cho riêng lĩnh vực nào.

## Cấu trúc thư mục

```text
pipeline/
├── orchestrator.py
├── step1_ingestion.py
├── step2_context.py
├── step3_stt.py
├── step4_translate.py
├── step5_tts.py
└── step6_mux.py
```

## Hai chế độ chạy

`orchestrator.run_pipeline(..., mode=...)`:

- **`end_to_end`**: chạy tuần tự bước 1 → 6, xuất video lồng tiếng cuối.
- **`translate_only`**: chạy bước 1 → 4 rồi DỪNG ở trạng thái `awaiting_review`, trả về `subtitles_vi.srt` + đường dẫn video gốc để người dùng xem & sửa. Sau khi sửa, gọi `orchestrator.resume_from_subtitles(hardsub=...)` để chạy tiếp bước 5 & 6 từ phụ đề đã chỉnh sửa.

## Chi tiết các Module

### orchestrator.py
Bộ não trung tâm: khởi tạo trình tự xử lý, quản lý chuyển trạng thái (`processing` → `awaiting_review`/`completed`/`failed`), gọi từng bước đúng thứ tự, phát tiến độ ra UI, và **dọn RAM/VRAM giữa các bước nặng** (vd giải phóng model STT trước khi gọi LLM). Cung cấp `run_pipeline`, `_finalize`, và `resume_from_subtitles`.

### step1_ingestion.py
Tách audio khỏi video gốc bằng FFmpeg, chuẩn hóa định dạng tối ưu cho STT, và tạo bản video không tiếng (`video_no_audio.mp4`) để ghép lại ở bước cuối.

### step2_context.py — Dynamic Context Layer
Chạy Whisper-tiny trên 30 giây đầu để lấy transcript thô, rồi dùng LLM (Ollama) nhận diện **chủ đề/ngành** và trích xuất **thuật ngữ đặc thù ngành** cần giữ nguyên tiếng Anh. Hỗ trợ **Domain Override** (người dùng chỉ định cứng ngành để tăng tốc, bỏ qua auto-detect).

### step3_stt.py
Speech-to-Text đầy đủ bằng Faster-Whisper. Dùng danh sách thuật ngữ từ bước 2 làm `initial_prompt` để nhận dạng đúng từ chuyên ngành. Xuất `subtitles_en.srt` với timestamp chính xác.

### step4_translate.py — Dịch Agentic (module phức tạp nhất)
- Dịch theo lô với **Sliding-Window Context** (ngữ cảnh gối đầu).
- **RAG từ điển ngành**: chèn nghĩa/ghi chú của thuật ngữ khớp vào prompt.
- **Khắc phục lỗi dịch sót** 3 lớp: tự kiểm tra trong lô → `PostTranslationValidator` (phát hiện từ tiếng Anh còn sót, đối chiếu câu gốc) → `TranslationAdjudicator` (LLM phán xử dịch/giữ nguyên, bảo vệ cụm thuật ngữ liền kề).
- **G2P**: phiên âm từ tiếng Anh sang tiếng Việt bồi cho TTS.
- **AI tự ghi nhận** thuật ngữ mới vào từ điển ngành (`added_by="AI"`).
- Xuất `subtitles_vi.srt` + metadata `subtitles_vi.json` (chứa `phonetic_text`).
- Xem thêm: [`../../docs/incomplete_translation_recovery.md`](../../docs/incomplete_translation_recovery.md).

### step5_tts.py
Tổng hợp giọng nói tiếng Việt bằng Microsoft Edge-TTS từ `phonetic_text`. Gộp các câu ngắn liên tiếp để đọc tự nhiên; áp dụng thuật toán **"mượn khoảng lặng"** + time-stretch (Edge rate + FFmpeg atempo) để khớp audio vào timeline gốc, tránh lệch hình.

### step6_mux.py
Dùng FFmpeg ghép (mux) audio tiếng Việt vào video gốc, xuất video cuối (`final_output.mp4`).

## Lưu ý tài nguyên (VRAM)

Các lệnh gọi LLM được cấu hình `num_ctx` nhỏ và tắt thinking mode (`think=False`) ở những bước chỉ cần ngữ cảnh ngắn (vd Adjudicator) để **giảm rủi ro tràn VRAM** trên GPU nhỏ. Bước STT (Whisper) và LLM được nạp/giải phóng tuần tự, không giữ đồng thời trên VRAM.
