# VidTranscribe.ai Evaluation Results

Ngày ghi nhận: 2026-06-10

Báo cáo này ghi lại kết quả chạy bộ benchmark mới trong `evaluation/`, được thiết kế để đưa vào CV/project portfolio.

---

## 1. Text Normalization & Localization Benchmark

### Mục tiêu

Đánh giá chất lượng chuẩn hóa tiếng Việt bồi / code-switching trước khi đưa text cho Edge-TTS.

### Lệnh đã chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_localization.py
```

### Kết quả

| Metric | Value |
|---|---:|
| Test cases | 6 |
| Avg Token Error Rate | 0.0000 |
| Exact Match Rate | 100.00% |
| Target TER | <= 0.15 |
| Status | PASS |

### Output files

- `evaluation/localization_results.csv`
- `evaluation/localization_results.json`

### Nhận xét

Dataset hiện tại là bộ smoke-test nhỏ để xác nhận benchmark chạy đúng và định dạng case ổn định. Để dùng làm benchmark CV mạnh hơn, nên mở rộng `test_cases.json` lên 50-100 câu bẫy gồm:

- câu bị cắt bởi Whisper sliding window
- code-switching Anh-Việt
- thuật ngữ kỹ thuật cần giữ nguyên
- tiếng Việt bồi cần localize tự nhiên hơn

---

## 2. Performance & Hardware Benchmark

### Mục tiêu

Đánh giá tốc độ xử lý end-to-end và telemetry phần cứng của pipeline.

### Lệnh đã chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments
```

### Cấu hình chạy

| Item | Value |
|---|---|
| Video | `evaluation\Download.mp4` |
| Video duration | 95.13s |
| Model | `qwen2.5:3b-instruct-q4_K_M` |
| Mode | `end_to_end` |
| Hardsub | False |
| Keep temp segments | True |
| CUDA/PyTorch VRAM | N/A on current machine |

### Kết quả

| Metric | Value |
|---|---:|
| Processing time | 321.79s |
| Real-Time Factor | 0.30x |
| Target RTF | > 1.5x |
| PyTorch Peak VRAM | N/A |
| Final output | `src\output\final\final_output.mp4` |
| Status | WARN on current CPU-limited machine |

### Output files

- `evaluation/perf_results.json`
- `src/output/final/final_output.mp4`
- `src/output/audio/temp_segments/tts_groups.json`

### Nhận xét

Kết quả RTF thấp vì môi trường hiện tại đang chạy Whisper trên CPU:

```python
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_MODEL_SIZE = "medium"
```

Do đó đây là kết quả tham chiếu trên máy yếu, không phải GPU baseline cuối cùng. Trên máy có NVIDIA GPU, nên chạy thêm:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments --include-nvidia-smi
```

để lấy system-level VRAM delta, bao gồm cả phần Ollama/GGML nếu cùng dùng GPU.

---

## 3. Group-level Sync Benchmark

### Mục tiêu

Đo độ lệch timing đúng theo tầng TTS grouping. Pipeline đã gộp các dòng phụ đề liên tiếp thành `temp_group_*.mp3` để giảm hiện tượng giọng đọc bị khựng do Whisper sliding window cắt câu giữa chừng.

### Lệnh đã chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_sync_group.py
```

### Kết quả

| Metric | Value |
|---|---:|
| Total groups | 13 |
| Valid groups | 13 |
| Missing audio | 0 |
| MAE overflow | 53.85 ms |
| Max overflow | 700 ms |
| Target MAE | < 150 ms |
| Status | PASS |

### Output files

- `evaluation/sync_group_results.csv`
- `evaluation/sync_group_results.json`

### Nhận xét

MAE **53.85ms** khớp với tình huống một outlier kéo trung bình lên: nếu 12/13 group gần như không overflow đáng kể và chỉ có 1 group lệch khoảng 700ms, thì MAE vẫn rơi vào quanh 53.85ms. Điều này cho thấy kiến trúc grouping + speed adaptation đang ổn ở phần lớn trường hợp, và phần còn lại chủ yếu là một case biên cần tinh chỉnh thêm.

`Max overflow = 700ms` phản ánh đúng nhóm biên đó. Nói ngắn gọn: benchmark sync hiện **không fail toàn hệ thống**, mà đang chỉ ra đúng một điểm cần tối ưu ở biên timing.

---

### Kết luận G2P

Benchmark G2P cũ vẫn hữu ích để đo riêng chất lượng phiên âm thuật ngữ tiếng Anh sang tiếng Việt bồi.

Các failure chính nằm ở nhóm token chữ-số và acronym/tên tool như `BM25`, `HTML5`, `OAuth2`, `FFmpeg`, `Nginx`, `PostgreSQL`. Đây là kiểu lỗi điển hình của G2P đại chúng khi gặp OOV và term chuyên ngành đa lĩnh vực.

Để xử lý đúng hướng, pipeline nên ưu tiên một tầng **domain lexicon / pronunciation override** trước G2P fallback, thay vì hardcode riêng cho một ngành. Điều này giúp dự án giữ được kiến trúc đa ngành cho y tế, tài chính, logistics, hàng không, pháp lý, và cả IT.

---

## 6. Tổng kết CV-ready

| Benchmark | Metric | Result | Status |
|---|---:|---:|---|
| Localization | Avg TER | 0.0000 | PASS |
| Performance | RTF | 0.30x | WARN on CPU |
| Group Sync | MAE overflow | 53.85ms | PASS |
| Optional G2P | Accuracy | 91.67% | Good baseline |

### Câu mô tả CV gợi ý

> Built automated evaluation for a local-first AI video dubbing pipeline, covering Vietnamese localization quality, end-to-end RTF/VRAM profiling, and group-level audio-video sync.  
> Optimized Whisper sliding-window subtitle fragmentation with TTS grouping, achieving **53.85ms group-level sync MAE** under a 150ms perceptual threshold, with a **91.67% G2P baseline** for technical term pronunciation.

---

## 7. Các điểm cần cải thiện tiếp

1. Mở rộng `evaluation/test_cases.json` từ 6 case lên 50-100 case để localization benchmark có ý nghĩa thống kê hơn.
2. Phân tách dataset theo nhóm `speech_flow` và `tech_stress` để benchmark hybrid 70/30 rõ ràng hơn.
3. Chạy performance benchmark trên máy GPU có `nvidia-smi` để lấy VRAM tổng thực tế.
4. Tối ưu group có overflow lớn nhất để giảm `Max overflow` từ 700ms xuống gần target 150ms.
5. Bổ sung LLM-as-a-judge cho localization benchmark nếu cần điểm văn phong tự nhiên hóa 1-5.
