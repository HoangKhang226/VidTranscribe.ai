# VidTranscribe.ai Evaluation Benchmarks

Thư mục `evaluation/` chứa bộ benchmark tự động để chứng minh pipeline xử lý được các bài toán khó của project:

1. **Performance & Hardware** - đo tốc độ xử lý end-to-end, RTF và VRAM.
2. **Group-level Sync** - đo độ lệch timing đúng theo tầng TTS group, không đo per-subtitle.
3. **Localization LLM Judge** - dùng Gemini 2.5 chấm chất lượng dịch/localization theo batch 20 câu với LangChain structured output.
4. **Optional G2P** - đo riêng chất lượng phiên âm thuật ngữ Anh → tiếng Việt bồi.

Các benchmark chạy bằng Python trong project `venv` hoặc trực tiếp trên Google Colab qua `Colab_Evaluation.ipynb`.

---

## 0. Chuẩn bị dataset hybrid 100 câu

Benchmark localization hiện dùng dataset hybrid:

| Nguồn | Tỷ lệ | Số câu | Vai trò |
|---|---:|---:|---|
| PhoST local | 70% | 70 | Spoken-flow / câu dịch tự nhiên |
| ViMedCSS | 30% | 30 | Medical code-switching / OOV stress test |

### PhoST local

Repo chứa sẵn 2 file song song:

```text
evaluation/data/1188.en
evaluation/data/1188.vi
```

Script sẽ ghép:

```text
1188.en dòng n -> 1188.vi dòng n
```

### ViMedCSS

Nguồn:

```text
tensorxt/ViMedCSS
```

Chỉ stream text metadata, không tải full audio:

```python
load_dataset("tensorxt/ViMedCSS", split="train", streaming=True)
```

Các cột được dùng:

| Cột | Vai trò |
|---|---|
| `segment_text` | `input_raw`, câu chứa code-switching/OOV |
| `cs_terms_list` | danh sách thuật ngữ bẫy |
| `segment_id` | ID case |
| `topic` | domain/topic |
| `original_video_link` | metadata nguồn |
| `start_time`, `end_time` | metadata segment |

### Build dataset

```powershell
venv\Scripts\python.exe evaluation\build_hybrid_dataset.py `
  --phost-limit 70 `
  --stress-limit 30 `
  --output evaluation\test_cases.json
```

Output:

```text
evaluation/test_cases.json
```

Schema chính:

```json
{
  "id": "...",
  "source": "PhoST/local hoặc tensorxt/ViMedCSS/streaming",
  "domain": "speech_flow hoặc medical_code_switching",
  "type": "spoken_translation hoặc medical_code_switching",
  "english_original": "...",
  "input_raw": "...",
  "ai_translation": "...",
  "expected_localization": "...",
  "cs_terms_list": []
}
```

---

## 1. Localization Benchmark - Gemini 2.5 LLM Judge

### File

- `benchmark_localization.py`
- input: `evaluation/test_cases.json`
- output:
  - `evaluation/localization_results.csv`
  - `evaluation/localization_results.json`

### Mục đích

Đo chất lượng localization/dịch theo tiêu chí gần với trải nghiệm người nghe hơn exact match:

- đúng nghĩa so với source/reference
- tiếng Việt tự nhiên
- xử lý code-switching/OOV
- phù hợp để đưa vào TTS

### Cách chạy local

Cần set Gemini API key:

```powershell
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

Chạy judge:

```powershell
venv\Scripts\python.exe evaluation\benchmark_localization.py `
  --judge `
  --judge-model gemini-2.5-flash-preview-05-20 `
  --judge-batch-size 20 `
  --judge-concurrency 5
```

### Thiết kế batch/parallel

Với 100 câu:

```text
100 cases / 20 cases per batch = 5 batches
```

Script dùng:

```python
asyncio.gather(...)
asyncio.Semaphore(judge_concurrency)
```

Do đó có thể gửi song song nhiều batch lên Gemini. Mặc định khuyến nghị trên Colab:

```text
--judge-batch-size 20
--judge-concurrency 5
```

Nếu bị rate limit, giảm xuống:

```text
--judge-concurrency 3
```

hoặc:

```text
--judge-batch-size 10 --judge-concurrency 5
```

### Structured output

Benchmark dùng LangChain:

```python
ChatGoogleGenerativeAI(...).with_structured_output(JudgeBatch)
```

Schema:

```python
class JudgeItem(BaseModel):
    id: str
    score: float  # 0..5
    reason: str

class JudgeBatch(BaseModel):
    results: list[JudgeItem]
```

### CSV output

CSV có các cột quan trọng:

| Cột | Ý nghĩa |
|---|---|
| `english_original` | câu gốc/source |
| `input_raw` | input raw |
| `ai_translation` | bản dịch/localization cần chấm |
| `expected_localization` | đáp án/reference |
| `token_error_rate` | TER tham khảo |
| `ai_score` | điểm Gemini 0-5 |
| `ai_reason` | lý do chấm |

---

## 2. Performance & Hardware Benchmark

### File

- `benchmark_perf.py`
- video mặc định: `evaluation/Download.mp4`
- output: `evaluation/perf_results.json`

### Mục đích

Đo pipeline có đủ thực tế để đưa vào production/demo hay không:

- tốc độ xử lý tổng thể
- output cuối có sinh thành công không
- PyTorch VRAM nếu có CUDA
- optional system-level VRAM delta bằng `nvidia-smi`

### Metric

```text
RTF = video_duration_sec / processing_time_sec
```

Target production-oriented:

```text
RTF > 1.5x
Peak VRAM < 4.5 GB
```

### Cách chạy

Full end-to-end và giữ TTS segments để chạy sync benchmark:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py `
  --mode end_to_end `
  --keep-temp-segments `
  --include-nvidia-smi
```

Translate-only để test nhanh:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode translate_only
```

> `torch.cuda.max_memory_allocated()` chỉ đo VRAM do PyTorch quản lý, không bao gồm Ollama/GGML. Muốn gần với tổng VRAM hơn, dùng `--include-nvidia-smi` trên máy NVIDIA.

---

## 3. Group-level Sync Benchmark

### File

- `benchmark_sync_group.py`
- input mặc định: `src/output/audio/temp_segments/tts_groups.json`
- output:
  - `evaluation/sync_group_results.csv`
  - `evaluation/sync_group_results.json`

### Mục đích

Pipeline không đọc từng dòng SRT đơn lẻ. `step5_tts.py` gộp các dòng phụ đề liên tiếp thành `temp_group_*.mp3` để giọng đọc không bị khựng do Whisper sliding window cắt câu giữa chừng.

Vì vậy benchmark sync đúng phải đo theo **TTS group**, không đo per-subtitle.

### Metric

Với mỗi group:

```text
allowed_window_ms = end_ms(last subtitle in group) - start_ms(first subtitle in group)
overflow_ms = max(0, audio_duration_ms - allowed_window_ms)
MAE = mean(overflow_ms)
```

Target:

```text
MAE < 150ms
```

### Cách chạy

Trước tiên chạy full pipeline và giữ temp segments:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments
```

Sau đó chạy sync group benchmark:

```powershell
venv\Scripts\python.exe evaluation\benchmark_sync_group.py
```

Tùy chỉnh input/output:

```powershell
venv\Scripts\python.exe evaluation\benchmark_sync_group.py `
  --groups-json src\output\audio\temp_segments\tts_groups.json `
  --target-mae-ms 150
```

---

## 4. Optional G2P Benchmark

File:

```text
evaluation/benchmark_g2p.py
```

Chạy:

```powershell
venv\Scripts\python.exe evaluation\benchmark_g2p.py
```

Benchmark này đo riêng chất lượng phiên âm thuật ngữ tiếng Anh/OOV sang tiếng Việt bồi. Các lỗi điển hình hiện nằm ở nhóm alphanumeric:

```text
BM25, HTML5, OAuth2, FFmpeg, Nginx, PostgreSQL
```

Hướng xử lý hiện tại trong pipeline:

```text
domain dictionary phonetic > domain/global phonetic cache > G2P fallback
```

---

## 5. Quy trình chạy đủ bộ benchmark CV local

```powershell
# 0. Build/check hybrid dataset 100 câu
venv\Scripts\python.exe evaluation\build_hybrid_dataset.py --phost-limit 70 --stress-limit 30 --output evaluation\test_cases.json

# 1. Localization LLM Judge
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
venv\Scripts\python.exe evaluation\benchmark_localization.py --judge --judge-model gemini-2.5-flash-preview-05-20 --judge-batch-size 20 --judge-concurrency 5

# 2. Full performance + preserve TTS group artifacts
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments --include-nvidia-smi

# 3. Group-level sync quality
venv\Scripts\python.exe evaluation\benchmark_sync_group.py

# 4. Optional G2P
venv\Scripts\python.exe evaluation\benchmark_g2p.py
```

---

## 6. Google Colab

Dùng notebook ở root project:

```text
Colab_Evaluation.ipynb
```

Notebook đã có sẵn các cell:

1. Mount Google Drive
2. Clone/pull repo
3. Cài FFmpeg + Ollama
4. Pull model Ollama
5. `pip install -r requirements.txt`
6. Nhập `GOOGLE_API_KEY`
7. Kiểm tra/build dataset 100 câu
8. Chạy Gemini 2.5 judge, 20 câu/batch, concurrency 5
9. Chạy performance benchmark
10. Chạy group sync benchmark
11. Copy kết quả về Drive

---

## 7. CV phrasing gợi ý

> Built a hybrid evaluation suite for a local-first AI video dubbing pipeline, combining public spoken translation data and medical code-switching stress tests to benchmark Vietnamese localization quality, end-to-end RTF/VRAM, and group-level audio-video synchronization.
>
> Implemented a Gemini 2.5 LLM-as-a-judge benchmark with LangChain structured output, batching 20 cases per request and running batches concurrently to score localization quality across 100 hybrid test cases.
