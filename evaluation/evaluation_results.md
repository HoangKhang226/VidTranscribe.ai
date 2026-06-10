# VidTranscribe.ai Evaluation Results

Ngày cập nhật: 2026-06-10

Báo cáo này mô tả bộ benchmark CV-ready hiện tại của VidTranscribe.ai sau khi chuyển localization benchmark sang mô hình **LLM-as-a-Judge bằng Gemini 2.5** và dataset hybrid **100 câu = 70% PhoST + 30% ViMedCSS**.

---

## 1. Benchmark Suite hiện tại

| # | Benchmark | Script | Mục tiêu |
|---:|---|---|---|
| 1 | Performance & Hardware | `evaluation/benchmark_perf.py` | Đo tốc độ end-to-end, RTF, VRAM, output video |
| 2 | Group-level Sync | `evaluation/benchmark_sync_group.py` | Đo overflow audio theo group TTS thay vì từng subtitle |
| 3 | Localization LLM Judge | `evaluation/benchmark_localization.py` | Gemini 2.5 chấm chất lượng localize/dịch theo batch 20 câu |

> Benchmark 3 là phần mới nhất: không chỉ so chuỗi bằng TER, mà dùng Gemini 2.5 làm giám khảo để chấm độ tự nhiên, đúng nghĩa, và khả năng xử lý code-switching/OOV.

---

## 2. Hybrid Localization Dataset

Dataset evaluation được tạo bởi:

```powershell
venv\Scripts\python.exe evaluation\build_hybrid_dataset.py --phost-limit 70 --stress-limit 30 --output evaluation\test_cases.json
```

### Nguồn dữ liệu

| Nguồn | Tỷ lệ | Số câu | Vai trò |
|---|---:|---:|---|
| PhoST local | 70% | 70 | Spoken-flow / câu dịch tự nhiên |
| ViMedCSS | 30% | 30 | Medical code-switching / OOV stress test |

### PhoST local

PhoST đã được đặt sẵn trong repo:

```text
evaluation/data/1188.en
evaluation/data/1188.vi
```

Script tự ghép song song:

```text
1188.en dòng n  ->  1188.vi dòng n
```

### ViMedCSS

Nguồn Hugging Face:

```text
tensorxt/ViMedCSS
```

Chỉ lấy các cột cần cho benchmark text:

| Cột | Vai trò |
|---|---|
| `segment_text` | `input_raw`, câu chứa code-switching term |
| `cs_terms_list` | danh sách term bẫy/OOV |
| `segment_id` | ID case |
| `topic` | domain/topic |
| `original_video_link` | metadata truy xuất nguồn |
| `start_time`, `end_time` | metadata segment |

Để tránh tải audio nặng, loader dùng:

```python
load_dataset("tensorxt/ViMedCSS", split="train", streaming=True)
```

### Schema `test_cases.json`

Mỗi case phục vụ LLM judge có các trường chính:

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

Trong CSV kết quả, benchmark xuất ra các cột quan trọng:

| Cột | Ý nghĩa |
|---|---|
| `english_original` | câu gốc/source |
| `ai_translation` | bản dịch/localization do hệ thống hoặc baseline sinh ra |
| `expected_localization` | đáp án/reference |
| `ai_score` | điểm Gemini judge 0-5 |
| `ai_reason` | lý do chấm ngắn |

---

## 3. Localization Benchmark bằng Gemini 2.5 Judge

### Mục tiêu

Đánh giá chất lượng localization ở cấp câu bằng LLM judge thay vì chỉ dựa vào exact match. Gemini 2.5 chấm theo các tiêu chí:

- đúng nghĩa so với source/reference
- tự nhiên tiếng Việt
- xử lý thuật ngữ code-switching/OOV
- phù hợp để đưa sang TTS

### Cách chạy local

Cần API key:

```powershell
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

Chạy benchmark:

```powershell
venv\Scripts\python.exe evaluation\benchmark_localization.py `
  --judge `
  --judge-model gemini-2.5-flash-preview-05-20 `
  --judge-batch-size 20 `
  --judge-concurrency 5
```

### Thiết kế batch/parallel

Benchmark localization hiện dùng dataset hybrid 100 câu:

```text
70 PhoST local + 30 ViMedCSS streaming = 100 cases
```

Mỗi batch chứa 20 câu:

```text
100 cases / 20 cases per batch = 5 batches
```

Sau đó các batch được gửi song song bằng `asyncio.gather` và giới hạn concurrency bằng `asyncio.Semaphore`:

```text
--judge-batch-size 20
--judge-concurrency 5
```

Nghĩa là trên Colab có thể gửi tối đa 5 batch song song, mỗi batch yêu cầu Gemini 2.5 trả về **20 điểm có structured output**.

### Structured output

Benchmark dùng LangChain để ép mô hình trả về schema cố định:

```python
ChatGoogleGenerativeAI(...).with_structured_output(JudgeBatch)
```

Schema chính:

```python
class JudgeItem(BaseModel):
    id: str
    score: float  # 0..5
    reason: str

class JudgeBatch(BaseModel):
    results: list[JudgeItem]
```

### Output files

- `evaluation/localization_results.csv`
- `evaluation/localization_results.json`

### Cột dữ liệu trong CSV

| Cột | Ý nghĩa |
|---|---|
| `english_original` | câu gốc/source |
| `input_raw` | input raw |
| `ai_translation` | bản dịch/localization do hệ thống sinh |
| `expected_localization` | đáp án/reference |
| `token_error_rate` | TER tham khảo |
| `ai_score` | điểm Gemini judge 0-5 |
| `ai_reason` | lý do chấm |

### Nhận xét

Benchmark 3 không còn là smoke-test đơn giản nữa mà đã trở thành **LLM-as-a-Judge benchmark** có thể chạy song song theo batch trên Colab. Cấu hình này phù hợp để lấy điểm nhanh hơn so với chấm từng câu đơn lẻ, đồng thời vẫn giữ được structured output để xuất báo cáo CV-ready.
    id: str
    score: float  # 0..5
    reason: str

class JudgeBatch(BaseModel):
    results: list[JudgeItem]
```

### Output files

```text
evaluation/localization_results.csv
evaluation/localization_results.json
```

---

## 4. Performance & Hardware Benchmark

### Lệnh chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments --include-nvidia-smi
```

### Kết quả hiện tại trên máy local CPU

| Metric | Value |
|---|---:|
| Processing time | 321.79s |
| Video duration | 95.13s |
| Real-Time Factor | 0.30x |
| PyTorch Peak VRAM | N/A |
| Status | WARN on CPU-limited machine |

### Nhận xét

Kết quả local hiện chỉ là CPU baseline. Trên Colab T4/A100, cần chạy lại để có:

- RTF thực tế khi có GPU
- VRAM qua PyTorch
- system-level VRAM delta qua `nvidia-smi`

---

## 5. Group-level Sync Benchmark

### Lệnh chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_sync_group.py
```

### Kết quả hiện tại

| Metric | Value |
|---|---:|
| Total groups | 13 |
| Valid groups | 13 |
| Missing audio | 0 |
| MAE overflow | 53.85 ms |
| Max overflow | 700 ms |
| Target MAE | < 150 ms |
| Status | PASS |

### Nhận xét

MAE **53.85ms** khớp với insight một outlier kéo trung bình lên:

```text
700 / 13 = 53.846ms
```

Điều này cho thấy grouping + speed adaptation không fail toàn hệ thống. Phần lớn group ổn, còn lại một case biên cần tối ưu thêm.

---

## 6. Optional G2P / Pronunciation Benchmark

Benchmark G2P vẫn hữu ích để đo riêng khả năng đọc thuật ngữ tiếng Anh sang tiếng Việt bồi.

| Metric | Value |
|---|---:|
| Total terms | 120 |
| PASS | 110 |
| FAIL | 10 |
| Accuracy | 91.67% |
| Target | >= 95.00% |
| Status | Good baseline, below strict target |

Các lỗi chính nằm ở nhóm alphanumeric/OOV:

```text
BM25, HTML5, OAuth2, FFmpeg, Nginx, PostgreSQL
```

Hướng xử lý hiện tại:

```text
domain dictionary phonetic > domain/global phonetic cache > G2P fallback
```

---

## 7. Colab Workflow

Notebook mới cần chạy theo thứ tự:

1. Mount Google Drive
2. Clone/pull repo
3. Cài system dependencies: FFmpeg, Ollama
4. Cài Python dependencies từ `requirements.txt`
5. Set `GOOGLE_API_KEY`
6. Kiểm tra `evaluation/test_cases.json` đã có 100 cases
7. Chạy Gemini localization judge
8. Chạy performance benchmark
9. Chạy group sync benchmark
10. Copy kết quả về Drive

---

## 8. Tổng kết CV-ready

| Benchmark | Metric | Current Result | Status |
|---|---:|---:|---|
| Localization LLM Judge | Gemini score | pending Colab run | Ready |
| Performance | RTF | 0.30x local CPU | Needs GPU rerun |
| Group Sync | MAE overflow | 53.85ms | PASS |
| Optional G2P | Accuracy | 91.67% | Good baseline |

### Câu mô tả CV gợi ý

> Built a hybrid evaluation suite for a local-first AI video dubbing pipeline, combining public spoken translation data and medical code-switching stress tests to evaluate Vietnamese localization quality, performance/VRAM, and group-level audio-video synchronization.
>
> Implemented a Gemini 2.5 LLM-as-a-judge benchmark with LangChain structured output, batching 20 cases per request and running batches concurrently to score localization quality across 100 hybrid test cases.

---

## 9. Việc cần làm sau khi chạy Colab

Sau khi chạy notebook trên Colab, cập nhật lại file này với:

1. `avg_ai_score`
2. số case đã chấm
3. thời gian chạy judge
4. RTF/VRAM trên GPU
5. sync result sau pipeline mới
