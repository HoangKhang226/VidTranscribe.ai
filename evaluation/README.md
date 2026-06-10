# VidTranscribe.ai Evaluation Benchmarks

Thư mục `evaluation/` chứa bộ benchmark tự động để chứng minh pipeline xử lý được ba bài toán khó của project:

1. **Group-level Sync** - đo độ lệch timing sau khi TTS gộp câu để khắc phục lỗi sliding window của Whisper.
2. **Performance & Hardware** - đo tốc độ xử lý end-to-end và tài nguyên VRAM.
3. **Text Localization** - đo chất lượng chuẩn hóa tiếng Việt bồi / code-switching trước khi đưa vào Edge-TTS.

Các benchmark chạy bằng Python trong project `venv`.

---

## 1. Group-level Sync Benchmark

### File

- `benchmark_sync_group.py`
- input mặc định: `src/output/audio/temp_segments/tts_groups.json`
- output:
  - `sync_group_results.csv`
  - `sync_group_results.json`

### Mục đích

Pipeline không đọc từng dòng SRT đơn lẻ. `step5_tts.py` gộp các dòng liên tiếp thành `temp_group_*.mp3` để giọng đọc không bị khựng do Whisper sliding window cắt câu giữa chừng.

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

## 2. Performance & Hardware Benchmark

### File

- `benchmark_perf.py`
- video mặc định: `evaluation/Download.mp4`
- output: `perf_results.json`

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

Full end-to-end:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end
```

Full end-to-end và giữ TTS segments để chạy sync benchmark:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments
```

Translate-only để test nhanh:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode translate_only
```

Đo thêm VRAM hệ thống bằng `nvidia-smi`:

```powershell
venv\Scripts\python.exe evaluation\benchmark_perf.py --include-nvidia-smi
```

> `torch.cuda.max_memory_allocated()` chỉ đo VRAM do PyTorch quản lý, không bao gồm Ollama/GGML. Muốn gần với tổng VRAM hơn, dùng `--include-nvidia-smi` trên máy NVIDIA.

---

## 3. Text Normalization & Localization Benchmark

### File

- `benchmark_localization.py`
- input: `test_cases.json`
- output:
  - `localization_results.csv`
  - `localization_results.json`

### Mục đích

Đo chất lượng sửa tiếng Việt bồi / code-switching trước khi đưa text cho Edge-TTS đọc.

Ví dụ case:

```json
{
  "input": "I am... studying AI agents trên Ollama.",
  "gold": "Tôi đang nghiên cứu các tác nhân trí tuệ nhân tạo trên nền tảng Ô-la-ma.",
  "predicted": "Tôi đang nghiên cứu các tác nhân trí tuệ nhân tạo trên nền tảng Ô-la-ma."
}
```

### Metric

Hiện tại dùng normalized token error rate, tương tự WER ở mức token:

```text
TER = levenshtein(pred_tokens, gold_tokens) / len(gold_tokens)
```

Target:

```text
Average TER <= 0.15
```

Có thể mở rộng thêm LLM-as-a-judge để chấm tự nhiên hóa văn phong theo thang 1-5, target `> 4.5/5`.

### Cách chạy

```powershell
venv\Scripts\python.exe evaluation\benchmark_localization.py
```

Tùy chỉnh dataset:

```powershell
venv\Scripts\python.exe evaluation\benchmark_localization.py --cases evaluation\test_cases.json
```

---

## Quy trình chạy đủ bộ benchmark CV

```powershell
# 1. Localization quality
venv\Scripts\python.exe evaluation\benchmark_localization.py

# 2. Full performance + preserve TTS group artifacts
venv\Scripts\python.exe evaluation\benchmark_perf.py --mode end_to_end --keep-temp-segments

# 3. Group-level sync quality
venv\Scripts\python.exe evaluation\benchmark_sync_group.py
```

Optional G2P benchmark cũ vẫn dùng được để đo riêng chất lượng phiên âm thuật ngữ:

```powershell
venv\Scripts\python.exe evaluation\benchmark_g2p.py
```

---

## CV phrasing gợi ý

> Built automated evaluation for a local-first AI dubbing pipeline: group-level audio-video sync MAE, end-to-end RTF/VRAM profiling, and Vietnamese localization quality via token error rate.  
> Optimized Whisper sliding-window subtitle fragmentation with TTS grouping and measured group-level sync MAE under the 150ms perceptual threshold.
