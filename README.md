# VidTranscribe.ai: Agentic Video Dubbing & Translation Pipeline

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

## Project Vision

Translating and dubbing technical videos across multiple domains is a complex challenge. This project goes beyond basic translation by building a **fully automated, Agentic AI-driven dubbing pipeline**.

By leveraging a dynamic multi-pass architecture, VidTranscribe.ai preserves complex industry terminology, strictly prevents literal translations, and intelligently synchronizes generated speech to video frames, turning standard videos into localized, professional-grade assets.

---

## Key Features (Production-Ready)

- **6-Stage Modular Pipeline**: Clean separation of core audio processing, transcription, translation, and video muxing tasks.
- **Dynamic Two-Pass Translation**: 
  - **Context Analyzer**: Automatically extracts domain-specific terminology (e.g., Medical, IT, Finance) directly from the raw transcript.
  - **Translator Injector**: Dynamically injects these extracted terms into the prompt's core rules, forcing the LLM to preserve technical keywords and context.
- **Robust Fallback System**: Specifically designed to handle "hallucinations" (e.g., Chinese character leakage, prompt leakage) common in smaller local LLMs. It degrades gracefully from Fast Batch JSON translation -> Paragraph Translation -> Sentence-by-Sentence Fallback.
- **Advanced G2P Engine**: A custom Grapheme-to-Phoneme engine that accurately transliterates English technical terms into Vietnamese phonetics (e.g., "Agent" -> "Ây dần") so the TTS engine reads them correctly without over-phonetizing regular conversational language.
- **"Cheating Gaps" Audio Sync**: An intelligent algorithm that calculates silence gaps before and after subtitles, borrowing this time to extend the TTS speaking window. This significantly reduces the need for aggressive audio speedups, ensuring natural-sounding voices that remain perfectly synced.

---

## Tech Stack & Architecture

| Category                | Tools                                           |
| :---------------------- | :---------------------------------------------- |
| **Language**            | Python 3.12                                     |
| **Core LLM**            | Qwen 2.5 7B (via Ollama)                        |
| **Orchestration**       | LangChain, Pydantic                             |
| **Speech-to-Text**      | Faster-Whisper                                  |
| **Text-to-Speech**      | Edge-TTS (Microsoft)                            |
| **Audio/Video Engine**  | FFmpeg, Pydub                                   |
| **Web Frameworks**      | FastAPI (Backend), Gradio (Frontend)            |

---

## The 6-Stage Lifecycle

Each stage is a self-contained module located in `pipeline/`:

1.  **Audio Extraction**: Isolates high-quality audio tracks from raw video files.
2.  **Speech-to-Text (STT)**: Transcribes the audio into original English subtitles using Faster-Whisper.
3.  **Synchronization**: Batches paragraphs and synchronizes initial timing markers.
4.  **Agentic Translation**: Executes the two-pass translation with dynamic terminology preservation and strict fallback handling.
5.  **Speech Synthesis (TTS)**: Processes the Vietnamese text through the G2P engine and generates perfectly-timed audio using the "Cheating Gaps" algorithm.
6.  **Video Muxing**: Merges the new audio track and hard/soft subtitles back into the final video output.

---

## Quick Start

### Local Development

```bash
# 1. Clone & Enter
git clone https://github.com/HoangKhang226/VidTranscribe.ai.git
cd VidTranscribe.ai

# 2. Setup Environment
python -m venv venv
source venv/Scripts/activate  # Or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Pull Local LLM via Ollama
ollama pull qwen2.5:7b-instruct-q4_K_M

# 4. Run the Full System (Backend API & Gradio UI)
python main.py --mode all

# Or run via CLI directly:
python main.py --mode cli --source "video.mp4" --model "qwen2.5:7b-instruct-q4_K_M"
```

---

## Project Navigation

```text
.
├── api.py                 # FastAPI Backend Service
├── app.py                 # Gradio Frontend Dashboard
├── main.py                # Central Pipeline Orchestrator
├── config.py              # Centralized Configuration
├── output/                # Stored Output Data & Subtitles
└── pipeline/              # The 6-Stage Modular Logic
    ├── step1_ingestion.py
    ├── step2_context.py
    ├── step3_stt.py
    ├── step4_translate.py
    ├── step5_tts.py
    └── step6_mux.py
└── utils/                 # G2P Engine and SRT utilities
```

---

## Contributing & Security

This project is maintained by **[HoangKhang226](https://github.com/HoangKhang226)**.

---
*Built with a focus on Agentic AI and dynamic localization standards.*
