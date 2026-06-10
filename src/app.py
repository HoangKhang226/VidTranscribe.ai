import os
import time
import threading
import pandas as pd
import gradio as gr
from src.pipeline.orchestrator import PipelineOrchestrator
from src.config import GRADIO_PORT
from src.utils.logger import logger
from src.db.db_manager import db, _now_str

# CSS tùy chỉnh cao cấp mô phỏng Google Gemini (Dark Mode)
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

body {
    background-color: #131314;
    color: #e3e3e3;
    font-family: 'Google Sans', 'Inter', system-ui, -apple-system, sans-serif;
    margin: 0;
    padding: 0;
}
.gradio-container {
    background-color: #131314 !important;
    border: none !important;
    max-width: 100% !important;
}
h1, h2, h3 {
    color: #e3e3e3 !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px;
}
.sidebar {
    background-color: #1e1f20 !important;
    border-right: none !important;
    border-radius: 16px;
    padding: 20px !important;
    height: 100vh;
    margin-right: 10px;
}
.primary-btn {
    background: #c2e7ff !important;
    color: #001d35 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease !important;
}
.primary-btn:hover {
    background: #a8d5f2 !important;
}
.nav-btn {
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 15px !important;
    font-size: 0.95em !important;
    background: transparent !important;
    border: none !important;
    border-radius: 20px !important;
    color: #e3e3e3 !important;
    transition: background 0.2s ease !important;
}
.nav-btn:hover {
    background: #282a2c !important;
}
.card-glass {
    background-color: #1e1f20 !important;
    border: none !important;
    border-radius: 16px;
    padding: 24px !important;
    box-shadow: none !important;
}
.progress-bar-wrap {
    border-radius: 8px !important;
    overflow: hidden;
}
"""

MODE_E2E = "Dịch & Lồng tiếng (End-to-End)"
MODE_TRANSLATE = "Chỉ dịch (Xem & sửa phụ đề)"

def _read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Không đọc được file {path}: {e}")
        return ""

def process_video_gradio(upload_file, ollama_model, domain_selection, mode_label):
    """
    Wrapper Gradio thực thi pipeline. Hỗ trợ 2 chế độ:
      - End-to-End: chạy tới video cuối.
      - Chỉ dịch: dừng sau bước dịch, hiển thị video gốc + phụ đề để sửa.

    Outputs (theo thứ tự):
      progress, status, final_video, srt_file, preview_video, subtitle_editor,
      review_group(update visible), orchestrator_state
    """
    mode = "translate_only" if mode_label == MODE_TRANSLATE else "end_to_end"

    if upload_file is None:
        yield (0.0, "Vui lòng upload file video!", None, None, None,
               gr.update(), gr.update(visible=False), None)
        return

    orchestrator = PipelineOrchestrator()

    thread = threading.Thread(
        target=orchestrator.run_pipeline,
        args=(upload_file, False, ollama_model, domain_selection, mode)
    )
    thread.start()

    # Poll tiến trình
    while thread.is_alive():
        status = orchestrator.get_status()
        progress = status.get("progress", 0.0)
        current_step = status.get("current_step", "Đang xử lý...")
        yield (progress, current_step, None, None, None,
               gr.update(), gr.update(visible=False), orchestrator)
        time.sleep(0.5)

    status = orchestrator.get_status()
    st = status["status"]
    results = status.get("results", {})

    if st == "completed":
        final_video = results.get("final_video")
        srt_vi = results.get("srt_vi")
        if final_video and os.path.exists(final_video):
            yield (1.0, f"Hoàn thành lồng tiếng trong {status.get('elapsed_time', 0.0):.1f}s!",
                   final_video, srt_vi, None, gr.update(value=""),
                   gr.update(visible=False), orchestrator)
        else:
            yield (0.0, "Lỗi: Không tìm thấy file đầu ra.", None, None, None,
                   gr.update(), gr.update(visible=False), orchestrator)

    elif st == "awaiting_review":
        # Chế độ chỉ dịch: hiện video gốc + nội dung phụ đề để sửa
        srt_vi = results.get("srt_vi")
        source_video = results.get("source_video")
        srt_content = _read_text_file(srt_vi) if srt_vi else ""
        yield (0.75, "Đã dịch xong! Xem video kèm phụ đề bên dưới, chỉnh sửa rồi bấm 'Lưu phụ đề'.",
               None, srt_vi, source_video, gr.update(value=srt_content),
               gr.update(visible=True), orchestrator)
    else:
        error_msg = status.get("error", "Lỗi không xác định")
        yield (0.0, f"Pipeline thất bại: {error_msg}", None, None, None,
               gr.update(), gr.update(visible=False), orchestrator)

def save_edited_subtitle(orchestrator, srt_text):
    """Ghi nội dung phụ đề người dùng đã sửa đè lên file srt_vi."""
    if orchestrator is None:
        return "Chưa có phiên xử lý nào. Hãy chạy 'Chỉ dịch' trước."
    srt_vi = orchestrator.get_status().get("results", {}).get("srt_vi")
    if not srt_vi:
        return "Không tìm thấy file phụ đề để lưu."
    try:
        with open(srt_vi, "w", encoding="utf-8") as f:
            f.write(srt_text)
        return f"Đã lưu phụ đề đã chỉnh sửa vào: {os.path.basename(srt_vi)}"
    except Exception as e:
        return f"Lỗi khi lưu phụ đề: {e}"

def continue_dubbing(orchestrator, srt_text):
    """Lưu phụ đề đã sửa rồi tiếp tục lồng tiếng + ghép video (Bước 6 & 7)."""
    if orchestrator is None:
        yield 0.75, "Chưa có phiên xử lý nào.", None, None
        return
    # Lưu phụ đề trước khi lồng tiếng
    save_edited_subtitle(orchestrator, srt_text)

    thread = threading.Thread(target=orchestrator.resume_from_subtitles, args=(False,))
    thread.start()
    while thread.is_alive():
        status = orchestrator.get_status()
        yield status.get("progress", 0.8), status.get("current_step", "Đang lồng tiếng..."), None, None
        time.sleep(0.5)

    status = orchestrator.get_status()
    if status["status"] == "completed":
        results = status.get("results", {})
        final_video = results.get("final_video")
        yield 1.0, f"Hoàn thành lồng tiếng trong {status.get('elapsed_time', 0.0):.1f}s!", final_video, results.get("srt_vi")
    else:
        yield 0.75, f"Lỗi khi lồng tiếng: {status.get('error', 'Không xác định')}", None, None

DICT_COLUMNS = ["Từ tiếng Anh", "Tiếng Việt bồi", "Giải thích", "Ngày bổ sung", "Người bổ sung", "Cập nhật cuối"]

def load_dict_to_df(domain):
    """Load JSON dictionary (schema mới) sang Pandas DataFrame cho Gradio."""
    if domain == "Auto Detect":
        return pd.DataFrame(columns=DICT_COLUMNS)
    data = db.load_dictionary(domain)
    rows = []
    for eng, d in data.items():
        rows.append({
            "Từ tiếng Anh": eng,
            "Tiếng Việt bồi": d.get("phonetic", ""),
            "Giải thích": d.get("explanation", ""),
            "Ngày bổ sung": d.get("date_added", ""),
            "Người bổ sung": d.get("added_by", ""),
            "Cập nhật cuối": d.get("last_updated", ""),
        })
    return pd.DataFrame(rows, columns=DICT_COLUMNS)

def save_df_to_dict(domain, df):
    """Lưu DataFrame ngược lại file JSON, tự quản lý ngày tháng & người bổ sung."""
    if domain == "Auto Detect":
        return "Vui lòng chọn một ngành cụ thể để lưu."

    existing = db.load_dictionary(domain)
    now = _now_str()
    data = {}
    for _, row in df.iterrows():
        eng = str(row.get("Từ tiếng Anh", "")).strip()
        if not eng or eng.lower() == "nan":
            continue
        phonetic = str(row.get("Tiếng Việt bồi", "")).strip()
        explanation = str(row.get("Giải thích", "")).strip()
        # Làm sạch giá trị 'nan' do pandas sinh ra ở ô trống
        if phonetic.lower() == "nan": phonetic = ""
        if explanation.lower() == "nan": explanation = ""
        added_by = str(row.get("Người bổ sung", "")).strip() or "Human"
        if added_by.lower() == "nan": added_by = "Human"

        prev = existing.get(eng)
        if prev:
            # Giữ nguyên ngày bổ sung & người bổ sung gốc; chỉ cập nhật last_updated nếu có thay đổi
            date_added = prev.get("date_added") or now
            added_by = prev.get("added_by") or added_by
            changed = (phonetic != prev.get("phonetic", "")) or (explanation != prev.get("explanation", ""))
            last_updated = now if changed else prev.get("last_updated", date_added)
        else:
            date_added = now
            last_updated = now

        data[eng] = {
            "phonetic": phonetic,
            "explanation": explanation,
            "date_added": date_added,
            "added_by": added_by,
            "last_updated": last_updated,
        }
    db.save_dictionary(domain, data)
    return f"Đã lưu thành công {len(data)} từ vựng vào kho {domain}! (Các dòng bị xóa cũng đã được loại bỏ.)"

DOMAINS = [
    "Auto Detect", "General",
    "Công nghệ thông tin", "Y tế & Sức khỏe", "Tài chính & Ngân hàng",
    "Giáo dục", "Pháp lý & Luật", "Khoa học", "Kỹ thuật & Cơ khí",
    "Marketing & Kinh doanh", "Du lịch & Ẩm thực", "Thể thao",
    "Nghệ thuật & Giải trí", "Lịch sử & Văn hóa", "Nông nghiệp & Môi trường"
]

# Xây dựng giao diện UI Gradio (Gradio 6.0: theme & css truyền ở launch())
with gr.Blocks(title="VidTranscribe.ai") as demo:
    
    with gr.Row():
        # ================= SIDEBAR (Kiểu Gemini) =================
        with gr.Column(scale=2, elem_classes=["sidebar"]):
            gr.Markdown("### ✨ VidTranscribe.ai")
            
            btn_studio = gr.Button("⊕ Tạo Video Lồng Tiếng Mới", elem_classes=["primary-btn"])
            
            gr.Markdown("<br>**Gần đây**")
            gr.Button("💬 Lịch sử Video 1", elem_classes=["nav-btn"])
            gr.Button("💬 Lịch sử Video 2", elem_classes=["nav-btn"])
            
            gr.Markdown("<br>**Cài đặt & Từ điển**")
            btn_kb = gr.Button("⚙️ Quản lý Domain & Từ Vựng", elem_classes=["nav-btn"])
            
            gr.Markdown("<br><br><br><br><br><br><br><br>")
            gr.Markdown("<small>*Powered by Gemma 4 & Faster-Whisper*</small>")

        # ================= MAIN CONTENT =================
        with gr.Column(scale=10):
            
            # --- VIEW 1: TRANSCRIBE STUDIO ---
            with gr.Group(visible=True) as view_studio:
                gr.Markdown("## Bắt đầu một Video mới")
                
                with gr.Row():
                    with gr.Column(scale=5, elem_classes=["card-glass"]):
                        gr.Markdown("### 1. Nguồn Video & Ngữ cảnh")
                        upload_file = gr.File(label="Tải lên Video Local", file_types=["video"])
                        
                        mode_selector = gr.Radio(
                            label="Chế độ xử lý",
                            choices=[MODE_E2E, MODE_TRANSLATE],
                            value=MODE_E2E,
                            info="End-to-End: xuất luôn video lồng tiếng. Chỉ dịch: dừng lại để bạn xem & sửa phụ đề trước khi lồng tiếng."
                        )
                        
                        domain_selector = gr.Dropdown(
                            label="Chỉ định Ngành (Domain Override)",
                            choices=DOMAINS,
                            value="Auto Detect",
                            info="Chọn Auto để AI tự đoán dựa trên 20 câu đầu. Hoặc chọn cứng để tăng tốc."
                        )
                        
                        ollama_model = gr.Dropdown(
                            label="Mô hình LLM dịch thuật",
                            choices=["gemma4:e4b", "hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"],
                            value="gemma4:e4b"
                        )
                        
                        submit_btn = gr.Button("Bắt đầu xử lý", elem_classes=["primary-btn"])
                        
                    with gr.Column(scale=5, elem_classes=["card-glass"]):
                        gr.Markdown("### 2. Trạng thái & Kết quả")
                        progress_bar = gr.Slider(label="Tiến độ", minimum=0.0, maximum=1.0, value=0.0, interactive=False)
                        status_text = gr.Textbox(label="Trạng thái", value="Chờ lệnh...", interactive=False)
                        
                        output_video = gr.Video(label="Video Lồng tiếng Tiếng Việt", interactive=False)
                        output_srt = gr.File(label="Tải về Phụ đề Tiếng Việt", interactive=False)

                # --- VÙNG REVIEW (chỉ hiện ở chế độ 'Chỉ dịch') ---
                with gr.Group(visible=False) as review_group:
                    gr.Markdown("### 3. Xem & Chỉnh sửa phụ đề tiếng Việt")
                    gr.Markdown("Xem video gốc kèm phụ đề đã dịch. Chỉnh sửa nội dung phụ đề (định dạng SRT) rồi bấm **Lưu phụ đề**, hoặc **Tiếp tục lồng tiếng** để xuất video cuối.")
                    with gr.Row():
                        preview_video = gr.Video(label="Video gốc", interactive=False)
                        subtitle_editor = gr.Textbox(
                            label="Phụ đề tiếng Việt (SRT) — có thể chỉnh sửa trực tiếp",
                            lines=18, interactive=True, max_lines=40
                        )
                    with gr.Row():
                        save_srt_btn = gr.Button("💾 Lưu phụ đề", elem_classes=["nav-btn"])
                        continue_btn = gr.Button("▶️ Tiếp tục lồng tiếng", elem_classes=["primary-btn"])
                    save_srt_status = gr.Textbox(label="Trạng thái lưu phụ đề", interactive=False)

                # State giữ orchestrator để tiếp tục lồng tiếng sau khi sửa phụ đề
                orchestrator_state = gr.State(value=None)

                submit_btn.click(
                    fn=process_video_gradio,
                    inputs=[upload_file, ollama_model, domain_selector, mode_selector],
                    outputs=[progress_bar, status_text, output_video, output_srt,
                             preview_video, subtitle_editor, review_group, orchestrator_state]
                )

                save_srt_btn.click(
                    fn=save_edited_subtitle,
                    inputs=[orchestrator_state, subtitle_editor],
                    outputs=[save_srt_status]
                )

                continue_btn.click(
                    fn=continue_dubbing,
                    inputs=[orchestrator_state, subtitle_editor],
                    outputs=[progress_bar, status_text, output_video, output_srt]
                )

            # --- VIEW 2: DOMAIN KNOWLEDGE BASE ---
            with gr.Group(visible=False) as view_kb:
                gr.Markdown("## 📚 Domain Knowledge Base (Từ điển ngành)")
                gr.Markdown("Quản lý định nghĩa tiếng Việt và cách phát âm bồi cho từng thuật ngữ chuyên ngành.")
                
                with gr.Row():
                    kb_domain_selector = gr.Dropdown(
                        label="Chọn Ngành để chỉnh sửa",
                        choices=[d for d in DOMAINS if d != "Auto Detect"],
                        value="General"
                    )
                    kb_save_btn = gr.Button("💾 Lưu Thay Đổi", elem_classes=["primary-btn"])
                    
                kb_status = gr.Textbox(label="Trạng thái lưu", interactive=False)
                
                kb_dataframe = gr.Dataframe(
                    headers=DICT_COLUMNS,
                    datatype=["str", "str", "str", "str", "str", "str"],
                    column_count=(6, "fixed"),
                    interactive=True,
                    row_count=5
                )
                
                # Logic: Khi đổi domain thì load lại DF
                kb_domain_selector.change(
                    fn=load_dict_to_df,
                    inputs=[kb_domain_selector],
                    outputs=[kb_dataframe]
                )
                
                # Logic: Nút Lưu
                kb_save_btn.click(
                    fn=save_df_to_dict,
                    inputs=[kb_domain_selector, kb_dataframe],
                    outputs=[kb_status]
                )

    # Đăng ký sự kiện Navigation Sidebar
    btn_studio.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[view_studio, view_kb])
    btn_kb.click(lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[view_studio, view_kb])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT,
                theme=gr.themes.Default(primary_hue="sky"), css=custom_css)
