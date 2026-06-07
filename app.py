import os
import time
import threading
import gradio as gr
from src.pipeline.orchestrator import PipelineOrchestrator
from src.config import GRADIO_PORT
from src.utils.logger import logger

# CSS tùy chỉnh cao cấp (Rich Aesthetics & Dark Glassmorphism)
custom_css = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.gradio-container {
    background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 30px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}
h1, h2 {
    color: #38bdf8 !important;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    font-weight: 800 !important;
}
.primary-btn {
    background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4) !important;
    transition: all 0.3s ease !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6) !important;
}
.card-glass {
    background: rgba(30, 41, 59, 0.5) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px;
}
"""

def process_video_gradio(upload_file, ollama_model):
    """Hàm wrapper cho Gradio, thực thi pipeline trên thread phụ và poll tiến trình."""
    if upload_file is None:
        yield 0.0, "Vui lòng upload file video!", None, None
        return

    orchestrator = PipelineOrchestrator()
    
    thread = threading.Thread(
        target=orchestrator.run_pipeline,
        args=(upload_file, False, ollama_model)
    )
    thread.start()
    
    # Vòng lặp poll tiến trình để cập nhật UI liên tục
    while thread.is_alive():
        status = orchestrator.get_status()
        progress = status.get("progress", 0.0)
        current_step = status.get("current_step", "Đang xử lý...")
        
        yield progress, f"{current_step}", None, None
        time.sleep(0.5)
        
    # Lấy trạng thái cuối cùng sau khi luồng kết thúc
    status = orchestrator.get_status()
    if status["status"] == "completed":
        final_video = status["results"].get("final_video")
        srt_vi = status["results"].get("srt_vi")
        
        # Đảm bảo file tồn tại
        if final_video and os.path.exists(final_video):
            yield (
                1.0,
                f"Hoàn thành lồng tiếng thành công trong {status.get('elapsed_time', 0.0):.1f}s!",
                final_video,
                srt_vi
            )
        else:
            yield 0.0, "Lỗi: Pipeline báo thành công nhưng không tìm thấy file đầu ra.", None, None
    else:
        error_msg = status.get("error", "Lỗi không xác định")
        yield 0.0, f"Pipeline thất bại: {error_msg}", None, None

# Xây dựng giao diện UI Gradio
with gr.Blocks(theme=gr.themes.Default(primary_hue="sky"), css=custom_css, title="VidTranscribe.ai") as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=8):
            gr.Markdown(
                """
                # VidTranscribe.ai
                ### Hệ thống lồng tiếng AI thông minh gối đầu ngữ cảnh.
                *Tối ưu hóa chạy êm mượt trên RTX 2050 (4GB VRAM) bằng cơ chế Sequential Model Loading.*
                """
            )
        with gr.Column(scale=1):
            pass

    with gr.Row():
        # Cột trái - Nhập dữ liệu đầu vào
        with gr.Column(scale=5, elem_classes=["card-glass"]):
            gr.Markdown("### 1. Đầu vào Video")
            
            upload_file = gr.File(
                label="Tải lên Video Local (.mp4, .mkv, .mov)",
                file_types=["video"]
            )
            
            gr.Markdown("### 2. Cấu hình & Tùy chọn")
            
            ollama_model = gr.Dropdown(
                label="Mô hình LLM dịch thuật (Ollama)",
                choices=["qwen2.5:3b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M"],
                value="qwen2.5:7b-instruct-q4_K_M",
                info="Mô hình 7B được khuyến nghị để dịch và phiên âm tối ưu nhất."
            )
            
            submit_btn = gr.Button("Bắt đầu dịch & Lồng tiếng", elem_classes=["primary-btn"])
            
        # Cột phải - Trạng thái và Kết quả đầu ra
        with gr.Column(scale=5, elem_classes=["card-glass"]):
            gr.Markdown("### 3. Tiến trình xử lý")
            
            progress_bar = gr.Slider(
                label="Tiến độ",
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                interactive=False
            )
            
            status_text = gr.Textbox(
                label="Trạng thái hiện tại",
                value="Chờ lệnh...",
                interactive=False
            )
            
            gr.Markdown("### 4. Kết quả đầu ra")
            output_video = gr.Video(
                label="Video Lồng tiếng Tiếng Việt (Không kèm phụ đề)",
                interactive=False
            )
            
            output_srt = gr.File(
                label="Tải về Phụ đề Tiếng Việt (.srt)",
                interactive=False
            )

    # Đăng ký sự kiện click nút xử lý
    submit_btn.click(
        fn=process_video_gradio,
        inputs=[upload_file, ollama_model],
        outputs=[progress_bar, status_text, output_video, output_srt]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_PORT)
