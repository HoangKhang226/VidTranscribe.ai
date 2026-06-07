import argparse
import sys
import threading
import uvicorn
from src.utils.logger import logger
from src.utils.memory import clean_memory
from src.pipeline.orchestrator import PipelineOrchestrator
from src import config

def start_api_server():
    """Khởi động server backend FastAPI."""
    logger.info(f"Đang khởi động Backend API tại http://{config.API_HOST}:{config.API_PORT} ...")
    uvicorn.run("src.api:app", host=config.API_HOST, port=config.API_PORT, reload=False)

def start_web_ui():
    """Khởi động giao diện Web UI Gradio."""
    logger.info(f"Đang khởi động Web UI Gradio tại http://127.0.0.1:{config.GRADIO_PORT} ...")
    from src.app import demo
    demo.launch(server_name="127.0.0.1", server_port=config.GRADIO_PORT)

def main():
    parser = argparse.ArgumentParser(description="VidTranscribe.ai - Pipeline dịch phụ đề và lồng tiếng AI.")
    
    parser.add_argument(
        "--source", 
        type=str, 
        help="Đường dẫn file video local hoặc URL YouTube để chạy CLI trực tiếp."
    )
    parser.add_argument(
        "--hardsub", 
        action="store_true", 
        default=True,
        help="Ghép phụ đề cứng vào video (mặc định: True)."
    )
    parser.add_argument(
        "--softsub",
        dest="hardsub",
        action="store_false",
        help="Ghép phụ đề mềm (không burn chữ vào hình)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen2.5:3b-instruct-q4_K_M", "qwen2.5:7b-instruct-q4_K_M"],
        default="qwen2.5:3b-instruct-q4_K_M",
        help="Mô hình Ollama dùng để dịch thuật (qwen2.5:3b-instruct-q4_K_M hoặc qwen2.5:7b-instruct-q4_K_M)."
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["cli", "api", "web", "all"],
        default=None,
        help="Chế độ hoạt động: cli (dòng lệnh), api (Backend), web (Frontend), hoặc all (cả hai)."
    )
    
    args = parser.parse_args()
    
    # Tự động quyết định chế độ nếu người dùng không chỉ định
    if args.mode is None:
        if args.source:
            args.mode = "cli"
        else:
            args.mode = "all"
            
    clean_memory()
    
    if args.mode == "cli":
        if not args.source:
            logger.error("Chế độ CLI yêu cầu tham số --source (URL YouTube hoặc file video).")
            sys.exit(1)
            
        if args.source.startswith("http://") or args.source.startswith("https://"):
            logger.error("Hệ thống chỉ chấp nhận tệp video local, không hỗ trợ tải từ URL/YouTube.")
            sys.exit(1)
            
        try:
            orchestrator = PipelineOrchestrator()
            final_video_path = orchestrator.run_pipeline(
                source=args.source,
                hardsub=args.hardsub,
                ollama_model=args.model
            )
            logger.info(f"Đã hoàn thành xử lý. Video đầu ra lưu tại: {final_video_path}")
        except Exception as e:
            logger.error(f"Đã xảy ra lỗi trong quá trình thực thi: {e}")
            sys.exit(1)
            
    elif args.mode == "api":
        start_api_server()
        
    elif args.mode == "web":
        start_web_ui()
        
    elif args.mode == "all":
        # Khởi động cả API Server (Background Thread) và Web UI (Main Thread)
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Chờ 1 giây để API khởi động xong trước khi bật UI
        import time
        time.sleep(1)
        
        start_web_ui()

if __name__ == "__main__":
    main()
