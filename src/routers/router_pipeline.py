import os
import uuid
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException
from src.pipeline.orchestrator import PipelineOrchestrator
from src.config import DOWNLOADS_DIR, API_HOST, API_PORT
from src.utils.logger import logger

router = APIRouter(prefix="/transcribe", tags=["Pipeline"])

# Lưu trữ các task lồng tiếng đang chạy (In-memory)
tasks = {}

def run_pipeline_task(task_id: str, source: str, hardsub: bool, model_name: str, domain: str, mode: str = "end_to_end"):
    """Hàm chạy ngầm trong Background Thread."""
    orchestrator = tasks[task_id]["orchestrator"]
    try:
        orchestrator.run_pipeline(source, hardsub=hardsub, ollama_model=model_name, domain_override=domain, mode=mode)
    except Exception as e:
        logger.error(f"Lỗi chạy Background Task {task_id}: {e}")

@router.post("/upload")
async def transcribe_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    hardsub: bool = Form(True),
    model: str = Form("gemma4:e4b"),
    domain: str = Form("Auto Detect"),
    mode: str = Form("end_to_end")
):
    """API Nhận file video upload trực tiếp từ client."""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận các định dạng video: .mp4, .avi, .mkv, .mov")
        
    task_id = str(uuid.uuid4())
    
    # Lưu file upload tạm thời
    file_ext = os.path.splitext(file.filename)[1]
    saved_filename = f"upload_{task_id}{file_ext}"
    saved_path = os.path.join(DOWNLOADS_DIR, saved_filename)
    
    try:
        with open(saved_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu file tải lên: {e}")
        
    orchestrator = PipelineOrchestrator()
    tasks[task_id] = {
        "orchestrator": orchestrator,
        "source": saved_path
    }
    
    # Khởi chạy pipeline ngầm
    background_tasks.add_task(run_pipeline_task, task_id, saved_path, hardsub, model, domain, mode)
    
    return {
        "message": "Task upload đã được xếp hàng.",
        "task_id": task_id,
        "status": "processing"
    }

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """API truy vấn tiến trình và trạng thái xử lý."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Không tìm thấy mã Task ID.")
        
    orchestrator = tasks[task_id]["orchestrator"]
    status_data = orchestrator.get_status().copy()
    
    # Định dạng lại kết quả cho Client dễ dùng
    results = status_data.get("results", {})
    if "final_video" in results:
        # Tạo URL download tĩnh
        filename = os.path.basename(results["final_video"])
        results["download_url"] = f"http://{API_HOST}:{API_PORT}/static/final/{filename}"
        
        if "srt_vi" in results:
            srt_filename = os.path.basename(results["srt_vi"])
            results["srt_vi_url"] = f"http://{API_HOST}:{API_PORT}/static/subtitles/{srt_filename}"
            
    return status_data
