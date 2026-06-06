import os
import uuid
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline.orchestrator import PipelineOrchestrator
from config import OUTPUT_DIR, DOWNLOADS_DIR, API_HOST, API_PORT
from utils.logger import logger

app = FastAPI(
    title="VidTranscribe.ai API",
    description="Backend API cho hệ thống lồng tiếng và dịch phụ đề AI tự động.",
    version="1.0.0"
)

# Kích hoạt CORS cho phép Web UI bên ngoài gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lưu trữ các task lồng tiếng đang chạy (In-memory)
tasks = {}

class TranscribeUrlRequest(BaseModel):
    url: str
    hardsub: bool = True
    model: str = "qwen2.5:3b-instruct-q4_K_M"

# Mount thư mục output làm Static Files để tải file trực tiếp qua URL
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

def run_pipeline_task(task_id: str, source: str, is_url: bool, hardsub: bool, model_name: str):
    """Hàm chạy ngầm trong Background Thread."""
    orchestrator = tasks[task_id]["orchestrator"]
    try:
        orchestrator.run_pipeline(source, is_url=is_url, hardsub=hardsub, ollama_model=model_name)
    except Exception as e:
        logger.error(f"❌ Lỗi chạy Background Task {task_id}: {e}")

@app.post("/api/v1/transcribe/url")
async def transcribe_url(request: TranscribeUrlRequest, background_tasks: BackgroundTasks):
    """API Nhận URL Youtube để xử lý lồng tiếng."""
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL không hợp lệ. Phải bắt đầu bằng http/https.")
        
    task_id = str(uuid.uuid4())
    orchestrator = PipelineOrchestrator()
    
    tasks[task_id] = {
        "orchestrator": orchestrator,
        "source": request.url,
        "is_url": True
    }
    
    # Đưa vào Background Tasks để tránh block request API
    background_tasks.add_task(run_pipeline_task, task_id, request.url, True, request.hardsub, request.model)
    
    return {
        "message": "Task lồng tiếng đã được xếp hàng.",
        "task_id": task_id,
        "status": "processing"
    }

@app.post("/api/v1/transcribe/upload")
async def transcribe_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    hardsub: bool = Form(True),
    model: str = Form("qwen2.5:3b-instruct-q4_K_M")
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
        "source": saved_path,
        "is_url": False
    }
    
    # Khởi chạy pipeline ngầm
    background_tasks.add_task(run_pipeline_task, task_id, saved_path, False, hardsub, model)
    
    return {
        "message": "Task upload đã được xếp hàng.",
        "task_id": task_id,
        "status": "processing"
    }

@app.get("/api/v1/status/{task_id}")
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
        # File final_output.mp4 nằm ở final/
        results["download_url"] = f"http://{API_HOST}:{API_PORT}/static/final/{filename}"
        
        # Thêm URL download phụ đề Việt
        if "srt_vi" in results:
            srt_filename = os.path.basename(results["srt_vi"])
            results["srt_vi_url"] = f"http://{API_HOST}:{API_PORT}/static/subtitles/{srt_filename}"
            
    return status_data

@app.get("/api/v1/health")
async def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của hệ thống."""
    return {"status": "healthy", "service": "VidTranscribe.ai Backend"}
