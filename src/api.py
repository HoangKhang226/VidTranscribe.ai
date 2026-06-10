import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.config import OUTPUT_DIR
from src.utils.logger import logger

# Import routers
from src.routers.router_pipeline import router as pipeline_router
from src.routers.router_dictionary import router as dictionary_router

app = FastAPI(
    title="VidTranscribe.ai Advanced API",
    description="Backend API Modular cho hệ thống AI Dubbing & Domain-Aware RAG.",
    version="2.0.0"
)

# Kích hoạt CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount thư mục tĩnh
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Đăng ký các Routers
app.include_router(pipeline_router, prefix="/api/v1")
app.include_router(dictionary_router, prefix="/api/v1")

@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của hệ thống."""
    return {"status": "healthy", "service": "VidTranscribe.ai Backend V2"}
