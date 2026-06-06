import gc
import psutil
from utils.logger import logger

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def log_memory_usage(stage_name="General"):
    """Ghi log lượng RAM và VRAM đang sử dụng."""
    # RAM
    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    ram_msg = f"RAM: {ram_usage_mb:.2f} MB"
    
    # VRAM
    vram_msg = "VRAM: N/A"
    if HAS_TORCH and torch.cuda.is_available():
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        vram_msg = f"VRAM Allocated: {vram_allocated_mb:.2f} MB, Reserved: {vram_reserved_mb:.2f} MB"
        
    logger.info(f"[{stage_name}] Memory Usage -> {ram_msg} | {vram_msg}")

def clean_memory():
    """Giải phóng RAM và VRAM chủ động."""
    logger.info("🧹 Bắt đầu dọn dẹp bộ nhớ (GC & PyTorch Cache)...")
    gc.collect()
    
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    log_memory_usage("Sau khi dọn dẹp")
