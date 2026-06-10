# -*- coding: utf-8 -*-
"""
Factory khởi tạo ChatOllama với cấu hình GIẢM THIỂU TRÀN VRAM dùng chung toàn dự án.

Tập trung mọi tham số an toàn VRAM tại một chỗ (đọc từ config) để dễ tinh chỉnh:
- num_ctx: giới hạn cửa sổ ngữ cảnh (mặc định nhỏ cho tác vụ câu ngắn).
- think:   tắt thinking mode của Gemma để tiết kiệm token/thời gian.
- num_gpu: số layer offload lên GPU (None = để Ollama tự quyết, an toàn nhất).
"""
from langchain_ollama import ChatOllama
from src.config import (
    OLLAMA_API_URL, OLLAMA_MODEL_NAME,
    LLM_NUM_CTX, LLM_THINK, LLM_NUM_GPU,
)


def make_chat_ollama(model_name: str = OLLAMA_MODEL_NAME,
                     temperature: float = 0.1,
                     num_ctx: int = LLM_NUM_CTX,
                     think: bool = LLM_THINK,
                     num_gpu=LLM_NUM_GPU) -> ChatOllama:
    """Khởi tạo ChatOllama với cấu hình an toàn VRAM."""
    base_url = OLLAMA_API_URL.replace("/api/generate", "")
    kwargs = dict(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        think=think,
    )
    # Chỉ truyền num_gpu khi được chỉ định rõ (None -> để Ollama tự quyết)
    if num_gpu is not None:
        kwargs["num_gpu"] = num_gpu
    return ChatOllama(**kwargs)
