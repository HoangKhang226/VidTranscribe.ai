import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger
from pipeline import step4_translate

def test_paragraph_translation():
    srt_path = "output/subtitles/subtitles_en.srt"
    
    if not os.path.exists(srt_path):
        logger.error(f"Test file not found: {srt_path}")
        return
        
    context = {"keywords": ["ai agents", "general purpose task agents", "workflow agents", "specialized research and analysis agents", "autonomous coding agents", "agent mode", "professional workflows"]}
    
    # We will test on just 5 sentences (1 paragraph) to make it quick, or test the whole thing
    logger.info("Bắt đầu test Paragraph Translation...")
    
    try:
        vi_srt = step4_translate.run(srt_path, context, model_name="qwen2.5:7b-instruct-q4_K_M")
        logger.info(f"Test thành công! File xuất ra tại: {vi_srt}")
    except Exception as e:
        logger.error(f"Test thất bại: {e}")

if __name__ == "__main__":
    test_paragraph_translation()
