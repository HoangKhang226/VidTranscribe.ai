import os
import asyncio
import re
import json
import nest_asyncio
import edge_tts
from pydub import AudioSegment
from utils.logger import logger
from utils.memory import clean_memory, log_memory_usage
from utils.srt_utils import parse_srt
from config import EDGE_TTS_VOICE, AUDIO_DIR

# Thích ứng môi trường chạy async trong môi trường đồng bộ
nest_asyncio.apply()

def clean_tts_text(text: str) -> str:
    """
    Làm sạch văn bản TTS:
    1. Thêm khoảng trắng sau các dấu câu nếu viết sát nhau (ví dụ: tốc.igy -> tốc. igy).
    2. Ánh xạ các ký tự không thuộc bảng chữ cái tiếng Việt (z, f, w, j, apostrophe) sang các âm đọc thuần Việt tương ứng.
    3. Loại bỏ các ký tự đặc biệt không thể đọc được.
    4. Chuẩn hóa khoảng trắng.
    """
    # 1. Thêm khoảng trắng sau các dấu câu nếu viết sát nhau
    text = re.sub(r'([.,?!;:])([a-zA-Z0-9])', r'\1 \2', text)
    
    # 2. Thay thế apostrophe 's bằng s hoặc bỏ đi để tránh lỗi từ đặc biệt
    text = text.replace("'s", "s").replace("'", "")
    
    # 3. Chuyển đổi các ký tự không có trong bảng chữ cái tiếng Việt (z -> d, f -> ph, j -> gi, w -> u)
    # để tránh việc bộ đọc tiếng Việt vi-VN từ chối sinh tiếng hoặc trả về rỗng.
    def replace_non_vietnamese_chars(match):
        char = match.group(0)
        mapping = {
            'z': 'd', 'Z': 'D',
            'f': 'ph', 'F': 'Ph',
            'j': 'gi', 'J': 'Gi',
            'w': 'u', 'W': 'U'
        }
        return mapping.get(char, char)
        
    text = re.sub(r'[zZfFjJwW]', replace_non_vietnamese_chars, text)
    
    # 4. Loại bỏ các ký hiệu đặc biệt khác ngoài chữ cái tiếng Việt, số và một số dấu câu cơ bản
    text = re.sub(r'[^\w\s\d.,?!;:\-\/\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def synthesize_all_tts(entries, metadata, voice, temp_dir) -> dict:
    """
    Sinh toàn bộ tiếng Việt cho tất cả các câu phụ đề một cách tuần tự
    nhưng dùng chung một event loop và một connection/communicate.
    """
    speech_segments = {}
    total = len(entries)
    
    for i, entry in enumerate(entries):
        temp_path = os.path.join(temp_dir, f"temp_{entry.index}.mp3")
        entry_meta = metadata.get(str(entry.index), {})
        tts_text = entry_meta.get("phonetic_text", entry.text)
        if not tts_text.strip():
            tts_text = entry.text
            
        # Làm sạch chuỗi trước khi gửi tới Edge-TTS
        tts_text = clean_tts_text(tts_text)
            
        # Tính toán khả năng mượn thời gian (Cheat gaps)
        prev_entry = entries[i-1] if i > 0 else None
        next_entry = entries[i+1] if i < total - 1 else None
        
        SAFE_GAP = 100 # Giữ ít nhất 100ms an toàn giữa các câu
        MAX_BORROW = 1500 # Mượn tối đa 1.5s mỗi đầu
        
        if prev_entry:
            gap_left = entry.start_ms - prev_entry.end_ms
        else:
            gap_left = entry.start_ms
        usable_left = max(0, min((gap_left - SAFE_GAP) / 2, MAX_BORROW))
        
        if next_entry:
            gap_right = next_entry.start_ms - entry.end_ms
        else:
            gap_right = MAX_BORROW
        usable_right = max(0, min((gap_right - SAFE_GAP) / 2, MAX_BORROW))
        
        orig_allowed = entry.end_ms - entry.start_ms
        if orig_allowed <= 0:
            orig_allowed = 500
            
        # Thử sinh tiếng
        speech = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(tts_text, voice, rate="+0%")
                await communicate.save(temp_path)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    speech = AudioSegment.from_file(temp_path, format="mp3")
                    break
            except Exception as e:
                logger.warning(f"Thử Edge-TTS lần {attempt+1}/{max_retries} thất bại cho '{tts_text[:30]}...'. Chi tiết: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff khi bị chặn
                    await asyncio.sleep(2 * (attempt + 1))
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                        
        if speech is None:
            speech = AudioSegment.silent(duration=1000)
            
        actual_ms = len(speech)
        
        # Bước B: Xử lý khớp thời gian với cơ chế Ăn Gian (Cheating Gaps)
        if actual_ms <= orig_allowed:
            allowed_ms = orig_allowed
            overlay_start = entry.start_ms
        else:
            extra_needed = actual_ms - orig_allowed
            # Ưu tiên mượn gap phía sau (nói lố sau khi sub kết thúc tự nhiên hơn)
            borrow_right = min(usable_right, extra_needed)
            remain = extra_needed - borrow_right
            borrow_left = min(usable_left, remain)
            
            allowed_ms = orig_allowed + borrow_left + borrow_right
            overlay_start = entry.start_ms - borrow_left
            
            if borrow_right > 0 or borrow_left > 0:
                logger.info(f"Câu {entry.index}: Ăn gian khoảng trống (Trái: {int(borrow_left)}ms, Phải: {int(borrow_right)}ms) để giảm ép tốc độ.")
        
        if actual_ms > allowed_ms:
            speed_factor = actual_ms / allowed_ms
            speed_factor = min(speed_factor, 1.8)
            rate_pct = int((speed_factor - 1.0) * 100)
            
            logger.info(f"Cần tăng tốc câu {entry.index} ({actual_ms}ms > {int(allowed_ms)}ms, {speed_factor:.2f}x). Sinh lại với rate=+{rate_pct}%...")
            
            # Sinh lại với tốc độ đã tăng
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(tts_text, voice, rate=f"+{rate_pct}%")
                    await communicate.save(temp_path)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        speech = AudioSegment.from_file(temp_path, format="mp3")
                        break
                except Exception as e:
                    logger.warning(f"Thử Edge-TTS với rate=+{rate_pct}% lần {attempt+1}/{max_retries} thất bại. Chi tiết: {e}")
                    if attempt < max_retries - 1:
                        # Exponential backoff khi bị chặn
                        await asyncio.sleep(2 * (attempt + 1))
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                            
            if speech is None or len(speech) == 0:
                speech = AudioSegment.silent(duration=1000)
            speech = speech[:int(allowed_ms)]
        else:
            padding = AudioSegment.silent(duration=int(allowed_ms - actual_ms))
            speech = speech + padding
            
        speech_segments[entry.index] = {
            "speech": speech,
            "overlay_start": int(overlay_start)
        }
        
        # Log tiến trình nhỏ
        if (i + 1) % 15 == 0 or (i + 1) == total:
            logger.info(f"Đã lồng tiếng: {i + 1}/{total} câu...")
            
        # Khoảng nghỉ 1.5s để tránh bị rate limit từ Microsoft
        await asyncio.sleep(1.5)
        
    return speech_segments

def run(srt_vi_path: str, audio_original_path: str) -> str:
    """
    Điểm chạy chính của Bước 6 (TTS).
    Tạo tiếng Việt, khớp mốc thời gian và gộp lại thành file audio chính thức.
    """
    logger.info("=== BƯỚC 6: SPEECH SYNTHESIS (EDGE-TTS PHONETIC) ===")
    log_memory_usage("TTS - Bắt đầu")
    
    # 1. Tính toán tổng thời lượng video gốc
    original_audio = AudioSegment.from_wav(audio_original_path)
    total_duration_ms = len(original_audio)
    logger.info(f"Tổng thời lượng âm thanh gốc: {total_duration_ms / 1000:.2f} giây")
    
    # Khởi tạo background audio tĩnh
    final_audio = AudioSegment.silent(duration=total_duration_ms)
    
    entries = parse_srt(srt_vi_path)
    if not entries:
        raise ValueError(f"Không có phụ đề tiếng Việt để sinh tiếng: {srt_vi_path}")
        
    # Thử load file metadata JSON song song chứa phonetic_text
    metadata = {}
    metadata_path = os.path.splitext(srt_vi_path)[0] + ".json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Đã tải file metadata phụ đề thành công: {metadata_path}")
        except Exception as e:
            logger.warning(f"Không load được metadata JSON: {e}")
            
    temp_dir = os.path.join(AUDIO_DIR, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    
    total = len(entries)
    logger.info(f"Đang thực hiện lồng tiếng cho {total} câu phụ đề...")
    
    # Chạy hàm async để sinh toàn bộ tiếng Việt
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        speech_segments = loop.run_until_complete(
            synthesize_all_tts(entries, metadata, EDGE_TTS_VOICE, temp_dir)
        )
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi tổng hợp TTS: {e}")
        speech_segments = {entry.index: {"speech": AudioSegment.silent(duration=entry.end_ms - entry.start_ms), "overlay_start": entry.start_ms} for entry in entries}
        
    for entry in entries:
        seg_data = speech_segments.get(entry.index)
        if seg_data and isinstance(seg_data, dict):
            speech = seg_data.get("speech")
            pos = seg_data.get("overlay_start", entry.start_ms)
            if speech:
                final_audio = final_audio.overlay(speech, position=pos)
            
    # Ghi file audio tiếng Việt hoàn chỉnh
    audio_vi_path = os.path.join(AUDIO_DIR, "audio_vietnamese.wav")
    final_audio.export(audio_vi_path, format="wav")
    logger.info(f"Đã xuất bản file âm thanh tiếng Việt hoàn chỉnh: {audio_vi_path}")
    
    # Dọn dẹp thư mục temp
    try:
        shutil = __import__('shutil')
        shutil.rmtree(temp_dir)
    except Exception:
        pass
        
    clean_memory()
    return audio_vi_path
