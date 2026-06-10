import os
import asyncio
import re
import json
import nest_asyncio
import edge_tts
from pydub import AudioSegment
from src.utils.logger import logger
from src.utils.memory import clean_memory, log_memory_usage
from src.utils.srt_utils import parse_srt
from src.config import EDGE_TTS_VOICE, AUDIO_DIR

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
    text = re.sub(r'[^\w\s\d.,?!;\-]', ' ', text)
    # Loại bỏ các dấu câu đứng trơ trọi hoặc liên tiếp
    text = re.sub(r'[.,?!;\-]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def synthesize_all_tts(entries, metadata, voice, temp_dir, preserve_segments: bool = False) -> dict:
    """
    Sinh toàn bộ tiếng Việt cho tất cả các câu phụ đề một cách tuần tự
    nhưng dùng chung một event loop và một connection/communicate.
    Đã bổ sung thuật toán: NHẬN DIỆN VÀ GỘP CÁC CÂU LIÊN TIẾP ĐỂ ĐỌC MƯỢT MÀ.
    """
    speech_segments = {}
    group_metadata = []
    
    # --- THUẬT TOÁN GỘP CÂU LỒNG TIẾNG ---
    groups = []
    current_group = []
    
    for entry in entries:
        if not current_group:
            current_group.append(entry)
        else:
            prev_entry = current_group[-1]
            gap = entry.start_ms - prev_entry.end_ms
            
            # Tính độ dài dự kiến của nhóm nếu gộp thêm câu này
            expected_group_duration = entry.end_ms - current_group[0].start_ms
            
            # Gộp nếu câu trước KHÔNG kết thúc bằng dấu ngắt câu (. ! ? :) VÀ khoảng cách <= 800ms
            # ĐỒNG THỜI tổng thời lượng nhóm không được vượt quá 10 giây (10000ms) để tránh lệch hình (desync)
            ends_with_break = bool(re.search(r'[.!?:]\s*$', prev_entry.text))
            
            if not ends_with_break and gap <= 800 and expected_group_duration <= 10000:
                current_group.append(entry)
            else:
                groups.append(current_group)
                current_group = [entry]
                
    if current_group:
        groups.append(current_group)
        
    total_groups = len(groups)
    logger.info(f"Đã gộp {len(entries)} dòng phụ đề thành {total_groups} nhóm câu để đọc tự nhiên, không bị ngắt.")
    
    SAFE_GAP = 100 # Giữ ít nhất 100ms an toàn giữa các cụm câu
    MAX_BORROW = 1500 # Mượn tối đa 1.5s mỗi đầu
    
    for i, group in enumerate(groups):
        first_entry = group[0]
        last_entry = group[-1]
        temp_path = os.path.join(temp_dir, f"temp_group_{i}.mp3")
        final_group_path = os.path.join(temp_dir, f"temp_group_{i}.mp3")
        
        # Ghép text của cả group lại
        tts_text_parts = []
        for e in group:
            e_meta = metadata.get(str(e.index), {})
            part = e_meta.get("phonetic_text", e.text)
            if not part.strip(): part = e.text
            tts_text_parts.append(part)
            
        tts_text = " ".join(tts_text_parts)
        tts_text = clean_tts_text(tts_text)
        
        # Tính khoảng trống bên trái (dựa trên nhóm trước)
        prev_group = groups[i-1] if i > 0 else None
        if prev_group:
            gap_left = first_entry.start_ms - prev_group[-1].end_ms
        else:
            gap_left = first_entry.start_ms
        usable_left = max(0, min((gap_left - SAFE_GAP) / 2, MAX_BORROW))
        
        # Tính khoảng trống bên phải (dựa trên nhóm sau)
        next_group = groups[i+1] if i < total_groups - 1 else None
        if next_group:
            gap_right = next_group[0].start_ms - last_entry.end_ms
        else:
            gap_right = MAX_BORROW
        usable_right = max(0, min((gap_right - SAFE_GAP) / 2, MAX_BORROW))
        
        orig_allowed = last_entry.end_ms - first_entry.start_ms
        if orig_allowed <= 0:
            orig_allowed = 500
            
        # Thử sinh tiếng
        speech = None
        
        # Nếu text chỉ chứa khoảng trắng hoặc dấu gạch ngang, không gọi Edge-TTS để tránh lỗi
        if not tts_text.strip() or tts_text.strip() == "-":
            speech = AudioSegment.silent(duration=100) # Im lặng siêu ngắn
        else:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(tts_text, voice)
                    await communicate.save(temp_path)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        speech = AudioSegment.from_file(temp_path, format="mp3")
                        break
                except Exception as e:
                    logger.warning(f"Thử Edge-TTS nhóm {i+1} thất bại. Chi tiết: {e}")
                    if attempt < max_retries - 1:
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
        
        if actual_ms <= orig_allowed:
            allowed_ms = orig_allowed
            overlay_start = first_entry.start_ms
        else:
            extra_needed = actual_ms - orig_allowed
            borrow_right = min(usable_right, extra_needed)
            remain = extra_needed - borrow_right
            borrow_left = min(usable_left, remain)
            
            allowed_ms = orig_allowed + borrow_left + borrow_right
            overlay_start = first_entry.start_ms - borrow_left
            
            if borrow_right > 0 or borrow_left > 0:
                logger.info(f"Nhóm {i+1}: Ăn gian khoảng trống (Trái: {int(borrow_left)}ms, Phải: {int(borrow_right)}ms)")
        
        if actual_ms > allowed_ms:
            speed_factor = actual_ms / allowed_ms
            edge_speed_factor = min(speed_factor, 1.5)
            rate_pct = int((edge_speed_factor - 1.0) * 100)
            
            logger.info(f"Cần tăng tốc nhóm {i+1} ({actual_ms}ms > {int(allowed_ms)}ms, Tổng: {speed_factor:.2f}x). Sinh Edge-TTS rate=+{rate_pct}%...")
            
            for attempt in range(max_retries):
                try:
                    communicate = edge_tts.Communicate(tts_text, voice, rate=f"+{rate_pct}%")
                    await communicate.save(temp_path)
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        speech = AudioSegment.from_file(temp_path, format="mp3")
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                        
            if speech is not None and len(speech) > 0 and speed_factor > 1.5:
                extra_speed = speed_factor / 1.5
                logger.info(f"Nhóm {i+1}: Áp dụng FFmpeg atempo thêm {extra_speed:.2f}x...")
                try:
                    import subprocess
                    temp_in = "temp_ffmpeg_in.wav"
                    temp_out = "temp_ffmpeg_out.wav"
                    speech.export(temp_in, format="wav")
                    
                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_in,
                        "-filter:a", f"atempo={extra_speed}",
                        temp_out
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
                        speech = AudioSegment.from_file(temp_out, format="wav")
                    else:
                        raise Exception("FFmpeg output failed")
                        
                    for tmp_f in [temp_in, temp_out]:
                        if os.path.exists(tmp_f): os.remove(tmp_f)
                        
                except Exception as e:
                    logger.warning(f"Lỗi thuật toán FFmpeg ép audio: {e}. Fallback sang chipmunk mode...")
                    new_rate = int(speech.frame_rate * extra_speed)
                    speech = speech._spawn(speech.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(speech.frame_rate)
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                            
            if speech is None or len(speech) == 0:
                speech = AudioSegment.silent(duration=1000)
            
            if len(speech) > allowed_ms:
                speech = speech[:int(allowed_ms)].fade_out(50)
        else:
            padding = AudioSegment.silent(duration=int(allowed_ms - actual_ms))
            speech = speech + padding
            
        # Lưu toàn bộ audio vào câu phụ đề đầu tiên của nhóm
        speech_segments[first_entry.index] = {
            "speech": speech,
            "overlay_start": int(overlay_start)
        }

        if preserve_segments:
            try:
                speech.export(final_group_path, format="mp3")
            except Exception as e:
                logger.warning(f"Không lưu được audio group {i}: {e}")

        group_metadata.append({
            "group_index": i,
            "audio_file": final_group_path,
            "subtitle_indices": [e.index for e in group],
            "start_ms": int(first_entry.start_ms),
            "end_ms": int(last_entry.end_ms),
            "overlay_start_ms": int(overlay_start),
            "allowed_window_ms": int(last_entry.end_ms - first_entry.start_ms),
            "audio_duration_ms": int(len(speech)),
            "text": " ".join(e.text for e in group),
            "tts_text": tts_text,
        })
        # Các câu phụ đề tiếp theo trong nhóm sẽ bị tắt tiếng, nhường chỗ cho khối âm thanh trên chạy vắt ngang qua
        for e in group[1:]:
            speech_segments[e.index] = {
                "speech": AudioSegment.silent(duration=0),
                "overlay_start": int(e.start_ms)
            }
        
        if (i + 1) % 15 == 0 or (i + 1) == total_groups:
            logger.info(f"Đã lồng tiếng: {i + 1}/{total_groups} nhóm câu...")
            
    return {
        "speech_segments": speech_segments,
        "group_metadata": group_metadata,
    }

def run(srt_vi_path: str, audio_original_path: str, keep_temp_segments: bool = False) -> str:
    """
    Điểm chạy chính của Bước 6 (TTS).
    Tạo tiếng Việt, khớp mốc thời gian và gộp lại thành file audio chính thức.

    Args:
        keep_temp_segments: True để giữ lại src/output/audio/temp_segments cho benchmark sync.
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
            
        tts_result = loop.run_until_complete(
            synthesize_all_tts(entries, metadata, EDGE_TTS_VOICE, temp_dir, preserve_segments=keep_temp_segments)
        )
        if isinstance(tts_result, dict) and "speech_segments" in tts_result:
            speech_segments = tts_result.get("speech_segments", {})
            group_metadata = tts_result.get("group_metadata", [])
        else:
            speech_segments = tts_result
            group_metadata = []
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi tổng hợp TTS: {e}")
        speech_segments = {entry.index: {"speech": AudioSegment.silent(duration=entry.end_ms - entry.start_ms), "overlay_start": entry.start_ms} for entry in entries}
        group_metadata = []
        
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
    
    # Dọn dẹp thư mục temp, trừ khi cần giữ lại để chạy sync benchmark.
    if keep_temp_segments:
        groups_json = os.path.join(temp_dir, "tts_groups.json")
        try:
            with open(groups_json, "w", encoding="utf-8") as f:
                json.dump(group_metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã ghi metadata group-level cho benchmark sync: {groups_json}")
        except Exception as e:
            logger.warning(f"Không ghi được metadata group-level: {e}")
        logger.info(f"Giữ lại thư mục audio segment để benchmark sync: {temp_dir}")
    else:
        try:
            shutil = __import__('shutil')
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        
    clean_memory()
    return audio_vi_path
