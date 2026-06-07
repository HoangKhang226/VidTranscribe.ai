import re
import os
from src.utils.logger import logger

class SRTEntry:
    def __init__(self, index: int, start_ms: int, end_ms: int, text: str):
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def __repr__(self):
        return f"SRTEntry({self.index}, {ms_to_srt_time(self.start_ms)} -> {ms_to_srt_time(self.end_ms)}, {self.text})"

def srt_time_to_ms(srt_time_str: str) -> int:
    """Chuyển đổi định dạng HH:MM:SS,mmm thành Mili giây."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", srt_time_str)
    if not match:
        raise ValueError(f"Định dạng thời gian SRT không đúng: {srt_time_str}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + milliseconds

def ms_to_srt_time(ms: int) -> str:
    """Chuyển đổi Mili giây thành định dạng HH:MM:SS,mmm."""
    if ms < 0:
        ms = 0
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_srt(srt_path: str) -> list[SRTEntry]:
    """Phân tích cú pháp file SRT thành danh sách SRTEntry."""
    if not os.path.exists(srt_path):
        logger.error(f"File SRT không tồn tại: {srt_path}")
        return []

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    entries = []
    # Chia file SRT bằng dòng trống
    blocks = re.split(r"\n\s*\n", content)
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                index = int(lines[0].strip())
                time_line = lines[1].strip()
                text = " ".join([l.strip() for l in lines[2:]])
                
                # Bóc tách start -> end
                times = time_line.split("-->")
                if len(times) == 2:
                    start_ms = srt_time_to_ms(times[0].strip())
                    end_ms = srt_time_to_ms(times[1].strip())
                    entries.append(SRTEntry(index, start_ms, end_ms, text))
            except Exception as e:
                logger.warning(f"Lỗi parse block SRT: {block}. Lỗi: {e}")
                
    return entries

def write_srt(entries: list[SRTEntry], output_path: str) -> str:
    """Ghi danh sách SRTEntry ra file SRT."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            f.write(f"{i + 1}\n")
            f.write(f"{ms_to_srt_time(entry.start_ms)} --> {ms_to_srt_time(entry.end_ms)}\n")
            f.write(f"{entry.text}\n\n")
    logger.info(f"Đã ghi file phụ đề tại: {output_path}")
    return output_path

def sanitize_srt_overlap(srt_path: str, min_gap_ms: int = 100) -> str:
    """
    Sanity Check 3: Khử lỗi Ghosting Subtitles
    Nếu mốc start_ms của câu sau nhỏ hơn hoặc trùng khít với end_ms của câu trước (< min_gap_ms),
    tự động điều chỉnh lùi/nhích mốc thời gian để tránh chồng lấp.
    """
    logger.info(f"Đang kiểm tra và khử overlap trong file phụ đề: {srt_path}")
    entries = parse_srt(srt_path)
    if not entries:
        return srt_path

    adjusted_count = 0
    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        
        # Nếu câu hiện tại bắt đầu trước khi câu trước kết thúc + min_gap_ms
        if curr.start_ms < prev.end_ms + min_gap_ms:
            # Tính toán độ lệch
            overlap = (prev.end_ms + min_gap_ms) - curr.start_ms
            
            # Thử co ngắn câu trước lại nếu thời lượng của nó cho phép (tối thiểu 500ms)
            if prev.duration_ms - overlap >= 500:
                prev.end_ms -= overlap
                adjusted_count += 1
            else:
                # Nếu không thể co ngắn câu trước, đẩy lùi câu sau
                curr.start_ms += overlap
                # Đảm bảo end_ms của câu sau cũng được kéo theo để giữ nguyên duration thoại
                curr.end_ms += overlap
                adjusted_count += 1

    if adjusted_count > 0:
        logger.info(f"Đã tự động vá và làm sạch {adjusted_count} điểm trùng lặp phụ đề để tránh Ghosting Subtitles.")
        write_srt(entries, srt_path)
    else:
        logger.info("Phụ đề sạch, không phát hiện lỗi trùng lặp.")
        
    return srt_path
