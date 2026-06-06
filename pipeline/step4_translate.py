import os
import json
import re
import concurrent.futures
from pydantic import BaseModel, Field
from typing import List, Dict
from utils.logger import logger
from utils.memory import clean_memory
from utils.srt_utils import parse_srt, write_srt
from utils.g2p_helper import transliterate_batch
from config import OLLAMA_API_URL, OLLAMA_MODEL_NAME, SUBTITLES_DIR

# Import LangChain components
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class TranslationResult(BaseModel):
    translated_text: str = Field(description="Câu dịch tiếng Việt ngắn gọn, tự nhiên, giữ nguyên các thuật ngữ tiếng Anh chuyên ngành.")

class EnglishExtractionResult(BaseModel):
    english_words: List[str] = Field(description="Danh sách các từ hoặc cụm từ tiếng Anh chuyên ngành xuất hiện trong câu dịch.")

class TransliterationResult(BaseModel):
    transliterations: Dict[str, str] = Field(description="Bản đồ map từ tiếng Anh sang cách đọc tiếng Việt. Ví dụ: {'qdrant': 'quát đrơnt'}")

# (Đã loại bỏ Batch Translation classes)

def extract_tech_terms(srt_en_path: str, model_name: str) -> list[str]:
    """
    Đọc tối đa 20 câu thoại đầu tiên từ kịch bản phụ đề tiếng Anh en.srt
    và dùng Ollama để trích xuất các thuật ngữ chuyên ngành đa ngành, từ viết tắt, tên riêng
    hoặc khái niệm khó dịch cần giữ nguyên dạng tiếng Anh gốc.
    """
    logger.info("Đang tự động trích xuất thuật ngữ chuyên ngành đa ngành từ kịch bản phụ đề...")
    entries = parse_srt(srt_en_path)
    if not entries:
        return []
        
    # Lấy tối đa 20 câu thoại đầu (khoảng 1 phút) để hiểu ngữ cảnh chung của video
    sample_text = " ".join([e.text for e in entries[:20]])
    
    ollama_base_url = OLLAMA_API_URL.replace("/api/generate", "")
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.1
    )
    
    class KeywordExtraction(BaseModel):
        keywords: List[str] = Field(description="Danh sách các thuật ngữ, từ viết tắt, tên công nghệ, từ mượn tiếng Anh quan trọng xuất hiện trong kịch bản (viết thường).")
        
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Bạn là một chuyên gia ngôn ngữ học đa ngành. Nhiệm vụ của bạn là phân tích đoạn kịch bản video tiếng Anh đầu vào "
            "(thuộc bất kỳ lĩnh vực nào như IT, khoa học, y tế, kinh tế, cơ khí, nghệ thuật...) và trích xuất các thuật ngữ "
            "chuyên ngành, từ viết tắt, hoặc cụm danh từ tiếng Anh mà khi dịch sang tiếng Việt NÊN GIỮ NGUYÊN (không dịch nghĩa) "
            "để đảm bảo tính tự nhiên và chính xác của phụ đề.\n"
            "Ví dụ:\n"
            "- Lĩnh vực IT: 'ai agents', 'workflow', 'llm', 'rag', 'overfitting', 'api'\n"
            "- Lĩnh vực sinh/y học: 'mrna', 'crispr', 'vaccine', 'dna'\n"
            "- Lĩnh vực kinh doanh: 'b2b', 'marketing', 'pitch deck', 'equity'\n"
            "Hãy trả về danh sách các thuật ngữ viết thường."
        )),
        ("human", "Hãy trích xuất thuật ngữ chuyên ngành từ đoạn kịch bản sau, nếu có quá ít, hãy tự dùng kiến thức có sẵn để trích ra, đoạn kịch bản \n{text}")
    ])
    
    chain = extraction_prompt | llm.with_structured_output(KeywordExtraction)
    
    extracted = []
    try:
        res = chain.invoke({"text": sample_text})
        extracted = [w.lower().strip() for w in res.keywords if w.strip()]
        logger.info(f"Đã trích xuất được các thuật ngữ: {extracted}")
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất thuật ngữ từ kịch bản: {e}")
        
    return extracted

TRANSLATION_RULES = """1. DỊCH THEO NGỮ CẢNH (Contextual): Phải hiểu ý đồ của cả đoạn để dịch thoát ý.
   - Ví dụ: "You think you know X? You don't." -> "Bạn nghĩ bạn hiểu X? Chưa chắc đâu / Bạn lầm rồi" (KHÔNG dịch là "Bạn không làm").
   - Nếu video nói "plain English", hãy dịch đúng là "tiếng Anh thông thường" hoặc "ngôn ngữ tự nhiên" (KHÔNG tự ý bản địa hóa thành "tiếng Việt").
2. GIỮ NGUYÊN TÊN RIÊNG & THUẬT NGỮ CỐT LÕI (Không dịch):
   - Tuyệt đối giữ nguyên tên sản phẩm thương mại, tên phần mềm (VD: Copilot Studio, ChatGPT).
   - Giữ nguyên các thuật ngữ kiến trúc/chuyên ngành mang tính toàn cầu (VD: front-end, back-end, Agent, AI, API).
   - Danh sách các từ CẦN GIỮ NGUYÊN: {tech_terms_str}.
   - Xưng hô "Agent" giữ nguyên hoặc dịch là "Tác nhân" (Tuyệt đối KHÔNG dịch là "Đại diện").
3. TỰ ĐỘNG SỬA LỖI STT: Nếu kịch bản có lỗi nghe nhầm (VD: "leap capture" -> "thu thập lead" / "lead capture"), hãy TỰ ĐỘNG dịch theo thuật ngữ đúng.
4. VĂN PHONG TỰ NHIÊN: Đảm bảo văn bản đầu ra là 100% TIẾNG VIỆT chuẩn xác. KHÔNG giữ lại tiếng Anh giao tiếp (VD: "seamlessly"). Chỉnh lại văn phong nếu LLM dịch sai (VD: sửa "Đã như bạn đã thuê..." thành "Cứ như bạn đã thuê...").
5. KHÔNG dùng teencode. KHÔNG dịch sang ngôn ngữ thứ 3."""

PARAGRAPH_TRANSLATE_SYSTEM_PROMPT = f"""Bạn là một biên dịch viên phụ đề song ngữ Anh-Việt cao cấp, chuyên dịch thuật đa ngành (IT, y tế, kinh tế, v.v.).
Nhiệm vụ của bạn là dịch NGUYÊN MỘT ĐOẠN VĂN TIẾNG ANH sang tiếng Việt sao cho tự nhiên, mượt mà như người bản xứ nói chuyện, tuyệt đối không dịch kiểu word-by-word (word-for-word).

QUY TẮC BẮT BUỘC:
{TRANSLATION_RULES}
6. CHỈ TRẢ VỀ bản dịch tiếng Việt, KHÔNG giải thích, KHÔNG thêm ngoặc kép bao quanh.
"""

def normalize_numbers_to_text(text: str) -> str:
    """Chuyển đổi số thành chữ tiếng Việt để tránh lệch timeline TTS."""
    text = text.replace("%", " phần trăm")
    def num_to_vi(match):
        num_str = match.group()
        if not num_str.isdigit(): return num_str
        n = int(num_str)
        units = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
        if n < 10: return units[n]
        if n < 100:
            tens, ones = n // 10, n % 10
            res = (units[tens] + " mươi") if tens > 1 else "mười"
            if ones == 1 and tens > 1: res += " mốt"
            elif ones == 5 and tens > 0: res += " lăm"
            elif ones > 0: res += " " + units[ones]
            return res
        if n < 1000:
            hundreds, rem = n // 100, n % 100
            res = units[hundreds] + " trăm"
            if rem == 0: return res
            if rem < 10: return res + " lẻ " + units[rem]
            tens, ones = rem // 10, rem % 10
            res += " " + ((units[tens] + " mươi") if tens > 1 else "mười")
            if ones == 1 and tens > 1: res += " mốt"
            elif ones == 5 and tens > 0: res += " lăm"
            elif ones > 0: res += " " + units[ones]
            return res
        return num_str
    return re.sub(r'\b\d+\b', num_to_vi, text)

PHONETIC_SYSTEM_PROMPT = """Bạn là một chuyên gia ngôn ngữ học và dịch thuật chuyên nghiệp.
Nhiệm vụ của bạn là phiên âm các từ/cụm từ tiếng Anh chuyên ngành sang cách đọc/phát âm bằng chữ tiếng Việt (chỉ dùng các âm đọc tiếng Việt thông dụng) để bộ đọc TTS tiếng Việt có thể phát âm chuẩn xác.

QUY TẮC BẮT BUỘC:
1. Đối với mỗi từ tiếng Anh trong danh sách đầu vào, hãy tạo ra cách phát âm chuẩn bằng tiếng Việt.
   Ví dụ:
   - "vector" -> "véc tơ"
   - "database" -> "đa ta bây"
   - "RAG" -> "rác"
   - "workflow" -> "quớt phờ lâu"
   - "LLM" -> "eo eo em"
   - "API" -> "a pi ai"
   - "Python" -> "pai thân"
   - "Qdrant" -> "quát đờ rần"
   - "payload" -> "pay lốt"
   - "index" -> "in đếch"
   - "default" -> "đì phau"
   - "semantic" -> "se man tíc"
   - "similarity" -> "si mi la ri ti"
   - "chunk" -> "chăng"
   - "size" -> "sai"
   - "either" -> "i đơ"
2. MỌI TỪ PHIÊN ÂM tiếng Việt phải tuân thủ đúng chính tả và cấu trúc âm tiết tiếng Việt thuần túy.
   - TUYỆT ĐỐI KHÔNG chứa các chữ cái không thuộc bảng chữ cái tiếng Việt như 'z', 'f', 'w', 'j'. Hãy thay thế chúng bằng: 'z' -> 'd', 'f' -> 'ph', 'j' -> 'gi', 'w' -> 'u' hoặc 'o'.
   - TUYỆT ĐỐI KHÔNG kết thúc từ bằng các phụ âm không hợp lệ trong tiếng Việt (như r, l, s, d, g, x, b, h, v, q, k). Ví dụ:
     * Không phiên âm "azure" thành "az u re", hãy phiên âm thành "a dơ" hoặc "a du re".
     * Không phiên âm "email" thành "ai maul" hay "i meo-l", hãy phiên âm thành "i meo" hoặc "e mai".
     * Không phiên âm "genspar" thành "gen spar", hãy phiên âm thành "gen xpa".
     * Không phiên âm "studio" thành "stoo do" hay "stu-di-o", hãy phiên âm thành "xtu đi ô".
     * Không phiên âm "tech" thành "teh", hãy phiên âm thành "tếch" hoặc "téc".
"""

def has_leakage(text: str) -> bool:
    """Kiểm tra xem văn bản có bị rò rỉ ký tự tiếng Trung/Cyrillic không."""
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    if re.search(r'[\u0400-\u04FF]', text):
        return True
    return False

def translate_paragraph(paragraph_text: str, context_text: str, llm, tech_terms_str: str) -> str:
    """Dịch một đoạn văn bản hoàn chỉnh sử dụng LangChain."""
    system_prompt = PARAGRAPH_TRANSLATE_SYSTEM_PROMPT.format(tech_terms_str=tech_terms_str)
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """[Bối cảnh các đoạn phía trước (CHỈ DÙNG THAM KHẢO, KHÔNG DỊCH):]
{context_text}
 
[Đoạn văn tiếng Anh cần dịch sang tiếng Việt:]
{target_text}
 
Hãy dịch đoạn văn trên sang tiếng Việt tự nhiên và trả về duy nhất chuỗi tiếng Việt đã dịch trong JSON.""")
    ])
    
    translation_chain = translate_prompt | llm.with_structured_output(TranslationResult)
    
    vietnamese_text = ""
    for attempt in range(2):
        try:
            res = translation_chain.invoke({
                "context_text": context_text if context_text else "[Bắt đầu video]",
                "target_text": paragraph_text
            })
            vietnamese_text = res.translated_text
            if not has_leakage(vietnamese_text) and vietnamese_text.strip() and vietnamese_text.lower().strip() != paragraph_text.lower().strip():
                return normalize_numbers_to_text(vietnamese_text)
            else:
                logger.warning(f"Phát hiện rò rỉ hoặc dịch lỗi ở paragraph (lần thử {attempt+1}): '{vietnamese_text}'. Đang dịch lại...")
        except Exception as e:
            logger.error(f"Lỗi LangChain khi dịch paragraph (lần thử {attempt+1}): {e}")
            
    if not vietnamese_text.strip() or has_leakage(vietnamese_text):
        logger.warning(f"Dịch JSON lỗi, chuyển sang dịch thô dự phòng...")
        try:
            raw_prompt = f"""Bạn là biên dịch viên phụ đề. Hãy dịch ĐOẠN VĂN tiếng Anh sau sang tiếng Việt tự nhiên nhất.
Quy tắc:
{TRANSLATION_RULES.replace('{tech_terms_str}', tech_terms_str)}
6. CHỈ trả về bản dịch tiếng Việt, KHÔNG giải thích.

Đoạn văn cần dịch: '{paragraph_text}'"""
            raw_res = llm.invoke(raw_prompt)
            raw_text = raw_res.content.strip()
            raw_text = re.sub(r'^["\']|["\']$', '', raw_text).strip()
            
            if raw_text and not has_leakage(raw_text):
                return raw_text
        except Exception as e:
            logger.error(f"Lỗi dịch thô: {e}")
            
    logger.warning("Dịch paragraph thô lỗi, chuyển sang dịch từng câu lẻ...")
    sentences = re.split(r'(?<=[.!?])\s+', paragraph_text)
    translated_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            raw_prompt = f"""Bạn là biên dịch viên phụ đề. Hãy dịch CÂU tiếng Anh sau sang tiếng Việt tự nhiên nhất.
Quy tắc:
{TRANSLATION_RULES.replace('{tech_terms_str}', tech_terms_str)}
6. Dịch dựa trên ngữ cảnh của Đoạn văn gốc: '{paragraph_text}'.
7. CHỈ trả về bản dịch tiếng Việt, KHÔNG giải thích.

Câu cần dịch: '{sentence}'"""
            raw_res = llm.invoke(raw_prompt)
            raw_text = raw_res.content.strip()
            raw_text = re.sub(r'^["\']|["\']$', '', raw_text).strip()
            
            if raw_text and not has_leakage(raw_text) and raw_text.lower() != sentence.lower():
                translated_sentences.append(raw_text)
            else:
                logger.warning(f"Bỏ qua câu bị rò rỉ: '{sentence}'")
        except Exception as e:
            logger.warning(f"Lỗi khi dịch lẻ câu: {e}")
            
    if translated_sentences:
        return normalize_numbers_to_text(" ".join(translated_sentences))
        
    logger.warning("Dùng đoạn gốc tiếng Anh làm fallback cuối cùng.")
    return normalize_numbers_to_text(paragraph_text)

def split_translation_to_entries(translated_paragraph: str, original_entries: list) -> list:
    """Tách bản dịch đoạn thành từng dòng theo tỷ lệ thời lượng."""
    total_duration = sum(e.duration_ms for e in original_entries)
    if total_duration == 0:
        total_duration = 1
        
    words = translated_paragraph.split()
    total_words = len(words)
    
    result = []
    word_cursor = 0
    for i, entry in enumerate(original_entries):
        ratio = entry.duration_ms / total_duration
        word_count = max(1, round(total_words * ratio))
        
        if i == len(original_entries) - 1:
            chunk_words = words[word_cursor:]
        else:
            chunk_words = words[word_cursor:word_cursor + word_count]
            
        result.append(" ".join(chunk_words))
        word_cursor += len(chunk_words)
        
    for i in range(len(result)):
        if not result[i].strip():
            result[i] = "-"
            
    return result

def extract_and_align_entry(entry, vietnamese_text, all_terms) -> list:
    """
    Trích xuất các từ tiếng Anh chuyên ngành bằng Regex và tập từ khóa (không dùng LLM).
    """
    orig_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', entry.text.lower()))
    
    # Tập hợp các từ tiếng Việt không dấu phổ biến để tránh nhận nhầm thành tiếng Anh
    vietnamese_no_diacritics = {"ai", "la", "co", "ba", "ca", "da", "nha", "cha", "khi", "cho", "thu", "thi", "nho", "ho", "sau", "chi", "con", "lon", "nhi", "minh", "nam", "ly", "ma", "ra", "xa", "to", "no", "va", "tu", "ta"}
    
    word_positions = []
    try:
        words_list = vietnamese_text.split()
        for pos, w in enumerate(words_list):
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if not clean_w: continue
            
            is_english = False
            # Nếu từ này nằm trong danh sách keyword đã biết
            if any(clean_w == t or clean_w in t.split() for t in all_terms):
                is_english = True
            # Hoặc nếu từ này rõ ràng là tiếng Anh (thuộc câu gốc) và không bị nhầm lẫn với tiếng Việt không dấu
            elif clean_w in orig_words and clean_w not in vietnamese_no_diacritics and not clean_w.isnumeric():
                if any(c in clean_w for c in 'fjwz') or len(clean_w) > 3:
                    is_english = True
                    
            if is_english and w.isascii():
                word_positions.append({"word": clean_w, "position": pos})
    except Exception as e:
        logger.error(f"Lỗi tính toán vị trí index cho dòng {entry.index}: {e}")
        
    return word_positions

def get_batch_transliterations_langchain(english_terms: list, model_name: str = OLLAMA_MODEL_NAME) -> dict:
    """Gọi Ollama phiên âm hàng loạt danh sách từ tiếng Anh sử dụng LangChain."""
    if not english_terms:
        return {}
        
    ollama_base_url = OLLAMA_API_URL.replace("/api/generate", "")
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.1
    )
    
    phonetic_prompt = ChatPromptTemplate.from_messages([
        ("system", PHONETIC_SYSTEM_PROMPT),
        ("human", "Hãy phiên âm danh sách từ/cụm từ tiếng Anh kỹ thuật sau sang cách đọc tiếng Việt:\n{english_terms}")
    ])
    
    structured_llm = llm.with_structured_output(TransliterationResult)
    phonetic_chain = phonetic_prompt | structured_llm
    
    try:
        result = phonetic_chain.invoke({
            "english_terms": json.dumps(english_terms, ensure_ascii=False)
        })
        return result.transliterations
    except Exception as e:
        logger.error(f"Lỗi LangChain khi phiên âm hàng loạt: {e}")
    return {}

def run(srt_en_path: str, context: dict, model_name: str = OLLAMA_MODEL_NAME, limit: int = None) -> str:
    """
    Điểm chạy chính của Bước 5 (Translation).
    Dịch song song theo batches với Sliding Window Context (lịch sử gối đầu), khống chế độ dài ký tự và phiên âm chuẩn xác.
    """
    logger.info("=== BƯỚC 5: DỊCH THUẬT PHỤ ĐỀ (DECOUPLED LANGCHAIN BATCH) ===")
    
    entries = parse_srt(srt_en_path)
    if not entries:
        raise ValueError(f"Không phân tích được dòng phụ đề nào từ: {srt_en_path}")
        
    if limit is not None:
        logger.info(f"Giới hạn dịch thử nghiệm {limit} câu đầu tiên.")
        entries = entries[:limit]
        
    translated_entries = []
    metadata_records = {}
    total = len(entries)

    # Tự động trích xuất các thuật ngữ chuyên ngành đa ngành từ kịch bản tiếng Anh
    extracted_terms = extract_tech_terms(srt_en_path, model_name)
    context_keywords = context.get("keywords", []) if isinstance(context, dict) else []
    all_terms = set(extracted_terms) | set(context_keywords)
    tech_terms_str = ", ".join([f"'{term}'" for term in sorted(all_terms)])
    logger.info(f"Tổng hợp các thuật ngữ chuyên ngành đa ngành cần giữ nguyên tiếng Anh: {tech_terms_str}")
    
    # Chia các câu thoại thành các đoạn (paragraphs), kích thước 5
    group_size = 5
    paragraphs = []
    for i in range(0, len(entries), group_size):
        group = entries[i : i + group_size]
        
        # Ngữ cảnh gối đầu: Lấy đoạn trước đó
        context_list = []
        if i >= group_size:
            prev_start = max(0, i - group_size)
            context_list = [e.text for e in entries[prev_start:i]]
            
        context_text = " ".join(context_list)
        full_text = " ".join([e.text for e in group])
        paragraphs.append((group, full_text, context_text))
        
    results_map = {}
    all_english_terms = set()
    
    ollama_base_url = OLLAMA_API_URL.replace("/api/generate", "")
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.1
    )
    
    def process_paragraph(group, full_text, context_text):
        """Hàm xử lý một paragraph: dịch -> tách -> trích xuất và gióng hàng."""
        translated_para = translate_paragraph(full_text, context_text, llm, tech_terms_str)
        split_texts = split_translation_to_entries(translated_para, group)
        
        batch_results = []
        for i, entry in enumerate(group):
            global_idx = entries.index(entry)
            vietnamese_text = split_texts[i]
            
            # Thực hiện trích xuất và căn chỉnh từ tiếng Anh cho từng dòng
            word_positions = extract_and_align_entry(entry, vietnamese_text, all_terms)
            logger.info(f"Đã dịch (Paragraph Split) câu {entry.index} ({global_idx+1}/{total}): '{vietnamese_text}' | English terms: {[w['word'] for w in word_positions]}")
            
            batch_results.append((entry.index, vietnamese_text, word_positions))
        return batch_results

    # Chạy tuần tự các paragraphs với 1 worker để tránh deadlock kết nối Ollama
    logger.info(f"Bắt đầu dịch {len(paragraphs)} paragraphs bằng mô hình {model_name}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_paragraph, p[0], p[1], p[2]): i
            for i, p in enumerate(paragraphs)
        }
        for future in concurrent.futures.as_completed(futures):
            para_idx = futures[future]
            try:
                batch_results = future.result()
                for idx, vietnamese_text, word_positions in batch_results:
                    results_map[idx] = (vietnamese_text, word_positions)
                    for w_info in word_positions:
                        clean_term = w_info["word"].strip().lower()
                        if clean_term and re.match(r'^[a-z0-9\s\-]+$', clean_term) and len(clean_term) >= 3:
                            all_english_terms.add(clean_term)
            except Exception as exc:
                logger.error(f"Paragraph {para_idx} phát sinh lỗi nghiêm trọng: {exc}")
                for entry in paragraphs[para_idx][0]:
                    results_map[entry.index] = (entry.text, [])
                    
    # 2. Sinh phiên âm hàng loạt cho danh sách từ tiếng Anh bằng G2P + MOP
    english_list = list(all_english_terms)
    logger.info(f"Đang tạo phiên âm (G2P) cho {len(english_list)} thuật ngữ tiếng Anh: {english_list}")
    transliterations_map = transliterate_batch(english_list)
    logger.info(f"Đã nhận được bản đồ phiên âm (G2P): {transliterations_map}")
    
    # Chuẩn hóa khóa của bản đồ về chữ thường
    transliterations_map = {k.lower().strip(): v.strip() for k, v in transliterations_map.items()}
    
    # Gán lại kết quả dịch và thay thế từ tiếng Anh bằng từ phiên âm trong phonetic_text theo đúng vị trí index
    for entry in entries:
        vietnamese_text, word_positions = results_map.get(entry.index, (entry.text, []))
        entry.text = vietnamese_text
        translated_entries.append(entry)
        
        # Tách câu dịch thành danh sách từ
        words = vietnamese_text.split()
        
        # Sắp xếp các từ tiếng Anh theo vị trí position giảm dần (phải sang trái)
        # để không làm thay đổi chỉ số index của các từ phía trước khi thay thế
        sorted_word_positions = sorted(word_positions, key=lambda x: x["position"], reverse=True)
        
        for info in sorted_word_positions:
            pos = info["position"]
            term = info["word"].strip().lower()
            if 0 <= pos < len(words):
                phonetic_val = transliterations_map.get(term)
                if phonetic_val:
                    # Giữ nguyên dấu câu kèm theo ở cuối từ gốc
                    orig_word = words[pos]
                    match = re.match(r'^([a-zA-Z0-9\s\-]+)([^a-zA-Z0-9\s\-]*)$', orig_word)
                    if match:
                        clean_word = match.group(1)
                        punctuation = match.group(2)
                        words[pos] = phonetic_val + punctuation
                    else:
                        words[pos] = phonetic_val
                        
        phonetic_text = " ".join(words)
        
        metadata_records[str(entry.index)] = {
            "translated_text": vietnamese_text,
            "phonetic_text": phonetic_text
        }
        
    # Ghi file phụ đề tiếng Việt chính thức
    srt_vi_path = os.path.join(SUBTITLES_DIR, "subtitles_vi.srt")
    write_srt(translated_entries, srt_vi_path)
    
    # Ghi file metadata JSON bổ trợ chứa phonetic_text
    metadata_path = os.path.join(SUBTITLES_DIR, "subtitles_vi.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Đã ghi file metadata tiếng Việt tại: {metadata_path}")
    
    # Dọn dẹp
    clean_memory()
    return srt_vi_path
