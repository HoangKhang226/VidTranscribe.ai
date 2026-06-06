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

# Class structures for Batch Translation
class BatchTranslationItem(BaseModel):
    index: int = Field(description="Mã số index của câu thoại đầu vào.")
    translated_text: str = Field(description="Bản dịch tiếng Việt tương ứng cho câu thoại đó, giữ nguyên thuật ngữ tiếng Anh chuyên ngành.")

class BatchTranslationResult(BaseModel):
    translations: List[BatchTranslationItem] = Field(description="Danh sách các câu đã dịch tương ứng với đầu vào.")

TRANSLATE_SYSTEM_PROMPT = """Bạn là một biên dịch viên phụ đề song ngữ Anh-Việt chuyên nghiệp.
Nhiệm vụ của bạn là dịch câu tiếng Anh mục tiêu sang tiếng Việt ngắn gọn và tự nhiên.

QUY TẮC BẮT BUỘC:
1. Dịch tự nhiên, đúng ngữ cảnh của câu. BẠN PHẢI DỊCH MỌI CÂU THOẠI ĐƯỢC CUNG CẤP, tuyệt đối không từ chối dịch hoặc trả về các câu trả lời giải thích, xin lỗi, từ chối.
2. GIỚI HẠN ĐỘ DÀI: Câu dịch trong "translated_text" KHÔNG ĐƯỢC VƯỢT QUÁ {max_chars} ký tự (bao gồm cả khoảng trắng). Hãy chọn lọc từ ngữ cực kỳ cô đọng, súc tích nhưng vẫn giữ nguyên nghĩa cốt lõi.
3. Trong "translated_text": Giữ nguyên dạng viết tiếng Anh của các thuật ngữ chuyên ngành, kỹ thuật, từ mượn hoặc tên riêng tiếng Anh (ví dụ: 'AI', 'marketing', 'B2B', 'DNA', 'API', 'slide', 'CEO') để hiển thị làm phụ đề chuyên nghiệp.
4. TUYỆT ĐỐI KHÔNG sử dụng các từ viết tắt, ký hiệu chat, teencode, hoặc viết tắt tiếng Việt không chuẩn (ví dụ: không được dùng 'đk', 'khng', 'mdl', 'kb', 'vs', 'đc'). Các từ tiếng Việt phải được viết đầy đủ, chính tả rõ ràng.
5. TUYỆT ĐỐI KHÔNG dịch sang tiếng Trung (không chứa bất kỳ chữ Hán hay chữ tượng hình nào), tiếng Nhật, tiếng Hàn hay ngôn ngữ khác ngoài tiếng Việt. Bản dịch bắt buộc phải sử dụng chữ cái Latinh tiếng Việt.
"""

BATCH_TRANSLATE_SYSTEM_PROMPT = """Bạn là một biên dịch viên phụ đề song ngữ Anh-Việt chuyên nghiệp.
Nhiệm vụ của bạn là dịch danh sách các câu tiếng Anh mục tiêu được cung cấp sang tiếng Việt ngắn gọn và tự nhiên.

QUY TẮC BẮT BUỘC:
1. Dịch tự nhiên, đúng ngữ cảnh của câu. BẠN PHẢI DỊCH MỌI CÂU THOẠI ĐƯỢC CUNG CẤP trong danh sách.
2. Với mỗi câu thoại trong danh sách, bản dịch trong "translated_text" phải giữ nguyên chỉ số "index" tương ứng từ đầu vào.
3. Giữ nguyên dạng viết tiếng Anh của các thuật ngữ chuyên ngành, kỹ thuật, từ mượn hoặc tên riêng tiếng Anh (ví dụ: 'AI', 'marketing', 'B2B', 'DNA', 'API', 'slide', 'CEO') để hiển thị làm phụ đề chuyên nghiệp.
4. TUYỆT ĐỐI KHÔNG sử dụng các từ viết tắt, ký hiệu chat, teencode, hoặc viết tắt tiếng Việt không chuẩn. Các từ tiếng Việt phải được viết đầy đủ, chính tả rõ ràng.
5. TUYỆT ĐỐI KHÔNG dịch sang tiếng Trung (không chứa bất kỳ chữ Hán hay chữ tượng hình nào), tiếng Nhật, tiếng Hàn hay ngôn ngữ khác ngoài tiếng Việt. Bản dịch bắt buộc phải sử dụng chữ cái Latinh tiếng Việt.
"""

EXTRACTION_SYSTEM_PROMPT = """Bạn là một trợ lý phân tích ngôn ngữ.
Nhiệm vụ của bạn là đọc câu tiếng Việt được cung cấp và trích xuất tất cả các từ hoặc cụm từ tiếng Anh chuyên ngành, kỹ thuật, từ mượn hoặc tên riêng (ví dụ: 'AI', 'marketing', 'B2B', 'DNA', 'API', 'slide', 'CEO') đang xuất hiện trong câu đó.

QUY TẮC BẮT BUỘC:
1. Chỉ trích xuất các từ hoặc cụm từ viết bằng tiếng Anh có mặt trong câu tiếng Việt đầu vào.
2. Không trích xuất các từ tiếng Việt hoặc từ đã được dịch sang tiếng Việt.
3. Các từ tiếng Anh trích xuất phải được đưa vào danh sách 'english_words'. Nếu không có từ tiếng Anh nào, hãy trả về danh sách rỗng [].
"""

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

def translate_single_entry(entry, idx, total, context_text, llm) -> tuple:
    """Dịch và trích xuất danh sách từ tiếng Anh sử dụng LangChain thông qua 2 bước LLM riêng biệt."""
    allowed_sec = (entry.end_ms - entry.start_ms) / 1000.0
    if allowed_sec <= 0:
        allowed_sec = 0.5
    max_chars = max(24, int(allowed_sec * 14))
    
    # Bước 1: Dịch thuật
    system_prompt = TRANSLATE_SYSTEM_PROMPT.format(max_chars=max_chars)
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """[Bối cảnh các câu thoại tiếng Anh phía trước (CHỈ DÙNG THAM KHẢO, TUYỆT ĐỐI KHÔNG DỊCH):]
{context_text}
 
[Câu thoại tiếng Anh mục tiêu cần dịch sang tiếng Việt:]
{target_text}
 
Hãy dịch câu thoại mục tiêu trên sang tiếng Việt tự nhiên và trả về duy nhất chuỗi tiếng Việt đã dịch trong JSON.""")
    ])
    
    translation_chain = translate_prompt | llm.with_structured_output(TranslationResult)
    
    vietnamese_text = ""
    # Thử dịch tối đa 2 lần nếu phát hiện rò rỉ ngôn ngữ khác
    for attempt in range(2):
        try:
            res = translation_chain.invoke({
                "context_text": context_text if context_text else "[Bắt đầu video]",
                "target_text": entry.text
            })
            vietnamese_text = res.translated_text
            if not has_leakage(vietnamese_text) and vietnamese_text.strip():
                break
            else:
                logger.warning(f"⚠️ Phát hiện rò rỉ ngôn ngữ lạ ở câu {entry.index} (lần thử {attempt+1}): '{vietnamese_text}'. Đang dịch lại...")
        except Exception as e:
            logger.error(f"❌ Lỗi LangChain khi dịch dòng {entry.index} (lần thử {attempt+1}): {e}")
            vietnamese_text = entry.text
            
    # Nếu dịch lỗi hoặc vẫn có lỗi rò rỉ, dùng prompt đơn giản bằng tiếng Anh để dịch lại (dịch thô dự phòng)
    if not vietnamese_text.strip() or has_leakage(vietnamese_text):
        logger.warning(f"⚠️ Dịch JSON có rò rỉ ngôn ngữ ở câu {entry.index}. Chuyển sang prompt dịch thô dự phòng...")
        try:
            raw_prompt = f"Translate the following English sentence into simple, natural Vietnamese. Keep technical terms like 'vector database', 'LLM', 'RAG', 'agent', 'ChatGPT' as is in English. Output ONLY the Vietnamese translation and nothing else. Sentence to translate: '{entry.text}'"
            raw_res = llm.invoke(raw_prompt)
            raw_text = raw_res.content.strip()
            # Loại bỏ ngoặc kép ở đầu và cuối nếu có
            raw_text = re.sub(r'^["\']|["\']$', '', raw_text).strip()
            
            if raw_text and not has_leakage(raw_text):
                vietnamese_text = raw_text
                logger.info(f"✅ Dịch thô dự phòng thành công câu {entry.index}: '{vietnamese_text}'")
        except Exception as e:
            logger.error(f"❌ Lỗi dịch thô dự phòng câu {entry.index}: {e}")
            
    # Nếu vẫn lỗi, dùng câu gốc tiếng Anh làm fallback cuối cùng
    if not vietnamese_text.strip() or has_leakage(vietnamese_text):
        logger.warning(f"⚠️ Không thể loại bỏ lỗi rò rỉ ngôn ngữ ở câu {entry.index}. Dùng câu gốc tiếng Anh làm fallback.")
        vietnamese_text = entry.text
        
    # Gióng hàng từ tiếng Anh
    word_positions = extract_and_align_entry(entry, vietnamese_text, llm)
    logger.info(f"✅ Đã dịch câu {entry.index} ({idx+1}/{total}): '{vietnamese_text}' | English terms: {[w['word'] for w in word_positions]}")
    return entry.index, vietnamese_text, word_positions

def translate_single_batch(batch_entries, context_text, llm) -> dict:
    """
    Dịch cả batch câu thoại cùng lúc sử dụng LangChain.
    Trả về: dict[int, str] mapping index -> translated_text, hoặc None nếu thất bại/rò rỉ.
    """
    targets = [{"index": e.index, "text": e.text} for e in batch_entries]
    target_list_str = json.dumps(targets, ensure_ascii=False, indent=2)
    
    batch_prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_TRANSLATE_SYSTEM_PROMPT),
        ("human", """[Bối cảnh các câu thoại tiếng Anh phía trước (CHỈ DÙNG THAM KHẢO, TUYỆT ĐỐI KHÔNG DỊCH):]
{context_text}

[Danh sách các câu thoại tiếng Anh mục tiêu cần dịch sang tiếng Việt:]
{target_list_str}

Hãy dịch toàn bộ danh sách câu thoại mục tiêu trên sang tiếng Việt tự nhiên và trả về cấu trúc JSON tương ứng với đúng số lượng phần tử và index.""")
    ])
    
    batch_chain = batch_prompt | llm.with_structured_output(BatchTranslationResult)
    
    for attempt in range(2):
        try:
            res = batch_chain.invoke({
                "context_text": context_text if context_text else "[Bắt đầu video]",
                "target_list_str": target_list_str
            })
            
            temp_map = {}
            has_error = False
            for item in res.translations:
                text = item.translated_text
                # Check for leakage
                if has_leakage(text):
                    logger.warning(f"⚠️ Phát hiện rò rỉ ngôn ngữ lạ ở batch (lần thử {attempt+1}): '{text}'")
                    has_error = True
                    break
                temp_map[item.index] = text
                
            # Kiểm tra xem có đủ câu dịch cho tất cả các index trong batch không
            if not has_error and all(e.index in temp_map for e in batch_entries):
                return temp_map
        except Exception as e:
            logger.error(f"❌ Lỗi LangChain khi dịch batch (lần thử {attempt+1}): {e}")
            
    return None

def extract_and_align_entry(entry, vietnamese_text, llm) -> list:
    """
    Trích xuất các từ tiếng Anh chuyên ngành từ câu dịch và tính toán vị trí index.
    """
    # Bước 2: Trích xuất các từ tiếng Anh có trong câu dịch
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        ("human", "Hãy trích xuất các từ tiếng Anh từ câu dịch sau:\n{vietnamese_text}")
    ])
    
    extraction_chain = extraction_prompt | llm.with_structured_output(EnglishExtractionResult)
    
    extracted_words = []
    try:
        res_ext = extraction_chain.invoke({"vietnamese_text": vietnamese_text})
        extracted_words = res_ext.english_words
    except Exception as e:
        logger.warning(f"⚠️ Không trích xuất được từ tiếng Anh của dòng {entry.index}: {e}")
        
    # Quét bổ sung bằng Python để tránh LLM bỏ sót từ
    scanned_words = []
    try:
        orig_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', entry.text.lower()))
        for w in vietnamese_text.split():
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if (clean_w in orig_words or any(clean_w in ow for ow in orig_words)) and len(clean_w) >= 3:
                scanned_words.append(clean_w)
    except Exception as e:
        logger.warning(f"⚠️ Lỗi quét Python bổ sung cho dòng {entry.index}: {e}")
        
    # Hợp nhất danh sách từ tiếng Anh
    merged_words = list(set([w.lower().strip() for w in extracted_words] + scanned_words))
    
    # Bước 3: Tính toán vị trí bằng Python
    word_positions = []
    try:
        words_list = vietnamese_text.split()
        for pos, w in enumerate(words_list):
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            # Chỉ khớp nếu từ gốc trong câu dịch là pure ASCII (để tránh khớp nhầm các từ tiếng Việt như 'vectơ')
            if clean_w in merged_words and w.isascii():
                word_positions.append({"word": clean_w, "position": pos})
    except Exception as e:
        logger.error(f"❌ Lỗi tính toán vị trí index cho dòng {entry.index}: {e}")
        
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
        logger.error(f"❌ Lỗi LangChain khi phiên âm hàng loạt: {e}")
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
        logger.info(f"⚠️ Giới hạn dịch thử nghiệm {limit} câu đầu tiên.")
        entries = entries[:limit]
        
    translated_entries = []
    metadata_records = {}
    total = len(entries)
    
    # Chia các câu thoại thành các batch kích thước 3
    batch_size = 3
    batches = []
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i : i + batch_size]
        
        # Ngữ cảnh gối đầu (Sliding Window Context): Lấy tối đa 2 câu thoại cuối của batch trước đó
        context_list = []
        if i >= 1:
            prev_start = max(0, i - 2)
            context_list = [entries[k].text for k in range(prev_start, i)]
            
        context_text = " | ".join(context_list)
        batches.append((batch_entries, context_text))
        
    results_map = {}
    all_english_terms = set()
    
    ollama_base_url = OLLAMA_API_URL.replace("/api/generate", "")
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_base_url,
        temperature=0.1
    )
    
    def process_batch(batch_entries, context_text):
        """Hàm xử lý một batch câu thoại: dịch batch -> fallback nếu lỗi -> trích xuất và gióng hàng."""
        # 1. Dịch batch
        batch_translation = translate_single_batch(batch_entries, context_text, llm)
        
        batch_results = []
        for entry in batch_entries:
            global_idx = entries.index(entry)
            
            vietnamese_text = None
            if batch_translation:
                vietnamese_text = batch_translation.get(entry.index)
                
            if not vietnamese_text or not vietnamese_text.strip():
                # Chuyển sang dịch đơn lẻ dự phòng nếu batch dịch lỗi
                logger.warning(f"⚠️ Dịch batch thất bại hoặc thiếu dòng cho index {entry.index}. Chuyển sang dịch đơn lẻ dự phòng.")
                _, vietnamese_text, word_positions = translate_single_entry(entry, global_idx, total, context_text, llm)
            else:
                # Nếu dịch thành công, thực hiện trích xuất và căn chỉnh từ tiếng Anh
                word_positions = extract_and_align_entry(entry, vietnamese_text, llm)
                logger.info(f"✅ Đã dịch (Batch) câu {entry.index} ({global_idx+1}/{total}): '{vietnamese_text}' | English terms: {[w['word'] for w in word_positions]}")
                
            batch_results.append((entry.index, vietnamese_text, word_positions))
        return batch_results

    # Chạy tuần tự các batches với 1 worker để tránh deadlock kết nối Ollama
    logger.info(f"🚀 Bắt đầu dịch {len(batches)} batches bằng mô hình {model_name}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_batch, b[0], b[1]): i
            for i, b in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                for idx, vietnamese_text, word_positions in batch_results:
                    results_map[idx] = (vietnamese_text, word_positions)
                    for w_info in word_positions:
                        clean_term = w_info["word"].strip().lower()
                        if clean_term and re.match(r'^[a-z0-9\s\-]+$', clean_term) and len(clean_term) >= 3:
                            all_english_terms.add(clean_term)
            except Exception as exc:
                logger.error(f"❌ Batch {batch_idx} phát sinh lỗi nghiêm trọng: {exc}")
                for entry in batches[batch_idx][0]:
                    results_map[entry.index] = (entry.text, [])
                    
    # 2. Sinh phiên âm hàng loạt cho danh sách từ tiếng Anh bằng G2P + MOP
    english_list = list(all_english_terms)
    logger.info(f"📚 Đang tạo phiên âm (G2P) cho {len(english_list)} thuật ngữ tiếng Anh: {english_list}")
    transliterations_map = transliterate_batch(english_list)
    logger.info(f"✅ Đã nhận được bản đồ phiên âm (G2P): {transliterations_map}")
    
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
        
    logger.info(f"✅ Đã ghi file metadata tiếng Việt tại: {metadata_path}")
    
    # Dọn dẹp
    clean_memory()
    return srt_vi_path
