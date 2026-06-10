import os
import json
import re
import concurrent.futures
from pydantic import BaseModel, Field
from typing import List, Dict
from src.utils.logger import logger
from src.utils.memory import clean_memory
from src.utils.srt_utils import parse_srt, write_srt
from src.utils.g2p_helper import transliterate_batch
from src.utils.translation_cache import TranslationCache
from src.utils.word_translator import WordLevelTranslator
from src.utils.translation_adjudicator import TranslationAdjudicator
from src.db.db_manager import db
from src.config import OLLAMA_MODEL_NAME, SUBTITLES_DIR
from src.utils.llm_factory import make_chat_ollama

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate

def extract_tech_terms(srt_en_path: str, model_name: str, context: dict = None) -> list[str]:
    """
    Đọc tối đa 20 câu thoại đầu tiên từ kịch bản phụ đề tiếng Anh en.srt
    và dùng Ollama để trích xuất các thuật ngữ chuyên ngành.
    """
    logger.info("Đang tự động trích xuất thuật ngữ chuyên ngành đa ngành từ kịch bản phụ đề...")
    entries = parse_srt(srt_en_path)
    if not entries:
        return []
        
    sample_text = " ".join([e.text for e in entries[:20]])
    
    # Sinh ngẫu nhiên ví dụ từ domain cache
    domain_name = context.get("topic", "General") if context else "General"
    cache = db.load_phonetic_cache(domain_name)
    example_str = (
        "- Công nghệ: 'api', 'workflow', 'kernel'\n"
        "- Y học: 'mri', 'antibody', 'hemoglobin'\n"
        "- Tài chính: 'etf', 'portfolio', 'hedge fund'\n"
        "- Pháp lý: 'plaintiff', 'subpoena'\n"
        "- Marketing: 'ctr', 'roi', 'funnel'\n"
    )
    if cache:
        import random
        sample_keys = random.sample(list(cache.keys()), min(15, len(cache)))
        example_str = f"- Lĩnh vực {domain_name}: " + ", ".join([f"'{k}'" for k in sample_keys]) + "\n"
    
    llm = make_chat_ollama(model_name=model_name, temperature=0.1)
    
    class KeywordExtraction(BaseModel):
        keywords: List[str] = Field(description="Danh sách ngắn gọn (tối đa 5-10 từ) các thuật ngữ, danh từ chuyên ngành cốt lõi cần giữ nguyên.")
        
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"Bạn là một chuyên gia ngôn ngữ học ĐA NGÀNH (công nghệ, y tế, tài chính, giáo dục, pháp lý, "
            f"khoa học, marketing, ẩm thực, thể thao, nghệ thuật, lịch sử...). Nhiệm vụ: phân tích kịch bản "
            f"video và trích xuất các thuật ngữ chuyên ngành, từ viết tắt, danh từ riêng/sản phẩm mà khi dịch "
            f"sang tiếng Việt NÊN GIỮ NGUYÊN tiếng Anh.\n"
            f"QUY TẮC:\n"
            f"1. CHỈ lấy danh từ chuyên môn cốt lõi, từ viết tắt, hoặc tên riêng đặc thù ngành.\n"
            f"2. TUYỆT ĐỐI KHÔNG lấy động từ, tính từ, từ giao tiếp thông thường, hoặc thành ngữ.\n"
            f"3. Trích xuất càng ít càng tốt, tối đa 5-10 từ quan trọng nhất.\n"
            f"4. Không bịa thêm thuật ngữ không xuất hiện trong văn bản.\n"
            f"Ví dụ thuật ngữ đặc thù theo ngành:\n"
            f"{example_str}"
        )),
        ("human", "Hãy trích xuất thuật ngữ chuyên ngành TỐI GIẢN NHẤT từ đoạn kịch bản sau:\n{text}")
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

TRANSLATION_RULES = """1. STRICT 1:1 MAPPING: Translate EXACTLY what is in the line.
2. DO NOT COMPLETE SENTENCES: If a line is cut off abruptly (e.g. "into a polished"), translate ONLY up to that word and STOP. NEVER guess or add the next word. NEVER finish the sentence.
3. KEEP DANGLING WORDS: If a line ends with "Number two,", KEEP IT. NEVER drop trailing words.
4. TECH TERMS: Keep {tech_terms_str} in English.
5. NO LEAKAGE: Use [PREVIOUS CONTEXT] to understand, but DO NOT output it.
6. VIETNAMESE ONLY."""

BATCH_TRANSLATE_SYSTEM_PROMPT = """Translate subtitles to Vietnamese. Output ONLY in this exact format:
[index] translated text
Do NOT add any other text or explanations.

RULES:
{translation_rules}
{local_rag_rules}
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



def has_leakage(text: str) -> bool:
    """Kiểm tra xem văn bản có bị rò rỉ ký tự tiếng Trung/Cyrillic không."""
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    if re.search(r'[\u0400-\u04FF]', text):
        return True
    return False

def translate_batch(group_entries: list, context_text: str, llm, tech_terms_str: str, local_rag_rules: str = "") -> dict:
    """Dịch một lô các câu thoại (line-by-line) sử dụng LangChain."""
    rules_text = TRANSLATION_RULES.replace("{tech_terms_str}", tech_terms_str)
    system_prompt = BATCH_TRANSLATE_SYSTEM_PROMPT.replace("{translation_rules}", rules_text).replace("{local_rag_rules}", local_rag_rules)
    
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """[CONTEXT (DO NOT TRANSLATE):]
{context_text}
 
[LINES TO TRANSLATE (Output ONLY in [index] format):]  
{target_text}""")
    ])
    
    # Use plain LLM instead of Structured Output - just get text
    translation_chain = translate_prompt | llm
    
    target_text = "\n".join([f"[{e.index}] {e.text}" for e in group_entries])
    
    best_result_map = {}
    
    for attempt in range(2):
        try:
            res = translation_chain.invoke({
                "context_text": context_text if context_text else "[Bắt đầu video]",
                "target_text": target_text
            })
            
            # Parse text using Regex: [index] translated text
            result_map = {}
            response_text = res.content if hasattr(res, 'content') else str(res)
            for match in re.finditer(r'\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|$)', response_text, re.DOTALL):
                idx = int(match.group(1))
                text = match.group(2).strip()
                if text and not has_leakage(text):
                    result_map[idx] = normalize_numbers_to_text(text)
                
            # Keep the result with the most translated lines
            if len(result_map) > len(best_result_map):
                best_result_map = result_map
            
            # Check if all indices are translated
            if len(result_map) == len(group_entries):
                # VALIDATION LỖI SÓT TỪ / GỘP CÂU (GENERIC MULTI-DOMAIN)
                # Nếu độ dài chuỗi tiếng Việt quá ngắn so với tiếng Anh (dưới 80%), khả năng cao LLM đã tự ý cắt bỏ từ ngữ lửng lơ hoặc câu phụ.
                for e in group_entries:
                    en_len = len(e.text.strip())
                    vi_len = len(result_map[e.index].strip())
                    if en_len > 15 and vi_len < en_len * 0.80:
                        logger.warning(f"Phát hiện lỗi rớt từ ở câu {e.index} (Tỷ lệ độ dài VI/EN: {vi_len}/{en_len}). Hủy lô để chạy Fallback...")
                        raise ValueError("Dropped trailing word in batch")
                        
                return result_map
            else:
                logger.warning(f"Thiếu câu trong bản dịch (được {len(result_map)}/{len(group_entries)}). Thử lại...")
        except Exception as e:
            logger.error(f"Lỗi LangChain khi dịch batch (lần thử {attempt+1}): {e}")
            
    # FALLBACK CẤP 2: Dịch Từng Câu (Single-Line Fallback)
    logger.warning("Mẻ dịch bị lỗi hoặc gộp câu. Kích hoạt chế độ Fallback Dịch Từng Câu...")
    fallback_map = {}
    current_context = context_text if context_text else "[Bắt đầu video]"
    
    for e in group_entries:
        if e.index in best_result_map:
            fallback_map[e.index] = best_result_map[e.index]
            current_context += f"\n[{e.index}] {e.text} -> {best_result_map[e.index]}"
            continue
            
        logger.info(f"Đang chạy Fallback cho Index {e.index}...")
        single_target = f"[{e.index}] {e.text}"
        
        fallback_text = "-"
        for attempt in range(3):
            try:
                single_res = translation_chain.invoke({
                    "context_text": current_context,
                    "target_text": single_target
                })
                
                # Parse single response using regex
                response_text = single_res.content if hasattr(single_res, 'content') else str(single_res)
                match = re.search(r'\[(\d+)\]\s*(.*?)(?=\n|$)', response_text)
                if match:
                    text = match.group(2).strip()
                    
                    en_len = len(e.text.strip())
                    vi_len = len(text.strip())
                    
                    if en_len > 15 and vi_len < en_len * 0.80:
                        logger.warning(f"Fallback rớt từ (VI/EN: {vi_len}/{en_len}). Thử lại (lần {attempt+1}/3)...")
                        # Nếu đã thử 3 lần mà vẫn rớt từ, đành lấy nguyên bản tiếng Anh để tránh mất dữ liệu (ví dụ: 'Number two,')
                        if attempt == 2:
                            fallback_text = text + " " + e.text.split()[-1]
                        continue
                        
                    if not has_leakage(text):
                        fallback_text = normalize_numbers_to_text(text)
                        break
            except Exception as ex:
                logger.error(f"Lỗi Fallback câu {e.index}: {ex}")
                
        fallback_map[e.index] = fallback_text
            
        if fallback_map[e.index] != "-":
            current_context += f"\n[{e.index}] {e.text} -> {fallback_map[e.index]}"
            
    return fallback_map

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

def run(srt_en_path: str, context: dict, model_name: str = OLLAMA_MODEL_NAME, limit: int = None, progress_callback=None) -> str:
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
    extracted_terms = extract_tech_terms(srt_en_path, model_name, context)
    context_keywords = context.get("keywords", []) if isinstance(context, dict) else []
    all_terms = set(extracted_terms) | set(context_keywords)
    tech_terms_str = ", ".join([f"'{term}'" for term in sorted(all_terms)])
    logger.info(f"Tổng hợp các thuật ngữ chuyên ngành cần giữ nguyên tiếng Anh: {tech_terms_str}")
    
    # [Targeted RAG] Lấy từ điển của Domain hiện tại (nếu có)
    domain_name = context.get("topic", "General")
    domain_dict = db.load_dictionary(domain_name)
    logger.info(f"Đã nạp Domain Dictionary '{domain_name}' với {len(domain_dict)} từ khóa.")
    
    # Initialize Translation Cache & Word-Level Translator
    translation_cache = TranslationCache()
    word_translator = WordLevelTranslator(model_name=model_name, cache=translation_cache)
    cache_stats = translation_cache.stats()
    logger.info(f"Translation Cache initialized: {cache_stats['total']} entries "
                f"({cache_stats['tech_terms']} tech, {cache_stats['words']} words)")
    
    # Chia các câu thoại thành các đoạn (paragraphs), kích thước 3
    # Sliding Window Context: 2 câu context + 3 câu dịch
    group_size = 3
    context_window_size = 2  # Chỉ lấy 2 câu trước làm context
    paragraphs = []
    for i in range(0, len(entries), group_size):
        group = entries[i : i + group_size]
        
        # Ngữ cảnh gối đầu: Lấy 2 câu trước đó
        context_list = []
        if i >= context_window_size:
            prev_start = max(0, i - context_window_size)
            context_list = [e.text for e in entries[prev_start:i]]
            
        context_text = " ".join(context_list)
        full_text = " ".join([e.text for e in group])
        paragraphs.append((group, full_text, context_text))
        
    results_map = {}
    all_english_terms = set()
    
    llm = make_chat_ollama(model_name=model_name, temperature=0.1)
    
    def process_batch(group, full_text, context_text):
        """Hàm xử lý một batch line-by-line: dịch -> trích xuất từ tiếng Anh."""
        # --- SPATIALLY-AWARE RAG ---
        local_rules = []
        full_text_lower = full_text.lower()
        for eng_term, details in domain_dict.items():
            if re.search(r'\b' + re.escape(eng_term.lower()) + r'\b', full_text_lower):
                # Ưu tiên phần giải thích (schema mới); tương thích ngược 'vi' (schema cũ)
                vi_meaning = details.get("explanation") or details.get("vi") or eng_term
                local_rules.append(f" - Giữ nguyên/giải nghĩa '{eng_term}': {vi_meaning}.")
                
        local_rag_str = ""
        if local_rules:
            local_rag_str = "Bắt buộc tuân thủ từ vựng sau:\n" + "\n".join(local_rules) + "\n\n"
            logger.info(f"🔥 Chunk-Level RAG Inject: {local_rules}")
            
        translated_map = translate_batch(group, context_text, llm, tech_terms_str, local_rag_str)
        
        batch_results = []
        for entry in group:
            vietnamese_text = translated_map.get(entry.index, entry.text)
            
            word_positions = extract_and_align_entry(entry, vietnamese_text, all_terms)
            logger.info(f"Đã dịch câu {entry.index} ({entries.index(entry)+1}/{total}): '{vietnamese_text}'")
            
            batch_results.append((entry.index, vietnamese_text, word_positions))
        return batch_results

    # Chạy tuần tự các paragraphs với 1 worker để tránh deadlock kết nối Ollama
    logger.info(f"Bắt đầu dịch {len(paragraphs)} paragraphs bằng mô hình {model_name}...")
    total_paras = len(paragraphs)
    processed_paras = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_batch, p[0], p[1], p[2]): i
            for i, p in enumerate(paragraphs)
        }
        for future in concurrent.futures.as_completed(futures):
            para_idx = futures[future]
            processed_paras += 1
            if progress_callback:
                progress_callback(processed_paras, total_paras)
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

    # 1.5 TRỌNG TÀI HẬU DỊCH (LLM Adjudicator) — ĐA NGÀNH
    # Rà soát các từ tiếng Anh còn sót KHÔNG thuộc thuật ngữ ngành đã biết.
    # LLM quyết định: dịch bổ sung (từ phổ thông bị bỏ quên) hoặc giữ nguyên (thuật ngữ mới).
    ai_discovered_terms = set()  # Thuật ngữ mới do AI xác nhận giữ nguyên -> sẽ ghi vào từ điển ngành
    try:
        logger.info("=== TRỌNG TÀI HẬU DỊCH: Rà soát từ tiếng Anh còn sót ===")
        adjudicator = TranslationAdjudicator(
            domain_terms=all_terms,
            model_name=model_name,
            domain=domain_name,
            cache=translation_cache,
        )
        entries_by_index = {e.index: e.text for e in entries}
        adj_result = adjudicator.process_results(results_map, entries_by_index)

        # Cập nhật văn bản đã dịch (các từ phổ thông được dịch bổ sung)
        for idx, new_text in adj_result["updated_texts"].items():
            _, old_positions = results_map[idx]
            results_map[idx] = (new_text, old_positions)

        # Bổ sung các thuật ngữ mới (LLM xác nhận giữ nguyên) vào tập term
        new_terms = adj_result["new_terms"]
        if new_terms:
            all_terms |= new_terms
            ai_discovered_terms |= new_terms
            logger.info(f"[Adjudicator] Thuật ngữ mới được xác nhận giữ nguyên: {sorted(new_terms)}")

        # Tính lại vị trí từ tiếng Anh & danh sách phiên âm dựa trên văn bản đã rà soát
        all_english_terms = set()
        for entry in entries:
            vi_text, _ = results_map.get(entry.index, (entry.text, []))
            word_positions = extract_and_align_entry(entry, vi_text, all_terms)
            results_map[entry.index] = (vi_text, word_positions)
            for w_info in word_positions:
                clean_term = w_info["word"].strip().lower()
                if clean_term and re.match(r'^[a-z0-9\s\-]+$', clean_term) and len(clean_term) >= 3:
                    all_english_terms.add(clean_term)
    except Exception as e:
        logger.error(f"Lỗi khi chạy Trọng tài hậu dịch (bỏ qua, dùng kết quả gốc): {e}")

    # 2. Sinh phiên âm hàng loạt cho danh sách từ tiếng Anh bằng G2P + MOP
    english_list = list(all_english_terms)
    logger.info(f"Đang tạo phiên âm (G2P) cho {len(english_list)} thuật ngữ tiếng Anh: {english_list}")
    transliterations_map = transliterate_batch(english_list, domain=domain_name)
    logger.info(f"Đã nhận được bản đồ phiên âm (G2P): {transliterations_map}")
    
    # Chuẩn hóa khóa của bản đồ về chữ thường
    transliterations_map = {k.lower().strip(): v.strip() for k, v in transliterations_map.items()}

    # 2.5 GHI NHẬN THUẬT NGỮ AI PHÁT HIỆN vào Từ điển ngành (đa ngành)
    # Mỗi thuật ngữ mới do Adjudicator xác nhận giữ nguyên sẽ được lưu lại kèm metadata
    # (người bổ sung = AI, ngày bổ sung, phiên âm bồi) để người dùng xem/sửa sau này.
    if ai_discovered_terms and domain_name and domain_name.lower() != "auto detect":
        for term in sorted(ai_discovered_terms):
            try:
                phonetic_val = transliterations_map.get(term.lower().strip(), "")
                db.upsert_term(domain_name, term, phonetic=phonetic_val, explanation="", added_by="AI")
            except Exception as e:
                logger.error(f"Không thể ghi thuật ngữ AI '{term}' vào từ điển ngành: {e}")
        logger.info(f"Đã ghi {len(ai_discovered_terms)} thuật ngữ AI phát hiện vào từ điển '{domain_name}'.")
    
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
    
    # Save translation cache to disk
    word_translator.save_cache()
    final_stats = translation_cache.stats()
    logger.info(f"Translation Cache saved: {final_stats['total']} entries "
                f"({final_stats['tech_terms']} tech, {final_stats['words']} words)")
    
    # Dọn dẹp
    clean_memory()
    return srt_vi_path
