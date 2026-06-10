# -*- coding: utf-8 -*-
"""
TranslationAdjudicator (Trọng tài Hậu Dịch) — ĐA NGÀNH
=======================================================
Sau khi dịch xong, một số từ tiếng Anh vẫn còn sót lại trong câu tiếng Việt.
Có 2 trường hợp:
  1. Từ thường (phổ thông) bị LLM bỏ quên chưa dịch          -> CẦN DỊCH.
  2. Thuật ngữ ngành / danh từ riêng mà mô hình suy luận ra
     (chưa nằm trong danh sách thuật ngữ ngành đã biết)      -> GIỮ NGUYÊN tiếng Anh.

Module này dùng LLM để "phán xử" (adjudicate) từng từ tiếng Anh còn sót mà
KHÔNG thuộc danh sách thuật ngữ ngành đã biết, để quyết định nên dịch hay giữ nguyên.

Thiết kế hoàn toàn ĐA NGÀNH (multi-domain): không hardcode bất kỳ lĩnh vực nào.
LLM tự suy ra ngành từ ngữ cảnh câu. Có tích hợp cache để tránh hỏi LLM lặp lại.
"""
import re
from typing import Dict, List, Set
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from src.utils.logger import logger
from src.config import OLLAMA_MODEL_NAME
from src.utils.llm_factory import make_chat_ollama
from src.utils.post_translation_validator import PostTranslationValidator
from src.utils.translation_cache import TranslationCache


class WordVerdict(BaseModel):
    """Phán quyết cho một từ tiếng Anh còn sót."""
    word: str = Field(description="Từ tiếng Anh đang được xét.")
    decision: str = Field(description="'translate' nếu là từ phổ thông cần dịch; 'keep' nếu là thuật ngữ/danh từ riêng nên giữ nguyên tiếng Anh.")
    vietnamese: str = Field(default="", description="Bản dịch tiếng Việt (1-4 từ). Chỉ điền khi decision='translate'.")


class SentenceVerdict(BaseModel):
    """Tập phán quyết cho tất cả từ tiếng Anh còn sót trong một câu."""
    verdicts: List[WordVerdict] = Field(description="Danh sách phán quyết cho từng từ.")


class TranslationAdjudicator:
    """Trọng tài hậu dịch dựa trên LLM, hoạt động cho mọi lĩnh vực."""

    def __init__(self,
                 domain_terms: Set[str] = None,
                 model_name: str = OLLAMA_MODEL_NAME,
                 domain: str = "General",
                 cache: TranslationCache = None):
        self.domain_terms = set(t.lower().strip() for t in (domain_terms or []) if t.strip())
        self.model_name = model_name
        self.domain = domain
        self.cache = cache or TranslationCache()
        self.validator = PostTranslationValidator(self.domain_terms)

        # Cấu hình an toàn VRAM (num_ctx nhỏ + tắt thinking). Xem src/utils/llm_factory.py
        self.llm = make_chat_ollama(model_name=model_name, temperature=0.1)

    # ------------------------------------------------------------------
    # Phát hiện ứng viên (candidate) cần phán xử trong một câu
    # ------------------------------------------------------------------
    def _adjacency_protected(self, vi_text: str, en_text: str) -> Set[str]:
        """
        Tìm các từ tiếng Anh thuộc một CỤM tiếng Anh liền kề có chứa thuật ngữ ngành.

        Quy tắc đa ngành: các từ tiếng Anh đứng liền nhau (chỉ cách nhau bởi dấu/space)
        tạo thành một cụm; nếu trong cụm đó có ÍT NHẤT một thuật ngữ ngành đã biết,
        thì những từ còn lại trong cụm là bổ nghĩa cho thuật ngữ -> GIỮ NGUYÊN cả cụm.

        Ví dụ: 'general purpose task agents' (agents là term) -> giữ nguyên cả 4 từ.
        Ngược lại 'Number two' (không có term) -> KHÔNG được bảo vệ -> để LLM phán xử.
        """
        en_tokens = set(m.group().lower() for m in self.validator._tokens(en_text)) if en_text else set()

        protected: Set[str] = set()

        # Quét theo token thật để xác định cụm liền kề (chỉ cách nhau bởi dấu/space,
        # không có từ tiếng Việt xen giữa).
        tokens = list(re.finditer(r'[A-Za-z][A-Za-z\-]*', vi_text))
        run: List[str] = []
        run_has_term = False

        def flush():
            nonlocal run, run_has_term
            if len(run) >= 2 and run_has_term:
                protected.update(run)
            run = []
            run_has_term = False

        prev_end = None
        for m in tokens:
            clean = m.group().lower().strip('-')
            # Token này có phải tiếng Anh không (xuất hiện trong câu gốc hoặc là thuật ngữ ngành)
            is_english = (clean in en_tokens) or self.validator.is_domain_term(clean)
            # Cụm chỉ được nối khi giữa 2 token CHỈ có khoảng trắng/gạch nối.
            # Dấu phẩy, chấm, hai chấm... là ranh giới ngữ nghĩa -> NGẮT cụm.
            gap_text = vi_text[prev_end:m.start()] if prev_end is not None else None
            gap_connects = gap_text is not None and re.fullmatch(r'[\s\-]+', gap_text) is not None

            if is_english and gap_connects:
                run.append(clean)
                if self.validator.is_domain_term(clean):
                    run_has_term = True
            else:
                flush()
                if is_english:
                    run = [clean]
                    run_has_term = self.validator.is_domain_term(clean)
            prev_end = m.end()
        flush()
        return protected

    def _find_candidates(self, vi_text: str, en_text: str = "") -> List[Dict]:
        """
        Tìm các từ tiếng Anh CHƯA thuộc thuật ngữ ngành còn sót trong câu tiếng Việt.
        - Dựa vào câu gốc tiếng Anh để loại bỏ tự động các từ tiếng Việt không dấu.
        - Loại các từ được "bảo vệ theo cụm liền kề" (đi với thuật ngữ ngành) -> giữ nguyên.
        """
        issues = self.validator.find_foreign_words(vi_text, en_text)
        protected = self._adjacency_protected(vi_text, en_text)

        candidates = []
        seen = set()
        for w in issues["candidates"]:
            clean = w["clean"]
            if clean in seen or len(clean) < 2:
                continue
            if clean in protected:
                # Thuộc cụm thuật ngữ nhiều từ -> giữ nguyên, không cần hỏi LLM
                continue
            seen.add(clean)
            candidates.append(w)
        return candidates

    # ------------------------------------------------------------------
    # Phán xử một câu bằng LLM (có dùng cache)
    # ------------------------------------------------------------------
    def adjudicate_sentence(self, vi_text: str, en_text: str = "") -> Dict[str, Dict]:
        """
        Phán xử các từ tiếng Anh còn sót trong một câu tiếng Việt.
        Return: { "<word_clean>": {"decision": "translate"|"keep", "vietnamese": str} }
        """
        candidates = self._find_candidates(vi_text, en_text)
        if not candidates:
            return {}

        verdicts: Dict[str, Dict] = {}
        words_to_ask: List[str] = []

        # 1) Ưu tiên tra cache để giảm số lần gọi LLM
        for c in candidates:
            clean = c["clean"]
            cached_term = self.cache.get(clean, word_type="tech")
            cached_word = self.cache.get(clean, word_type="word")
            if cached_term is not None:
                verdicts[clean] = {"decision": "keep", "vietnamese": ""}
            elif cached_word:
                verdicts[clean] = {"decision": "translate", "vietnamese": cached_word}
            else:
                words_to_ask.append(clean)

        # 2) Hỏi LLM cho các từ chưa có trong cache
        if words_to_ask:
            llm_verdicts = self._ask_llm(vi_text, en_text, words_to_ask)
            for clean, v in llm_verdicts.items():
                verdicts[clean] = v
                if v["decision"] == "keep":
                    self.cache.set(clean, clean, word_type="tech")
                elif v["decision"] == "translate" and v.get("vietnamese"):
                    self.cache.set(clean, v["vietnamese"], word_type="word")

        return verdicts

    def _ask_llm(self, vi_text: str, en_text: str, words: List[str]) -> Dict[str, Dict]:
        """Gọi LLM để phán xử danh sách từ trong ngữ cảnh câu (đa ngành)."""
        word_list_str = ", ".join([f"'{w}'" for w in words])

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Bạn là chuyên gia ngôn ngữ học song ngữ Anh-Việt, làm việc với nội dung ĐA NGÀNH "
                "(mọi lĩnh vực: y tế, tài chính, pháp lý, giáo dục, khoa học, công nghệ, marketing, "
                "ẩm thực, thể thao, nghệ thuật, lịch sử...). Bạn đang rà soát một câu dịch phụ đề.\n"
                "Câu tiếng Việt đã được dịch nhưng vẫn còn sót lại vài TỪ TIẾNG ANH. "
                "Với MỖI từ tiếng Anh được liệt kê, hãy quyết định:\n"
                "- 'translate': nếu đó là TỪ PHỔ THÔNG (động từ, tính từ, danh từ thông thường) "
                "bị bỏ quên chưa dịch. Cung cấp bản dịch tiếng Việt ngắn gọn (1-4 từ) HỢP NGỮ CẢNH.\n"
                "- 'keep': nếu đó là THUẬT NGỮ CHUYÊN NGÀNH, TÊN SẢN PHẨM/THƯƠNG HIỆU, TÊN RIÊNG, "
                "TỪ VIẾT TẮT, hoặc là một phần của cụm thuật ngữ tiếng Anh nhiều từ — những thứ mà "
                "người Việt thường giữ nguyên tiếng Anh.\n"
                "QUY TẮC:\n"
                "1. Chỉ xét đúng các từ được liệt kê, không thêm từ khác.\n"
                "2. Khi decision='keep', để trống trường vietnamese.\n"
                "3. Dịch phải tự nhiên, đúng ngữ cảnh và đúng ngành (tự suy ra ngành từ câu).\n"
                "4. Nếu một từ đứng cạnh các từ tiếng Anh khác tạo thành cụm thuật ngữ, hãy 'keep'.\n"
                "5. Khi không chắc chắn, ưu tiên 'keep' để tránh dịch sai thuật ngữ chuyên môn."
            )),
            ("human", (
                "[CÂU GỐC TIẾNG ANH]:\n{en_text}\n\n"
                "[CÂU TIẾNG VIỆT ĐÃ DỊCH]:\n{vi_text}\n\n"
                "[CÁC TỪ TIẾNG ANH CẦN PHÁN XỬ]: {word_list}\n\n"
                "Hãy trả về phán quyết cho từng từ."
            ))
        ])

        chain = prompt | self.llm.with_structured_output(SentenceVerdict)
        result: Dict[str, Dict] = {}
        try:
            res = chain.invoke({
                "en_text": en_text if en_text else "(không có)",
                "vi_text": vi_text,
                "word_list": word_list_str,
            })
            valid_words = set(words)
            for v in res.verdicts:
                clean = v.word.lower().strip()
                if clean not in valid_words:
                    continue
                decision = "keep" if v.decision.lower().strip().startswith("keep") else "translate"
                vietnamese = re.sub(r'["\']', '', v.vietnamese or "").strip()
                if decision == "translate" and not vietnamese:
                    decision = "keep"  # bảo toàn: không có bản dịch thì giữ nguyên
                result[clean] = {"decision": decision, "vietnamese": vietnamese}
        except Exception as e:
            logger.error(f"Lỗi khi phán xử từ bằng LLM: {e}")
            # Fallback an toàn: giữ nguyên tất cả (coi như thuật ngữ)
            for w in words:
                result[w] = {"decision": "keep", "vietnamese": ""}

        return result

    # ------------------------------------------------------------------
    # Áp dụng phán quyết: thay từ đã dịch vào câu
    # ------------------------------------------------------------------
    @staticmethod
    def apply_verdicts(vi_text: str, verdicts: Dict[str, Dict]) -> str:
        """
        Thay thế các từ được phán 'translate' bằng bản dịch tiếng Việt trong câu.
        Các từ 'keep' giữ nguyên. So khớp không phân biệt hoa thường, giữ dấu câu.
        """
        if not verdicts:
            return vi_text

        def repl(match):
            token = match.group(0)
            v = verdicts.get(token.lower())
            if v and v["decision"] == "translate" and v.get("vietnamese"):
                return v["vietnamese"]
            return token

        return re.sub(r'[A-Za-z][A-Za-z\-]*', repl, vi_text)

    def process_results(self,
                        results_map: Dict[int, tuple],
                        entries_by_index: Dict[int, str]) -> Dict:
        """
        Xử lý toàn bộ kết quả dịch.

        Args:
            results_map: {index: (vietnamese_text, word_positions)}
            entries_by_index: {index: english_text}

        Return: {
            "updated_texts": {index: new_vietnamese_text},
            "new_terms": set(...),           # thuật ngữ mới được xác nhận giữ nguyên
            "translated_words": {word: vi},
            "stats": {...}
        }
        """
        updated_texts: Dict[int, str] = {}
        new_terms: Set[str] = set()
        translated_words: Dict[str, str] = {}
        num_translated = 0
        num_kept = 0
        num_sentences_touched = 0

        for index in sorted(results_map.keys()):
            vi_text = results_map[index][0]
            en_text = entries_by_index.get(index, "")

            verdicts = self.adjudicate_sentence(vi_text, en_text)
            if not verdicts:
                continue

            num_sentences_touched += 1
            for word, v in verdicts.items():
                if v["decision"] == "keep":
                    new_terms.add(word)
                    num_kept += 1
                else:
                    translated_words[word] = v["vietnamese"]
                    num_translated += 1

            new_text = self.apply_verdicts(vi_text, verdicts)
            if new_text != vi_text:
                updated_texts[index] = new_text
                logger.info(f"[Adjudicator] Câu {index}: '{vi_text}' -> '{new_text}'")

        stats = {
            "sentences_touched": num_sentences_touched,
            "translated": num_translated,
            "kept_as_term": num_kept,
        }
        logger.info(
            f"[Adjudicator] Hoàn tất: {num_sentences_touched} câu được rà soát, "
            f"{num_translated} từ được dịch bổ sung, {num_kept} thuật ngữ mới được giữ nguyên."
        )
        return {
            "updated_texts": updated_texts,
            "new_terms": new_terms,
            "translated_words": translated_words,
            "stats": stats,
        }
