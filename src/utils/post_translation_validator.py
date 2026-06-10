# -*- coding: utf-8 -*-
"""
PostTranslationValidator — Rà soát hậu dịch (ĐA NGÀNH)
======================================================
Phát hiện các từ gốc tiếng Anh còn sót lại trong câu tiếng Việt sau khi dịch,
và phân loại chúng dựa trên danh sách thuật ngữ ngành (domain_terms).

Thiết kế KHÔNG phụ thuộc vào bất kỳ lĩnh vực cụ thể nào:
- domain_terms có thể đến từ BẤT KỲ ngành nào (y tế, tài chính, pháp lý,
  công nghệ, ẩm thực, thể thao...). Nó chỉ là tập "thuật ngữ cần giữ nguyên".
- Tín hiệu để xác định một token có phải "tiếng Anh còn sót" hay không KHÔNG
  dựa vào danh sách từ tiếng Việt hardcode (luôn thiếu sót và thiên ngôn ngữ),
  mà dựa vào việc token đó CÓ XUẤT HIỆN trong câu gốc tiếng Anh hay không.
  Đây là cách tổng quát và chính xác cho mọi ngành.
"""
import re
from typing import Dict, List, Set
from src.utils.logger import logger


class PostTranslationValidator:
    def __init__(self, domain_terms: Set[str] = None):
        """
        domain_terms: Tập thuật ngữ ngành cần giữ nguyên (đến từ bất kỳ lĩnh vực nào).
        """
        self.domain_terms = set(t.lower().strip() for t in (domain_terms or []) if t.strip())

    # ------------------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> List[re.Match]:
        """Lấy các token chữ Latin (có thể có gạch nối) trong văn bản."""
        return list(re.finditer(r'[A-Za-z][A-Za-z\-]*', text))

    def extract_english_words(self, text: str) -> List[Dict]:
        """
        Trích xuất tất cả token chữ Latin trong văn bản.
        Return: [{"word":..., "clean":..., "position":..., "in_context":...}]
        """
        matches = []
        for m in self._tokens(text):
            word = m.group()
            clean = word.lower().strip('-')
            if not clean:
                continue
            matches.append({
                "word": word,
                "clean": clean,
                "position": m.start(),
                "in_context": text[max(0, m.start() - 12):m.end() + 12],
            })
        return matches

    # ------------------------------------------------------------------
    def is_domain_term(self, word: str) -> bool:
        """
        Một từ thuộc thuật ngữ ngành nếu:
        - Trùng khớp một thuật ngữ trong domain_terms, hoặc
        - Là một thành phần của thuật ngữ ghép (vd 'machine' trong 'machine learning').
        Hoàn toàn dựa trên domain_terms truyền vào -> đúng cho MỌI ngành.
        """
        clean = word.lower().strip('-')
        if clean in self.domain_terms:
            return True
        for term in self.domain_terms:
            if clean in term.split():
                return True
        return False

    def classify_word(self, word: str) -> str:
        """'term' (thuật ngữ giữ nguyên) hoặc 'candidate' (ứng viên cần phán xử)."""
        return "term" if self.is_domain_term(word) else "candidate"

    # ------------------------------------------------------------------
    def find_foreign_words(self, vi_text: str, en_text: str = "") -> Dict[str, List[Dict]]:
        """
        Tìm các từ tiếng Anh còn sót lại trong câu tiếng Việt.

        Quy tắc tổng quát (đa ngành):
        - Chỉ tính là "tiếng Anh còn sót" nếu token đó CÓ trong câu gốc tiếng Anh
          (en_text). Điều này loại bỏ tự động các từ tiếng Việt không dấu trùng
          dạng Latin (vd 'vi', 'kinh', 'doanh') mà không cần danh sách hardcode.
        - Nếu không có en_text, fallback: nhận mọi token Latin (kém chính xác hơn).

        Return: {
            "terms": [...],        # đã biết là thuật ngữ ngành -> giữ nguyên
            "candidates": [...],   # cần LLM phán xử (dịch hay giữ nguyên)
            "total_english": int
        }
        """
        en_tokens = set(m.group().lower() for m in self._tokens(en_text)) if en_text else None

        english_words = self.extract_english_words(vi_text)
        terms, candidates = [], []

        for info in english_words:
            clean = info["clean"]
            # Lọc theo nguồn: phải xuất hiện trong câu gốc tiếng Anh
            if en_tokens is not None and clean not in en_tokens:
                continue
            if self.is_domain_term(clean):
                terms.append(info)
            else:
                candidates.append(info)

        return {
            "terms": terms,
            "candidates": candidates,
            "total_english": len(terms) + len(candidates),
        }

    # ------------------------------------------------------------------
    def validate_translation(self, vi_text: str, en_text: str = "",
                             domain_terms: Set[str] = None) -> Dict:
        """Rà soát một câu dịch và trả về tóm tắt vấn đề."""
        if domain_terms:
            self.domain_terms = set(t.lower().strip() for t in domain_terms if t.strip())

        issues = self.find_foreign_words(vi_text, en_text)
        has_issues = len(issues["candidates"]) > 0

        if issues["candidates"]:
            logger.warning(
                f"Phát hiện {len(issues['candidates'])} từ tiếng Anh nghi chưa dịch: "
                f"{[c['clean'] for c in issues['candidates']]}"
            )

        return {
            "is_valid": not has_issues,
            "english_words_found": issues["total_english"],
            "candidates": issues["candidates"],
            "terms_in_sentence": [t["clean"] for t in issues["terms"]],
            "details": issues,
        }
