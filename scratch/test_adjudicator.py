# -*- coding: utf-8 -*-
"""
Test TranslationAdjudicator (đa ngành) với dữ liệu thật từ subtitles_vi.json.
Kiểm tra: từ phổ thông -> dịch, thuật ngữ/tên riêng -> giữ nguyên,
và từ tiếng Việt không dấu (vi, kinh, doanh) KHÔNG bị nhận nhầm.
"""
import sys, json
sys.path.insert(0, r'd:\Project\VidTranscribe.ai')

from src.utils.translation_adjudicator import TranslationAdjudicator
from src.utils.post_translation_validator import PostTranslationValidator
from src.utils.srt_utils import parse_srt

# ---- Unit test Lớp 2 (không cần LLM): lọc theo câu gốc tiếng Anh ----
print("=" * 80)
print("UNIT TEST — Lớp 2: PostTranslationValidator.find_foreign_words")
print("=" * 80)
v = PostTranslationValidator(domain_terms={"workflow", "agents"})
vi = "Chúng kết nối liền mạch với các công cụ kinh doanh. Number two, workflow agents."
en = "These connect to your business tools seamlessly. Number two, workflow agents."
res = v.find_foreign_words(vi, en)
print("candidates:", [c["clean"] for c in res["candidates"]])
print("terms     :", [t["clean"] for t in res["terms"]])
assert "kinh" not in [c["clean"] for c in res["candidates"]], "FAIL: 'kinh' bị nhận nhầm!"
assert "doanh" not in [c["clean"] for c in res["candidates"]], "FAIL: 'doanh' bị nhận nhầm!"
assert "workflow" in [t["clean"] for t in res["terms"]], "FAIL: 'workflow' phải là term"
print("PASS: từ tiếng Việt không dấu bị loại, thuật ngữ ngành phân loại đúng.\n")

# ---- Unit test adjacency: cụm thuật ngữ nhiều từ được bảo vệ ----
print("=" * 80)
print("UNIT TEST — _adjacency_protected: cụm thuật ngữ nhiều từ giữ nguyên")
print("=" * 80)
adj_probe = TranslationAdjudicator(domain_terms={"agents"}, model_name="gemma4:e4b")
vi2 = "Loại một, general purpose task agents. Điều này giống như có"
en2 = "Type one, general purpose task agents. This is like having"
cand = [c["clean"] for c in adj_probe._find_candidates(vi2, en2)]
print("candidates còn lại (sẽ hỏi LLM):", cand)
assert "general" not in cand and "purpose" not in cand and "task" not in cand, \
    f"FAIL: cụm 'general purpose task' phải được giữ nguyên, nhưng còn: {cand}"
print("PASS: cụm 'general purpose task agents' được bảo vệ (không bị dịch lẻ).\n")

# Dấu phẩy phải NGẮT cụm: 'Number two, workflow agents' -> number/two vẫn là candidate
adj_probe2 = TranslationAdjudicator(domain_terms={"workflow", "agents"}, model_name="gemma4:e4b")
vi3 = "không cần bạn quản lý vi mô nó. Number two, workflow agents."
en3 = "all without you micromanaging it. Number two, workflow agents."
cand3 = [c["clean"] for c in adj_probe2._find_candidates(vi3, en3)]
print("candidates (Number two,...):", cand3)
assert "number" in cand3 and "two" in cand3, f"FAIL: dấu phẩy phải ngắt cụm, còn: {cand3}"
print("PASS: dấu phẩy ngắt cụm, 'Number two' vẫn được đưa đi phán xử.\n")

# ---- Unit test apply_verdicts ----
print("=" * 80)
print("UNIT TEST — apply_verdicts (giữ dấu câu, không phân biệt hoa thường)")
print("=" * 80)
out = TranslationAdjudicator.apply_verdicts(
    "Number two, workflow agents.",
    {"number": {"decision": "translate", "vietnamese": "Số"},
     "two": {"decision": "translate", "vietnamese": "hai"},
     "workflow": {"decision": "keep", "vietnamese": ""},
     "agents": {"decision": "keep", "vietnamese": ""}}
)
print("->", out)
assert out == "Số hai, workflow agents.", f"FAIL: {out}"
print("PASS\n")

# ---- Integration test Lớp 3 (có LLM) ----
print("=" * 80)
print("INTEGRATION TEST — Lớp 3: Adjudicator + LLM (gemma4:e4b), 5 câu đầu")
print("=" * 80)

KNOWN_TERMS = {"ai", "agent", "agents", "workflow", "api", "chatgpt"}
with open(r'd:\Project\VidTranscribe.ai\src\output\subtitles\subtitles_vi.json', encoding='utf-8') as f:
    vi_data = json.load(f)
en_entries = {e.index: e.text for e in parse_srt(r'd:\Project\VidTranscribe.ai\src\output\subtitles\subtitles_en.srt')}

results_map = {int(k): (rec["translated_text"], []) for k, rec in vi_data.items()}

# Dùng cache riêng để test sạch (không lẫn entry cũ)
from src.utils.translation_cache import TranslationCache
clean_cache = TranslationCache(cache_file=r'd:\Project\VidTranscribe.ai\scratch\_test_cache.json')
clean_cache.cache = {}

adj = TranslationAdjudicator(
    domain_terms=KNOWN_TERMS,
    model_name="gemma4:e4b",
    domain="General",
    cache=clean_cache,
)

limited = {k: v for k, v in results_map.items() if k <= 5}
limited_en = {k: v for k, v in en_entries.items() if k <= 5}
result = adj.process_results(limited, limited_en)

print("\n--- CÂU ĐƯỢC CẬP NHẬT ---")
for idx, new_text in result["updated_texts"].items():
    print(f"[{idx}] {vi_data[str(idx)]['translated_text']}")
    print(f"  -> {new_text}\n")
print("--- TỪ DỊCH BỔ SUNG ---", result["translated_words"])
print("--- THUẬT NGỮ MỚI (GIỮ NGUYÊN) ---", sorted(result["new_terms"]))
print("--- THỐNG KÊ ---", result["stats"])
print("=" * 80)
