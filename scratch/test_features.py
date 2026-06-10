# -*- coding: utf-8 -*-
"""Test nhanh: import UI/orchestrator + schema dictionary mới (upsert, migration)."""
import sys, os, json, tempfile
sys.path.insert(0, r'd:\Project\VidTranscribe.ai')

# 1) Import smoke test (đảm bảo không lỗi cú pháp/import)
import src.app as app
from src.pipeline.orchestrator import PipelineOrchestrator
from src.db.db_manager import db, normalize_dict_entry
print("PASS: import app, orchestrator, db OK")

# 2) Migration schema cũ {vi, pho} -> schema mới
old = {"vi": "quy trình làm việc", "pho": "uơc phlô"}
new = normalize_dict_entry(old)
assert new["phonetic"] == "uơc phlô" and new["explanation"] == "quy trình làm việc"
assert set(new.keys()) == {"phonetic", "explanation", "date_added", "added_by", "last_updated"}
print("PASS: migration {vi,pho} -> schema mới:", new)

# 3) upsert_term: thêm mới (AI) rồi cập nhật
test_domain = "Zzz Test Domain"
try:
    e1 = db.upsert_term(test_domain, "workflow", phonetic="uơc phlô", added_by="AI")
    assert e1["added_by"] == "AI" and e1["date_added"]
    import time; time.sleep(1)
    e2 = db.upsert_term(test_domain, "workflow", explanation="luồng công việc", added_by="Human")
    # added_by gốc (AI) phải được giữ, chỉ explanation + last_updated thay đổi
    assert e2["added_by"] == "AI", f"added_by phải giữ gốc AI: {e2}"
    assert e2["explanation"] == "luồng công việc"
    assert e2["date_added"] == e1["date_added"], "date_added phải giữ nguyên"
    print("PASS: upsert thêm mới (AI) + cập nhật giữ metadata gốc:", e2)
finally:
    # dọn file domain test
    p = db.get_dictionary_path(test_domain)
    if os.path.exists(p):
        os.remove(p)
        print("Đã dọn file test:", os.path.basename(p))

# 4) Kiểm tra MODE labels và hàm tồn tại
assert hasattr(PipelineOrchestrator, "resume_from_subtitles")
assert app.MODE_E2E and app.MODE_TRANSLATE
print("PASS: orchestrator.resume_from_subtitles + MODE labels tồn tại")
print("ALL FEATURE TESTS PASSED")
