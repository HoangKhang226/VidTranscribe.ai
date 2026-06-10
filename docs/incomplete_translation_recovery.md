# Thiết kế: Khắc phục lỗi LLM dịch sót / dịch không hết (Incomplete Translation Recovery)

> Tài liệu thiết kế kỹ thuật cho pipeline dịch phụ đề đa ngành của VidTranscribe.ai.
> Mô tả cách phát hiện, phân loại, và sửa các token/cụm tiếng Anh còn sót sau bước dịch
> (`src/pipeline/step4_translate.py`) bằng một chuỗi kiểm tra nhiều lớp: tự kiểm tra trong lô,
> validator hậu dịch, LLM adjudicator, cache, và domain lexicon.

---

## 1. Bối cảnh & Phạm vi

Pipeline lồng tiếng AI dịch phụ đề tiếng Anh (`subtitles_en.srt`) sang tiếng Việt theo batch bằng LLM (Ollama / Gemini judge tùy chế độ). Đầu ra chính gồm:

- `subtitles_vi.srt`: phụ đề tiếng Việt sau khi rà soát.
- `subtitles_vi.json`: metadata chứa `translated_text`, `phonetic_text`, và các term đã được xác nhận.

Tài liệu này tập trung vào **lỗi dịch sót**: LLM có trả về bản dịch nhưng vẫn để sót token/cụm tiếng Anh trong câu VI, hoặc giữ nhầm một cụm đáng lẽ phải được dịch. Đây là lỗi ảnh hưởng trực tiếp đến trải nghiệm người xem vì tạo cảm giác câu nói bị ngắt mạch, còn lẫn tiếng Anh chưa xử lý.

Yêu cầu xuyên suốt: **mọi cơ chế phải đa ngành** (y tế, tài chính, pháp lý, công nghệ, ẩm thực, logistics, hàng không...). Không hardcode từ vựng hay luật riêng cho một lĩnh vực.

---

## 2. Phân loại lỗi "dịch không hết"

| # | Loại lỗi | Ví dụ (EN → VI) | Mong muốn |
|---|----------|-----------------|-----------|
| A | **Sót từ phổ thông** | "all without you micromanaging" → "...mà không cần bạn micromanaging nó" | Phải dịch: "quản lý vi mô" |
| B | **Sót cụm chỉ dẫn/đếm** | "Number two, ..." → "Number two, ..." | "Số hai, ..." |
| C | **Thuật ngữ ngành được giữ đúng** | "workflow agents" → "workflow agents" | Giữ nguyên |
| D | **Thuật ngữ mới / tên riêng / acronym** | "Genspark super agent", "BM25", "OAuth2" | Cần quyết định giữ hay dịch theo ngữ cảnh đa ngành |
| E | **Rớt từ cuối câu (truncation)** | "into a polished" → "thành một bản" | Giữ đủ ý, không bịa |
| F | **Gộp câu / lệch index** | 3 dòng dịch gộp thành 1 | Tách lại theo index |

Trọng tâm tài liệu: **A, B, D**. C là hành vi đúng. E và F đã được xử lý ở lớp 1.

---

## 3. Nguyên nhân gốc

1. **LLM bỏ sót do batch dài hoặc ngữ cảnh rộng**: model đôi khi giữ nguyên token tiếng Anh vì tưởng là thuật ngữ hoặc vì không ưu tiên dịch từng chi tiết.
2. **Ranh giới thuật ngữ mơ hồ**: không biết từ nào nên giữ nguyên, từ nào phải dịch. Đây là bản chất nhập nhằng của bài toán hậu dịch.
3. **Từ Latin hợp lệ trong tiếng Việt**: ví dụ "vi", "kinh", "doanh" không phải tiếng Anh nhưng trông giống token Latin, dễ bị nhận nhầm nếu chỉ quét bề mặt.
4. **Cụm thuật ngữ nhiều từ**: nhiều term đúng phải được giữ cả cụm, không được tách lẻ từng token.

Giải pháp mới xử lý gốc lỗi bằng cách kết hợp `PostTranslationValidator` + `TranslationAdjudicator` + cache + domain lexicon, thay vì phụ thuộc vào một lần prompt duy nhất.

---

## 4. Kiến trúc giải pháp nhiều lớp

Giải pháp gồm **3 lớp phòng thủ**, chạy nối tiếp. Mỗi lớp bắt một loại lỗi khác
nhau; lỗi lọt qua lớp trước sẽ được lớp sau xử lý.

```
        ┌─────────────────────────────────────────────────────────┐
        │  LỚP 1: Dịch theo lô + Tự kiểm tra ngay (in-batch)        │
        │  - Kiểm độ đầy đủ index (đủ số câu?)                       │
        │  - Kiểm tỉ lệ độ dài VI/EN (chống rớt từ cuối - lỗi E)    │
        │  - Fallback dịch từng câu khi lô lỗi (lỗi F)              │
        └───────────────────────────┬─────────────────────────────┘
                                     │ (bản dịch thô)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  LỚP 2: Phát hiện từ tiếng Anh còn sót (không LLM)         │
        │  PostTranslationValidator.find_foreign_words(vi, en)      │
        │  - Token Latin trong VI ∩ token trong EN gốc             │
        │  - Loại thuật ngữ ngành đã biết (domain_terms) -> "term"  │
        │  - Phần còn lại -> "candidate" (nghi sót - lỗi A/B/D)     │
        └───────────────────────────┬─────────────────────────────┘
                                     │ (danh sách candidate)
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  LỚP 3: Trọng tài hậu dịch bằng LLM (Adjudicator)         │
        │  TranslationAdjudicator.adjudicate_sentence(vi, en)       │
        │  - Với mỗi candidate: LLM quyết định translate / keep     │
        │  - translate -> thay từ đã dịch vào câu (lỗi A/B)         │
        │  - keep      -> ghi nhận thuật ngữ mới (lỗi D) -> G2P     │
        │  - Có cache để không hỏi lại cùng một từ                  │
        └─────────────────────────────────────────────────────────┘
```

### 4.1. Lớp 1 — Tự kiểm tra trong lô (đã có)

Nằm trong `translate_batch()`:

- **Đủ index**: parse theo định dạng `[index] text`. Nếu thiếu câu → thử lại,
  rồi rơi xuống **fallback dịch từng câu**.
- **Tỉ lệ độ dài**: nếu `len(VI) < 0.80 * len(EN)` với câu đủ dài → nghi rớt từ
  cuối (lỗi E) → huỷ lô, chạy fallback. Ở fallback, sau 3 lần vẫn rớt thì ghép
  bù token cuối của câu gốc để không mất dữ liệu.
- **Chống rò rỉ**: loại bản dịch lẫn ký tự Trung/Cyrillic (`has_leakage`).

> Lớp này xử lý tốt E & F nhưng **không** bắt được A/B/D (vì câu vẫn "đủ dài" và
> "đủ index", chỉ là còn vài từ tiếng Anh lẫn vào).

### 4.2. Lớp 2 — Phát hiện từ còn sót (đa ngành, không LLM)

`PostTranslationValidator.find_foreign_words(vi_text, en_text)`.

Nguyên tắc cốt lõi **không thiên ngành, không thiên ngôn ngữ**:

> Một token chữ Latin trong câu tiếng Việt chỉ được coi là "tiếng Anh còn sót"
> **nếu nó cũng xuất hiện trong câu gốc tiếng Anh**.

- Ưu điểm: tự động loại "vi", "kinh", "doanh"... (không có trong câu EN gốc) mà
  **không cần** danh sách từ tiếng Việt hardcode (vốn luôn thiếu và thiên ngữ).
- Sau đó phân loại token còn lại:
  - Thuộc `domain_terms` (hoặc là thành phần của cụm thuật ngữ) → `term` (giữ).
  - Còn lại → `candidate` (đưa lên Lớp 3).

`domain_terms` đến từ Bước 2/4 (auto-detect chủ đề + trích thuật ngữ) nên **đúng
cho mọi ngành**, không cố định danh sách.

### 4.3. Lớp 3 — Trọng tài hậu dịch bằng LLM

`TranslationAdjudicator`. Với mỗi câu có `candidate`:

0. **Bảo vệ cụm thuật ngữ liền kề (adjacency, không LLM)** — chạy trước khi hỏi LLM:
   - Quét các chuỗi token tiếng Anh **liền kề** trong câu VI (chỉ nối khi giữa hai
     token chỉ có khoảng trắng/gạch nối; dấu phẩy, chấm, hai chấm... sẽ NGẮT cụm).
   - Nếu một chuỗi liền kề chứa ÍT NHẤT một thuật ngữ ngành đã biết, thì **mọi từ
     trong chuỗi** được coi là bổ nghĩa cho thuật ngữ → `keep` (loại khỏi candidate).
   - Ví dụ: `general purpose task agents` (có `agents`) → giữ nguyên cả 4 từ.
     `Number two, workflow agents` → dấu phẩy ngắt: `Number two` vẫn được phán xử,
     còn `workflow agents` giữ nguyên.
   - Quy tắc này **đa ngành tuyệt đối**: chỉ dựa vào `domain_terms` (đến từ ngành
     bất kỳ) và cấu trúc câu, không hardcode từ vựng.
1. **Tra cache trước** (`TranslationCache`):
   - đã đánh dấu `tech` (giữ) → quyết định `keep`.
   - đã có bản dịch `word` → dùng lại bản dịch.
   - chưa có → đưa vào danh sách hỏi LLM.
2. **Hỏi LLM** (structured output) cho từng từ trong ngữ cảnh **cả câu EN và VI**:
   - `translate`: từ phổ thông bị bỏ quên → trả bản dịch 1–4 từ hợp ngữ cảnh.
   - `keep`: thuật ngữ ngành / tên riêng / viết tắt / phần của cụm thuật ngữ.
   - **Quy tắc an toàn**: khi không chắc → `keep` (tránh dịch sai thuật ngữ).
3. **Áp dụng**:
   - `translate` → thay token bằng bản dịch (giữ dấu câu, không phân biệt hoa thường).
   - `keep` → thêm vào tập thuật ngữ → được sinh phiên âm G2P ở bước sau.
4. **Ghi cache** để các lần chạy sau không phải hỏi lại.

Cấu hình LLM cho Lớp 3 (tiết kiệm tài nguyên):
- `think=False` (tắt suy luận nội bộ của Gemma), `num_ctx=2048` (mỗi lần 1 câu).
- Chạy tuần tự để tránh tràn VRAM trên GPU nhỏ.

---

## 5. Vị trí tích hợp trong pipeline

Trong `step4_translate.run()`:

1. Dịch toàn bộ lô → `results_map[index] = (vietnamese_text, word_positions)`.
2. **(Mới)** Chạy `TranslationAdjudicator.process_results(results_map, en_by_index)`:
   - Cập nhật `updated_texts` (các câu có từ được dịch bổ sung).
   - Thu `new_terms` (thuật ngữ mới được xác nhận giữ) → hợp vào `all_terms`.
3. **Tính lại** `word_positions` và `all_english_terms` trên văn bản đã rà soát.
4. Sinh phiên âm G2P → ghi `subtitles_vi.srt` + `subtitles_vi.json`.

> Đặt Lớp 3 **trước** bước G2P để thuật ngữ mới (`keep`) kịp được phiên âm, còn
> từ được dịch (`translate`) thì biến mất khỏi danh sách cần phiên âm.

---

## 6. Hợp đồng dữ liệu (Data Contracts)

`find_foreign_words(vi_text, en_text) -> dict`:
```json
{
  "terms":      [{"word","clean","position","in_context"}],
  "candidates": [{"word","clean","position","in_context"}],
  "total_english": 0
}
```

`adjudicate_sentence(vi_text, en_text) -> dict`:
```json
{ "<word_clean>": {"decision": "translate|keep", "vietnamese": "..."} }
```

`process_results(results_map, entries_by_index) -> dict`:
```json
{
  "updated_texts":    {"<index>": "câu VI mới"},
  "new_terms":        ["term1", "term2"],
  "translated_words": {"word": "bản dịch"},
  "stats": {"sentences_touched": 0, "translated": 0, "kept_as_term": 0}
}
```

---

## 7. Trường hợp biên & cách xử lý

| Trường hợp biên | Xử lý |
|-----------------|-------|
| LLM bảo `translate` nhưng không cho bản dịch | Hạ xuống `keep` (bảo toàn, không mất từ) |
| LLM lỗi/timeout | Fallback: `keep` toàn bộ candidate (không phá câu) |
| Cụm thuật ngữ nhiều từ ("general purpose task") | **Bảo vệ theo cụm liền kề** (mục 4.3.0): cả cụm chứa thuật ngữ ngành được giữ nguyên |
| Từ tiếng Việt không dấu trùng dạng Latin | Bị loại ở Lớp 2 nhờ đối chiếu câu EN gốc |
| Token < 2 ký tự | Bỏ qua ở Lớp 3 |
| Từ đã phán xử trước đó | Tra cache, không gọi lại LLM |

---

## 8. Kế hoạch kiểm thử

### 8.1. Unit test — Lớp 2 (không cần LLM)
- **Đối chiếu nguồn**: VI="...quản lý vi mô..." + EN không chứa "vi" → "vi" KHÔNG
  vào candidate.
- **Phân loại thuật ngữ**: `domain_terms={"workflow"}`, VI chứa "workflow" → vào
  `terms`, không vào `candidates`.
- **Cụm thuật ngữ**: `domain_terms={"machine learning"}`, VI chứa "machine" →
  nhận là `term` (thành phần của cụm).

### 8.2. Unit test — `apply_verdicts` (không cần LLM)
- Thay đúng token, giữ dấu câu: "Number two," + {number→Số, two→hai} →
  "Số hai,".
- Token `keep` giữ nguyên.
- Không phân biệt hoa thường khi so khớp.

### 8.3. Integration test — Lớp 3 (có LLM, dữ liệu thật)
- Nạp `subtitles_vi.json` thật, chạy `process_results` trên 5 câu đầu.
- Khẳng định:
  - "Number two" → dịch ("Số hai").
  - "micromanaging" (nếu sót) → dịch.
  - Tên riêng/sản phẩm ("Copilot", "Microsoft", "Studio") → `keep`.
- Đo thời gian/câu để theo dõi chi phí LLM.

### 8.4. Test hồi quy đa ngành
Chuẩn bị mẫu cho ≥3 ngành khác nhau (vd y tế, tài chính, ẩm thực) với
`domain_terms` tương ứng, xác nhận không có luật nào "ăn gian" cho riêng công nghệ.

### 8.5. Tiêu chí chấp nhận
- Tỉ lệ từ phổ thông còn sót sau Lớp 3 giảm về ~0 trên tập mẫu.
- Không có thuật ngữ ngành bị dịch sai (đo thủ công trên tập mẫu).
- Không làm rớt/đổi nghĩa từ đã dịch đúng ở Lớp 1.

---

## 9. Rủi ro & Đánh đổi

- **Chi phí LLM**: thêm 1 lượt hỏi/câu có candidate. Giảm nhờ cache + chỉ hỏi từ
  chưa biết + tắt `think`.
- **Sai phán quyết của LLM**: giảm rủi ro bằng quy tắc "không chắc thì keep" và
  cache để con người có thể chỉnh sửa từ điển ngành về sau.
- **Phụ thuộc chất lượng câu EN gốc**: Lớp 2 dựa vào câu EN; nếu STT sai chính tả
  nặng, một số token có thể lọt/lọc nhầm — chấp nhận được vì Lớp 3 vẫn phán xử.

---

## 10. Mở rộng tương lai

- Phán xử **toàn bộ cụm** bằng LLM (hiện cụm chứa thuật ngữ ngành đã được giữ
  nguyên bằng quy tắc adjacency; có thể bổ sung hỏi LLM cho các cụm tiếng Anh dài
  KHÔNG chứa thuật ngữ ngành nào để dịch trọn cụm thay vì từng từ).
- Tự động **đề xuất bổ sung từ điển ngành** (`db/dictionaries/<domain>.json`) từ
  các quyết định `keep` lặp lại nhiều lần.
- Bảng điều khiển review: liệt kê các quyết định của Adjudicator để người dùng duyệt.
