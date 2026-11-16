# Kế hoạch thử nghiệm và tiêu chí chọn (chốt 2026-11-16, trước khi chạy dev)

## Điều kiện so sánh chung

- Corpus: 178 chunk chiến lược B của Day 6 (paragraph 300–1000 ký tự), một tài liệu World Bank 41 trang nội dung.
- Index dense: pgvector Day 6, `text-embedding-3-small` 1536 chiều, cosine exact. Không re-embed tài liệu.
- Benchmark: `data/benchmark/questions.json` + `cases.json`; dev 26 có đáp án + 3 thiếu bằng chứng; test 8 + 3.
- Query: câu hỏi gốc cho bảng retrieval; agent tự viết query và được báo riêng.
- Filter: chỉ `section`, và chỉ khi câu hỏi có `filter` do người dùng cung cấp (q23–q26, t08). Áp dụng như nhau cho mọi phương pháp.
- Hit: chunk đúng `source_id`, `page` và chứa nguyên văn evidence (chuẩn hóa khoảng trắng); evidence thay thế được chấp nhận.

## Phương pháp retrieval

| Tên | Mô tả |
|---|---|
| `dense` | Embedding query → cosine trên pgvector Day 6 |
| `bm25_uni` | BM25 (k1=1.5, b=0.75) trên âm tiết tiếng Việt |
| `bm25_bi` | BM25 trên âm tiết + cặp âm tiết liền kề (xấp xỉ tách từ 2 âm tiết) |
| `hybrid` | RRF (k=60) của `dense` top-20 và `bm25_bi` top-20 |
| `hybrid_rerank` | `hybrid` top-15 → `gpt-4.1-mini` chấm liên quan 0–3, giữ thứ tự gốc khi hòa |

## Giả thuyết cải thiện

1. **BM25 bigram > unigram:** âm tiết đơn lẻ ("tỷ", "giá") quá phổ biến; cặp âm tiết bắt được từ ghép và cụm số.
2. **Hybrid > dense:** câu hỏi chứa số liệu, tên văn bản (Thông tư 02/2023), viết tắt (SBIMIS) mà dense dễ bỏ lỡ.
3. **Rerank > hybrid ở Hit@1:** model đọc được chunk chứa đúng con số thay vì chunk cùng chủ đề.
4. **Agent top-k 5 + budget lớn hơn > top-k 3:** context chứa evidence nhiều hơn, đổi lại tốn token.

## Tiêu chí chọn cấu hình agent (chỉ dùng dev)

1. Context Evidence Rate của agent cao nhất (coi là hòa nếu chênh ≤ 1 câu).
2. Grounded answer trung bình /2 cao hơn, tính cả abstention đúng trên câu thiếu bằng chứng.
3. Chi phí và thời gian trung vị thấp hơn.

Sau khi chốt, ghi `results/selection.json`; chỉ khi đó script mới cho chạy test. Chạy baseline và cấu hình chọn trên test đúng một lần.
