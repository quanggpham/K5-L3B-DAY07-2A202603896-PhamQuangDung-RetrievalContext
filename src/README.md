# Code

| File | Vai trò |
|---|---|
| `common.py` | Đường dẫn, biến môi trường, bảng giá, chuẩn hóa text, định nghĩa hit |
| `export_day6.py` | Xuất 178 chunk chiến lược B từ pgvector Day 6 sang định dạng chunk Day 7 |
| `build_benchmark.py` | Ghép benchmark Day 6 + câu bổ sung Day 7, kiểm evidence nằm trọn trong một chunk |
| `retrievers.py` | `dense` (pgvector), `bm25_uni`/`bm25_bi`, `hybrid` (RRF), `hybrid_rerank` (LLM rerank) |
| `eval_retrieval.py` | Hit@1, Evidence Hit@3, Hit@5, MRR@10, Context Evidence Rate với query gốc |
| `agent.py` | Agent tool-calling: `search_report`, `submit_answer`; dedup, budget, kiểm citation |
| `run_agent.py` | Chạy agent trên một split, lưu trace và metric tự động |

Test chỉ chạy sau khi `results/selection.json` tồn tại. Cách chạy: [README](../README.md#bài-làm-setup-và-chạy).
