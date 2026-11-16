# Day 7 — Agentic RAG

## Bài làm: setup và chạy

Phạm Quang Dũng · 2A202603896 · L3B. Báo cáo: [report/individual/2A202603896_PhamQuangDung.md](report/individual/2A202603896_PhamQuangDung.md).

**Yêu cầu:** Python 3.11+ (đã chạy với 3.13), Docker, `OPENAI_API_KEY`, và repo Day 6 [K5-L3B-DAY06-2A202603896-DataIngestion](https://github.com/quanggpham/K5-L3B-DAY06-2A202603896-DataIngestion) đã ingest (bảng `chunks`, chiến lược B, `text-embedding-3-small`). PDF và chunk text không có trong repo vì bản quyền; xem `data/manifest.json` của Day 6.

```powershell
# 1. Vector store Day 6 (chạy trong thư mục starter/ của repo Day 6)
docker compose -p day6_student_starter up -d --wait

# 2. Môi trường Day 7 (thư mục gốc repo này)
py -3.13 -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env        # điền OPENAI_API_KEY
$env:PYTHONUTF8 = "1"              # Windows: tránh lỗi cp1252 khi in/đọc tiếng Việt

# 3. Dữ liệu và benchmark
python -m src.export_day6          # pgvector -> data/day6/chunks.json (gitignored)
python -m src.build_benchmark      # -> data/benchmark/questions.json, cases.json (kiểm evidence)

# 4. Thử nghiệm trên dev
python -m src.eval_retrieval --split dev
python -m src.run_agent --split dev --configs agent_dense_k3 agent_bm25_k3 agent_hybrid_rerank_k3 agent_hybrid_rerank_k5
python starter/agent_metrics.py --cases data/benchmark/cases.json --traces results/agent/dev/agent_hybrid_rerank_k3/traces.json --reviews results/agent/dev/agent_hybrid_rerank_k3/reviews.json --split dev --output results/agent/dev/agent_hybrid_rerank_k3/metrics.json

# 5. Test (chỉ chạy được khi đã có results/selection.json)
python -m src.eval_retrieval --split test --configs dense_k3 hybrid_rerank_k3
python -m src.run_agent --split test --configs agent_dense_k3 agent_hybrid_rerank_k3

# Unit test (mock, không đo chất lượng)
python -m unittest discover -s tests -v
```

| Nội dung | Đường dẫn |
|---|---|
| Code | `src/` ([mô tả](src/README.md)), test `tests/test_day7.py` |
| Benchmark | `data/benchmark/` (`day7_additions.json` là câu viết mới cho Day 7) |
| Kế hoạch, tiêu chí chọn | `docs/EXPERIMENT_PLAN.md`, `results/selection.json` |
| Retrieval dev/test | `results/retrieval/{dev,test}/*.json`, `*_summary.json` |
| Agent: trace, nhãn, metric | `results/agent/{dev,test}/<config>/{traces,reviews,metrics,summary}.json` |
| Lần chạy dev trước khi sửa agent | `results/agent/dev_v1_fullid/`, `results/agent/dev_v2_shortid/` |

Kết quả chạy lại sẽ khác đôi chút vì model không hoàn toàn tất định dù `temperature=0`; query embedding và điểm rerank được cache trong `.cache/` (gitignored).

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
