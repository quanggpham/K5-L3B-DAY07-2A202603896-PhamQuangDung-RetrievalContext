# Báo cáo Day 7 — Phạm Quang Dũng (2A202603896, L3B)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** World Bank, *Báo cáo Điểm lại tháng 8/2023: Để đầu tư công thúc đẩy tăng trưởng* (bản tiếng Việt, doc ID 099417508092322367), 58 trang vật lý, 41 trang nội dung được index. Dùng cho môn học, phi thương mại, có ghi nguồn; PDF và chunk text không đưa lên repo (chi tiết quyền và sha256 trong `data/manifest.json` của repo Day 6). Index: **178 chunk** chiến lược B của Day 6 (paragraph, 300–1000 ký tự, trung bình 669). Bảng và nhãn biểu đồ đã bị loại khi ingest Day 6, nên benchmark chỉ hỏi về văn xuôi.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker, Day 6), bảng `chunks`, `strategy='B'`, cosine exact (không ANN). `text-embedding-3-small`, 1536 chiều. Query được embed cùng model; `export_day6.py` kiểm tra model của index trùng `EMBEDDING_MODEL`. Không re-embed tài liệu nên không có chi phí index mới.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-4.1-mini-2025-04-14` (OpenAI chat completions, tool calling, `temperature=0`). Dev = 26 câu có đáp án của Day 6 (20 dev + 6 test cũ) + 3 câu thiếu bằng chứng; test = **8 câu có đáp án + 3 câu thiếu bằng chứng viết mới** cho Day 7 (`data/benchmark/day7_additions.json`) trước mọi thử nghiệm Day 7. Test Day 6 được chuyển vào dev vì Day 7 xây hệ thống mới; test mới hỏi các sự kiện mà không câu dev nào hỏi.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold` = `{source_id, page, evidence}`, evidence là trích nguyên văn; `build_benchmark.py` kiểm mọi evidence (kể cả evidence thay thế) nằm trọn trong một chunk đúng trang. Hit = chunk đúng nguồn, đúng trang và chứa evidence (chuẩn hóa khoảng trắng). Câu thiếu evidence là câu gần chủ đề (GDP 2026, dự báo tỷ giá, xuất khẩu gạo, vốn Vành đai 4, lãi suất cho vay bình quân, CPI tháng 7/2023); đã tìm trên text gốc của cả 58 trang PDF (gồm nhãn bảng/biểu đồ mà index bỏ) để chắc không có câu trả lời, cách kiểm ghi trong `absence_reason`. Câu thiếu evidence không vào mẫu số của metric retrieval.

## Retrieval và thử nghiệm

Query gốc (câu hỏi nguyên văn), cùng corpus, index và nhãn. Filter `section` chỉ áp dụng cho câu có `filter` do người dùng cung cấp (q23–q26, t08), như nhau cho mọi phương pháp. Context budget tính bằng ký tự (gồm dòng header của mỗi chunk). Thời gian là trung vị mỗi câu ở lần chạy không cache; chi phí tính bằng USD cho cả split theo giá OpenAI (embedding $0.02/1M token; gpt-4.1-mini $0.40/$1.60 trên 1M token vào/ra).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense_k3` | Dense pgvector, top-3, budget 4000 | Baseline Day 6 | dev | 19/26 (73%) | 25/26 (96%) | 25/26 (96%) | 244 ms / $0.00002 |
| Phương pháp 2 `bm25_bi_k3` | BM25 âm tiết + cặp âm tiết, top-3, 4000 | Câu hỏi chứa số, tên văn bản, viết tắt mà dense dễ bỏ lỡ | dev | 24/26 (92%) | 25/26 (96%) | 25/26 (96%) | 6 ms / $0 |
| Phương pháp 3 `hybrid_k3` | RRF (k=60) dense top-20 + BM25 top-20, top-3, 4000 | Kết hợp ưu điểm hai bên | dev | 21/26 (81%) | 24/26 (92%) | 24/26 (92%) | ≈ dense + 10 ms / $0.00002 |
| Cải thiện 1 `bm25_uni_k3` (ablation) | BM25 chỉ âm tiết đơn | Bỏ cặp âm tiết sẽ kém hơn | dev | 24/26 (92%) | 25/26 (96%) | 25/26 (96%) | 3 ms / $0 |
| Cải thiện 2 `hybrid_rerank_k3` | Hybrid top-15 → gpt-4.1-mini chấm 0–3, top-3, 4000 | Model đọc được chunk chứa đúng con số | dev | **26/26 (100%)** | **26/26 (100%)** | **26/26 (100%)** | 4.9 s / $0.057 |
| Cải thiện 3 `*_k5` | Như trên nhưng top-5, budget 6000 | Context rộng hơn chứa evidence nhiều hơn | dev | (như k3) | (như k3) | dense 25/26, hybrid 25/26, rerank 26/26 | context TB 4.1–4.2k ký tự so với 2.5k |
| Cấu hình đã chọn `hybrid_rerank_k3` | như Cải thiện 2 | chốt trên dev | test | 8/8 (100%) | 8/8 (100%) | 8/8 (100%) | 5.1 s / $0.018 |
| Baseline `dense_k3` | như Baseline | | test | 7/8 (88%) | 8/8 (100%) | 8/8 (100%) | 483 ms / $0.00001 |

MRR@10 trên dev: dense 0.820, BM25 0.942, hybrid 0.880, hybrid+rerank 1.000. Kết quả từng câu (rank, score, hit, base_rank trước rerank) nằm trong `results/retrieval/<split>/<config>.json`.

**Kết quả từng giả thuyết:** (1) cặp âm tiết không cải thiện so với âm tiết đơn (cùng 24/26 và 25/26) và chậm gấp đôi, nên giả thuyết không được xác nhận; (2) hybrid RRF tốt hơn dense ở Hit@1 (21 so với 19) nhưng kém BM25 thuần: trọng số bằng nhau khiến các chunk dense xếp sai kéo chunk đúng của BM25 xuống (q03 hạng 7, q11 hạng 4); (3) rerank đưa mọi câu lên hạng 1, đổi lại +4.7 s và khoảng $0.002 mỗi query; (4) top-5 không tăng Context Evidence Rate của retrieval một lượt nhưng làm agent bịa ở câu thiếu bằng chứng (xem bên dưới).

**Lỗi dev 1 — từ vựng lệch với BM25 (q03).** "Đầu tư công năm 2023 có thể đạt bao nhiêu phần trăm GDP và với điều kiện gì?" BM25 không có evidence trong top-10: tài liệu viết "7,1% GDP… nếu được triển khai đầy đủ", không có "phần trăm"/"điều kiện", còn "đầu tư công" xuất hiện ở hầu hết chunk nên IDF thấp. Dense xếp hạng 3, hybrid hạng 7, rerank hạng 1. Ở mức agent, `agent_bm25_k3` trả lời sai "báo cáo không nêu % GDP" (điểm 0). Cách sửa đã thử: thêm tín hiệu ngữ nghĩa bằng hybrid, sau đó rerank; `agent_hybrid_rerank_k3` trả lời đúng 7,1% kèm điều kiện.

**Lỗi dev 2 — dense bỏ lỡ chunk số liệu (q11, q20, q12).** Với "tỷ lệ nợ xấu tăng từ mức nào lên mức nào", dense xếp chunk p31:c00 ở hạng 7; agent dense tìm 3 lượt với query viết lại vẫn không lấy được nên từ chối. Tương tự, q20 (tỷ lệ 90-10, p50:c04) và q12. `agent_dense_k3` từ chối nhầm 3/26 câu; BM25 còn 1/26 (q20), hybrid+rerank 0/26.

**Lỗi dev 3 — agent không chép được citation ID (lỗi lớn nhất, sửa trong 2 vòng).** Lần chạy đầu (`results/agent/dev_v1_fullid/`) context có evidence ở 22–26/26 câu nhưng agent chỉ trả lời được 13–17/26: model viết đúng đáp án nhưng không chèn được ID dài như `[wb-vn-taking-stock-2023-08-vi:p20:c00]` vào câu trả lời, khi được nhắc lại thì rút gọn thành `[p20:c00]`, nên bị loại vì citation không hợp lệ. Giả thuyết: ID dài khó chép. Vòng 2 (`dev_v2_shortid/`): hiển thị mã `S1, S2…` và ánh xạ ngược về ID thật, số câu được trả lời tăng lên 21–25/26. Lỗi còn lại là viết gộp `[S1, S2]` hoặc chép cả header `[S1 | trang 22 | chuong_1]`. Vòng 3: chuẩn hóa nhóm ngoặc chỉ khi chứa mã có trong context (mã ngoài context vẫn bị loại), lời nhắc sửa lỗi có ví dụ cụ thể, và quy định rõ phải `abstain=true` khi báo cáo không có đúng chỉ tiêu/năm được hỏi, cho kết quả 23–26/26.

| Dev, câu có đáp án được trả lời | v1 ID dài | v2 mã ngắn | v3 (code đã chốt) |
|---|---|---|---|
| agent_dense_k3 | 13/26 | 21/26 | 23/26 |
| agent_bm25_k3 | 14/26 | 25/26 | 25/26 |
| agent_hybrid_rerank_k3 | 16/26 | 22/26 | 26/26 |
| agent_hybrid_rerank_k5 | 17/26 | 24/26 | 26/26 |

**Lỗi dev 4 — context rộng làm agent bịa ở câu thiếu bằng chứng.** `agent_hybrid_rerank_k5` (top-5, budget 9000) trả lời u01 "GDP 2026 là 6,0%" (đó là dự báo 2025) và u03 "xuất khẩu gạo giảm 12%" (đó là mức giảm tổng xuất khẩu). Bản k3 với cùng prompt từ chối đúng cả 3 câu.

**Lý do chọn cấu hình.** Tiêu chí công bố trước trong `docs/EXPERIMENT_PLAN.md`: (1) Context Evidence Rate của agent cao nhất, coi là hòa nếu chênh ≤ 1 câu; (2) nhiều câu đúng nhất, tính câu điểm 2 cộng câu từ chối đúng; (3) chi phí và thời gian thấp hơn. Hybrid+rerank k3 và k5 hòa ở (1) với 26/26; k3 thắng ở (2) với 28/29 so với 27/29. Chọn `agent_hybrid_rerank_k3`, baseline là `agent_dense_k3`; ghi vào `results/selection.json` rồi mới chạy test, mỗi cấu hình một lần.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** hai tool là `search_report(query)` (gọi retriever của cấu hình, top-k cố định, filter là ngữ cảnh người dùng do hệ thống áp vào, model không tự đặt) và `submit_answer(answer, citations, abstain)`. `tool_choice="required"`, mỗi lượt một tool call. Agent dừng khi gọi `submit_answer`; sau 3 lượt tìm thì bị ép gọi `submit_answer`; giới hạn tổng 6 lượt model. Context tích lũy qua các lượt, không lặp chunk đã thấy (dedup), mỗi chunk mang mã S-n, trang và chương; chunk không vừa budget (6000 ký tự cho k3, 9000 cho k5) bị bỏ và ghi `dropped_budget`. Citation phải là mã có trong context và xuất hiện trong câu trả lời; nếu sai được sửa 1 lần, sai tiếp thì chuyển thành từ chối. Lỗi retriever được trả cho model dưới dạng thông báo lỗi và ghi vào trace, không thay bằng kết quả giả; lỗi provider ghi `status: error`. Không có lỗi model hay vector store trong các lần chạy đã báo cáo.

| Cấu hình / split | Tool use | Số lượt gọi (phân bố, TB) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| agent_dense_k3 / dev | 29/29 | 1:23, 2:1, 3:5 · 1.38 | 3/0/23 · 1.77 | 27/27 | 24/24 | 2/3 | 3/26 |
| agent_bm25_k3 / dev | 29/29 | 1:23, 2:2, 3:4 · 1.34 | 2/0/24 · 1.85 | 37/38 | 25/25 | 3/3 | 1/26 |
| agent_hybrid_rerank_k3 / dev | 29/29 | 1:25, 2:1, 3:3 · 1.24 | 0/1/25 · 1.96 | 40/41 | 26/26 | 3/3 | 0/26 |
| agent_hybrid_rerank_k5 / dev | 29/29 | 1:26, 3:3 · 1.21 | 0/0/26 · 2.00 | 44/48 | 28/28 | 1/3 | 0/26 |
| **agent_hybrid_rerank_k3 / test** | 11/11 | 1:9, 2:1, 3:1 · 1.27 | 0/0/8 · **2.00** | 12/12 | 8/8 | **3/3** | 0/8 |
| agent_dense_k3 / test (baseline) | 11/11 | 1:7, 2:1, 3:3 · 1.64 | 0/0/8 · 2.00 | 9/9 | 8/8 | 3/3 | 0/8 |

Thời gian và chi phí của agent (cả split, gồm embedding và rerank): dev dense k3 4.2 s trung vị / $0.044; BM25 k3 4.2 s / $0.044; hybrid+rerank k3 4.9 s / $0.124; k5 4.3 s / $0.128. Test baseline 5.7 s / $0.020; cấu hình chọn 10.1 s / $0.047. Context Evidence Rate của agent (evidence nằm trong context cuối): dev 23/26, 24/26, 26/26, 26/26; test 8/8 cho cả hai cấu hình.

**Chấm context trên test** (`results/agent/test/context_review.json`, cả 8 câu có đáp án, cho baseline và cấu hình chọn): mức 2 = 8/8 ở cả hai cấu hình, mức 1 và mức 0 = 0. Mọi context cuối đều có chunk chứa nguyên văn evidence, đúng trang, kèm đơn vị và điều kiện.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t01 (test, chọn) | Có đáp án | `search_report("lãi suất tái cấp vốn quý 2 năm 2023")` → p29:c01 (gold) hạng 1, p12:c03, p13:c02 | "…lãi suất tái cấp vốn… 4,5%, giảm 150-200 điểm cơ bản qua bốn đợt cắt giảm từ tháng 3 đến tháng 6 năm 2023" [p29:c01][p12:c03] | 2; cả hai citation đều chứa 4,5% và 150-200 bps |
| u05 (test, chọn) | Thiếu bằng chứng | 2 lượt tìm "lãi suất cho vay bình quân tháng 6/2023" → chỉ có lãi suất chính sách | "Báo cáo không cung cấp số liệu cụ thể về lãi suất cho vay bình quân…, chỉ nêu các mức lãi suất chính sách" (abstain) | Từ chối đúng, không đưa con số thay thế |
| q03 (dev, bm25) | Có đáp án, lỗi | 1 lượt BM25 → p13:c01, p23:c00; không có p14/p35 | "…báo cáo không nêu rõ tỷ lệ phần trăm đầu tư công trên GDP" | 0; lệch từ vựng (Lỗi 1) |
| q11 (dev, dense) | Có đáp án, lỗi | 3 lượt dense ("tỷ lệ nợ xấu quý 1 năm 2023…") → không lần nào có p31:c00 | Từ chối | 0; từ chối nhầm (Lỗi 2) |
| u03 (dev, k5) | Thiếu bằng chứng, lỗi | 3 lượt "xuất khẩu gạo…" → p27:c00, p11:c01, p20:c01 (không có gạo) | "Kim ngạch xuất khẩu gạo… giảm 12%" [p27:c00][p11:c01][p20:c01] | Không từ chối, 3 citation không hỗ trợ (Lỗi 4) |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** trace agent `results/agent/<split>/<config>/traces.json` (mỗi tool call có query, filter, top-k, kết quả kèm rank/score, chunk thêm/lặp/bị bỏ do budget, câu trả lời gốc trước khi ánh xạ mã, context cuối, token, chi phí, thời gian); nhãn `reviews.json`; tổng hợp `metrics.json` (từ `starter/agent_metrics.py`) và `summary.json`. Lệnh chạy ở [README](../../README.md#bài-làm-setup-và-chạy). Tổng chi phí cho mọi lần chạy Day 7 (gồm v1, v2) khoảng $1.2 tính theo token; số tiền thực trả thấp hơn vì một phần embedding và rerank lấy từ cache (`cache_hits` trong summary).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trên dev, cấu hình chọn loại bỏ được 3 câu từ chối nhầm của baseline (1.77 → 1.96/2) và giữ abstention 3/3, đổi lại chi phí gấp khoảng 2.8 lần và thêm một lượt gọi LLM để rerank mỗi lần tìm.
  - **Trên test, cấu hình chọn không tốt hơn baseline về chất lượng câu trả lời** (cả hai đạt 8/8 điểm 2, 3/3 từ chối đúng); chỉ Hit@1 của retrieval tăng từ 7/8 lên 8/8, trong khi chi phí tăng khoảng 2.3 lần và thời gian trung vị từ 5.7 s lên 10.1 s. Test 8 câu hỏi về các sự kiện rõ ràng, nên chưa đủ để xác nhận mức cải thiện thấy trên dev. Nếu chi phí là ưu tiên, `agent_bm25_k3` là lựa chọn rẻ hơn với kết quả dev gần tương đương (27/29 câu đúng).
  - Bộ test nhỏ, mỗi cấu hình chạy một lần; model không hoàn toàn tất định dù `temperature=0` (ví dụ BM25 k3 từ chối đúng u01 ở v3 nhưng không ở v2 dù cùng retrieval).
  - Rerank dùng cùng model với agent; chưa thử reranker chuyên dụng hoặc RRF có trọng số.
  - Benchmark dev và các câu thiếu bằng chứng do cùng người viết; câu hỏi Day 6 có thể chịu ảnh hưởng bởi cách diễn đạt evidence, có lợi cho BM25.
  - Index Day 6 bỏ thân bảng và nhãn biểu đồ, nên câu hỏi về bảng hay biểu đồ sẽ bị từ chối thay vì được trả lời.
  - Citation hợp lệ được chấm bằng cách đọc chunk; một số khẳng định phụ dài (ví dụ q05, q24) được kiểm theo chunk gold đã trích, không kiểm từng câu một.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
