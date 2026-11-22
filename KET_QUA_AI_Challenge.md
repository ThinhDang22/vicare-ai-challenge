# [AI Challenge 24h] - ViCare: cải thiện truy xuất RAG

Notebook: ViCare_Bootcamp24h_IMPROVED.ipynb (chạy trên Colab T4, Ollama qwen2.5:3b)

## Cấu hình cải tiến (theo hướng BTC khuyến nghị)
- Hướng 2: kết hợp BM25 (từ + bigram âm tiết tiếng Việt) với embedding multilingual-e5-base, gộp bằng RRF
- Hướng 3: re-rank top-20 bằng cross-encoder BAAI/bge-reranker-v2-m3
- Đã thử và loại: gộp 3 embedding (e5-small, e5-base, bge-m3), tăng trọng số BM25, rerank top-40
- Giữ nguyên: chunk 200 từ/overlap 40, k=5, prompt trích dẫn [n], ngưỡng từ chối MIN_SCORE 0.86, qwen2.5:3b

## Kết quả (cùng 100 câu dev, seed 2026)
| Chỉ số | Trước | Sau | Thay đổi |
|---|---|---|---|
| Recall@1 | 36% | 48% | +12 điểm |
| Recall@3 | 46% | 59% | +13 điểm |
| Recall@5 | 50% | 64% | +14 điểm |
| Recall@10 | 54% | 67% | +13 điểm |
| Judge@30 | 83,3% | 83,3% | 0 |
| Tỷ lệ từ chối | 20% | 16,7% | -3,3 điểm |

## Lưu ý trung thực
- Judge do LLM 3B chấm, rất nhiễu (N=10 dao động 50-80%); đo ở N=30 thì không đổi.
- Chưa thử đổi prompt/k/model qwen2.5 1.5b/7b.
- Cell cuối notebook ghi đè `retrieve` và phụ thuộc cell thí nghiệm trước đó; chạy tuần tự trong cùng phiên.

## Thí nghiệm bổ sung: LLM và k (hướng 5), Judge@30, giám khảo cố định qwen2.5:3b
| Model | k | Judge@30 | Từ chối |
|---|---|---|---|
| qwen2.5:3b | 3 | 83,3% | 13,3% |
| qwen2.5:3b | 5 (cấu hình cuối) | 83,3% | 16,7% |
| qwen2.5:3b | 8 | 86,7% | 13,3% |
| qwen2.5:7b | 3 | 70,0% | 30,0% |
| qwen2.5:7b | 5 | 70,0% | 33,3% |
| qwen2.5:7b | 8 | 66,7% | 36,7% |

- qwen2.5:7b từ chối nhiều hơn (30-37%) nên Judge thấp hơn: giữ qwen2.5:3b.
- k=8 hơn k=5 đúng 1 câu trên 30 (+3,3 điểm), nằm trong nhiễu nên không đổi mặc định k=5.

## Thí nghiệm bổ sung: prompt (hướng 4), qwen2.5:3b, k=5, Judge@60, giám khảo cố định qwen2.5:3b
| Prompt | Judge@60 | 30 câu đầu | 30 câu sau | Từ chối |
|---|---|---|---|---|
| A: prompt gốc của BTC | 83,3% | 83,3% | 83,3% | 16,7% |
| B: thêm "có tài liệu liên quan thì trả lời phần liên quan nhất; chỉ từ chối khi không có tài liệu nào liên quan" | 96,7% | 96,7% | 96,7% | 3,3% |
- Kiểm tra an toàn: 8/8 câu ngoài phạm vi y tế (giá vàng, nấu ăn, bóng đá, code, chứng khoán...) vẫn bị từ chối.
- Cổng từ chối theo cosine (MIN_SCORE 0,86), dev, seed, cách chấm Judge và Recall đều giữ nguyên như BTC.
- Lưu ý: Judge do LLM 3B chấm nên có thể lạc quan; mức tăng nhất quán ở cả hai nửa dev nhưng vẫn là số tham khảo.
- File notebook mới nhất: ViCare_Bootcamp24h_IMPROVED.ipynb (SYSTEM_PROMPT = PROMPT_B được đặt ở cell cuối).

## Thí nghiệm bổ sung: chunk size / overlap (hướng 1), hybrid e5-base + BM25 (+ rerank top-20), 100 câu dev
| Chunk / overlap | Số chunk | Hybrid R@5 | Hybrid+rerank R@1 / R@3 / R@5 / R@10 |
|---|---|---|---|
| 100 / 20 | 16.331 | 58% | 48 / 59 / 60 / 64 |
| 150 / 30 | 12.601 | 55% | 49 / 60 / 67 / 69 |
| 200 / 40 (BTC, giữ nguyên) | 11.227 | 56% | 48 / 59 / 64 / 67 |
| 300 / 60 | 10.363 | 56% | 48 / 59 / 65 / 67 |
| 400 / 80 | 10.117 | 55% | 48 / 59 / 65 / 67 |
- Khi đã có reranker (chấm lại trên toàn văn bài), kích thước chunk ảnh hưởng rất ít. 150/30 nhỉnh nhất (+3 điểm R@5, +2 điểm R@10 so với 200/40) nhưng nằm trong mức dao động của 100 câu.
- Không đổi sang 150/30: phải dựng lại chỉ mục ChromaDB (dùng cho cổng từ chối) mà lợi ích không chắc chắn. Giữ 200/40.

## Thí nghiệm nâng cao truy xuất (không có cải thiện thêm)
Thử pool rerank 10/20/30, trộn điểm rerank với điểm RRF (hệ số 0 / 0,3 / 1), và các tổ hợp e5-base, bge-m3, e5-small với BM25.
Tốt nhất vẫn là hybrid e5-base + BM25, rerank thuần top-20 (R@1/3/5/10 = 48 / 59 / 63-64 / 67). Trộn thêm điểm RRF làm kém đi. Truy xuất đã chạm trần với bộ công cụ này.

## KẾT QUẢ CUỐI: Judge trên toàn bộ 100 câu dev (giám khảo cố định qwen2.5:3b)
| Cấu hình | Judge@100 | Từ chối |
|---|---|---|
| Baseline BTC (vector e5-small, prompt gốc) | 75,0% | 24,0% |
| Cuối (hybrid e5-base+BM25, rerank top-20, prompt B) | 94,0% | 4,0% |
Cùng với Recall@5 tăng từ 50% lên 64% (Recall@1/3/10: 36→48, 46→59, 54→67).
Notebook mới nhất: ViCare_Bootcamp24h_IMPROVED.ipynb.

## Thí nghiệm dài: embedding mới và reranker thứ hai (hướng 2, 3), 100 câu dev
- Embedding thử thêm: AITeamVN/Vietnamese_Embedding, Qwen/Qwen3-Embedding-0.6B. Qwen3-Embedding đứng riêng chỉ đạt R@1/3/5/10 = 37 / 46 / 50 / 54 (không hơn baseline).
- Tổ hợp tốt nhất sau rerank bge-reranker: e5b+vie+q3e+bm25 pool30 = 47 / 56 / 62 / 70; e5b+bge+vie+bm25 pool30 = 47 / 56 / 58 / 71. R@10 nhỉnh hơn 3-4 điểm nhưng R@1/3/5 không hơn cấu hình hiện tại (e5b+bm25, rerank top-20 = 48 / 59 / 63-64 / 67).
- Reranker AITeamVN/Vietnamese_Reranker kém hơn bge-reranker-v2-m3 rõ rệt (R@1/3/5 = 37-39 / 51-52 / 58 / 64-66) và kết hợp (ensemble) với bge cũng làm kém đi.
- Kết luận: không có cấu hình nào cải thiện Recall@5; giữ cấu hình cuối.

## Thí nghiệm: rerank theo chunk và HyDE (100 câu dev), không cải thiện
- Rerank theo chunk (trung bình chỉ 1,13 chunk/bài nên gần như giống rerank cả bài): max chunk = 47 / 57 / 63 / 66-67; cả bài (đang dùng) = 48 / 59 / 63 / 67. Không hơn.
- HyDE (qwen2.5:3b viết đoạn trả lời giả định để mở rộng câu hỏi): vector e5b + HyDE tụt từ 54% xuống 48% R@5, BM25 + HyDE tụt từ 47% xuống 42%. Tốt nhất sau rerank là 48 / 59 / 64 / 66, bằng cấu hình hiện tại (48 / 59 / 64 / 67). LLM 3B sinh nội dung giả định sai nên làm nhiễu truy xuất.

## Thử nghiệm fine-tune embedding e5-base trên split TRAIN (không dùng dev, 7.007 cặp câu hỏi - đáp án; đã loại cặp có đáp án trùng đáp án đúng của dev)
Loss InfoNCE đối xứng, batch 32, lr 2e-5, 2 epoch (~4 phút trên T4), đóng băng bảng từ vựng. Đo trên 100 câu dev (R@1 / R@3 / R@5 / R@10):
| Cấu hình | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| e5-base gốc, chỉ vector | 38 | 50 | 54 | 59 |
| e5-base fine-tune 1 epoch, chỉ vector | 39 | 56 | 58 | 66 |
| e5-base fine-tune 2 epoch, chỉ vector | 40 | 58 | 64 | 66 |
| fine-tune 2 epoch + 0,3 BM25 (không rerank) | 42 | 58 | 64 | 67 |
| fine-tune 2 epoch + BM25 + rerank top-20 | 46 | 58 | 61 | 67 |
| fine-tune 2 epoch + e5b + BM25 + rerank top-20 | 48 | 58 | 62 | 68 |
| huấn luyện thêm đến 5 epoch, chỉ vector | 42 | 59 | 61 | 67 |
| (cấu hình đang dùng: e5b + BM25 + rerank top-20) | 48 | 59 | 63-64 | 67 |
- Fine-tune nâng embedding đứng riêng thêm ~10 điểm R@5 (54 → 64), nhưng khi kết hợp với BM25 và reranker thì không vượt cấu hình đang dùng (R@5 vẫn ~64%, R@10 67-68%). Huấn luyện thêm epoch không tăng (R@5 61-62%).
- Reranker làm kém đi embedding đã fine-tune (reranker chưa được fine-tune cùng miền).


## Kiểm chứng lại & đóng gói
- Chạy lại toàn bộ `ViCare_Bootcamp24h_IMPROVED.ipynb` từ đầu trên Colab T4 (22/11/2026): Recall@1/3/5/10 = 36/46/50/54 → 48/59/64/67; Judge@100 75% → 94%; từ chối 25% → 4%. Số baseline ở trên đã cập nhật theo lần chạy này (các lần thử trước ghi 37/47/51/55 do dao động nhỏ giữa các lần chạy).
- Đóng gói theo gợi ý BTC: thư mục `vicare_package/` (`rag.py`, `api.py` FastAPI, `ui.py` Gradio, `README.md`). Hàm thuần đã kiểm thử; pipeline đóng gói chưa chạy end-to-end.
- Lần chạy lại cuối (có thêm mục 12 kiểm thử gói đóng gói): Recall giống hệt (36/46/50/54 → 48/59/64/67); Judge@100 75% → 95%, từ chối 25% → 3% (lần trước 94%/4%: dao động nhẹ của LLM). Gói `vicare_package` đo trên 100 câu dev cho Recall@1/3/5/10 = 48/59/64/67 (khớp notebook), API `/ask` trả 200 và câu ngoài lĩnh vực bị từ chối đúng.
- Đóng gói chạy thật trên Colab: `uvicorn api:app` (GET /health, POST /ask = 200) và `python ui.py` (Gradio, gọi qua gradio_client) đều trả lời đúng.
- Lần chạy lại sạch cuối cùng (Colab T4, chạy từ đầu bằng "Khởi động lại phiên và chạy tất cả", không lỗi): Recall@1/3/5/10 = 36/46/50/54 → 48/59/64/67; Judge@100 76% → 95%; từ chối 24% → 3%. Gói `vicare_package`: Recall 48/59/64/67, `uvicorn` và Gradio đều trả lời đúng.
