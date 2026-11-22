# ViCare – Trợ lý hỏi đáp sức khỏe tiếng Việt (RAG) – AI Challenge 24h

Bài làm cho **AI Challenge 24h** (AI4LIFE – ĐH Bách Khoa Hà Nội & Rikkei). Dựa trên notebook ViCare của ban tổ chức (ViHealthQA + multilingual-e5 + ChromaDB + Ollama qwen2.5:3b), nhóm giữ nguyên baseline và **thêm phần cải tiến, đo bằng đúng cách chấm của ban tổ chức** (cùng 100 câu `dev`, Recall@K, Judge@100, tỷ lệ từ chối).

> ViCare là sản phẩm DEMO học thuật, **không phải công cụ tư vấn y tế**.

## Kết quả (100 câu dev, Colab T4)

| Chỉ số | Baseline (BTC) | Cải tiến | Thay đổi |
|---|---|---|---|
| Recall@1 | 36% | 48% | +12 |
| Recall@3 | 46% | 59% | +13 |
| Recall@5 | 50% | 64% | +14 |
| Recall@10 | 54% | 67% | +13 |
| Judge@100 | 76% | 95% | +19 |
| Tỷ lệ từ chối | 24% | 3% | −21 |

Recall giống hệt giữa các lần chạy; Judge dao động nhẹ (baseline 75–76%, cải tiến 94–95%) vì mô hình sinh không hoàn toàn tất định. Judge là mô hình 3B nên có xu hướng lạc quan.

## Phần cải tiến
- **Hybrid retrieval:** BM25 (âm tiết + bigram tiếng Việt) kết hợp vector `multilingual-e5-base`, hợp nhất bằng Reciprocal Rank Fusion.
- **Reranker:** `BAAI/bge-reranker-v2-m3` xếp hạng lại top-20 để chọn top-5.
- **Prompt:** trả lời từ phần liên quan nhất khi có tài liệu liên quan, chỉ từ chối khi không có tài liệu nào liên quan. Cổng từ chối vẫn dùng cosine của e5-small với `MIN_SCORE` gốc.
- **Đóng gói:** `vicare_package/` (`rag.py`, `api.py` FastAPI, `ui.py` Gradio).
- Đã thử nhưng không giúp thêm (chi tiết trong `KET_QUA_AI_Challenge.md`): đổi chunk size, thêm embedding/reranker khác, HyDE, rerank theo chunk, fine-tune embedding, LLM 7b.

## Cách chạy
1. Mở `ViCare_Bootcamp24h_IMPROVED.ipynb` trên Google Colab, chọn Runtime **T4 GPU**.
2. Bấm **Chạy tất cả** (khoảng 30 phút). Mục 6 và 8 lưu baseline, mục 8B chạy cải tiến, mục 11 in bảng so sánh, mục 12 kiểm thử gói đóng gói.
3. Muốn bấm thử giao diện: chạy cell mục 9 (Gradio) và mở link `gradio.live`.

Chạy gói đóng gói riêng trên máy: xem [`vicare_package/README.md`](vicare_package/README.md).

## Cấu trúc
```
ViCare_Bootcamp24h_IMPROVED.ipynb   # notebook: baseline BTC (mục 0–8) + cải tiến (8B) + so sánh (11) + đóng gói (12)
vicare_package/                  # rag.py, api.py, ui.py, test_rag.py, requirements.txt, README.md
KET_QUA_AI_Challenge.md          # báo cáo chi tiết, các thử nghiệm và ghi chú trung thực
```

## Hạn chế
- Chỉ đánh giá trên 100 câu dev; Judge là LLM 3B nên chưa thay thế đánh giá của con người.
- Kho tài liệu là ViHealthQA, có thể lỗi thời; không dùng để chẩn đoán, kê đơn hay đưa quyết định y tế.
