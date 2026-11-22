# ViCare – bản đóng gói (rag.py · api.py · ui.py)

Đóng gói cấu hình cải tiến của notebook (mục 8B) thành mã chạy độc lập, theo gợi ý của BTC
("đóng gói: `rag.py`, `api.py` (FastAPI), `ui.py`").

| File | Vai trò |
|---|---|
| `rag.py` | Lõi RAG: corpus → chunk 200/40 → BM25 + e5-base → RRF → rerank bge-reranker-v2-m3 → cổng từ chối (e5-small, MIN_SCORE 0.86) → Ollama qwen2.5:3b |
| `api.py` | REST API FastAPI: `POST /ask`, `GET /health` |
| `ui.py` | Giao diện chat Gradio |
| `test_rag.py` | Kiểm thử nhanh các hàm thuần (chunk, tokenize, RRF, cảnh báo cấp cứu) |

## Chạy
```bash
pip install -r requirements.txt
ollama serve &            # và: ollama pull qwen2.5:3b
python test_rag.py        # kiểm thử nhanh
python ui.py              # giao diện http://127.0.0.1:7860
uvicorn api:app --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?"}'
```
Lần chạy đầu mã hoá toàn kho (vài phút trên GPU, lâu hơn trên CPU) và lưu vào `cache/`; các lần sau nạp lại ngay.

## Lưu ý
- Khác notebook: chỉ mục vector dùng numpy (tích vô hướng toàn kho) thay cho ChromaDB; kết quả tương đương vì ChromaDB cũng xếp theo cosine. Cổng từ chối vẫn dùng cosine của e5-small như BTC.
- Đã kiểm thử end-to-end trên Colab T4 (mục 12 của notebook): Recall@1/3/5/10 của gói = 48/59/64/67% trên 100 câu dev, khớp với mục 8B; `POST /ask` (qua TestClient) trả 200; câu ngoài lĩnh vực bị từ chối. Cũng đã chạy thật `uvicorn api:app` (/health và POST /ask trả 200, câu ngoài lĩnh vực bị từ chối) và `python ui.py` (Gradio phản hồi qua gradio_client) trên Colab.
- ViCare là sản phẩm DEMO học thuật, không phải công cụ tư vấn y tế.
