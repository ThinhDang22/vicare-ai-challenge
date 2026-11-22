"""Giao diện chat Gradio cho ViCare.   Chạy:  python ui.py   (mở http://127.0.0.1:7860)"""
import gradio as gr

from rag import ViCare

bot = ViCare()


def chat_fn(message, history):
    result = bot.ask(message)
    refs = []
    for i, s in enumerate(result["sources"], 1):
        url = str(s["url"] or "")
        refs.append(f"[{i}] {url if url.startswith(('http://', 'https://')) else s['doc_id']}")
    return result["answer"] + ("\n\n**Tài liệu tham khảo:**\n" + "\n".join(refs) if refs else "")


demo = gr.ChatInterface(
    fn=chat_fn,
    title="ViCare – Hỏi đáp sức khỏe (DEMO HỌC THUẬT, không tư vấn y tế)",
    description="Không nhập thông tin cá nhân hoặc dùng để đưa ra quyết định y tế.",
)

if __name__ == "__main__":
    demo.launch()
