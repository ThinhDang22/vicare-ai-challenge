"""API REST cho ViCare (FastAPI).   Chạy:  uvicorn api:app --host 0.0.0.0 --port 8000

POST /ask  {"question": "...", "k": 5}  ->  {"answer", "sources"}
GET  /health
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import ViCare

bot: ViCare | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    bot = ViCare()          # nạp một lần khi khởi động
    yield


app = FastAPI(title="ViCare API (DEMO học thuật, không tư vấn y tế)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok" if bot is not None else "loading"}


@app.post("/ask")
def ask(req: AskRequest):
    result = bot.ask(req.question, req.k)
    return {"answer": result["answer"], "sources": result["sources"]}
