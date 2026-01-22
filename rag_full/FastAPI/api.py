from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import answer_with_sources

app = FastAPI(title="Customer Intelligence API", version="1.0")

class QueryRequest(BaseModel):
    question: str
    k: int = 4

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok", "message": "Customer Intelligence API is running"}

@app.post("/ask", response_model=QueryResponse)
def ask(query: QueryRequest):
    result = answer_with_sources(query.question, query.k)
    return result
