from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_engine import answer_with_sources

app = FastAPI(
    title="Customer Intelligence RAG API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    k: int = Field(default=4, ge=1, le=10)

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "RAG API"}

@app.post("/ask", response_model=QueryResponse)
def ask(query: QueryRequest):
    try:
        return answer_with_sources(query.question, query.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
