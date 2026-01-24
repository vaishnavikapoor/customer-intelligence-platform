import os
import pickle
import logging
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_ENGINE")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PKL_PATH = os.path.join(BASE_DIR, "df_chunks.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set")


def load_dataframe(path: str) -> pd.DataFrame:
    logger.info("Loading dataframe from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_faiss_index(path: str):
    logger.info("Loading FAISS index from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing FAISS index: {path}")
    return faiss.read_index(path)


df_chunks = load_dataframe(PKL_PATH)
index = load_faiss_index(INDEX_PATH)

embedder = SentenceTransformer(EMBEDDING_MODEL)
llm_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)

logger.info("Models and data loaded successfully.")


def retrieve(query: str, k: int = 4) -> pd.DataFrame:
    """
    Retrieve top-k relevant chunks from FAISS index
    """
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = df_chunks.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    # Distance thresholding (tunable)
    results = results[results["distance"] < 1.2]

    logger.info("Retrieved %d chunks for query.", len(results))
    return results

def format_sources(results: pd.DataFrame) -> List[str]:
    if results.empty:
        return []
    return [
        f"Complaint {row['complaint_id']} (chunk {row['chunk_id']})"
        for _, row in results.iterrows()
    ]

def build_prompt(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
You are a financial customer support analyst.

Use ONLY the context below to answer the question.
If insufficient info is available, respond accordingly.

Context:
{context}

Question:
{query}

Answer in clear, concise bullet points (3–5 max).
"""

def generate_answer(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return "Not enough relevant data found to answer this query."

    context_chunks = results["text"].astype(str).tolist()
    prompt = build_prompt(query, context_chunks)

    response = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a professional financial assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.3
    )

    return response.choices[0].message["content"]

def answer_with_sources(query: str, k: int = 4) -> Dict:
    """
    Full RAG pipeline
    """
    results = retrieve(query, k)
    answer = generate_answer(query, results)
    sources = format_sources(results)

    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }
