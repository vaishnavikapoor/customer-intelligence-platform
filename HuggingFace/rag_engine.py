import os
import pickle
import logging
from functools import lru_cache
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_ENGINE")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, "df_chunks.pkl")
INDEX_PATH = os.path.join(BASE_DIR, "faiss.index")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable not set")

@lru_cache(maxsize=1)
def get_df_chunks() -> pd.DataFrame:
    logger.info("Loading DataFrame...")
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"Missing file: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def get_faiss_index():
    logger.info("Loading FAISS index...")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    import torch

    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    model.eval()
    return model

from groq import Groq

@lru_cache(maxsize=1)
def get_llm_client():
    logger.info("Initializing Groq client...")
    return Groq(api_key=GROQ_API_KEY)

def retrieve(query: str, k: int = 3) -> pd.DataFrame:
    embedder = get_embedder()
    df_chunks = get_df_chunks()
    index = get_faiss_index()

    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)

    if indices[0][0] == -1:
        return pd.DataFrame()

    results = df_chunks.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    # optional threshold filter
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
If insufficient info is available, say so clearly.

Context:
{context}

Question:
{query}

Answer in clear, concise bullet points (4–8 max).
"""

import traceback

def generate_answer(query: str, results: pd.DataFrame) -> str:
    if results.empty:
        return "Not enough relevant data found to answer this query."

    context_chunks = results["text"].astype(str).tolist()
    context_chunks = context_chunks[:3]  # prevent context overflow
    prompt = build_prompt(query, context_chunks)

    try:
        client = get_llm_client()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[
                {"role": "system", "content": "You are a professional financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq LLM failed: {e}")
        return f"Groq error: {str(e)}"

def answer_with_sources(query: str, k: int = 3) -> Dict:
    results = retrieve(query, k)
    answer = generate_answer(query, results)
    sources = format_sources(results)
    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }