import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

with open(os.path.join(ASSETS_DIR, "df_chunks.pkl"), "rb") as f:
    df_chunks = pickle.load(f)

index = faiss.read_index(os.path.join(ASSETS_DIR, "faiss.index"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HF_TOKEN")
)

def retrieve(query, k=4):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = df_chunks.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    # Filter weak matches
    results = results[results["distance"] < 1.2]

    return results

def format_sources(results):
    if results.empty:
        return []
    return results.apply(
        lambda row: f"Complaint {row['complaint_id']} (chunk {row['chunk_id']})",
        axis=1
    ).tolist()

def generate_answer(query, results):
    if results.empty:
        return "No relevant complaints found for this query."

    context = "\n\n".join(results["text"].astype(str).tolist())

    prompt = f"""
You are a customer support analyst.

Context:
{context}

Question:
{query}

Answer in 3–5 bullet points.
"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3
    )

    return response.choices[0].message["content"]

def answer_with_sources(query, k):
    results = retrieve(query, k)
    answer = generate_answer(query, results)

    return {
        "question": query,
        "answer": answer,
        "sources": format_sources(results)
    }
