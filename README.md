# Customer Intelligence Platform (RAG)

A **Retrieval-Augmented Generation (RAG)** system for answering customer complaint queries with **grounded, source-backed responses**.

The system retrieves relevant complaint snippets using vector similarity search and generates concise answers strictly based on retrieved context.

---
## Live Demo (Hugging Face Spaces)

**Live App:** [text](https://huggingface.co/spaces/vaishnavikapoor/customer-intelligence-rag)

## Tech Stack

- Python  
- FAISS (Vector Store)  
- SentenceTransformers (Embeddings)  
- Hugging Face Inference API  
- FastAPI (Backend)  
- Streamlit (Frontend)  

---

## Testing & Deployment

- The project was **tested locally** using:
  - Streamlit for the user interface
  - FastAPI for backend API validation

- The application was deployed using **Hugging Face Spaces**, ensuring:
  - Lightweight assets
  - Free-tier compatibility
  - Fast startup and stable inference

---

## Demo – HuggingFace Space

Example:

![Out of Scope](Screenshots/screenshot3.png)

---

## Demo – Streamlit

### In-Scope Queries

![Query 1](Screenshots/screenshot1.png) 

### Out-of-Scope Handling

![Out of Scope](Screenshots/screenshot3.png)

---

## Demo – FastAPI

![API Docs](Screenshots/screenshot4.png)  
![API Response](Screenshots/screenshot5.png)

---

## Why This Project Matters

- Prevents hallucination through document grounding  
- Returns traceable sources with each answer  
- Explicit handling of out-of-scope queries  
- Clean separation between retrieval, generation, and serving layers  
- Demonstrates an end-to-end, deployable RAG system  

---

## Author

Vaishnavi Kapoor
