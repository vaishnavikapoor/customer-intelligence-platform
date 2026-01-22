# Customer Intelligence Platform (RAG)

A Retrieval-Augmented Generation (RAG) system for answering customer complaint queries with **grounded, source-backed responses**.

Includes:
- **Full RAG** → high-quality retrieval + generation  
- **RAG Lite** → lightweight, free-tier deployable version

---

## Tech Stack

- Python
- FastAPI (Backend)
- Streamlit (UI)
- FAISS (Vector Store)
- SentenceTransformers (Embeddings)
- HuggingFace Models
- SQL (Complaint Data Storage)
- LLM APIs / Local Models

---

## Demo – Streamlit (Full RAG)

### In-Scope Queries

![Query 1](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/rag_full/Streamlit/screenshots/Screenshot%202026-01-23%20005322.png)
![Query 2](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/rag_full/Streamlit/screenshots/Screenshot%202026-01-23%20005502.png)

### Out-of-Scope Handling

![Out of Scope](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/rag_full/Streamlit/screenshots/Screenshot%202026-01-23%20005404.png)

---

## Demo – FastAPI

![API Docs](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/rag_full/FastAPI/Screenshots/Screenshot%202026-01-23%20004646.png)
![API Response](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/rag_full/FastAPI/Screenshots/Screenshot%202026-01-23%20004826.png)

---

## Why This Project Matters

- Prevents hallucination via document grounding  
- Returns traceable sources  
- Explicit out-of-scope handling  
- Backend + frontend separation  
- Scales from prototype → production  

---

## Author

Vaishnavi Kapoor
