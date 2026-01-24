# Customer Intelligence Platform (RAG)

A **Retrieval-Augmented Generation (RAG)** system for answering customer complaint queries with **grounded, source-backed responses**.

The system retrieves relevant complaint snippets using vector similarity search and generates concise answers strictly based on retrieved context.

---
## Live Demo (Hugging Face Spaces)

**Live App:** https://huggingface.co/spaces/vaishnavikapoor/customer-intelligence-rag

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

![Out of Scope](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/Screenshots/Screenshot%202026-01-25%20021851.png)

---

## Demo – Streamlit

### In-Scope Queries

![Query 1](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/Screenshots/Screenshot%202026-01-23%20005322.png) 

### Out-of-Scope Handling

![Out of Scope](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/Screenshots/Screenshot%202026-01-23%20005404.png)

---

## Demo – FastAPI

![API Docs](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/Screenshots/Screenshot%202026-01-23%20004646.png)  
![API Response](https://github.com/vaishnavikapoor/customer-intelligence-platform/blob/main/Screenshots/Screenshot%202026-01-23%20004826.png)

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
