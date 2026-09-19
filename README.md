# LunorAI

**Mini AI Knowledge Assistant — Ask your documents. Get grounded answers.**

LunorAI is a lightweight Retrieval-Augmented Generation (RAG) assistant that lets users upload documents and ask questions about their content.

## Features

- Supports PDF, DOCX, TXT & Markdown
- Text extraction and intelligent chunking
- NVIDIA Nemotron embeddings
- FAISS vector similarity search
- NVIDIA LLM for grounded answers
- Source and page references
- Document upload and deletion
- Deployed frontend and backend

## Tech Stack

| Layer      | Technologies                      |
| ---------- | --------------------------------- |
| Frontend   | React, Vite, JavaScript, CSS      |
| Backend    | Python, FastAPI                   |
| RAG        | NVIDIA Embeddings, FAISS, NumPy   |
| Documents  | PyMuPDF, python-docx              |
| Deployment | Vercel + Render                   |

## RAG Pipeline

```text
Document
   ↓
Extract & Chunk
   ↓
NVIDIA Embeddings
   ↓
FAISS Vector Search
   ↓
Relevant Context
   ↓
NVIDIA LLM
   ↓
Answer + Sources
```

## Live Demo

- **Frontend:** https://lunor-ai-ai-knowledge-assistant.vercel.app/
- **Backend:** https://lunorai-ai-knowledge-assistant.onrender.com/

## Local Setup

```bash
git clone <repository-url>
cd LunorAI

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd ../frontend
npm install
npm run dev
```

## Environment Variables

### Backend `.env`

```env
NVIDIA_API_KEY=your_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=your_model
```

### Frontend `.env`

```env
VITE_API_URL=http://localhost:8000
```