"""FastAPI application entry point for LunorAI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router


app = FastAPI(
    title="LunorAI",
    description="Mini AI Knowledge Assistant with RAG",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    documents_router,
)

app.include_router(
    chat_router,
)


@app.get("/")
def root() -> dict:
    """Return basic API information."""

    return {
        "name": "LunorAI",
        "description": "Mini AI Knowledge Assistant with RAG",
        "status": "running",
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""

    return {
        "status": "healthy",
    }