"""HTTP interface for the Southeast University assistant."""
import logging
import os
import sqlite3
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from zipfile import BadZipFile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.freshness import potentially_outdated
from src.runtime import create_retriever, create_service, save_feedback
from src.service import source_url
from src.store import INDEX_FILE, index_directory, load_index

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
logger = logging.getLogger(__name__)
app = FastAPI(title="Chatbot-SEU")
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])


class Turn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=5)
    language: Literal["auto", "en", "bn", "banglish"] = "auto"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value):
        if not value.strip():
            raise ValueError("A question is required.")
        return value.strip()


class Citation(BaseModel):
    id: int
    title: str | None = None
    url: str | None = None
    page: int | None = None
    source_id: str | None = None
    admission_session: str | None = None
    potentially_outdated: bool = True


class ChatResponse(BaseModel):
    request_id: UUID
    answer: str
    route: str
    citations: list[Citation]
    retrieved_count: int
    elapsed_ms: float
    intents: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    request_id: UUID
    helpful: bool


@lru_cache(maxsize=1)
def get_retriever():
    return create_retriever(load_index(index_directory()))


class LazyRetriever:
    def search(self, question, limit):
        return get_retriever().search(question, limit)

    def search_support(self, topic):
        return get_retriever().search_support(topic)

    def search_program_fees(self, programs, fee_type=None):
        return get_retriever().search_program_fees(programs, fee_type)


@lru_cache(maxsize=1)
def get_service():
    return create_service(LazyRetriever())


@app.get("/health")
def health():
    return {"status": "ok", "index_present": (index_directory() / INDEX_FILE).is_file()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = get_service().chat(request.question,
                                    [(turn.question, turn.answer) for turn in request.history],
                                    request.language)
        return ChatResponse(request_id=uuid4(), **asdict(result))
    except Exception as exc:
        logger.error("Chat request failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Chat is temporarily unavailable.") from None


@app.get("/sources")
def sources():
    from src.store import read_manifest

    try:
        documents = read_manifest(index_directory())["documents"]
    except (OSError, ValueError, BadZipFile, KeyError) as exc:
        logger.error("Source manifest unavailable (%s)", type(exc).__name__)
        raise HTTPException(503, "Sources are temporarily unavailable.") from None
    summaries = {}
    for document in documents:
        metadata = document["metadata"]
        key = (metadata.get("source_url"), metadata.get("title"))
        summaries[key] = {
            "url": source_url(key[0]), "title": key[1], "category": metadata.get("category"),
            "source_id": metadata.get("id"), "admission_session": metadata.get("admission_session"),
            "program": metadata.get("program"), "fee_type": metadata.get("fee_type"),
            "academic_session": metadata.get("academic_session"), "effective_date": metadata.get("effective_date"),
            "document_type": metadata.get("document_type"), "superseded_by": metadata.get("superseded_by"), "retrieved_at": metadata.get("retrieved_at"),
            "updated_at": metadata.get("updated_at"), "potentially_outdated": potentially_outdated(metadata.get("updated_at")),
        }
    return list(summaries.values())


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        save_feedback(request.request_id, request.helpful)
    except (OSError, sqlite3.Error) as exc:
        logger.error("Feedback storage failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Feedback is temporarily unavailable.") from None
    return {"saved": True}
