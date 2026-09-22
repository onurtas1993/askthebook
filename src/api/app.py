"""Thin synchronous API. One model-processing request at a time."""

from contextlib import contextmanager
import logging
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.documents.service import DocumentStore
from src.rag.pipeline import ask


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("question", "document_id", "model")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Must not be blank.")
        return value


def create_app(config_path: Path) -> FastAPI:
    config_path = config_path.resolve()
    store = DocumentStore(config_path)
    lock = Lock()
    app = FastAPI(title="AskTheBook", version="1.0", docs_url=None, redoc_url=None)
    app.state.processing_lock = lock

    @contextmanager
    def processing():
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Backend is busy. Retry after the current request finishes.")
        try:
            yield
        except HTTPException:
            raise
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (OSError, RuntimeError) as error:
            logging.exception("Local processing failed")
            raise HTTPException(503, str(error)) from error
        finally:
            lock.release()

    @app.get("/documents")
    def documents():
        return {"documents": store.list()}

    @app.post("/documents", status_code=201)
    def add_document(file: UploadFile):
        with processing():
            return store.prepare(file.file, file.filename or "")

    @app.post("/ask")
    def answer(request: Question):
        with processing():
            result = ask(request.question, config_path=config_path,
                         index_dir=store.index_for(request.document_id),
                         model=request.model, top_k=request.top_k)
            # Clients receive evidence, not the server's filesystem paths.
            return {"document_id": request.document_id,
                    **{key: result[key] for key in ("question", "answer", "sources", "warnings", "model_requested", "model_returned", "finish_reason", "usage")}}

    return app
