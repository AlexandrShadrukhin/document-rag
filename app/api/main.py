from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.generation.llm import LLMUnavailableError
from app.runtime import ApplicationContainer
from app.schemas import IndexRequest, IndexResponse, QueryRequest, QueryResponse

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
container = ApplicationContainer(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    container.close()


app = FastAPI(
    title="Document RAG API",
    version="0.1.0",
    description="Local hybrid RAG backend for PDF and TXT documents.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest) -> IndexResponse:
    try:
        stats = await run_in_threadpool(container.index, request.path, False)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logging.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {error}") from error
    return IndexResponse(
        files_discovered=stats.files_discovered,
        files_indexed=stats.files_indexed,
        files_skipped=stats.files_skipped,
        pages_parsed=stats.pages_parsed,
        chunks_created=stats.chunks_created,
        embedding_batches=stats.embedding_batches,
        timings=stats.timings(),
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        service = container.rag_service(request.mode)
        return await run_in_threadpool(service.answer, request.query.strip())
    except LLMUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logging.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {error}") from error
