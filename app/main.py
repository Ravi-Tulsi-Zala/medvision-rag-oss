"""
FastAPI Backend — MedVision RAG (Vertex AI edition)

Key change from previous version:
- Removed model preloading at startup (no local model to load)
- Startup is now instant — just validates env vars
- All inference happens via HTTP calls to Vertex AI
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate Vertex AI config on startup. Fast — no model loading."""
    required = ["GCP_PROJECT_ID", "GCP_LOCATION", "VERTEX_ENDPOINT_ID"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        logger.warning(f"Missing env vars: {missing}. /analyze will fail until set.")
    else:
        logger.info("Vertex AI config present. Ready to serve requests.")
    yield


app = FastAPI(
    title="MedVision RAG API",
    description="MedGemma via Vertex AI · FAISS RAG · LangGraph",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class ImageFindings(BaseModel):
    study_type: str
    anatomy:    str
    findings:   str
    impression: str


class AnalysisResponse(BaseModel):
    image_findings: ImageFindings
    context:        str
    summary:        str
    context_used:   str
    answer:         str
    red_flags:      list[str]
    recommendation: str


class HealthResponse(BaseModel):
    status:   str
    version:  str
    backend:  str
    endpoint: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="3.0.0",
        backend="Vertex AI",
        endpoint=os.getenv("VERTEX_ENDPOINT_ID", "not set"),
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    image_bytes = None
    if file:
        if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(status_code=422, detail="Only JPG/PNG supported.")
        image_bytes = await file.read()

    from app.graph import pipeline
    try:
        result = pipeline.invoke({
            "question":    question,
            "image_bytes": image_bytes,
            "findings":    None,
            "context":     None,
            "answer":      None,
        })
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    findings = result.get("findings", {})
    answer   = result.get("answer", {})

    return AnalysisResponse(
        image_findings=ImageFindings(**findings),
        context=result.get("context", ""),
        summary=answer.get("summary", ""),
        context_used=answer.get("context_used", ""),
        answer=answer.get("answer", ""),
        red_flags=answer.get("red_flags", []),
        recommendation=answer.get("recommendation", ""),
    )
