"""FastAPI backend for the sentiment+RAG demo.

Loads klue-bert-qwen-sentiment-v4 (classifier) and the news embedding index once
at startup, then serves POST /analyze for the Next.js frontend to call.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np

from rag_pipeline import SentimentRAG

app = FastAPI(title="SentiQuant RAG demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None


@app.on_event("startup")
def load_models():
    global rag
    rag = SentimentRAG()


class AnalyzeRequest(BaseModel):
    text: str
    top_k: int = 5


class ExplainRequest(BaseModel):
    text: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "loaded": rag is not None}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = rag.analyze(req.text, top_k=req.top_k)
    cases = result["similar_cases"].replace({np.nan: None}).to_dict(orient="records")
    historical_avg = {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                       for k, v in result["historical_avg"].items()}
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "similar_cases": cases,
        "historical_avg": historical_avg,
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    result = rag.analyze(req.text, top_k=req.top_k)
    explanation = rag.explain(req.text, result["label"], result["confidence"], result["similar_cases"])
    return {"explanation": explanation}
