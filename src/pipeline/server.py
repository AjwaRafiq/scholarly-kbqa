import sys
import time
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline.kbqa_pipeline import ScholarlyKBQA, DEFAULT_CONFIG

app = FastAPI(title="Scholarly KBQA API", version="1.0")

print("Loading KBQA pipeline...")
pipeline = ScholarlyKBQA(DEFAULT_CONFIG)
print("Pipeline ready!")

class QuestionRequest(BaseModel):
    question: str
    verbose: bool = False

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    source: str
    evidence: list
    latency_ms: float

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    start = time.time()
    try:
        result = pipeline.answer(req.question, verbose=req.verbose)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start) * 1000

    return AnswerResponse(
        question=req.question,
        answer=result.get("natural_answer", result.get("answer", "No answer found")),
        confidence=result["confidence"],
        source=result["source"],
        evidence=result["evidence"],
        latency_ms=round(latency, 1)
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
