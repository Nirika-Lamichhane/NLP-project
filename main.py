import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from registry import registry
from pipeline import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.initialize()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        data = run_pipeline(req.url)  # ← now returns { results, stats }
        return data                   # ← return directly, not wrapped
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ready" if registry.is_ready() else "loading"}