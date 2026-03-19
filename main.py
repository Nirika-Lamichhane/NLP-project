from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from registry import registry
from pipeline import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once at startup — loads all models into memory
    registry.initialize()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        results = run_pipeline(req.url)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"status": "ready" if registry.is_ready() else "loading"}