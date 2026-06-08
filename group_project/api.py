from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path

# Thêm root path để import được src/rag_engine
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_engine import generate_with_citation_dynamic

app = FastAPI(title="Drug Law RAG API", description="Backend API for retrieving and generating drug law questions with citation.")

# Cấu hình CORS để Chainlit UI ở port khác có thể gửi request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str # 'user' hoặc 'assistant'
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.3
    rerank_method: Optional[str] = "cross_encoder"

class SourceChunk(BaseModel):
    content: str
    score: float
    metadata: dict
    source: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    retrieval_source: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Chuyển chat_history từ Pydantic models sang dict list cho RAG engine
        history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        
        result = generate_with_citation_dynamic(
            query=request.query,
            chat_history=history,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            rerank_method=request.rerank_method
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
