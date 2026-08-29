from fastapi import APIRouter
from pydantic import BaseModel

from app.llm.ollama_llm import llm


# ==================================================
# Router
# ==================================================

router = APIRouter(
    prefix="/ollama",
    tags=["Ollama"],
)


# ==================================================
# Request Schema
# ==================================================

class OllamaRequest(BaseModel):

    question: str


# ==================================================
# Ollama Endpoint
# ==================================================

@router.post("/chat")
def ollama_chat(request: OllamaRequest):

    response = llm.invoke(
        request.question
    )

    return {
        "question": request.question,
        "model": "qwen2.5:1.5b",
        "provider": "Ollama",
        "answer": response.content,
    }