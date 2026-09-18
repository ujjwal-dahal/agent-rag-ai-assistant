from fastapi import FastAPI
from pydantic import BaseModel

from app.api.ollama import router as ollama_router
from app.graph.ollama_agent import run_agent
from app.schemas.response import AssistantResponse


app = FastAPI(
    title="AI Assistant API",
    description="Production-ready AI Assistant",
    version="2.0.0",
)


app.include_router(
    ollama_router
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat", response_model=AssistantResponse)
def chat(request: ChatRequest):
    return run_agent(request.question)


@app.get("/")
def root():

    return {
        "message": "AI Assistant API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }