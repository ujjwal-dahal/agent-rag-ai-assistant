from fastapi import FastAPI

from app.api.ollama import router as ollama_router


app = FastAPI(
    title="AI Assistant API",
    description="Production-ready AI Assistant",
    version="2.0.0",
)


app.include_router(
    ollama_router
)


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