from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, ToolMessage

from app.graph.ollama_agent import graph
from app.schemas.response import AssistantResponse
from app.api.ollama import router as ollama_router


# --------------------------------------------------
# 1. FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="AI Assistant API",
    description="RAG and Tool Calling AI Assistant",
    version="1.0.0",
)


# --------------------------------------------------
# 2. Register Ollama Router
# --------------------------------------------------

app.include_router(ollama_router)


# --------------------------------------------------
# 3. Request Schema
# --------------------------------------------------

class ChatRequest(BaseModel):

    question: str


# --------------------------------------------------
# 4. Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Assistant API is running."
    }


# --------------------------------------------------
# 5. Main Chat Endpoint
# --------------------------------------------------

@app.post(
    "/chat",
    response_model=AssistantResponse
)
def chat(request: ChatRequest):

    # --------------------------------------------------
    # Invoke LangGraph
    # --------------------------------------------------

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=request.question
                )
            ]
        }
    )

    messages = result["messages"]


    # --------------------------------------------------
    # Find which tool was used
    # --------------------------------------------------

    tool_used = None

    for message in messages:

        if isinstance(message, ToolMessage):

            tool_used = message.name

            break


    # --------------------------------------------------
    # Determine whether RAG source was used
    # --------------------------------------------------

    source_used = tool_used == "rag_search"


    # --------------------------------------------------
    # Get final AI response
    # --------------------------------------------------

    final_message = messages[-1]

    answer = final_message.content


    # --------------------------------------------------
    # Return structured response
    # --------------------------------------------------

    return AssistantResponse(
        answer=answer,
        source_used=source_used,
        tool_used=tool_used,
    )