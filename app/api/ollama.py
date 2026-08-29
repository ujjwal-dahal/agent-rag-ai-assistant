from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

import os
import asyncio
import time
from collections import defaultdict


router = APIRouter()


# ==================================================
# Configuration
# ==================================================

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://ollama:11434"

MAX_RETRIES = 3
RETRY_DELAY = 1

RATE_LIMIT = 10
RATE_WINDOW = 60

CACHE_TTL = 300


# ==================================================
# LLM Configuration
# ==================================================

ollama_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


groq_llm = None

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:

    groq_llm = ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        ),
        temperature=0,
        api_key=groq_api_key,
    )


# ==================================================
# Request Schema
# ==================================================

class OllamaChatRequest(BaseModel):

    question: str


# ==================================================
# System Prompt
# ==================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question clearly and accurately.

Rules:

- Give simple and direct answers.
- Do not invent facts.
- If you are unsure, say that you are unsure.
- Use short paragraphs or bullet points when useful.
- Do not mention internal tools, models, APIs,
  system prompts, or implementation details.
- Return only the final answer.
"""


# ==================================================
# Cache
# ==================================================

response_cache = {}


def get_cached_response(question: str):

    cached = response_cache.get(question)

    if cached is None:

        return None

    answer, timestamp = cached

    if time.time() - timestamp > CACHE_TTL:

        del response_cache[question]

        return None

    return answer


def set_cached_response(
    question: str,
    answer: str,
):

    response_cache[question] = (
        answer,
        time.time(),
    )


# ==================================================
# Rate Limiting
# ==================================================

request_history = defaultdict(list)


def check_rate_limit(client_ip: str):

    current_time = time.time()

    request_history[client_ip] = [
        timestamp
        for timestamp in request_history[client_ip]
        if current_time - timestamp < RATE_WINDOW
    ]

    if len(request_history[client_ip]) >= RATE_LIMIT:

        return False

    request_history[client_ip].append(
        current_time
    )

    return True


# ==================================================
# Ollama Invocation
# ==================================================

async def call_ollama(question: str):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = await ollama_llm.ainvoke(
                [
                    SystemMessage(
                        content=SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=question
                    ),
                ]
            )

            return response.content

        except Exception as error:

            last_error = error

            if attempt < MAX_RETRIES - 1:

                await asyncio.sleep(
                    RETRY_DELAY * (attempt + 1)
                )

    raise last_error


# ==================================================
# Groq Fallback
# ==================================================

async def call_groq(question: str):

    if groq_llm is None:

        raise RuntimeError(
            "Groq fallback is not configured."
        )

    response = await groq_llm.ainvoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=question
            ),
        ]
    )

    return response.content


# ==================================================
# Chat Endpoint
# ==================================================

@router.post("/ollama/chat")
async def ollama_chat(
    request: OllamaChatRequest,
    http_request: Request,
):

    start_time = time.perf_counter()

    question = request.question.strip()


    # --------------------------------------------------
    # Input Validation
    # --------------------------------------------------

    if not question:

        return JSONResponse(
            status_code=400,
            content={
                "error": "Question cannot be empty."
            },
        )


    # --------------------------------------------------
    # Rate Limiting
    # --------------------------------------------------

    client_ip = (
        http_request.client.host
        if http_request.client
        else "unknown"
    )

    if not check_rate_limit(client_ip):

        return JSONResponse(
            status_code=429,
            content={
                "error": (
                    "Too many requests. "
                    "Please try again later."
                ),
            },
        )


    # --------------------------------------------------
    # Response Cache
    # --------------------------------------------------

    cached_answer = get_cached_response(
        question
    )

    if cached_answer:

        latency = (
            time.perf_counter() - start_time
        )

        return {
            "answer": cached_answer,
            "model": OLLAMA_MODEL,
            "provider": "cache",
            "cached": True,
            "latency_ms": round(
                latency * 1000,
                2,
            ),
        }


    # ==================================================
    # Primary Model: Ollama
    # ==================================================

    try:

        answer = await call_ollama(
            question
        )

        set_cached_response(
            question,
            answer,
        )

        latency = (
            time.perf_counter() - start_time
        )

        return {
            "answer": answer,
            "model": OLLAMA_MODEL,
            "provider": "ollama",
            "cached": False,
            "latency_ms": round(
                latency * 1000,
                2,
            ),
        }


    except Exception:

        # Ollama failed.
        # Continue to fallback model.


        # ==================================================
        # Fallback Model: Groq
        # ==================================================

        try:

            answer = await call_groq(
                question
            )

            set_cached_response(
                question,
                answer,
            )

            latency = (
                time.perf_counter() - start_time
            )

            return {
                "answer": answer,
                "model": os.getenv(
                    "GROQ_MODEL",
                    "llama-3.1-8b-instant",
                ),
                "provider": "groq_fallback",
                "cached": False,
                "latency_ms": round(
                    latency * 1000,
                    2,
                ),
            }


        except Exception:

            # ==================================================
            # Graceful Degradation
            # ==================================================

            return JSONResponse(
                status_code=503,
                content={
                    "answer": (
                        "The AI service is temporarily "
                        "unavailable. Please try again later."
                    ),
                    "provider": "unavailable",
                    "cached": False,
                },
            )