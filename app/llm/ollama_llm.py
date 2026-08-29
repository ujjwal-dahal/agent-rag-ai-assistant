from langchain_ollama import ChatOllama
from app.config.config import env_file

# --------------------------------------------------
# 1. Ollama Server URL
# --------------------------------------------------

OLLAMA_BASE_URL = env_file.OLLAMA_BASE_URL


# --------------------------------------------------
# 2. Create Ollama LLM
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)