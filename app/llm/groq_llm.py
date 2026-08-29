
from langchain_groq import ChatGroq

from app.config.config import env_file


# --------------------------------------------------
# Create Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model=env_file.MODEL_NAME,
    temperature=0,
    max_tokens=1024,
    api_key=env_file.MODEL_API_KEY,
)