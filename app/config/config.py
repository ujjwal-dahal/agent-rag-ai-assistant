from dotenv import load_dotenv
import os
load_dotenv()


class ENVIRONMENT_FILE:
  def __init__(self):
    self.MODEL_API_KEY = os.getenv("GROQ_API_KEY")
    self.MODEL_NAME = os.getenv("MODEL_NAME")
    self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


env_file = ENVIRONMENT_FILE()
  
  