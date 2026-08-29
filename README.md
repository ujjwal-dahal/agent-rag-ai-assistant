# AI Assistant

A containerized AI Assistant built using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), tool calling, structured JSON output, and local LLM inference using Ollama.

## 1. Project Overview

This project implements an AI Assistant capable of answering user questions, retrieving relevant information from a knowledge base, performing mathematical calculations, and returning structured responses through a FastAPI API.

The system combines a cloud-based LLM with local open-source LLM inference. LangGraph is used to manage the agent workflow and tool-calling process.

The application is containerized using Docker for consistent and reproducible deployment.

## 2. Features

- LLM integration using Groq
- Local LLM inference using Ollama
- Retrieval-Augmented Generation (RAG)
- Tool calling
- Calculator tool
- Vector-based document retrieval
- Structured JSON responses
- Prompt engineering
- FastAPI REST API
- Interactive Swagger API documentation
- Docker containerization
- LangGraph-based agent workflow

## 3. Technologies Used

| Technology      | Purpose                               |
| --------------- | ------------------------------------- |
| Python          | Application development               |
| FastAPI         | REST API development                  |
| LangChain       | LLM and tool integration              |
| LangGraph       | Agent workflow and tool orchestration |
| Groq            | Cloud LLM provider                    |
| Ollama          | Local LLM inference                   |
| Qwen 2.5 1.5B   | Local open-source LLM                 |
| RAG             | Knowledge retrieval                   |
| Vector Database | Storage and retrieval of embeddings   |
| Docker          | Containerization                      |
| Pydantic        | Request and response validation       |

## 4. System Architecture

The system follows an agent-based architecture.

```text
                         User
                           |
                           v
                    FastAPI REST API
                           |
             +-------------+-------------+
             |                           |
             v                           v
       AI Assistant API          Ollama Chat API
             |                           |
             v                           v
         LangGraph                 Ollama Server
             |                           |
             v                           v
        Groq LLM                  Qwen 2.5 1.5B
             |
             v
       Tool Calling
             |
       +-----+------+
       |            |
       v            v
   RAG Search    Calculator
       |
       v
  Vector Database
       |
       v
 Retrieved Context
       |
       v
     LLM
       |
       v
 Structured JSON Response
```

## 5. Agent Workflow

The LangGraph agent follows the following workflow:

```text
User Question
      |
      v
     LLM
      |
      v
Tool Required?
   /       \
 Yes        No
  |          |
  v          v
ToolNode    Final
  |
  v
Tool Result
  |
  v
  LLM
  |
  v
Final Answer
```

When the user asks a question that requires information from the knowledge base, the LLM calls the `rag_search` tool.

For mathematical questions, the LLM calls the `calculator` tool.

After receiving the tool result, the LLM generates the final answer.

## 6. RAG Pipeline

The RAG pipeline allows the assistant to answer questions using information stored in an external knowledge base.

The pipeline consists of the following stages:

```text
Documents
    |
    v
Document Ingestion
    |
    v
Document Chunking
    |
    v
Text Embeddings
    |
    v
Vector Database
    |
    v
Similarity Search
    |
    v
Relevant Context
    |
    v
LLM
    |
    v
Final Answer
```

### Document Ingestion

Documents are loaded into the application and prepared for retrieval.

### Chunking

Large documents are divided into smaller text chunks. Chunking allows the retrieval system to find only the relevant sections instead of processing the entire document.

### Embeddings

Each text chunk is converted into a numerical vector representation using an embedding model.

### Vector Database

The generated embeddings are stored in a vector database.

When a user asks a question, the question is also converted into an embedding. The system then searches for the most similar document chunks.

### Retrieval

The most relevant chunks are returned to the LLM as context.

### Generation

The LLM uses the retrieved context to generate a grounded response.

## 7. Tool Calling

The assistant supports multiple tools.

### RAG Search

The `rag_search` tool searches the knowledge base and returns relevant information.

Example:

```text
User:
What is Retrieval-Augmented Generation?

LLM:
Calls rag_search

RAG:
Returns relevant knowledge

LLM:
Generates final answer
```

### Calculator

The `calculator` tool is used for mathematical calculations.

Example:

```text
User:
What is 25 * 40?

LLM:
Calls calculator

Calculator:
Returns 1000

LLM:
Returns the final answer
```

## 8. Structured Output

The `/chat` endpoint returns a structured JSON response.

Example:

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) combines information retrieval with a Large Language Model to generate answers using retrieved external information.",
  "source_used": true,
  "tool_used": "rag_search"
}
```

The response contains:

- `answer`: Final response generated by the LLM
- `source_used`: Indicates whether the RAG knowledge source was used
- `tool_used`: Name of the tool used by the agent

## 9. Prompt Engineering

A system prompt is used to control the behavior of the AI Assistant.

The prompt defines:

- When to use RAG
- When to use the calculator
- How to use retrieved information
- How to avoid hallucinating knowledge-base information
- How to generate concise responses
- How to handle unavailable information
- How to avoid exposing internal tool calls

The LLM is configured with a low temperature to provide more deterministic responses.

## 10. Local LLM Deployment

Ollama is used to run an open-source LLM locally.

The currently configured local model is:

```text
qwen2.5:1.5b
```

The model is served through the Ollama API.

The Docker container communicates with the Ollama server running on the host machine using:

```text
http://host.docker.internal:11434
```

The available Ollama models can be checked using:

```bash
ollama list
```

The Ollama API can be tested using:

```bash
curl http://localhost:11434/api/tags
```

## 11. FastAPI API

The application provides REST API endpoints.

### Root Endpoint

```http
GET /
```

Response:

```json
{
  "message": "AI Assistant API is running."
}
```

### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "question": "What is Artificial Intelligence?"
}
```

Response:

```json
{
  "answer": "Artificial Intelligence is a branch of computer science...",
  "source_used": true,
  "tool_used": "rag_search"
}
```

### Ollama Chat Endpoint

```http
POST /ollama/chat
```

This endpoint sends the user query directly to the locally hosted Ollama model.

Example request:

```json
{
  "question": "Explain Artificial Intelligence simply."
}
```

## 12. API Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://localhost:8000/docs
```

The Swagger interface can be used to test the available endpoints.

## 13. Project Structure

```text
ai-assistant/
│
├── app/
│   ├── api/
│   │   ├── main.py
│   │   └── ollama.py
│   │
│   ├── graph/
│   │   └── ollama_agent.py
│   │
│   ├── llm/
│   │   ├── groq_llm.py
│   │   └── ollama_llm.py
│   │
│   ├── tools/
│   │   ├── calculator.py
│   │   └── rag_tool.py
│   │
│   └── schemas/
│       └── response.py
│
├── data/
│   └── documents/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 14. Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Do not commit the `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
__pycache__/
*.pyc
.venv/
```

## 15. Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ai-assistant
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 16. Ollama Setup

Install Ollama and make sure the Ollama server is running.

Pull the required model:

```bash
ollama pull qwen2.5:1.5b
```

Verify the installed model:

```bash
ollama list
```

Test the Ollama API:

```bash
curl http://localhost:11434/api/tags
```

## 17. Running Without Docker

Start the FastAPI application:

```bash
uvicorn app.api.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## 18. Docker Deployment

Build the Docker image:

```bash
docker build -t ai-assistant .
```

Run the container:

```bash
docker run -d \
  --name ai-assistant \
  -p 8000:8000 \
  --env-file .env \
  ai-assistant
```

Check running containers:

```bash
docker ps
```

Check application logs:

```bash
docker logs ai-assistant
```

## 19. Docker Compose

If Docker Compose is configured, start the application using:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

Stop the application:

```bash
docker compose down
```

## 20. Testing the API

The `/chat` endpoint can be tested using Swagger, Postman, or curl.

Example:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"What is Artificial Intelligence?\"}"
```

Example calculation:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"Calculate 25 * 40\"}"
```

Example RAG question:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"What is Retrieval-Augmented Generation?\"}"
```

## 21. Verification

The following components were verified during development:

### Ollama Model Availability

The Ollama API successfully returned the available local models, including:

```text
qwen2.5:1.5b
gemma3:4b
aiden_lu/minicpm-v2.6:Q4_K_M
moondream:1.8b
llama3.2-vision:11b
gemma3:1b
deepseek-r1:1.5b
```

### RAG Tool Calling

The LangGraph agent successfully:

1. Received the user question.
2. Selected the `rag_search` tool.
3. Retrieved relevant knowledge.
4. Passed the retrieved information back to the LLM.
5. Generated the final answer.

### FastAPI

The FastAPI application successfully starts on:

```text
http://0.0.0.0:8000
```

The Swagger documentation is available at:

```text
http://localhost:8000/docs
```

## 22. Limitations

- Local LLM performance depends on available CPU, RAM, and system resources.
- The quality of RAG responses depends on document quality, chunking, embeddings, and retrieval performance.
- The current local model is relatively small and may provide less capable responses than larger models.
- The application requires the Ollama server to be running for local-model requests.

## 23. Future Improvements

Possible improvements include:

- Adding more external tools
- Improving document chunking strategies
- Adding metadata filtering to RAG
- Implementing conversation memory
- Adding streaming responses
- Adding authentication and authorization
- Adding automated tests
- Adding monitoring and logging
- Deploying the application to a cloud platform
- Using a production-grade vector database
- Evaluating different embedding models
- Supporting multiple local LLMs

## 24. Conclusion

This project demonstrates the development of a modern AI Assistant using LLM integration, RAG, tool calling, structured responses, LangGraph, FastAPI, Ollama, and Docker.

The system can combine external knowledge retrieval with LLM generation, perform tool-based operations, and expose the complete functionality through REST APIs.

The project provides a foundation for building more advanced agentic AI applications with additional tools, memory, retrieval strategies, and locally deployed models.
