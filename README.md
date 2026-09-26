# AI Assistant

## Environment & Reproducibility (uv)

This Track B implementation uses `uv` as the single environment resolver. Run `uv sync` from a clean clone, then run commands with `uv run`; the committed `uv.lock` pins the LangChain/LangGraph, Groq SDK, Chroma, MLflow, and Evidently dependency graph. This matters here because the earlier W15/W16 assistant used a moving Groq model configuration and was affected by provider deprecation/compatibility changes. The lockfile makes that provider stack reproducible instead of allowing a later transitive upgrade to silently change tool-calling behavior.

The current workspace verified `uv sync` and Python compilation successfully. Provider-backed runs still require `GROQ_API_KEY` in `.env` and a current Groq model name in `MODEL_NAME`; no credentials are committed.

## Experiment Tracking Strategy (MLflow)

`prompts/v1.txt`, `prompts/v2.txt`, and `prompts/v3.txt` are loaded by `app/config/prompt_versions.py`. The agent records every model decision and raw tool result in `run_agent_versioned`, and `mlops/evaluate.py` logs those traces, prompt text, configuration parameters, and harness metrics to a local MLflow file store. The fixed cases are the RAG definition, calculator, missing-knowledge, and no-tool queries.

Version changes are documented in `prompts/CHANGELOG.md`: v2 adds explicit empty-evidence handling and reduces repeated retries; v3 prevents identical retries and increases the retrieval chunk size to preserve complete facts. These are hypotheses to verify from the preceding run traces, not claimed observations. The run creates `mlruns_artifacts/<version>/traces.json`, `prompt.txt`, and `comparison.json`; it also creates `golden_responses.json` from the best actual run.

| Version   | Task success | Avg iterations | Avg tokens | Evidently tests |
| --------- | -----------: | -------------: | ---------: | --------------: |
| prompt_v1 |       100.0% |           2.00 |   2,579.50 |           75.0% |
| prompt_v2 |       100.0% |           1.50 |   1,802.00 |           87.5% |
| prompt_v3 |        75.0% |           1.00 |   1,119.25 |           50.0% |

These figures come from the real Groq-backed runs saved under `mlruns_artifacts/` and are also available in the combined dashboard below.

## Monitoring & Regression Testing (Evidently AI)

Evidently is pinned through `evidently[llm]`. Its installed 0.7.23 API supports custom providers through `@llm_provider` and `LLMWrapper`; `mlops/evidently_groq.py` registers `groq`, uses the existing `GROQ_API_KEY`, and calls the live Groq model `qwen/qwen3.8-27b` (the account's current catalog exposed this model but no general Llama instruct model). The wrapper passes Evidently's `BinaryClassificationPromptTemplate` requests through Groq and preserves the parsed category and reasoning.

The native Evidently Test Suite runs two checks per version: reference-based correctness and tool appropriateness. The golden answers come from the previously executed best assistant run in `golden_responses.json`. Each suite uses a 75% per-check threshold, while `pct_tests_passed` is the per-case average across both checks.

| Version   | Correctness failures                                                             | Tool-choice failures           | Total passed | `pct_tests_passed` |
| --------- | -------------------------------------------------------------------------------- | ------------------------------ | -----------: | -----------------: |
| prompt_v1 | none                                                                             | `no_tool`                      |          6/8 |              75.0% |
| prompt_v2 | none                                                                             | `no_tool`                      |          7/8 |              87.5% |
| prompt_v3 | `missing_knowledge` (judge returned `UNKNOWN` for an empty answer) and `no_tool` | `missing_knowledge`, `no_tool` |          4/8 |              50.0% |

The Groq judge's verdicts were manually sanity-checked. For v1, marking the no-tool answer incorrect was reasonable because the answer said the knowledge base lacked information instead of giving the requested prompt-writing tip. For v2, marking the RAG definition and calculator answers correct matched the saved responses exactly, while the tool-choice failure correctly identified an unnecessary `rag_search`. For v3, the `UNKNOWN` verdict on the empty missing-knowledge answer was appropriately cautious rather than falsely calling it correct. The MLflow comparison therefore favors v2 on regression quality, while v3 remains cheaper but is not promoted because it loses a required answer.

### Combined comparison dashboard

Open the single-file [overall comparison report](reports/overall_comparison_report.html) to compare v1, v2, and v3 side by side. It contains task success, average iterations, average tokens, Evidently pass rates, the v2 promotion decision, failure patterns, and links/embedded views of each individual Evidently report. Evidently 0.7.23 can render a `Report` or `TestSuite` snapshot, but does not expose a clean static multi-version comparison export; this dashboard is generated from the real `comparison.json` and per-version report artifacts and is logged on the MLflow `overall-comparison` run.

## Running Track B

```powershell
uv sync
$env:GROQ_API_KEY = "..."
$env:MODEL_NAME = "openai/gpt-oss-20b"
uv run run-mlops-evaluation
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The MLflow UI shows one run per prompt version and the trace artifacts. The Evidently reports generated in this run are real Groq-backed HTML reports; no OpenAI key is required. The v2 run is the promoted regression winner because it passes 7/8 judge cases, while v3 is rejected despite lower token cost because it fails 4/8.

A production-oriented, containerized AI Assistant built using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), tool calling, structured responses, local LLM inference, FastAPI, Streamlit, and Docker.

The project is divided into two stages:

- **Task 1:** Build an AI Assistant using LLM, RAG, tool calling, structured output, and local LLM deployment.
- **Task 2:** Productionize the AI Assistant with a web UI, reliability mechanisms, performance improvements, fallback models, rate limiting, error handling, and Docker Compose deployment.

---

# 1. Project Overview

This project implements an AI Assistant capable of:

- Answering general user questions
- Retrieving information from a knowledge base using RAG
- Performing mathematical calculations using tools
- Generating structured responses
- Using a cloud-based LLM provider
- Using a locally hosted open-source LLM through Ollama
- Providing a web-based user interface
- Handling temporary model failures using retries
- Falling back to a local model when the primary provider is unavailable
- Applying rate limiting to API requests
- Returning graceful error responses
- Running the complete system using Docker Compose

The system uses **LangGraph** to manage the agent workflow and **LangChain** to integrate LLMs and tools.

---

# 2. Tasks Implemented

## Task 1: Build an AI Assistant

The following requirements were implemented:

- LLM integration
- Prompt engineering
- Structured response generation
- Tool calling
- RAG pipeline
- Document ingestion
- Document chunking
- Embeddings
- Vector database
- Local LLM deployment
- FastAPI backend
- Docker containerization

## Task 2: Productionize the AI Assistant

The following production-oriented features were added:

- Streamlit web UI
- Frontend-to-backend communication
- Docker Compose
- Concurrent request handling through FastAPI
- Retry mechanism
- Rate limiting
- Primary/fallback model architecture
- Error handling
- Graceful degradation
- Local Ollama inference
- Backend health checking
- Containerized frontend and backend
- Production-oriented API structure

---

# 3. Features

## AI Features

- LLM integration using Groq
- Local LLM inference using Ollama
- Qwen 2.5 1.5B local model
- Retrieval-Augmented Generation (RAG)
- Tool calling
- Calculator tool
- RAG search tool
- Vector-based document retrieval
- Structured JSON responses
- Prompt engineering
- LangGraph agent workflow

## Production Features

- Streamlit web interface
- FastAPI backend
- Retry mechanism
- Rate limiting
- Fallback model
- Error handling
- Graceful degradation
- Concurrent request handling
- Docker Compose deployment
- Health-checkable services
- Separate frontend, backend, and model services

---

# 4. Technologies Used

| Technology      | Purpose                              |
| --------------- | ------------------------------------ |
| Python          | Application development              |
| FastAPI         | REST API backend                     |
| Streamlit       | Web UI                               |
| LangChain       | LLM and tool integration             |
| LangGraph       | Agent workflow and orchestration     |
| Groq            | Primary cloud LLM provider           |
| Ollama          | Local LLM inference                  |
| Qwen 2.5 1.5B   | Local fallback LLM                   |
| RAG             | Knowledge retrieval                  |
| Embeddings      | Text vectorization                   |
| Vector Database | Vector storage and similarity search |
| Pydantic        | Request and response validation      |
| Docker          | Containerization                     |
| Docker Compose  | Multi-container deployment           |

---

# 5. System Architecture

The productionized system follows a layered architecture.

```text
                              USER
                                |
                                v
                       Streamlit Web UI
                                |
                                | HTTP Request
                                v
                       FastAPI Backend
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
              Rate Limiting          Error Handling
                    |                       |
                    +-----------+-----------+
                                |
                                v
                         AI Assistant
                          LangGraph
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
              Primary LLM             Fallback LLM
                 Groq                    Ollama
                    |                 Qwen 2.5 1.5B
                    |                       |
                    +-----------+-----------+
                                |
                                v
                         Tool Calling
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
                RAG Search             Calculator
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
             Structured Response
                    |
                    v
               FastAPI API
                    |
                    v
              Streamlit UI
```

---

# 6. Task 2 Production Architecture

The production architecture separates the application into independent services.

```text
+-------------------------------------------------------------+
|                         Docker Compose                      |
|                                                             |
|  +------------------+       +----------------------------+  |
|  |    Frontend      |       |          Backend           |  |
|  |    Streamlit     |-----> |           FastAPI          |  |
|  |    Port: 8501    | HTTP  |           Port: 8000       |  |
|  +------------------+       +-------------+--------------+  |
|                                           |                 |
|                                           | HTTP            |
|                                           v                 |
|                               +--------------------------+  |
|                               |          Ollama           |  |
|                               |        Port: 11434        |  |
|                               |      Qwen 2.5 1.5B       |  |
|                               +--------------------------+  |
|                                                             |
+-------------------------------------------------------------+
```

The three main services are:

1. **Frontend**
   - Streamlit
   - Provides the user interface
   - Sends requests to the FastAPI backend

2. **Backend**
   - FastAPI
   - Handles API requests
   - Performs rate limiting
   - Executes AI workflows
   - Handles retries and fallback logic

3. **Ollama**
   - Runs the local open-source model
   - Provides local inference through the Ollama API

---

# 7. Agent Workflow

The LangGraph agent follows this workflow:

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

The assistant can select tools based on the user's question.

For example:

```text
User Question
      |
      v
      LLM
      |
      +----> RAG Required
      |          |
      |          v
      |      rag_search
      |          |
      |          v
      |      Retrieved Data
      |
      +----> Calculation Required
                 |
                 v
             calculator
                 |
                 v
             Calculation
                 |
                 v
             Final LLM
                 |
                 v
             Final Answer
```

---

# 8. RAG Pipeline

The RAG system allows the assistant to retrieve information from a knowledge base before generating an answer.

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

## Document Ingestion

Documents are loaded into the application and prepared for retrieval.

## Document Chunking

Large documents are divided into smaller chunks so that relevant sections can be retrieved efficiently.

## Embeddings

Each document chunk is converted into a numerical vector representation.

## Vector Database

The generated embeddings are stored in a vector database.

## Similarity Search

When the user asks a question, the question is converted into an embedding and compared against stored document vectors.

## Context Retrieval

The most relevant document chunks are returned to the LLM.

## Generation

The LLM generates an answer using the retrieved context.

---

# 9. Tool Calling

The assistant currently supports multiple tools.

## RAG Search

The `rag_search` tool retrieves relevant information from the knowledge base.

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

## Calculator

The `calculator` tool performs mathematical calculations.

Example:

```text
User:
What is 25 * 40?

LLM:
Calls calculator

Calculator:
Returns 1000

LLM:
Returns final answer
```

---

# 10. Structured Output

The `/chat` endpoint returns a structured JSON response.

Example:

```json
{
  "answer": "Artificial Intelligence is a field of computer science...",
  "source_used": true,
  "tool_used": "rag_search"
}
```

The response contains:

- `answer`: Final generated answer
- `source_used`: Indicates whether RAG was used
- `tool_used`: Indicates which tool was used

Pydantic is used to validate the response structure.

---

# 11. Prompt Engineering

A system prompt controls the behavior of the AI Assistant.

The prompt provides instructions regarding:

- When to use RAG
- When to use the calculator
- How to use retrieved information
- How to avoid hallucinating information
- How to generate concise answers
- How to handle unavailable information
- How to avoid exposing internal tool execution details

The local Ollama model is configured with:

```text
temperature = 0
```

A low temperature makes responses more deterministic.

---

# 12. Primary and Fallback Model

Task 2 introduces a fallback architecture to improve reliability.

The system uses:

```text
Primary Model
     |
     v
    Groq
     |
  Success?
   /    \
 Yes     No
 |        |
 v        v
Answer   Retry
           |
       Still Fails?
         /    \
       No      Yes
       |        |
       v        v
     Answer   Ollama
                |
                v
          Qwen 2.5 1.5B
                |
                v
             Answer
```

## Primary Model

The primary model is provided through Groq.

## Fallback Model

If the primary provider fails after the configured retry attempts, the application uses the local Ollama model.

The fallback model is:

```text
qwen2.5:1.5b
```

This provides continued operation even when the cloud provider is temporarily unavailable.

---

# 13. Retry Mechanism

A retry mechanism is implemented for temporary LLM failures.

The general flow is:

```text
LLM Request
     |
     v
Request Failed?
   /       \
 No        Yes
 |          |
 v          v
Answer    Retry
             |
             v
        Retry Limit?
          /      \
        No        Yes
        |          |
        v          v
      Retry     Fallback
```

The system retries temporary failures before switching to the fallback model.

This reduces failures caused by temporary network or provider problems.

---

# 14. Rate Limiting

Rate limiting is implemented to prevent excessive API requests.

The purpose of rate limiting is to:

- Prevent API abuse
- Reduce unnecessary model calls
- Protect backend resources
- Control provider usage
- Improve system stability

The general flow is:

```text
Incoming Request
       |
       v
Rate Limiter
       |
       v
Limit Exceeded?
    /       \
  No         Yes
  |           |
  v           v
Process     HTTP 429
Request     Response
```

When the request limit is exceeded, the API returns an appropriate HTTP error instead of executing another model request.

---

# 15. Error Handling and Graceful Degradation

The application handles failures at multiple levels.

Possible failures include:

- Cloud LLM unavailable
- Ollama unavailable
- Network connection failure
- Invalid API request
- Tool execution failure
- RAG retrieval failure
- Model timeout

Instead of exposing internal exceptions directly to users, the API returns controlled error responses.

Example:

```json
{
  "detail": "AI service is temporarily unavailable. Please try again later."
}
```

The fallback model also provides graceful degradation.

For example:

```text
Groq unavailable
      |
      v
Retry
      |
      v
Still unavailable
      |
      v
Ollama
      |
      v
Qwen 2.5 1.5B
      |
      v
Response
```

---

# 16. Performance Engineering

FastAPI provides asynchronous request handling capabilities and can serve multiple requests concurrently.

The system is structured so that:

```text
Request 1 ──> Backend ──> Model
Request 2 ──> Backend ──> Model
Request 3 ──> Backend ──> Model
Request 4 ──> Backend ──> Model
```

This allows multiple users to access the API without requiring a separate backend process for every request.

Performance improvements include:

- Local model inference
- Low-temperature deterministic generation
- Rate limiting
- Retry control
- Lightweight Qwen model
- Separate frontend and backend services
- Dockerized service architecture

---

# 17. Model Optimization

ONNX conversion was not applied to the LLM because the project uses:

- Groq-hosted LLM inference
- Ollama-based GGUF model inference

The local model is already distributed in a quantized GGUF format.

The configured local model uses:

```text
Q4_K_M
```

quantization, which reduces memory requirements and makes local inference more practical on CPU-based systems.

Therefore, ONNX conversion is not necessary for the current architecture.

---

# 18. Local LLM Deployment

Ollama is used to serve the local open-source model.

Current model:

```text
qwen2.5:1.5b
```

The Docker backend communicates with the Ollama container using:

```text
http://ollama:11434
```

This is different from using `localhost`.

Inside Docker Compose, services communicate using their service names.

For example:

```text
Backend
   |
   v
http://ollama:11434
   |
   v
Ollama Container
```

Available models can be checked using:

```bash
docker exec ollama ollama list
```

The Ollama API can be tested using:

```bash
curl http://localhost:11434/api/tags
```

---

# 19. FastAPI API

The backend exposes REST API endpoints.

## Root Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "AI Assistant API is running."
}
```

## Chat Endpoint

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
  "source_used": false,
  "tool_used": null
}
```

## Ollama Chat Endpoint

```http
POST /ollama/chat
```

Request:

```json
{
  "question": "Explain Artificial Intelligence simply."
}
```

The endpoint sends the request to the locally hosted Ollama model.

---

# 20. Streamlit Web UI

Task 2 introduces a simple web interface using Streamlit.

The frontend provides:

- Question input
- Submit button
- AI response display
- Backend communication
- Error message display

The architecture is:

```text
Browser
   |
   v
Streamlit
   |
   | HTTP
   v
FastAPI
   |
   v
AI Assistant
   |
   v
Response
   |
   v
Streamlit
   |
   v
Browser
```

The Streamlit application runs on:

```text
http://localhost:8501
```

---

# 21. API Documentation

FastAPI automatically generates interactive API documentation.

After starting the application, open:

```text
http://localhost:8000/docs
```

The Swagger interface can be used to test:

```text
GET /
POST /chat
POST /ollama/chat
```

---

# 22. Project Structure

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
├── frontend/
│   └── streamlit_app.py
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

---

# 23. Environment Variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
OLLAMA_BASE_URL=http://ollama:11434
BACKEND_URL=http://backend:8000
```

Do not commit `.env` to GitHub.

The `.gitignore` file contains:

```text
.env
```

---

# 24. Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ai-assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 25. Ollama Setup

Install Ollama if running the model outside Docker.

Pull the required model:

```bash
ollama pull qwen2.5:1.5b
```

Verify:

```bash
ollama list
```

Test:

```bash
curl http://localhost:11434/api/tags
```

When using Docker Compose, the Ollama service is started as part of the application stack.

---

# 26. Running Without Docker

Start the FastAPI backend:

```bash
uvicorn app.api.main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Start Streamlit separately:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend will be available at:

```text
http://localhost:8501
```

---

# 27. Docker Deployment

Build the backend image:

```bash
docker build -t ai-assistant .
```

Run the backend:

```bash
docker run -d \
  --name ai-assistant \
  -p 8000:8000 \
  --env-file .env \
  ai-assistant
```

Check the container:

```bash
docker ps
```

View logs:

```bash
docker logs ai-assistant
```

---

# 28. Docker Compose Deployment

The complete application can be started using Docker Compose.

Build and start all services:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

Check services:

```bash
docker compose ps
```

Expected services:

```text
backend
frontend
ollama
```

Stop the complete application:

```bash
docker compose down
```

---

# 29. Service Ports

| Service  |  Port | Purpose          |
| -------- | ----: | ---------------- |
| Frontend |  8501 | Streamlit Web UI |
| Backend  |  8000 | FastAPI REST API |
| Ollama   | 11434 | Local LLM API    |

The application can therefore be accessed through:

```text
Frontend:
http://localhost:8501

Backend:
http://localhost:8000

Swagger:
http://localhost:8000/docs

Ollama:
http://localhost:11434
```

---

# 30. Testing the API

The `/chat` endpoint can be tested using Swagger, Postman, or curl.

Example:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"What is Artificial Intelligence?\"}"
```

Calculator example:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"Calculate 25 * 40\"}"
```

RAG example:

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"What is Retrieval-Augmented Generation?\"}"
```

Local Ollama example:

```bash
curl -X POST "http://localhost:8000/ollama/chat" \
-H "Content-Type: application/json" \
-d "{\"question\":\"Explain Artificial Intelligence simply.\"}"
```

---

# 31. Verification

The following components were verified during development.

## Backend

FastAPI successfully runs on:

```text
http://0.0.0.0:8000
```

The root endpoint returns:

```json
{
  "message": "AI Assistant API is running."
}
```

## Frontend to Backend Communication

The Streamlit container communicates with the backend using the Docker Compose service name:

```text
http://backend:8000
```

This allows communication between containers without using `localhost`.

## Ollama Connectivity

The backend can communicate with the Ollama service using:

```text
http://ollama:11434
```

The Ollama API successfully returns the available local model.

Current verified model:

```text
qwen2.5:1.5b
```

## RAG Tool Calling

The LangGraph agent can:

1. Receive the user question.
2. Determine whether RAG is required.
3. Call `rag_search`.
4. Retrieve relevant information.
5. Pass the retrieved context to the LLM.
6. Generate the final answer.

## Fallback

When the primary model/provider is unavailable, the system can use the local Ollama model as a fallback.

---

# 32. Error Handling Flow

The production request flow is:

```text
User Request
     |
     v
FastAPI
     |
     v
Rate Limiter
     |
     v
Primary LLM
     |
     +---- Success ----> Response
     |
     +---- Failure
             |
             v
           Retry
             |
             +---- Success ----> Response
             |
             +---- Failure
                     |
                     v
                  Ollama
                     |
                     v
               Qwen 2.5 1.5B
                     |
                     +---- Success ----> Response
                     |
                     +---- Failure
                              |
                              v
                       Graceful Error
```

This prevents a single model/provider failure from immediately causing the complete application to fail.

---

# 33. Limitations

- Local LLM performance depends on CPU, RAM, and available system resources.
- Qwen 2.5 1.5B is a relatively small model.
- The fallback model may produce lower-quality responses than the primary cloud model.
- RAG quality depends on document quality, chunking, embedding model, and retrieval configuration.
- Rate limiting configuration may need adjustment for production workloads.
- The current deployment does not include cloud deployment.
- Authentication and authorization are not currently implemented.
- Persistent conversation memory is not currently implemented.

---

# 34. Future Improvements

Possible improvements include:

- Adding authentication and authorization
- Adding conversation memory
- Implementing response caching
- Adding streaming responses
- Improving RAG retrieval
- Adding metadata filtering
- Adding automated tests
- Adding monitoring and observability
- Adding request tracing
- Improving logging
- Adding Prometheus and Grafana monitoring
- Deploying to AWS, Azure, or GCP
- Adding multiple fallback providers
- Using a production-grade vector database
- Evaluating different embedding models
- Implementing batch inference
- Adding load balancing
- Adding HTTPS and reverse proxy support

---

# 35. Conclusion

This project demonstrates the development and productionization of an AI Assistant using modern AI engineering technologies.

Task 1 focuses on building the core AI system using:

- LLM integration
- RAG
- Tool calling
- Structured output
- LangGraph
- FastAPI
- Ollama
- Docker

Task 2 extends the system into a more production-oriented application by adding:

- Streamlit Web UI
- Frontend-backend communication
- Retry mechanism
- Rate limiting
- Fallback model
- Error handling
- Graceful degradation
- Concurrent API request handling
- Docker Compose deployment

The resulting architecture provides a modular foundation for developing reliable and scalable Agentic AI applications.

---

# 36. Agentic Loop Extension

## Agentic Architecture

The graph now verifies every generated answer and loops back through retrieval or calculation when the evidence is insufficient. `MAX_ITERATIONS` is fixed at 3; after the cap, the best available answer is returned with `verified: false` and an explicit verification note.

```text
User Question
  |
  v
     LLM
  |
  v
Tool required? -------------------- No
   | Yes                              |
   v                                  v
ToolNode                         Draft answer
   |                                  |
   v                                  v
Record tool result ------------> verify_answer
              |
         +------------+------------+
         |                         |
       sufficient                   insufficient
         |                         |
         v                         v
        Final answer          iterations < 3?
                   |
              +------------+------------+
              |                         |
             Yes                        No
              |                         |
              v                         v
          Clear old RAG chunks      Final answer
          and refine query       verified: false
              |
              +------> LLM
```

## Context Engineering Technique

The technique is **clearing tool results**. It is applied in `prepare_retry` in `app/graph/ollama_agent.py`: before another RAG call, previous raw `ToolMessage` chunks are removed and replaced with a short insufficiency summary. This prevents stale, duplicate retrieval text from growing the context window across iterations.

## Agentic Pattern

This is a **SINGLE-AGENT LOOP**: one LangGraph graph uses one LLM role that switches between answer and verify prompts. A multi-agent split was not needed because this task is small enough that context isolation, parallelization, and specialization would not pay for their coordination cost; a single agent avoids a sequential bottleneck and extra coordination.

## Evaluation Harness

`eval_harness.py` runs nine plain-Python cases covering factual RAG, retries, calculation, missing knowledge, ambiguity, and no-tool responses. It measures task completion, tool-call correctness, trajectory length, and total tokens, and writes the detailed Markdown table to `eval_report.md`.

| Aggregate        | Completion rate | Average trajectory | Total tokens |
| ---------------- | --------------: | -----------------: | -----------: |
| Current Groq run |          100.0% |               2.78 |        38202 |

The current aggregate is from the Groq-only run. Ollama was bypassed for this evaluation, so the report records real Groq-backed completions and usage metadata.

### Skill vs. Agent

Verification could have been a simple Skill or fixed prompt template, but it is a node because it must make a control-flow decision: loop or stop.

### Token and Cost Accounting

See the `total_tokens` column in `eval_report.md`; the harness uses provider usage metadata when present and a small estimate otherwise. There is no multi-agent baseline because this system is single-agent by design.

### Failure Injection Test

`python eval_harness.py --inject-failure` monkeypatches the missing-KB query's RAG result. The verifier is designed to detect empty or malformed evidence, retry until the hard cap, and return `verified: false` rather than confidently accepting junk data. The recorded local run could not reach that branch because Ollama was unavailable.

### Tool vs. Agent Boundary

`rag_search` and `calculator` are bounded tool calls: each accepts one request and returns one result, with no internal state or multi-turn reasoning. They do not make independent decisions, so they are modeled as tools rather than agents; the LangGraph LLM decides when to call them and the verifier decides whether to continue.
