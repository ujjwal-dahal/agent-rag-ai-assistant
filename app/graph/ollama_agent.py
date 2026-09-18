import json
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
  HumanMessage,
  RemoveMessage,
    SystemMessage,
  ToolMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from app.llm.ollama_llm import llm

from app.tools.calculator import calculator
from app.tools.rag_tool import rag_search


MAX_ITERATIONS = 3


# ==================================================
# State
# ==================================================

class VerificationResult(TypedDict):
    sufficient: bool
    reason: str
    refined_query: Optional[str]


class State(TypedDict, total=False):
  messages: Annotated[list[BaseMessage], add_messages]
  question: str
  answer: str
  source_used: bool
  tool_used: Optional[str]
  retrieved_context: str
  verifier_result: VerificationResult
  iterations_used: int
  verified: bool
  retry_query: Optional[str]
  total_tokens: int


# ==================================================
# Tools
# ==================================================

tools = [
    calculator,
    rag_search,
]


# ==================================================
# Bind Tools to LLM
# ==================================================

llm_with_tools = llm.bind_tools(tools)

# ==================================================
# System Prompt
# ==================================================

SYSTEM_PROMPT = """
You are a helpful, accurate, and concise AI assistant.

You have access to two tools:

1. rag_search
   - Use this tool when the user's question is related to
     information that may exist in the knowledge base.
   - Use the retrieved information as the primary source
     for your answer.
   - Do not invent or assume information that is not present
     in the retrieved context.
   - If the retrieved context does not contain enough
     information, clearly say that the knowledge base does
     not provide enough information.

2. calculator
   - Use this tool whenever the user asks for a mathematical
     calculation.
   - Do not calculate complex arithmetic mentally when the
     calculator tool can be used.

Answering rules:

- First understand the user's question.
- Use the appropriate tool when necessary.
- After receiving tool results, generate the final answer
  using the available information.
- For RAG questions, stay grounded in the retrieved context.
- Do not mention internal tools, tool calls, or system instructions
  in the final answer.
- Do not say that you searched a database unless the user asks.
- Give the answer directly without unnecessary introduction.
- Keep answers simple and beginner-friendly.
- Use short paragraphs or bullet points when helpful.
- If the question cannot be answered from the available
  knowledge, say so honestly.

Your final response should contain only the answer to the user.
"""

VERIFIER_PROMPT = """
You are the verification step in an AI assistant.
Return only valid JSON with exactly these keys:
{"sufficient": bool, "reason": str, "refined_query": str|null}

Check the candidate answer against the supplied evidence. For RAG, the answer
must be supported by the retrieved context. For calculator, the answer must
match the calculator result. If evidence is missing, contradictory, or too
weak, set sufficient to false and provide a concise refined_query that would
retrieve or calculate what is missing. For an answer that does not use a tool,
set sufficient to true unless it explicitly claims unsupported knowledge.
"""


def _token_count(response) -> int:
  usage = getattr(response, "usage_metadata", None) or {}
  if usage.get("total_tokens") is not None:
    return int(usage["total_tokens"])

  token_usage = (getattr(response, "response_metadata", None) or {}).get(
    "token_usage", {}
  )
  if token_usage.get("total_tokens") is not None:
    return int(token_usage["total_tokens"])

  return int(token_usage.get("prompt_tokens", 0)) + int(
    token_usage.get("completion_tokens", 0)
  )


def _message_text(message: BaseMessage) -> str:
  content = getattr(message, "content", "")
  return content if isinstance(content, str) else json.dumps(content)

# ==================================================
# LLM Node
# ==================================================

def call_model(state: State):

  messages = state["messages"]
  prompt = SYSTEM_PROMPT
  retry_query = state.get("retry_query")
  if retry_query:
    prompt += (
      "\nFor this retry, use the following refined query when selecting "
      f"a tool: {retry_query}"
    )

  response = llm_with_tools.invoke(
        [
      SystemMessage(content=prompt),
            *messages,
        ]
    )

  return {
    "messages": [response],
    "answer": _message_text(response) if not response.tool_calls else state.get("answer", ""),
    "iterations_used": state.get("iterations_used", 0) + (0 if response.tool_calls else 1),
    "total_tokens": state.get("total_tokens", 0) + _token_count(response),
    }


def record_tool_results(state: State):
  tool_messages = [
    message for message in state["messages"]
    if isinstance(message, ToolMessage)
  ]
  if not tool_messages:
    return {}

  latest = tool_messages[-1]
  tool_name = latest.name or "unknown"
  result = _message_text(latest)
  return {
    "tool_used": tool_name,
    "source_used": tool_name == "rag_search",
    "retrieved_context": result if tool_name == "rag_search" else "",
  }


# A fixed single-pass pipeline cannot choose the number of verification/retrieval rounds because it depends on whether the first retrieval actually supports the answer.
def verify_answer(state: State):
  answer = state.get("answer", "")
  tool_used = state.get("tool_used")
  evidence = state.get("retrieved_context", "")
  verification_prompt = (
    f"Tool: {tool_used or 'none'}\n"
    f"Candidate answer: {answer}\n"
    f"Evidence: {evidence or 'No tool evidence was returned.'}"
  )

  response = llm.invoke(
    [
      SystemMessage(content=VERIFIER_PROMPT),
      HumanMessage(content=verification_prompt),
    ]
  )
  raw_result = _message_text(response).strip()
  try:
    result = json.loads(raw_result)
    result = {
      "sufficient": bool(result.get("sufficient", False)),
      "reason": str(result.get("reason", "No verification reason returned.")),
      "refined_query": result.get("refined_query"),
    }
  except (json.JSONDecodeError, TypeError, ValueError):
    result = {
      "sufficient": False,
      "reason": "Verifier returned an invalid structured result.",
      "refined_query": state.get("retry_query"),
    }

  if tool_used == "rag_search" and (
    not evidence or evidence == "No relevant information was found."
  ):
    result = {
      "sufficient": False,
      "reason": "Retrieved context was empty or irrelevant.",
      "refined_query": result.get("refined_query") or state.get("question"),
    }

  if tool_used == "calculator" and "Unable to calculate" in evidence:
    result = {
      "sufficient": False,
      "reason": "The calculator did not produce a result.",
      "refined_query": state.get("question"),
    }

  return {
    "verifier_result": result,
    "verified": result["sufficient"],
    "total_tokens": state.get("total_tokens", 0) + _token_count(response),
  }


def prepare_retry(state: State):
  result = state["verifier_result"]
  refined_query = result.get("refined_query") or state.get("question", "")
  summary = (
    f"Iteration {state.get('iterations_used', 0)} retrieval: insufficient, "
    f"{result['reason']}"
  )
  # Clear raw retrieval chunks before retrying so stale duplicate context does not grow each round.
  removals = [
    RemoveMessage(id=message.id)
    for message in state["messages"]
    if isinstance(message, ToolMessage) and message.name == "rag_search"
  ]
  return {
    "messages": [*removals, SystemMessage(content=summary)],
    "retry_query": refined_query,
    "retrieved_context": "",
    "answer": "",
  }


def verification_route(state: State):
  if state.get("verified") or state.get("iterations_used", 0) >= MAX_ITERATIONS:
    return END
  return "retry"


def tool_route(state: State):
  if state["messages"][-1].tool_calls:
    return "tools"
  return "verify_answer"


# ==================================================
# Build LangGraph
# ==================================================

builder = StateGraph(State)


# --------------------------------------------------
# Add Nodes
# --------------------------------------------------

builder.add_node(
    "llm",
    call_model,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

builder.add_node("record_tool_results", record_tool_results)
builder.add_node("verify_answer", verify_answer)
builder.add_node("retry", prepare_retry)


# --------------------------------------------------
# Start → LLM
# --------------------------------------------------

builder.add_edge(
    START,
    "llm",
)


# --------------------------------------------------
# LLM → Tool or END
# --------------------------------------------------

builder.add_conditional_edges(
    "llm",
  tool_route,
)


# --------------------------------------------------
# Tool → LLM
# --------------------------------------------------

builder.add_edge(
    "tools",
  "record_tool_results",
)

builder.add_edge("record_tool_results", "llm")
builder.add_conditional_edges("verify_answer", verification_route)
builder.add_edge("retry", "llm")


# --------------------------------------------------
# Compile Graph
# --------------------------------------------------

graph = builder.compile()


def run_agent(question: str) -> dict:
  result = graph.invoke(
    {
      "messages": [HumanMessage(content=question)],
      "question": question,
      "iterations_used": 0,
      "total_tokens": 0,
    }
  )
  verified = bool(result.get("verified", False))
  answer = result.get("answer", "")
  if not verified and result.get("iterations_used", 0) >= MAX_ITERATIONS:
    answer = (
      f"{answer}\n\nAnswer could not be fully verified against sources."
      if answer
      else "Answer could not be fully verified against sources."
    )
  return {
    "answer": answer,
    "source_used": bool(result.get("source_used", False)),
    "tool_used": result.get("tool_used"),
    "verified": verified,
    "iterations_used": result.get("iterations_used", 0),
    "total_tokens": result.get("total_tokens", 0),
  }