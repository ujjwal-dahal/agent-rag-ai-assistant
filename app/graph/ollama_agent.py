import json
import math
import re
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config.config import env_file
from app.config.prompt_versions import get_prompt_config
from app.tools.calculator import calculator
from app.tools.rag_tool import configure_retriever, rag_search


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
  prompt_version: str
  trace: list[dict]
  termination_reason: str


# ==================================================
# Tools
# ==================================================

tools = [
    calculator,
    rag_search,
]


# ==================================================
# Bind Tools to Groq LLM
# ==================================================

groq_llm = ChatGroq(
  model=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"),
  temperature=0,
  max_tokens=1024,
  api_key=env_file.MODEL_API_KEY,
)

llm_with_tools = groq_llm.bind_tools(tools)

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
must be supported by the retrieved context. For calculator, a successful
numeric result in the evidence is sufficient when the candidate answer states
that result, even if the original expression is not repeated in the evidence.
If evidence is missing, contradictory, or too weak, set sufficient to false
and provide a concise refined_query that would retrieve or calculate what is
missing. For an answer that does not use a tool, set sufficient to true unless
it explicitly claims unsupported knowledge.
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
  config = get_prompt_config(state.get("prompt_version", "prompt_v1"))
  prompt = config.prompt
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

  tool_calls = [
    {"id": call.get("id"), "name": call["name"], "args": call.get("args", {})}
    for call in response.tool_calls
  ]
  trace = [*state.get("trace", []), {
    "step": len(state.get("trace", [])) + 1,
    "kind": "model",
    "reasoning": (_message_text(response) or
             "Provider returned no textual reasoning; tool selection was recorded from the tool call."),
    "tool_calls": tool_calls,
    "tool_call_ids": [call.get("id") for call in response.tool_calls],
  }]
  return {
    "messages": [response],
    "answer": _message_text(response) if not response.tool_calls else state.get("answer", ""),
    "iterations_used": state.get("iterations_used", 0) + (0 if response.tool_calls else 1),
    "total_tokens": state.get("total_tokens", 0) + _token_count(response),
    "trace": trace,
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
  trace = [*state.get("trace", []), {
    "step": len(state.get("trace", [])) + 1,
    "kind": "tool",
    "tool": tool_name,
    "args": next((call["args"] for step in reversed(state.get("trace", []))
            if step.get("kind") == "model"
            for call in step.get("tool_calls", [])
            if call.get("id") == latest.tool_call_id), {}),
    "result": result,
    "reasoning": "Model requested this tool; raw tool output was recorded.",
  }]
  return {
    "tool_used": tool_name,
    "source_used": tool_name == "rag_search",
    "retrieved_context": result,
    "trace": trace,
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

  if tool_used == "rag_search" and (
    not evidence or evidence == "No relevant information was found."
  ):
    return {
      "verifier_result": {
        "sufficient": False,
        "reason": "Retrieved RAG evidence was empty or unavailable.",
        "refined_query": state.get("question"),
      },
      "verified": False,
      "total_tokens": state.get("total_tokens", 0),
    }

  if tool_used == "calculator" and (
    not evidence or "Unable to calculate" in evidence
  ):
    return {
      "verifier_result": {
        "sufficient": False,
        "reason": "The calculator did not return a usable result.",
        "refined_query": state.get("question"),
      },
      "verified": False,
      "total_tokens": state.get("total_tokens", 0),
    }

  response = groq_llm.invoke(
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

  if tool_used == "calculator" and evidence and "Unable to calculate" not in evidence:
    answer_numbers = [
      float(value)
      for value in re.findall(r"[-+]?\d+(?:\.\d+)?", answer)
    ]
    try:
      calculator_matches = any(
        math.isclose(float(evidence.strip()), value)
        for value in answer_numbers
      )
    except ValueError:
      calculator_matches = evidence.strip() in answer
    result = {
      "sufficient": calculator_matches,
      "reason": (
        "The calculator result appears in the candidate answer."
        if calculator_matches
        else "The candidate answer does not contain the calculator result."
      ),
      "refined_query": None if calculator_matches else state.get("question"),
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
  max_iterations = get_prompt_config(state.get("prompt_version", "prompt_v1")).max_iterations
  if state.get("verified"):
    state["termination_reason"] = "success"
    return END
  if state.get("iterations_used", 0) >= max_iterations:
    state["termination_reason"] = "max_iterations"
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
  return run_agent_versioned(question, "prompt_v1")


def run_agent_versioned(question: str, prompt_version: str = "prompt_v1") -> dict:
  config = get_prompt_config(prompt_version)
  configure_retriever(config.top_k)
  result = graph.invoke(
    {
      "messages": [HumanMessage(content=question)],
      "question": question,
      "iterations_used": 0,
      "total_tokens": 0,
      "prompt_version": prompt_version,
      "trace": [],
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
  termination_reason = "success" if verified else (
    "max_iterations" if result.get("iterations_used", 0) >= config.max_iterations else "error"
  )
  return {
    "answer": answer,
    "source_used": bool(result.get("source_used", False)),
    "tool_used": result.get("tool_used"),
    "verified": verified,
    "iterations_used": result.get("iterations_used", 0),
    "total_tokens": result.get("total_tokens", 0),
    "trace": result.get("trace", []),
    "termination_reason": termination_reason,
    "prompt_version": prompt_version,
  }