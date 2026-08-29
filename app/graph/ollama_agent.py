from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
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


# ==================================================
# State
# ==================================================

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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

# ==================================================
# LLM Node
# ==================================================

def call_model(state: State):

    messages = state["messages"]

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *messages,
        ]
    )

    return {
        "messages": [response]
    }


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
    tools_condition,
)


# --------------------------------------------------
# Tool → LLM
# --------------------------------------------------

builder.add_edge(
    "tools",
    "llm",
)


# --------------------------------------------------
# Compile Graph
# --------------------------------------------------

graph = builder.compile()