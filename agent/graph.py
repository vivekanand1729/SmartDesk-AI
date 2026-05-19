"""
LangGraph agent graph for SmartDesk AI.

Graph topology:
  START → agent_node → [tool_node | END]
                ↑
           tool_node
"""

import uuid
import re
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS

# ─────────────────────────────────────────────
# LLM setup with tool binding
# ─────────────────────────────────────────────
_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
_llm_with_tools = _llm.bind_tools(TOOLS)

# ─────────────────────────────────────────────
# Email extraction helper (to update state)
# ─────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _extract_email(text: str) -> str | None:
    match = _EMAIL_RE.search(text or "")
    return match.group(0).lower() if match else None


# ─────────────────────────────────────────────
# Node: agent
# ─────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current message history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)

    # Opportunistically capture email from user messages
    user_email = state.get("user_email")
    if user_email is None:
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                found = _extract_email(msg.content)
                if found:
                    user_email = found
                    break

    return {
        "messages": [response],
        "user_email": user_email,
    }


# ─────────────────────────────────────────────
# Conditional edge: continue or finish?
# ─────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─────────────────────────────────────────────
# Convenience: create initial state
# ─────────────────────────────────────────────
def new_session() -> AgentState:
    return AgentState(
        messages=[],
        user_email=None,
        session_id=str(uuid.uuid4()),
    )
