"""
Tools for SmartDesk AI:
  - search_knowledge_base: RAG-based search over IT/HR knowledge base
  - create_support_ticket:  Creates a ticket in the local JSON ticket store
  - get_ticket_status:      Retrieves tickets for a given employee email
"""

import os
import json
import uuid
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

import faiss
import numpy as np

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
_FAISS_PATH = _BASE / os.getenv("FAISS_INDEX_PATH", "index/faiss_index")
_TICKETS_PATH = _BASE / os.getenv("TICKETS_FILE", "tickets/tickets.json")
_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

# ─────────────────────────────────────────────
# Lazy-load FAISS index and document store
# ─────────────────────────────────────────────
_index: Optional[faiss.Index] = None
_docs: Optional[list] = None
_embeddings: Optional[OpenAIEmbeddings] = None


def _load_index() -> tuple:
    global _index, _docs, _embeddings

    if _index is not None:
        return _index, _docs, _embeddings

    faiss_file = str(_FAISS_PATH) + ".faiss"
    docs_file  = str(_FAISS_PATH) + ".pkl"

    if not Path(faiss_file).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_file}. "
            "Run `python build_index.py` first."
        )

    _index = faiss.read_index(faiss_file)
    with open(docs_file, "rb") as f:
        _docs = pickle.load(f)

    _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _index, _docs, _embeddings


# ─────────────────────────────────────────────
# Tool 1: search_knowledge_base
# ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the NovaTech IT and HR knowledge base for an answer to the employee's question.
    Returns relevant context and whether a confident answer was found.
    Always call this tool first for any IT or HR question before deciding to escalate.
    """
    try:
        index, docs, embeddings = _load_index()

        query_vector = np.array(
            embeddings.embed_query(query), dtype="float32"
        ).reshape(1, -1)

        k = 3
        distances, indices = index.search(query_vector, k)

        results = []
        best_score = 0.0

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # FAISS inner product (normalized) gives cosine similarity
            score = float(dist)
            if score > best_score:
                best_score = score
            doc = docs[idx]
            results.append({
                "score": round(score, 4),
                "category": doc.get("category", ""),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
            })

        if not results or best_score < _THRESHOLD:
            return (
                "NO_ANSWER_FOUND\n\n"
                "The knowledge base does not contain a confident answer to this query. "
                f"Best similarity score: {best_score:.3f} (threshold: {_THRESHOLD}). "
                "Consider escalating to a support ticket."
            )

        top = results[0]
        context_blocks = "\n\n---\n\n".join(
            f"[{r['category']}] {r['title']} (relevance: {r['score']})\n{r['content']}"
            for r in results
        )

        return (
            f"ANSWER_FOUND\n\n"
            f"Most relevant result: [{top['category']}] {top['title']} "
            f"(similarity: {top['score']})\n\n"
            f"Context:\n{context_blocks}"
        )

    except FileNotFoundError as exc:
        return f"TOOL_ERROR: {exc}"
    except Exception as exc:
        return f"TOOL_ERROR: Knowledge base search failed — {exc}"


# ─────────────────────────────────────────────
# Ticket helper utilities
# ─────────────────────────────────────────────
def _load_tickets() -> dict:
    if not _TICKETS_PATH.exists():
        return {"tickets": []}
    with open(_TICKETS_PATH, "r") as f:
        return json.load(f)


def _save_tickets(data: dict) -> None:
    _TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TICKETS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _next_ticket_id(category: str) -> str:
    prefix = "IT" if "IT" in category.upper() else "HR"
    data = _load_tickets()
    existing = [
        t["id"] for t in data["tickets"] if t["id"].startswith(prefix)
    ]
    if existing:
        nums = [int(t.replace(prefix + "-", "")) for t in existing if "-" in t]
        next_num = max(nums) + 1 if nums else 1001
    else:
        next_num = 1001
    return f"{prefix}-{next_num}"


# ─────────────────────────────────────────────
# Tool 2: create_support_ticket
# ─────────────────────────────────────────────
@tool
def create_support_ticket(
    email: str,
    title: str,
    description: str,
    category: str,
    priority: str = "Medium",
) -> str:
    """
    Create a support ticket in the NovaTech ticket system.

    IMPORTANT: Only call this tool AFTER you have:
    1. Collected the employee's email address.
    2. Presented a ticket summary to the employee.
    3. Received their explicit confirmation to proceed.

    Args:
        email:       Employee's work email address.
        title:       Short one-line title for the ticket.
        description: Detailed description of the issue.
        category:    'IT Support' or 'HR' (infer from context).
        priority:    'Low', 'Medium', 'High', or 'Critical'. Default: 'Medium'.

    Returns:
        Ticket ID and confirmation message.
    """
    try:
        ticket_id = _next_ticket_id(category)
        ticket = {
            "id": ticket_id,
            "email": email.lower().strip(),
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "status": "Open",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "comments": [],
        }

        data = _load_tickets()
        data["tickets"].append(ticket)
        _save_tickets(data)

        return (
            f"TICKET_CREATED\n\n"
            f"Ticket ID: {ticket_id}\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Priority: {priority}\n"
            f"Status: Open\n"
            f"Created: {ticket['created_at']}\n\n"
            f"The employee can check the ticket status by asking 'What is the status of my ticket?' "
            f"or by referencing ticket ID {ticket_id}."
        )
    except Exception as exc:
        return f"TICKET_ERROR: Failed to create ticket — {exc}"


# ─────────────────────────────────────────────
# Tool 3: get_ticket_status
# ─────────────────────────────────────────────
@tool
def get_ticket_status(email: str) -> str:
    """
    Retrieve all support tickets for an employee by their email address.
    Use this when the employee asks about the status of their tickets.

    Args:
        email: The employee's work email address.

    Returns:
        A formatted list of tickets and their current statuses.
    """
    try:
        data = _load_tickets()
        email_lower = email.lower().strip()
        tickets = [
            t for t in data["tickets"]
            if t.get("email", "").lower() == email_lower
        ]

        if not tickets:
            return (
                f"NO_TICKETS_FOUND\n\n"
                f"No tickets were found for {email}. "
                "The employee may not have raised any tickets yet, "
                "or they may have used a different email address."
            )

        lines = [f"TICKETS_FOUND — {len(tickets)} ticket(s) for {email}:\n"]
        for t in tickets:
            lines.append(
                f"• [{t['id']}] {t['title']}\n"
                f"  Status: {t['status']} | Priority: {t['priority']} | "
                f"Category: {t['category']}\n"
                f"  Created: {t['created_at'][:10]}"
            )
            if t.get("comments"):
                lines.append(f"  Latest update: {t['comments'][-1]['text']}")

        return "\n".join(lines)

    except Exception as exc:
        return f"STATUS_ERROR: Failed to retrieve tickets — {exc}"


# ─────────────────────────────────────────────
# Exported list
# ─────────────────────────────────────────────
TOOLS = [search_knowledge_base, create_support_ticket, get_ticket_status]
