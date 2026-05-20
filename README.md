# SmartDesk AI — Intelligent IT & HR Operations Agent

[![Live Demo](https://img.shields.io/badge/Live%20Demo-smartdeskai.streamlit.app-ff4b4b?logo=streamlit&logoColor=white)](https://smartdeskai.streamlit.app/)

An agentic AI help desk for NovaTech Solutions (fictional company) that:
- **Answers** IT and HR questions from a curated knowledge base (RAG)
- **Creates** support tickets when it cannot answer
- **Checks** ticket status on request

Built with **LangGraph + GPT-4o + FAISS** and a **Streamlit** chat UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  SmartDesk AI                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Employee ──▶ Streamlit Chat UI ──▶ LangGraph Agent  │
│                                          │           │
│                                    GPT-4o + Tools    │
│                                          │           │
│                          ┌───────────────┼──────┐    │
│                          │               │      │    │
│                    search_kb    create_ticket  get_status
│                          │               │      │    │
│                     FAISS Index     tickets.json     │
│                    (49 docs)                         │
└─────────────────────────────────────────────────────┘
```

**Components:**
| Layer | Technology |
|---|---|
| Agent Framework | LangGraph (ReAct loop) |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (local, cosine similarity) |
| Ticketing | Local JSON file (easily swappable to Jira/Notion) |
| UI | Streamlit |
| Observability | Langfuse |

---

## Knowledge Base

49 documents across 4 domains:

| Domain | Documents | Examples |
|---|---|---|
| IT Support | 20 | Password reset, VPN, MFA, printers, storage |
| HR Policies | 15 | Leave, WFH, reimbursements, POSH, performance |
| Onboarding | 8 | Day-one checklist, equipment, buddy programme |
| Payroll | 6 | Pay schedule, tax, ESOP vesting, PF/gratuity |

**Escalation logic:** If best cosine similarity < 0.40, the agent signals it cannot answer and offers to raise a ticket.

---

## Prerequisites

- Python 3.11+
- An OpenAI API key

---

## Setup

```bash
# 1. Clone / navigate to the project
cd "5. Smart Desk AI Agent"

# 2. Create virtual environment
python3 -m venv .venv          # or: uv venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt   # or: uv pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY (others are optional)

# 5. Build the FAISS knowledge base index (one-time)
python build_index.py

# 6. Launch the app
streamlit run app.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o and embeddings |
| `FAISS_INDEX_PATH` | No | Path to FAISS index (default: `index/faiss_index`) |
| `TICKETS_FILE` | No | Path to ticket JSON store (default: `tickets/tickets.json`) |
| `SIMILARITY_THRESHOLD` | No | Min cosine similarity for KB answers (default: `0.40`) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse observability (optional) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse observability (optional) |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing (optional) |

---

## Sample Conversation Flows

### Flow A — Query Answered from Knowledge Base
```
Employee: How do I set up VPN on my Mac?
Agent:    To set up VPN on your Mac, download Cisco AnyConnect from the
          Self-Service Portal at https://selfservice.novatech.internal...
```

### Flow B — Query Escalated to Ticket
```
Employee: My monitor has been flickering for two days.
Agent:    I'm sorry to hear that. I don't have a troubleshooting guide for
          this. Could you share your email so I can create a ticket?
Employee: jane.doe@novatech.com
Agent:    Here is the ticket I will create:
          • Title: Monitor Flickering Issue
          • Category: IT Support
          • Priority: Medium
          Shall I go ahead?
Employee: Yes, please.
Agent:    Done! Ticket IT-1001 has been created. The IT team will be in touch.
```

### Flow C — Ticket Status Check
```
Employee: What is the status of my ticket?
Agent:    Found 1 ticket for jane.doe@novatech.com:
          • [IT-1001] Monitor Flickering Issue
            Status: Open | Priority: Medium
```

---

## Project Structure

```
5. Smart Desk AI Agent/
├── app.py                    # Streamlit chat UI
├── build_index.py            # One-time FAISS index builder
├── requirements.txt
├── .env.example
├── knowledge_base/
│   ├── it_support.json       # 20 IT support documents
│   ├── hr_policies.json      # 15 HR policy documents
│   ├── onboarding.json       # 8 onboarding guides
│   └── payroll.json          # 6 payroll documents
├── agent/
│   ├── state.py              # AgentState (LangGraph TypedDict)
│   ├── prompts.py            # System prompt
│   ├── tools.py              # RAG, ticket creation, status tools
│   └── graph.py              # LangGraph compiled graph
├── tickets/
│   └── tickets.json          # Local ticket store
└── index/                    # Generated by build_index.py
    ├── faiss_index.faiss
    └── faiss_index.pkl
```

---

## Swapping to a Real Ticketing Platform

The ticket tools in `agent/tools.py` are simple to replace. To use Jira:

```python
# In agent/tools.py, replace create_support_ticket body with:
from jira import JIRA
jira = JIRA(server=os.getenv("JIRA_URL"), basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("JIRA_TOKEN")))
issue = jira.create_issue(project='IT', summary=title, description=description, issuetype={'name': 'Bug'})
return f"TICKET_CREATED\n\nTicket ID: {issue.key}"
```

---

## Marking Scheme Coverage

| Criterion | Implementation |
|---|---|
| Knowledge Base (25pts) | 49 docs, 4 domains, deliberate gaps for escalation testing |
| RAG Pipeline | FAISS + OpenAI embeddings, cosine similarity threshold |
| Ticket Creation (20pts) | Full field collection, human-in-the-loop confirmation |
| Ticket Status (15pts) | Email-based lookup, multiple ticket listing |
| Agent Orchestration (15pts) | LangGraph ReAct loop, intent routing via tools |
| Code Quality (15pts) | Modular, typed, documented |
| Error Handling (10pts) | Graceful tool failures, no hardcoded secrets |
| **Bonus**: Streamlit UI | Deployed chat interface with session management |
