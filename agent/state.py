from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State maintained across agent turns."""
    messages: Annotated[list, add_messages]
    user_email: Optional[str]        # collected during the session
    session_id: str                   # unique session identifier
