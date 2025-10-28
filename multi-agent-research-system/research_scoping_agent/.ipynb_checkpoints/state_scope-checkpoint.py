# state_scope.py

from typing import Optional, Sequence, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.graph.message import MessagesState


# Input state schema — holds user messages
class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass


# Main state schema for the research agent
class AgentState(MessagesState):
    """Main state for the full multi-agent research system."""
    research_brief: Optional[str] = None
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], list.__add__] = []
    notes: Annotated[list[str], list.__add__] = []
    final_report: Optional[str] = None


# Structured schema for clarification step
class ClarifyWithUser(BaseModel):
    """Schema for determining if user request needs clarification."""
    need_clarification: bool = Field(description="Whether clarification is needed")
    question: str = Field(description="Question to ask the user for clarification")
    verification: str = Field(description="Verification message confirming understanding")


# Structured schema for research brief generation
class ResearchQuestion(BaseModel):
    """Schema for generating a structured research brief."""
    research_brief: str = Field(description="Comprehensive research brief")