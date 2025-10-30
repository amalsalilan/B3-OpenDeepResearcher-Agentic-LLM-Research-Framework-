from typing import Optional, Sequence, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import operator

# Define the input state schema
class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

# Define the main agent state schema
class AgentState(MessagesState):
    """Main state for the full multi-agent research system."""
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], operator.add]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    final_report: str

# Define structured output schemas
class ClarifyWithUser(BaseModel):
    """Schema for determining if user request needs clarification."""
    need_clarification: bool = Field(description="Whether clarification is needed")
    question: str = Field(description="Question to ask the user for clarification")
    verification: str = Field(description="Verification message confirming understanding")

class ResearchQuestion(BaseModel):
    """Schema for generating a structured research brief."""
    research_brief: str = Field(description="Comprehensive research brief")
