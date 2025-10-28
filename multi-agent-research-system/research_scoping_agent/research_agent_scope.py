# research_agent_scope.py

from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph.state import Command
from langgraph.graph import END
from state_scope import AgentState, ClarifyWithUser, ResearchQuestion
from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt


# Utility function for date formatting
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")


# Initialize the language model
model = init_chat_model(
    model="gemini-2.5-flash-lite",
    model_provider="google_genai",
    temperature=0.0
)


# Step 1: Clarification node
def clarify_with_user(state: AgentState) -> Command:
    """Determine if the user's request needs clarification."""
    structured_output_model = model.with_structured_output(ClarifyWithUser)
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    if response.need_clarification:
        # Ask user for clarification
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        # Proceed to write research brief
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


# Step 2: Research brief generation node
def write_research_brief(state: AgentState):
    """Transform conversation history into a research brief."""
    structured_output_model = model.with_structured_output(ResearchQuestion)
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }
