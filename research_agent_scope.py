from datetime import datetime
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from state_scope import AgentState, AgentInputState, ClarifyWithUser, ResearchQuestion
from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt, summarize_research_prompt
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

# Utility function for date formatting
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")

# Initialize the LLM
model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Tavily search tool
tavily_tool = TavilySearch(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)

# Implement the clarification node
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """Determine if the user's request needs clarification."""
    print(f"Clarify with user: messages = {state['messages']}")
    structured_output_model = model.with_structured_output(ClarifyWithUser)
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])
    print(f"Response: {response}")

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})

# Implement the research brief generation node
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

# Implement the research execution node
def perform_research(state: AgentState):
    """Perform research using Tavily and summarize with Gemini."""
    # Use the research_brief as the query
    research_brief = state.get("research_brief", "")
    if not research_brief:
        return {"final_report": "No research brief provided."}

    query = research_brief

    # Use Tavily to search for the query
    search_results = tavily_tool.invoke({"query": query})

    # Extract content from search results
    content = ""
    if isinstance(search_results, dict):
        results = search_results.get("results", [])
        for result in results:
            content += f"Title: {result.get('title', 'N/A')}\n"
            content += f"Content: {result.get('content', 'N/A')}\n\n"
    else:
        content = str(search_results)

    # Use Gemini to summarize the search results into a comprehensive report
    summary_prompt = summarize_research_prompt.format(query=query, search_results=content)
    summary_response = model.invoke([HumanMessage(content=summary_prompt)])

    return {"final_report": summary_response.content}

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("perform_research", perform_research)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("clarify_with_user", "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "perform_research")
deep_researcher_builder.add_edge("perform_research", END)

# Compile the workflow
scope_research = deep_researcher_builder.compile()
