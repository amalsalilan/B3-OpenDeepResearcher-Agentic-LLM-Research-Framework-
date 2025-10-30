from datetime import datetime
from typing import Literal, Annotated
import os 
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string, BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv 
from state_scope import AgentState, AgentInputState, ClarifyWithUser, ResearchQuestion
from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt

load_dotenv() 

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")
# Initialize the chat model (Gemini 2.5 Flash Lite)
model = init_chat_model(
    model="gemini-2.5-flash-lite",
    model_provider="google_genai",
    temperature=0.0,
    google_api_key=os.environ.get("GEMINI_API_KEY") 
)
# Implement the clarification node
def clarify_with_user(state: AgentState) -> dict:
    """Determine if the user's request needs clarification and updates state."""
    structured_output_model = model.with_structured_output(ClarifyWithUser)    
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages=state["messages"]), 
        date=get_today_str()
    )   
    response = structured_output_model.invoke([HumanMessage(content=prompt_content)])
    if response.need_clarification:
        return {"messages": [AIMessage(content=response.question)]}
    else:
        return {"messages": [AIMessage(content=response.verification)]}
# Implement the research brief generation node
def write_research_brief(state: AgentState) -> dict:
    """Transform conversation history into a research brief."""
    structured_output_model = model.with_structured_output(ResearchQuestion)
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = structured_output_model.invoke([HumanMessage(content=prompt_content)])
    
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"Research Brief:\n{response.research_brief}")]
    }


def should_continue(state: AgentState) -> str:
    """Decide whether to proceed to brief generation or stop for user input."""
    last_message = state['messages'][-1]
    # Check for the verification phrases returned by the LLM in clarify_with_user
    if last_message.content.lower().startswith("i will research") or last_message.content.lower().startswith("i'm starting research"):
        return "write_research_brief"
    else:
        return END
# --- Graph Construction ---
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_conditional_edges(
    "clarify_with_user",
    should_continue,
    {
        "write_research_brief": "write_research_brief", 
        END: END                                        
    }
)

deep_researcher_builder.add_edge("write_research_brief", END)

checkpointer = InMemorySaver()
scope = deep_researcher_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    
    thread_id = "live_research_thread" 

    user_initial_question = input("USER (Ask a  question): ")
    
    initial_input = {
        "messages": [HumanMessage(content=user_initial_question)]
    }

    print("\n--- AGENT PROCESSING: CLARIFICATION CHECK ---")
    
    turn_1_result = scope.invoke(
        initial_input,
        config={"configurable": {"thread_id": thread_id}} 
    )

    clarification_question = turn_1_result['messages'][-1].content
    
    print(f"\nAGENT RESPONSE:\n{clarification_question}")
    
    if turn_1_result.get('research_brief'):
        print("\nError: Brief generated unexpectedly.")
        exit() 

    user_clarification = input("\nUSER (Provide clarification): ")
    turn_2_input = {
        "messages": [HumanMessage(content=user_clarification)]
    }
    print("\n--- AGENT PROCESSING: RESEARCH BRIEF GENERATION ---")
    
    final_result = scope.invoke(
        turn_2_input,
        config={"configurable": {"thread_id": thread_id}}
    )
    
    print("\nAGENT VERIFICATION:")
    print(f"{final_result['messages'][-1].content}")
    print("\n--- GENERATED RESEARCH BRIEF ---")
    print(f"{final_result.get('research_brief')}")
    print("==================================================")