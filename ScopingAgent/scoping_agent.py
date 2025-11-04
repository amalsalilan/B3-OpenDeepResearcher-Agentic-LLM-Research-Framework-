import os
import dotenv
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Optional, Literal, Dict, Any, List
import operator
import uuid
import json 
import re
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import  HumanMessage, AIMessage, BaseMessage, get_buffer_string

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI  

dotenv.load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file. Please add it.")

MAX_CLARIFICATIONS = 5


class AgentState(MessagesState):
    """Main state for the scoping agent."""
    research_brief: Optional[Dict[str, Any]] = None
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    clarify_needed: Optional[bool] = None
    verification: Optional[str] = None
    clarification_count: int = 0 # <-- NEW

class ClarifyWithUser(BaseModel):
    """Schema for determining if user request needs clarification."""
    need_clarification: bool = Field(description="Whether clarification is needed")
    question: str = Field(description="Question to ask the user for clarification")
    verification: str = Field(description="Verification message confirming understanding")

class AgentInputState(TypedDict):
    """Input schema for the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

class ResearchQuestion(BaseModel):
    """Schema for generating a structured research brief."""
    research_brief: str = Field(description="Comprehensive research brief")


clarify_with_user_instructions = """You are a research assistant helping to clarify research requests.
You have already asked {clarification_count} questions.
If you have enough information, *strongly* prefer to set `need_clarification` to `false` and generate a verification.
Only ask another question if it is *absolutely impossible* to proceed.

Given the conversation history, determine if you need more information.
Conversation History:
{messages}
Today's date: {date}
If the request is clear enough to begin research, respond with:
- need_clarification: false
- verification: A brief confirmation of what you'll research
- question: "N/A"
If you need more information, respond with:
- need_clarification: true
- verification: "N/A"
- question: A specific question to clarify the research needs
Respond *only* with the JSON object for the tool call."""

generate_research_brief_instructions = """You are a senior research analyst. Your task is to generate a comprehensive research brief based on the confirmed understanding of the user's request.
Use the conversation history and the final verification message to create a detailed brief.
The brief should outline the main research question, key objectives, and the scope of the investigation.
Conversation History:
{messages}

Confirmed Research Topic:
{verification}

Respond *only* with a JSON object containing the 'research_brief' key. This key's value should be *another* JSON object containing the brief's details (title, main_research_question, key_objectives, etc.)."""


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.0,
    convert_system_message_to_human=True
)

def clarify_with_user(state: AgentState) -> Dict[str, Any]:
    """
    Decide whether we need clarification. Returns a partial state dict.
    This node updates the state with the model's decision.
    """
    print("--- [Node]: clarify_with_user ---")
    messages = state.get("messages", [])
    count = state.get("clarification_count", 0)

    if count >= MAX_CLARIFICATIONS:
        print(f"--- Clarification limit ({MAX_CLARIFICATIONS}) reached. Forcing brief generation. ---")
        
        # create a verification message based on history
        history = get_buffer_string(messages=messages)
        verification_prompt = f"Based on this entire conversation, create a single, comprehensive 1-sentence verification message confirming the user's final request. Conversation: {history}"
        
        try:
            verification_msg = model.invoke([HumanMessage(content=verification_prompt)]).content
        except Exception as e:
            print(f"Error generating forced verification: {e}")
            verification_msg = "I will proceed based on our conversation."
        
        print(f"Forced Verification: {verification_msg}")
        ai_msg = AIMessage(content=verification_msg)
        
        return {
            "messages": [ai_msg],
            "clarify_needed": False,
            "verification": verification_msg,
            "clarification_count": count + 1 
        }

    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages=messages),
        date=get_today_str(),
        clarification_count=count
    )
    
    try:
        ai_response = model.invoke([HumanMessage(content=prompt)])
        response_text = ai_response.content
        
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_string = match.group(0)
            response_data = json.loads(json_string)
        else:
            raise ValueError("No JSON block found in LLM response.")
            
        # print(f"\nAnalysis (raw dict): {response_data}\n")

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        response_data = {
            "need_clarification": True,
            "question": "I'm sorry, I had an error processing your request. Could you rephrase it?"
        }

    if response_data.get("need_clarification") == True:
        ai_msg_content = response_data.get("question", "Could you please clarify your request?")
        ai_msg = AIMessage(content=ai_msg_content)
        clarify_needed = True
        verification = None
    else:
        ai_msg_content = response_data.get("verification", "Got it. Proceeding with the research.")
        ai_msg = AIMessage(content=ai_msg_content)
        clarify_needed = False
        verification = ai_msg_content
        
    return {
        "messages": [ai_msg],
        "clarify_needed": clarify_needed,
        "verification": verification,
        "clarification_count": count + 1, 
    }

def format_brief(brief_dict: Dict[str, Any]) -> str:
    """Converts the research brief dictionary into a clean, human-readable string."""
    try:
        lines = []
        lines.append(f"## {brief_dict.get('title', 'Research Brief')}")
        lines.append(f"**Date:** {brief_dict.get('date', 'N/A')}  ")
        lines.append(f"**To:** {brief_dict.get('prepared_for', 'N/A')}  ")
        lines.append(f"**From:** {brief_dict.get('prepared_by', 'N/A')}")
        
        lines.append("\n### 1. Main Research Question")
        lines.append(brief_dict.get('main_research_question', 'N/A'))
        
        lines.append("\n### 2. Key Objectives")
        objectives = brief_dict.get('key_objectives', [])
        if objectives and isinstance(objectives, list):
            for obj in objectives:
                lines.append(f"* **{obj.get('objective_id', 'Objective')}:** {obj.get('description', 'N/A')}")
        else:
            lines.append("N/A")
            
        lines.append("\n### 3. Scope of Investigation")
        scope = brief_dict.get('scope_of_investigation', {})
        if not isinstance(scope, dict):
            raise ValueError("Scope is not a dictionary.")
            
        lines.append("\n**In Scope:**")
        in_scope = scope.get('in_scope', [])
        if in_scope and isinstance(in_scope, list):
            for item in in_scope:
                lines.append(f"* {str(item).replace('**', '')}") # Clean up just in case
        else:
            lines.append("N/A")

        lines.append("\n**Out of Scope:**")
        out_of_scope = scope.get('out_of_scope', [])
        if out_of_scope and isinstance(out_of_scope, list):
            for item in out_of_scope:
                lines.append(f"* {str(item).replace('**', '')}") # Clean up just in case
        else:
            lines.append("N/A")

        return "\n".join(lines)
    except Exception as e:
        print(f"Error formatting brief: {e}")
        # Fallback if formatting fails
        return json.dumps(brief_dict, indent=2)


def generate_research_brief(state: AgentState) -> Dict[str, Any]:
    """
    Generate a comprehensive research brief based on the conversation.
    """
    print("--- [Node]: generate_research_brief ---")
    messages = state.get("messages", [])
    verification = state.get("verification", "No verification message found.")
    
    prompt = generate_research_brief_instructions.format(
        messages=get_buffer_string(messages=messages),
        verification=verification
    )
    
    try:
        ai_response = model.invoke([HumanMessage(content=prompt)])
        response_text = ai_response.content
        
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_string = match.group(0)
            response_data = json.loads(json_string)
            research_brief_data = response_data.get("research_brief", {"error": "Key 'research_brief' not found in LLM response."})
            if not isinstance(research_brief_data, dict):
                research_brief_data = {"error": "LLM returned 'research_brief' but it was not a valid dictionary.", "content": research_brief_data}
        else:
            raise ValueError("No JSON block found in LLM response.")
            
        # print(f"\nGenerated Brief (raw dict):\n{research_brief_data}\n")
        
        formatted_brief = format_brief(research_brief_data)
        ai_msg_content = f"I have generated the research brief:\n\n{formatted_brief}\n\nI will now proceed with the next steps."
        ai_msg = AIMessage(content=ai_msg_content)

        return {
            "messages": [ai_msg],
            "research_brief": research_brief_data # Store the dict in the state
        }
    except Exception as e:
        print(f"Error generating research brief: {e}")
        ai_msg = AIMessage(content="I'm sorry, I encountered an error while generating the research brief. Please try rephrasing your request.")
        return {
            "messages": [ai_msg],
            "research_brief": None
        }

# --- Router function---
def should_continue(state: AgentState) -> Literal["generate_brief", "end"]:
    """
    Router to decide the next step after clarification check.
    """
    print("--- [Router]: should_continue ---")
    if state.get("clarify_needed") == False:
        # print("Decision: Proceed to generate_brief")
        return "generate_brief"
    else:
        # print("Decision: End run, wait for user clarification")
        return "end"

# --- Graph Construction ---

print("Assembling graph...")

class AgentInputState(TypedDict):
    """Input schema for the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Build
builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("generate_research_brief", generate_research_brief)

# Add workflow edges
builder.add_edge(START, "clarify_with_user")

builder.add_conditional_edges(
    "clarify_with_user",
    should_continue,
    {
        "generate_brief": "generate_research_brief",
        "end": END
    }
)

builder.add_edge("generate_research_brief", END)

checkpointer = MemorySaver()
scope_research = builder.compile(checkpointer=checkpointer)

print("Graph compiled successfully!")

def run_chat():
    print("=========================================")
    print("Research Scoping Agent Initialized.")
    print("Type 'q' end.")
    print("=========================================")
    
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'q':
                print("Chatbot: Goodbye!")
                break
                
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            print("\Agent:")
            for event in scope_research.stream(inputs, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                
                if isinstance(last_message, AIMessage):
                    print(last_message.content)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Restarting conversation thread.")
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

if __name__ == "__main__":
    run_chat()