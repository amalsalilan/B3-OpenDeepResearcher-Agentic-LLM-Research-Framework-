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


class AgentState(MessagesState):
    """Main state for the scoping agent."""
    research_brief: Optional[str] = None
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] = []
    clarify_needed: Optional[bool] = None
    verification: Optional[str] = None


class ClarifyWithUser(BaseModel):
    """Schema for determining if user request needs clarification."""
    need_clarification: bool = Field(description="Whether clarification is needed")
    question: str = Field(description="Question to ask the user for clarification")
    verification: str = Field(description="Verification message confirming understanding")

class AgentInputState(TypedDict):
    """Input schema for the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# class ResearchQuestion(BaseModel):
#     """Schema for generating a structured research brief."""
#     research_brief: str = Field(description="Comprehensive research brief")


clarify_with_user_instructions = """You are a research assistant helping to clarify research requests.
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


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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
    
    # Build prompt based on message buffer
    prompt = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages=messages),
        date=get_today_str()
    )
    
    try:
        ai_response = model.invoke([HumanMessage(content=prompt)])
        response_text = ai_response.content
        # print('\n',response_text,'\n')    
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
    }

# --- 6. Graph Construction ---

print("Assembling graph...")

class AgentInputState(TypedDict):
    """Input schema for the graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Build the scoping workflow
builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
builder.add_node("clarify_with_user", clarify_with_user)

# Add workflow edges
builder.add_edge(START, "clarify_with_user")
builder.add_edge("clarify_with_user", END)

checkpointer = MemorySaver()
scope_research = builder.compile(checkpointer=checkpointer)

print("Graph compiled successfully!")

# --- 7. Testing the Agent ---

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
            
            print("\nChatbot:")
            # .stream() will run until it hits an END
            # If clarification is needed, the router sends to END,
            # so the loop stops and waits for the next user input.
            for event in scope_research.stream(inputs, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                
                # We only want to print messages from the AI
                if isinstance(last_message, AIMessage):
                    print(last_message.content)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Restarting conversation thread.")
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

if __name__ == "__main__":
    run_chat()