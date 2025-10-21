"""
Assignment 1: Simple Chatbot using LangGraph and Gemini 2.0 Flash

This script demonstrates a basic chatbot implementation that:
1. Takes user messages
2. Sends them to Google Gemini 2.0 Flash model
3. Returns and prints the model's response
"""

# Import required modules
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


# Step 1: Define the state structure
# The state holds all chat messages and uses add_messages to merge them
class State(TypedDict):
    """
    State structure for the chatbot.
    
    Attributes:
        messages: List of chat messages that accumulates over the conversation.
                  Using Annotated with add_messages ensures messages are merged
                  rather than replaced.
    """
    messages: Annotated[list, add_messages]


# Step 2: Load environment variables and initialize the model
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )

# Initialize Gemini 2.0 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=api_key,
    temperature=0.7
)


# Step 3: Create the chatbot node function
def chatbot(state: State):
    """
    Chatbot node that processes messages.
    
    Args:
        state: Current state containing the message history
        
    Returns:
        Dictionary with the model's response message
    """
    # Get the current messages from state
    messages = state["messages"]
    
    # Send messages to the model and get response
    response = llm.invoke(messages)
    
    # Return the response (it will be merged into the state)
    return {"messages": [response]}


# Step 4: Build the graph
# Initialize the graph builder with our state
graph_builder = StateGraph(State)

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Step 5: Define the flow: START → chatbot → END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Step 6: Compile the graph
graph = graph_builder.compile()


# Step 7: Run the chatbot with a test message
def run_chatbot(user_message: str):
    """
    Run the chatbot with a user message and print the response.
    
    Args:
        user_message: The message from the user
    """
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}\n")
    
    # Invoke the graph with the user message
    result = graph.invoke({
        "messages": [("user", user_message)]
    })
    
    # Extract and print the final response
    final_message = result["messages"][-1]
    print(f"Chatbot: {final_message.content}\n")
    print(f"{'='*60}\n")
    
    return final_message.content


if __name__ == "__main__":
    # Test the chatbot with the example question
    test_message = "What do you know about LangGraph?"
    run_chatbot(test_message)
    
    # Optional: You can add more test messages
    # run_chatbot("Can you explain how state management works in LangGraph?")
