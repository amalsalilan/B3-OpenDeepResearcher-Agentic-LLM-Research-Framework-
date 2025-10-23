from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from dotenv import load_dotenv # type: ignore
import os

# Step 1: Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Step 2: Initialize Gemini 2.0 Flash model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key
)

# Step 3: Define state structure
class ChatState(TypedDict):
    messages: Annotated[list, "List of messages"]

# Step 4: Define chatbot node
def chatbot_node(state: ChatState):
    user_message = state["messages"][-1]
    response = model.invoke(user_message)
    return {"messages": state["messages"] + [response.content]}

# Step 5: Build LangGraph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Step 6: Interactive chat loop
print("🤖 Gemini Chatbot (type 'exit' or 'quit' to stop)\n")

state = {"messages": []}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Add user message to conversation
    state["messages"].append(user_input)

    # Run graph
    result = graph.invoke(state)

    # Get latest bot reply
    bot_reply = result["messages"][-1]
    print(f"🤖: {bot_reply}")

    # Keep full conversation in memory
    state = result
