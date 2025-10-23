# langgraph_chatbot.py
"""
Interactive LangGraph -> Google Gemini 2.0 Flash chatbot.
Sends conversation history to Gemini and prints only the model's final reply.
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
# Load .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
else:
    print("API key loaded successfully!")


# Define state as TypedDict (no need for langgraph.types.State)
class ChatState(TypedDict):
    messages: List[dict]

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)

# Define the chatbot node
def chatbot_node(state: ChatState) -> ChatState:
    user_messages = state["messages"]
    response = model.invoke(user_messages)
    # Ensure we append the assistant’s reply to the message list
    reply_content = getattr(response, "content", str(response))
    return {"messages": user_messages + [{"role": "assistant", "content": reply_content}]}



# --- Build the LangGraph flow ---
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

# --- Interactive console loop ---
def run_interactive():
    current_state = {"messages": []}  # Initialize state

    print("\nLangGraph + Gemini Chatbot (type 'exit' to quit)\n")

    while True:
        try:
            user_text = input("You: ")
        except KeyboardInterrupt:
            print("\nExiting chatbot...")
            break

        if user_text.lower() in ("exit", "quit"):
            break

        # Append user message
        current_state["messages"].append({"role": "user", "content": user_text})

        # Get model reply
        response_state = chatbot_node(current_state)
        assistant_reply = response_state["messages"][-1]["content"]

        # Print assistant reply
        print("AI:", assistant_reply)


if __name__ == "__main__":
    run_interactive()
