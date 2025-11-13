# langgraph_chatbot_colab.py

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import os
# Set your Google API key directly
GOOGLE_API_KEY = "AIzaSyCTo80o8jZG3ej4FnTyNxMGKaVk6Hby1Tk"  # Replace with your real key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Define the chatbot state
class ChatState(TypedDict):
    messages: Annotated[List, "Holds the list of chat messages"]
# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)
# Define chatbot node function
def chatbot_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return {"messages": messages}
# Build the LangGraph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

chat_graph = graph_builder.compile()
# Run test message

initial_state = {"messages": [HumanMessage(content="What do you know about LangGraph?")]}
result = chat_graph.invoke(initial_state)

print("\n Model Response:\n")
print(result["messages"][-1].content)