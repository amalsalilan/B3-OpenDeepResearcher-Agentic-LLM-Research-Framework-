# 🧠 Assignment 1: Build a Simple Chatbot using LangGraph and Gemini 2.0 Flash

### 🎯 Objective
Create a basic chatbot using **LangGraph** that takes a user message, sends it to the **Google Gemini 2.0 Flash** model, and prints the model’s reply in the console.

---

## 🚀 Project Overview
This chatbot demonstrates how to integrate **LangGraph** with **Google’s Gemini 2.0 Flash model** via **LangChain**.  
The chatbot accepts a user message, processes it through Gemini, and displays the response — showcasing state-based flow management using LangGraph.

---

## 🧩 Technologies Used
- **Python 3.10+**
- **LangGraph** (for graph-based state management)
- **LangChain Google GenAI** (for Gemini model integration)
- **dotenv** (for API key loading)
- **Google Gemini 2.0 Flash**

---

## 📁 Project Structure

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <your-repo-link>
cd Simple-LangGraph-Chatbot
## **2️⃣ Create and Activate a Virtual Environment**
python -m venv venv
venv\Scripts\activate
## **3️⃣ Install Dependencies**
pip install langgraph langchain-google-genai python-dotenv
## 4️⃣** Add Your Google API Key**
GOOGLE_API_KEY=your_google_api_key_here

## **🧠 How the Chatbot Works**

#State Definition:
A ChatState object stores all chat messages in a list.

#State Merging:
New messages are appended instead of replacing old ones, preserving full conversation context.

#LangGraph Setup:
The chatbot flow connects START → chatbot → END, ensuring a single, clear message pass.

#Gemini Model Integration:
The chatbot node reads user input, sends it to Gemini 2.0 Flash, and stores the reply in the state.

#Console Output:
The final model response is printed directly to the terminal.

## **▶️ Run the Chatbot**
Option 1: Single Message Mode (default)
python chatbot_langgraph.py


Console output example:

✅ API key loaded!

🤖 Chatbot Reply:
LangGraph is a Python framework that helps structure and manage AI workflows...

Option 2: Interactive Chat Mode (optional)

You can replace the last section with this interactive loop:

print("\n💬 Type 'exit' to end the chat.\n")
chat_history = {"messages": []}

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    chat_history["messages"].append(user_input)
    chat_history = graph.invoke(chat_history)
    print("🤖:", chat_history["messages"][-1])

## **📸 Example Output**
✅ API key loaded!

🤖 Chatbot Reply:
LangGraph is a powerful framework for building structured AI workflows that integrate LLMs and external tools efficiently.

## **🏁 Summary**

This assignment demonstrates:
How to integrate LangGraph with Gemini 2.0 Flash
How to define and merge conversational state
How to create a clean, stateful chatbot workflow using LangChain’s Google GenAI module

Author: Mukul
Date: October 2025
Assignment: Springboard AI Internship – Assignment 1
