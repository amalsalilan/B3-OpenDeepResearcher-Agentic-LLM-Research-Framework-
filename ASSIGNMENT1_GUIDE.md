# Assignment 1 - Quick Reference Guide

## For Interns: How to Complete This Assignment

### What You Need to Submit

1. **GitHub Link**: Link to your implementation of `assignment1_chatbot.py`
2. **Console Screenshot**: Screenshot showing the chatbot's response
3. **Brief Explanation** (2-3 lines): How your chatbot flow works

---

## Step-by-Step Instructions

### 1. Fork or Clone This Repository

```bash
git clone https://github.com/amalsalilan/B3-OpenDeepResearcher-Agentic-LLM-Research-Framework-.git
cd B3-OpenDeepResearcher-Agentic-LLM-Research-Framework-
```

### 2. Set Up Your Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Your API Key

1. Get your Google AI API key from: https://makersuite.google.com/app/apikey
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### 4. Review the Code

Open `assignment1_chatbot.py` and understand:

- **Lines 11-17**: Imports for typing, LangGraph, and model
- **Lines 22-31**: State definition with message merging
- **Lines 35-50**: Environment loading and model initialization
- **Lines 53-67**: Chatbot node function
- **Lines 70-82**: Graph building (node, edges, compilation)
- **Lines 85-103**: Run function and test execution

### 5. Run the Chatbot

```bash
python assignment1_chatbot.py
```

### 6. Take a Screenshot

Capture the console output showing:
- Your test message
- The chatbot's response

### 7. Write Your Explanation

Example format:

> "My chatbot uses LangGraph's StateGraph to manage conversation state. When invoked, it passes user messages through a chatbot node that calls Gemini 2.0 Flash. The model's response is added to the state and returned. Flow: START → chatbot node → END."

### 8. Submit Your Work

Comment on the GitHub issue with:
- Link to your code (fork/your repo)
- Screenshot of the console output
- Your 2-3 line explanation

---

## Key Concepts Explained

### State Management
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
- `TypedDict`: Defines the structure of your state
- `Annotated[list, add_messages]`: Messages are merged, not replaced

### LangGraph Flow
```python
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```
- Creates a linear flow: START → chatbot → END
- The chatbot node processes messages and returns responses

### Model Invocation
```python
response = llm.invoke(messages)
```
- Sends all messages to Gemini 2.0 Flash
- Returns the AI's response

---

## Customization Ideas

Want to experiment? Try:

1. **Change the test message**:
   ```python
   test_message = "Explain quantum computing in simple terms"
   ```

2. **Adjust temperature** (creativity level):
   ```python
   llm = ChatGoogleGenerativeAI(
       model="gemini-2.0-flash-exp",
       temperature=0.9  # More creative (0.0-1.0)
   )
   ```

3. **Add multiple questions**:
   ```python
   run_chatbot("What is LangGraph?")
   run_chatbot("How does it compare to other frameworks?")
   ```

---

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure you created `.env` file (not `.env.example`)
- Check that your API key is correctly pasted

### "ModuleNotFoundError"
- Run `pip install -r requirements.txt`
- Make sure your virtual environment is activated

### "Model not found"
- The model name might have changed
- Check Google AI Studio for current model names
- Try `gemini-pro` as an alternative

---

## Good Luck! 🚀

Remember:
- ✅ Test your code before submitting
- ✅ Take a clear screenshot
- ✅ Write a concise explanation
- ✅ Submit before the deadline

If you have questions, ask in the issue comments!
