# Assignment 1: Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Assignment 1 Chatbot                      │
│                  LangGraph + Gemini 2.0 Flash                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Component Flow                          │
└─────────────────────────────────────────────────────────────┘

1. User Input
   └─> "What do you know about LangGraph?"

2. State Initialization
   └─> State(messages=[("user", "What do you know about LangGraph?")])

3. Graph Execution Flow
   ┌─────────┐     ┌──────────┐     ┌─────┐
   │  START  │────>│ chatbot  │────>│ END │
   └─────────┘     └──────────┘     └─────┘
                         │
                         ▼
                  [Reads State]
                         │
                         ▼
              [Invokes Gemini 2.0 Flash]
                         │
                         ▼
                  [Returns Response]
                         │
                         ▼
              [Updates State with Response]

4. Output
   └─> Chatbot: [AI Response]
```

## Code Structure Breakdown

### 1. Imports and Setup (Lines 11-17)
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
```

**Purpose**: Import all necessary libraries for:
- Type hints and state definition
- LangGraph components
- Gemini model integration
- Environment variable management

---

### 2. State Definition (Lines 22-31)
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Key Concept**: 
- `TypedDict`: Provides type safety for state structure
- `Annotated[list, add_messages]`: Special annotation that tells LangGraph to **merge** new messages into the list rather than **replace** the entire list

**Why it matters**: This allows conversation history to accumulate naturally.

---

### 3. Environment & Model Setup (Lines 35-50)
```python
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=api_key,
    temperature=0.7
)
```

**Components**:
- `load_dotenv()`: Loads `.env` file
- `os.getenv()`: Retrieves API key securely
- `ChatGoogleGenerativeAI`: LangChain wrapper for Gemini
- `temperature=0.7`: Controls response creativity (0.0 = deterministic, 1.0 = creative)

---

### 4. Chatbot Node (Lines 53-67)
```python
def chatbot(state: State):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

**Flow**:
1. Extract messages from current state
2. Send all messages to Gemini model
3. Get AI response
4. Return response (automatically merged into state)

**Note**: LangGraph automatically handles state merging because of the `add_messages` annotation.

---

### 5. Graph Construction (Lines 70-82)
```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

**Graph Structure**:
```
START ──[edge]──> chatbot ──[edge]──> END
         │                      │
         │                      └─> Node that processes messages
         └─> Entry point
```

**Purpose**: Creates a simple linear flow where:
1. Execution starts at START
2. Flows to the chatbot node
3. Completes at END

---

### 6. Execution Function (Lines 85-103)
```python
def run_chatbot(user_message: str):
    result = graph.invoke({
        "messages": [("user", user_message)]
    })
    final_message = result["messages"][-1]
    print(f"Chatbot: {final_message.content}")
```

**Process**:
1. Package user message into initial state
2. Invoke the compiled graph
3. Extract the last message (AI response)
4. Print formatted output

---

## State Management Deep Dive

### Without add_messages (Wrong):
```python
# Each update replaces the entire messages list
State 1: {"messages": [user_msg]}
State 2: {"messages": [ai_response]}  # ❌ Lost user message!
```

### With add_messages (Correct):
```python
# Each update merges into the messages list
State 1: {"messages": [user_msg]}
State 2: {"messages": [user_msg, ai_response]}  # ✅ History preserved!
```

---

## Data Flow Example

### Input
```python
graph.invoke({"messages": [("user", "What is LangGraph?")]})
```

### Step-by-Step Execution

1. **Initial State**:
   ```python
   {
     "messages": [
       HumanMessage(content="What is LangGraph?")
     ]
   }
   ```

2. **After Chatbot Node**:
   ```python
   {
     "messages": [
       HumanMessage(content="What is LangGraph?"),
       AIMessage(content="LangGraph is a framework...")
     ]
   }
   ```

3. **Final Output**:
   ```
   Chatbot: LangGraph is a framework for building stateful, 
   multi-agent applications with LLMs...
   ```

---

## Key Design Decisions

### 1. Why TypedDict?
- Provides type hints for better IDE support
- Ensures state structure consistency
- Enables static type checking

### 2. Why LangGraph?
- Built for stateful agent applications
- Native support for message history
- Easy to extend with more nodes/edges
- Production-ready framework

### 3. Why Gemini 2.0 Flash?
- Latest Google AI model
- Fast response times
- Cost-effective
- Strong performance on conversational tasks

### 4. Why This Flow Pattern?
- Simple: Easy to understand for beginners
- Extensible: Can add more nodes later
- Standard: Follows LangGraph best practices

---

## Extension Possibilities

### Add Memory
```python
def chatbot(state: State):
    # Now has access to full conversation history
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

### Add Multiple Nodes
```
START -> input_processor -> chatbot -> output_formatter -> END
```

### Add Conditional Routing
```python
def should_continue(state):
    if needs_clarification(state):
        return "clarify"
    return "end"

graph_builder.add_conditional_edges("chatbot", should_continue)
```

---

## Testing Checklist

- [ ] API key properly loaded from `.env`
- [ ] State merges messages correctly
- [ ] Graph compiles without errors
- [ ] Model responds to test message
- [ ] Output formatted correctly
- [ ] No runtime errors

---

## Performance Considerations

### Latency
- **Network Call**: ~1-3 seconds to Gemini API
- **Graph Execution**: <100ms overhead
- **Total**: ~1-3 seconds per message

### Token Usage
- Input tokens: Length of conversation history
- Output tokens: Length of AI response
- Cost: Based on Google's pricing for Gemini 2.0 Flash

### Optimization Tips
1. Keep conversation history manageable
2. Use streaming for long responses
3. Cache common queries
4. Implement rate limiting

---

## Common Patterns

### Single Turn Conversation
```python
run_chatbot("Hello!")
# No context from previous messages
```

### Multi-Turn Conversation
```python
# Build state with history
state = {"messages": [
    ("user", "My name is John"),
    ("assistant", "Nice to meet you John!"),
    ("user", "What's my name?")
]}
result = graph.invoke(state)
# AI can access full history
```

---

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect State
```python
result = graph.invoke({"messages": [...]})
print(f"Final state: {result}")
print(f"Message count: {len(result['messages'])}")
```

### Test Model Separately
```python
# Test LLM without graph
response = llm.invoke([("user", "Test")])
print(response.content)
```

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Google GenAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [Google AI Studio](https://makersuite.google.com/)
- [TypedDict Documentation](https://docs.python.org/3/library/typing.html#typing.TypedDict)

---

**Next Steps**: Complete Assignment 2 to learn about multi-node workflows!
