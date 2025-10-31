# Test script for the research scoping agent

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from research_agent_scope import deep_researcher_builder

# Compile with in-memory checkpointer for testing
checkpointer = InMemorySaver()
scope = deep_researcher_builder.compile(checkpointer=checkpointer)

# Test cases
test_cases = [
    {"name": "Research on AI advancements", "messages": [HumanMessage(content="Tell me about recent advancements in artificial intelligence.")]},
    {"name": "Unclear business query", "messages": [HumanMessage(content="I need help with my business.")]},
    {"name": "Health and wellness", "messages": [HumanMessage(content="What are some effective ways to improve mental health?")]},
    {"name": "Technology comparison", "messages": [HumanMessage(content="Compare Python and JavaScript for web development.")]},
    {"name": "Travel planning with partial details", "messages": [
        HumanMessage(content="Plan a vacation to Europe."),
        AIMessage(content="To plan your vacation, could you specify which countries or cities in Europe you're interested in, your travel dates, budget, and group size?"),
        HumanMessage(content="Paris and Rome, next summer, $3000 budget, solo traveler.")
    ]},
    {"name": "Education research", "messages": [HumanMessage(content="Research online learning platforms for data science.")]},
    {"name": "Empty query", "messages": [HumanMessage(content="")]},
    {"name": "Invalid symbols", "messages": [HumanMessage(content="!@#$%^&*()")]},
]

for i, test_case in enumerate(test_cases):
    print(f"\n--- Test {i+1}: {test_case['name']} ---")
    try:
        result = scope.invoke({
            "messages": test_case["messages"]
        }, config={"configurable": {"thread_id": f"test-thread-{i}"}})
        print("Research Brief:")
        brief = result.get("research_brief", "No brief generated")
        print(brief)
        if "No brief generated" in brief:
            print("Clarification or error occurred.")
    except Exception as e:
        print(f"Error: {e}")
