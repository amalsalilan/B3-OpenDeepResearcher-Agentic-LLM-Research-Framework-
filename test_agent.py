# Test script for the research scoping agent

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from research_agent_scope import deep_researcher_builder

# Compile with in-memory checkpointer for testing
checkpointer = InMemorySaver()
scope = deep_researcher_builder.compile(checkpointer=checkpointer)

# Test with a sample query
result = scope.invoke({
    "messages": [HumanMessage(content="find recent research on renewable energy technologies.")]
}, config={"configurable": {"thread_id": "test-thread"}})

print("Research Brief:")
print(result.get("research_brief", "No brief generated"))
print("\nFinal Report:")
print(result.get("final_report", "No report generated"))
