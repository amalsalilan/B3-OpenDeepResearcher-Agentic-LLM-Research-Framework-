# prompts.py

clarify_with_user_instructions = """
You are a research assistant helping to clarify research requests.
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
"""


transform_messages_into_research_topic_prompt = """
Based on the conversation history, create a comprehensive research brief.
The brief should include:
- Clear research objectives
- Key questions to answer
- Scope and limitations
- Any specific requirements mentioned

Conversation History:
{messages}

Today's date: {date}
"""
