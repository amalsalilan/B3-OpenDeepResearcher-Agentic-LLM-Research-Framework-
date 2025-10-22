
# Create README.md file content
readme_content = '''# Simple Chatbot with LangGraph and Gemini 2.0 Flash

A lightweight conversational AI chatbot built using LangGraph framework and Google's Gemini 2.0 Flash model. This implementation demonstrates stateful message management and clean integration with Google's generative AI capabilities.

## Features

 State management using TypedDict with message accumulation  
 LangGraph implementation with nodes and edges  
 Integration with Google Gemini 2.0 Flash model  
 Environment variable management for API keys  
 Clean console output formatting

## How It Works

The chatbot follows a simple flow:

1. **State Definition**: Uses a State TypedDict with annotated messages that merge instead of replace
2. **Model Initialization**: Loads Google API key from .env and initializes Gemini 2.0 Flash
3. **Chatbot Node**: Processes incoming messages and invokes the LLM to get responses
4. **Graph Flow**: START → chatbot → END
5. **Execution**: Takes user input, processes through the graph, and prints the response

## Project Structure

```
.
├── assignment1_chatbot.py    # Main chatbot implementation
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables file
├── .gitignore               # Git ignore patterns
├── README.md                # This file
└── LICENSE                  # MIT License
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Google API key for Gemini

### Setup Steps

1. **Clone or download this repository**
   ```bash
   cd chatbot-langgraph
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\\Scripts\\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   
   To get your API key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key and paste it in your `.env` file

## Usage

Run the chatbot:
```bash
python assignment1_chatbot.py
```

The script will execute with a default test message and display the conversation in the console.

### Customize User Input

Modify the `user_message` variable in `assignment1_chatbot.py`:
```python
user_message = "Your custom question here"
```

### Interactive Mode (Optional Enhancement)

For continuous conversation, you can modify the main block to include a loop:
```python
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        
        result = app.invoke({"messages": [("user", user_input)]})
        response = result["messages"][-1].content
        print(f"Assistant: {response}\\n")
```

## Dependencies

- **langgraph**: Framework for building stateful agent applications
- **langchain**: LLM application framework
- **langchain-google-genai**: Google Generative AI integration
- **python-dotenv**: Environment variable management
- **langchain-core**: Core LangChain functionality

## Implementation Details

### State Management

The chatbot uses a TypedDict-based state schema with message annotation:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The `add_messages` annotation ensures that new messages are appended to the existing list rather than replacing it, enabling conversation history.

### Graph Structure

The LangGraph implementation creates a simple linear flow:
- **START node**: Entry point
- **chatbot node**: Processes messages and generates responses
- **END node**: Exit point

### Model Configuration

The chatbot uses Gemini 2.0 Flash experimental model with:
- Model: `gemini-2.0-flash-exp`
- Temperature: 0.7 (balanced creativity)
- API authentication via environment variable

## Troubleshooting

### API Key Issues
- **Error**: "GOOGLE_API_KEY not found"
- **Solution**: Ensure `.env` file exists in the project root and contains the correct API key

### Import Errors
- **Error**: "No module named 'langgraph'"
- **Solution**: Activate virtual environment and run `pip install -r requirements.txt`

### Model Not Found
- **Error**: Model name not recognized
- **Solution**: The model name is `gemini-2.0-flash-exp` for the experimental version

## Contributing

This is an educational project. Feel free to fork and modify for your learning purposes.

## License

MIT License - Feel free to use this code for learning and development.

## Resources
These resources were helpful in developing this chatbot:

LangGraph Documentation,Official documentation for building stateful agent workflows.,https://docs.langchain.com/oss/python/langgraph/overview
LangChain Documentation,Official documentation for the core LLM application framework.,https://docs.langchain.com/
Google AI Studio,Platform for managing your API key and prototyping with Gemini models.,https://ai.google.dev/aistudio
Gemini API Documentation,Official documentation for integrating with the Gemini family of models.,https://ai.google.dev/docs/gemini_api_overview

## Acknowledgments

LangGraph,GitHub repository for the agent orchestration framework.,https://github.com/langchain-ai/langgraph
Google Gemini 2.5 Flash,Official product page and technical overview.,https://deepmind.google/technologies/gemini/
LangChain,GitHub repository for the core LLM framework.,https://github.com/langchain-ai/langchain