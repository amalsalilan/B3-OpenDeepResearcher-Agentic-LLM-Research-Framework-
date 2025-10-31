# B3 OpenDeepResearcher - Agentic LLM Research Framework

This repository contains assignments and implementations for building AI agents using LangGraph and various LLM models.

## Table of Contents
- [Assignment 1: Simple Chatbot with LangGraph and Gemini 2.0 Flash](#assignment-1-simple-chatbot-with-langgraph-and-gemini-20-flash)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)

## Assignment 1: Simple Chatbot with LangGraph and Gemini 2.0 Flash

### Objective
Create a basic chatbot using LangGraph that takes a user message, sends it to the Google Gemini 2.0 Flash model, and prints the model's reply in the console.

### Features
- ✅ State management using TypedDict with message accumulation
- ✅ LangGraph implementation with nodes and edges
- ✅ Integration with Google Gemini 2.0 Flash model
- ✅ Environment variable management for API keys
- ✅ Clean console output formatting

### How It Works

The chatbot follows a simple flow:

1. **State Definition**: Uses a `State` TypedDict with annotated messages that merge instead of replace
2. **Model Initialization**: Loads Google API key from `.env` and initializes Gemini 2.0 Flash
3. **Chatbot Node**: Processes incoming messages and invokes the LLM to get responses
4. **Graph Flow**: START → chatbot → END
5. **Execution**: Takes user input, processes through the graph, and prints the response

### File: `assignment1_chatbot.py`

The implementation includes:
- Import statements for typing, LangGraph, and model initialization
- State structure with message merging capability
- Environment variable loading
- Gemini 2.0 Flash model initialization
- Chatbot node function that reads state and invokes the model
- Graph builder with node and edge definitions
- Test execution with example question

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Google AI API key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amalsalilan/B3-OpenDeepResearcher-Agentic-LLM-Research-Framework-.git
   cd B3-OpenDeepResearcher-Agentic-LLM-Research-Framework-
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env and add your Google API key
   # GOOGLE_API_KEY=your_actual_api_key_here
   ```

### Running Assignment 1

```bash
python assignment1_chatbot.py
```

**Expected Output**:
```
============================================================
User: What do you know about LangGraph?
============================================================

Chatbot: [Model's response about LangGraph will appear here]

============================================================
```

### Customizing the Chatbot

You can modify the chatbot by:

1. **Changing the test message** in the `__main__` section:
   ```python
   test_message = "Your custom question here"
   run_chatbot(test_message)
   ```

2. **Adjusting model parameters** in the initialization:
   ```python
   llm = ChatGoogleGenerativeAI(
       model="gemini-2.0-flash-exp",
       google_api_key=api_key,
       temperature=0.7  # Adjust creativity (0.0-1.0)
   )
   ```

3. **Adding multiple interactions**:
   ```python
   run_chatbot("First question")
   run_chatbot("Follow-up question")
   ```

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

## Dependencies

- **langgraph**: Framework for building stateful agent applications
- **langchain**: LLM application framework
- **langchain-google-genai**: Google Generative AI integration
- **python-dotenv**: Environment variable management
- **langchain-core**: Core LangChain functionality

## Assignment Submission Guidelines

For interns submitting Assignment 1, please include:

1. **GitHub Link**: Link to your Python script (fork or your own repository)
2. **Console Screenshot**: Screenshot showing the chatbot's response
3. **Brief Explanation**: 2-3 lines explaining your implementation flow

### Example Explanation:
> "My chatbot uses LangGraph to create a state graph with a single chatbot node. When invoked, it takes the user message from state, passes it to Gemini 2.0 Flash via LangChain, and returns the AI response which gets added back to the state. The flow is: START → chatbot node → END."

## Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   ValueError: GOOGLE_API_KEY not found in environment variables
   ```
   **Solution**: Make sure you've created a `.env` file with your API key.

2. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'langgraph'
   ```
   **Solution**: Run `pip install -r requirements.txt`

3. **Model Not Found**:
   ```
   Error: Model not found
   ```
   **Solution**: Verify you're using the correct model name `gemini-2.0-flash-exp` or check available models.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)

## Contact

For questions or support, please open an issue in this repository.

---

**Happy Coding! 🚀**
