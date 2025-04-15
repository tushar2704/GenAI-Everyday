# Simple AI Agent using Anthropic API

A lightweight, pure Python implementation of an AI agent that leverages the Anthropic Claude API without any external AI agent libraries. This project demonstrates fundamental concepts of AI agents including task planning, tool usage, memory management, and reasoning.

## Overview

This project provides a simple yet powerful framework for building AI agents from scratch. It's designed to be educational, showing how the core components of AI agents work without the abstraction of existing frameworks.

### Key Features

- **Pure Python**: No dependencies on AI agent libraries
- **Tool System**: Extensible framework for adding capabilities to your agent
- **Memory Management**: Conversation history and important fact storage
- **Step-by-Step Reasoning**: Transparent decision-making process
- **Anthropic Claude Integration**: Leverages state-of-the-art LLMs

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Component Breakdown](#component-breakdown)
  - [Tool Class](#tool-class)
  - [Memory Class](#memory-class)
  - [SimpleAgent Class](#simpleagent-class)
- [How It Works](#how-it-works)
- [Advanced Usage](#advanced-usage)
- [Customization](#customization)
- [Example Scenarios](#example-scenarios)
- [Common Issues](#common-issues)
- [Best Practices](#best-practices)
- [Contributing](#contributing)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/simple-ai-agent.git
cd simple-ai-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install anthropic
```

## Setup

1. Obtain an API key from Anthropic: [https://console.anthropic.com/](https://console.anthropic.com/)

2. Set your API key as an environment variable:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

Alternatively, you can modify the `main()` function to read the API key from a configuration file or pass it directly.

## Usage

### Basic Usage

The repository contains a single Python file (`ai_agent.py`) with a working example. Run it with:

```bash
python ai_agent.py
```

This will start an interactive session where you can chat with your agent. Type `exit` to quit.

### Example Interaction

```
SimpleAgent initialized. Type 'exit' to quit.

You: What's the current time?

Agent: I'll get the current time for you.

Reasoning:
You've asked for the current time, which I can retrieve using the get_time tool that's available to me. This tool doesn't require any parameters.

Current time: 2025-04-15 14:32:17

You: Can you calculate 15 * 24 + 7?

Agent: I'll calculate that expression for you.

Reasoning:
You've asked me to calculate the mathematical expression 15 * 24 + 7. I'll use the calculate tool for this.

The calculation 15 * 24 + 7 equals 367.
First, I multiply 15 by 24 to get 360, then add 7 to get 367.

You: exit
```

### Creating Your Own Script

Here's a minimal script to create and use the agent:

```python
from ai_agent import SimpleAgent, Tool

# Create the agent
agent = SimpleAgent(api_key="your-api-key-here")

# Define a simple tool
def hello_world(name="World"):
    return f"Hello, {name}!"

# Register the tool
agent.register_tool(Tool(
    "hello", 
    "Say hello to someone. Parameters: name (str, optional)",
    hello_world
))

# Use the agent
response = agent.process_input("Can you say hello to John?")
print(response)
```

## Code Structure

The implementation consists of three main classes:

1. **Tool**: Represents a function that the agent can use to interact with external systems
2. **Memory**: Manages conversation history and important facts
3. **SimpleAgent**: Main agent class that processes inputs and generates responses

## Component Breakdown

### Tool Class

The `Tool` class represents a capability that the agent can use to perform actions or retrieve information.

```python
Tool(
    name="search",  # Name used to call the tool
    description="Search the web for information. Parameters: query (str)",  # Explains how to use it
    function=search_web  # The actual function to call
)
```

Each tool has:
- A **name** for identification
- A **description** that explains its purpose and parameters
- A **function** that performs the actual work

### Memory Class

The `Memory` class stores:
- Conversation history (limited to the most recent exchanges)
- Important facts that should persist

```python
# Store user input
memory.add_interaction("user", "What's the weather today?")

# Remember important information
memory.remember_fact("user_location", "New York")

# Retrieve context for decision making
context = memory.get_context()
```

### SimpleAgent Class

The `SimpleAgent` class ties everything together:

1. Manages tools and memory
2. Processes user inputs
3. Calls the Anthropic API
4. Parses responses to execute tools when needed
5. Handles follow-up interactions

Key methods:
- `register_tool()`: Adds a new tool to the agent
- `process_input()`: Handles user requests
- `_parse_tool_call()`: Extracts tool usage from responses
- `_execute_tool()`: Runs tools when needed
- `_generate_follow_up()`: Creates follow-up responses after tool execution

## How It Works

1. **User Input Processing**:
   - User sends a message to the agent
   - The message is stored in memory
   - The agent formulates a prompt including the system instructions, memory context, and user message

2. **LLM Response Generation**:
   - The prompt is sent to the Anthropic API
   - The model generates a response that may include reasoning and tool usage

3. **Tool Usage (if needed)**:
   - If the response includes a tool call, the agent parses it
   - The tool is executed with the specified parameters
   - The result is stored in memory

4. **Follow-Up Handling**:
   - If a tool was used, the agent generates a follow-up response incorporating the tool results
   - The final response is returned to the user and stored in memory

## Advanced Usage

### Custom System Prompt

The system prompt defines the agent's behavior. You can customize it by modifying the `system_prompt` attribute:

```python
agent = SimpleAgent(api_key)
agent.system_prompt = """
You are a specialized financial advisor agent. Your role is to:
1. Help users understand their financial situation
2. Provide investment advice
3. Use tools to calculate financial metrics
4. Always explain financial terms in simple language

When you need to use a tool, respond in the following format:
...
"""
```

### Creating Complex Tools

Tools can perform complex operations and integrate with external services:

```python
import requests

def weather_tool(location: str) -> str:
    """Get weather information for a location."""
    api_key = "your-weather-api-key"
    url = f"https://api.weatherservice.com/current?location={location}&key={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return f"Weather in {location}: {data['temp']}°C, {data['condition']}"
    except Exception as e:
        return f"Error getting weather for {location}: {str(e)}"

agent.register_tool(Tool(
    "weather",
    "Get current weather. Parameters: location (str)",
    weather_tool
))
```

### Persistent Memory

To make the agent remember information across sessions:

```python
import json

# Save memory to file
def save_memory(agent, filename="memory.json"):
    with open(filename, "w") as f:
        json.dump({
            "history": agent.memory.history,
            "facts": agent.memory.important_facts
        }, f)

# Load memory from file
def load_memory(agent, filename="memory.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            agent.memory.history = data["history"]
            agent.memory.important_facts = data["facts"]
    except FileNotFoundError:
        print("No saved memory found.")
```

## Customization

### Creating Specialized Agents

You can extend the `SimpleAgent` class to create specialized agents:

```python
class ResearchAgent(SimpleAgent):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.system_prompt = """You are a research assistant..."""
        
        # Register research-specific tools
        self.register_tool(Tool("search_papers", "Search academic papers", search_papers))
        self.register_tool(Tool("summarize", "Summarize text", summarize_text))
```

### Adding Authentication to Tools

For tools that need authentication:

```python
def secured_tool(api_key=None, **kwargs):
    """A tool that requires authentication."""
    if not api_key:
        return "Error: Authentication required"
        
    # Continue with authenticated operation
    return "Operation completed successfully"

# When registering, wrap with a function that supplies the API key
agent.register_tool(Tool(
    "secure_operation",
    "Perform a secure operation. Parameters: param1 (str)",
    lambda **kwargs: secured_tool(api_key="my-secret-key", **kwargs)
))
```

## Example Scenarios

### Creating a Research Assistant

```python
# Create research tools
def search_papers(query, limit=5):
    # Simulate paper search
    return f"Found {limit} papers on '{query}'"

def extract_summary(paper_id):
    # Simulate extracting summary
    return f"Summary of paper {paper_id}: This research shows..."

# Create the agent
agent = SimpleAgent(api_key)

# Register tools
agent.register_tool(Tool("search_papers", "Search academic papers. Parameters: query (str), limit (int, optional)", search_papers))
agent.register_tool(Tool("get_summary", "Get paper summary. Parameters: paper_id (str)", extract_summary))

# Example usage
response = agent.process_input("Find me papers about quantum computing and summarize the most relevant one")
```

### Creating a Data Analysis Assistant

```python
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Create data analysis tools
def load_csv(url):
    """Load CSV data."""
    df = pd.read_csv(url)
    return f"Loaded data with {len(df)} rows and columns: {', '.join(df.columns)}"

def generate_plot(data_url, x_col, y_col):
    """Generate a plot from data."""
    df = pd.read_csv(data_url)
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    
    # Save plot to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"Plot generated. Data: data:image/png;base64,{image_base64}"

# Register tools
agent.register_tool(Tool("load_csv", "Load CSV data. Parameters: url (str)", load_csv))
agent.register_tool(Tool("plot", "Create a plot. Parameters: data_url (str), x_col (str), y_col (str)", generate_plot))
```

## Common Issues

### API Key Issues

**Problem**: "API key not found" or authentication errors
**Solution**: Ensure your API key is correctly set as an environment variable or passed directly to the SimpleAgent constructor.

### Tool Execution Errors

**Problem**: Tool fails to execute properly
**Solution**: 
- Make sure all required parameters are provided
- Add error handling in your tool functions
- Check the format of the tool call in the agent's response

### Memory Limitations

**Problem**: Agent forgets important information
**Solution**:
- Increase the `max_history` parameter in the Memory class
- Use `remember_fact` to store critical information explicitly
- Implement persistent memory storage

### Anthropic API Limits

**Problem**: Hitting rate limits or token limits
**Solution**:
- Add retry logic with exponential backoff
- Optimize prompts to use fewer tokens
- Split complex tasks into smaller interactions

## Best Practices

1. **Tool Design**:
   - Keep tools focused on a single responsibility
   - Provide clear error messages when tools fail
   - Document required parameters and expected outputs

2. **System Prompt Engineering**:
   - Be explicit about the agent's role and capabilities
   - Include clear instructions for tool usage format
   - Specify error handling procedures

3. **Memory Management**:
   - Store only essential information as important facts
   - Summarize long conversations to save context space
   - Implement a caching strategy for frequently used data

4. **Security Considerations**:
   - Never expose API keys or sensitive data in responses
   - Validate user inputs before executing tools
   - Limit the capabilities of tools to prevent misuse

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
