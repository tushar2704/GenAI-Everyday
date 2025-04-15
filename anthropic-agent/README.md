# Simple AI Agent Implementation: Code Instructions
#### © 2025 All rights reserved by Author [Tushar Aggarwal](https://www.linkedin.com/in/tusharaggarwalinseec/)


## Understanding the Core Components

This implementation consists of three main classes:

1. **Tool**: Represents functionalities the agent can use
2. **Memory**: Manages the agent's conversation history and knowledge
3. **SimpleAgent**: The main agent implementation that coordinates everything

Let's break down each component in detail:

## Tool Class

The `Tool` class is designed to be a wrapper around any function that you want your agent to use. Each tool has:

- **Name**: Used to identify and call the tool
- **Description**: Explains what the tool does and its parameters
- **Function**: The actual implementation that will be executed

The class has a simple interface:
```python
tool = Tool(
    "calculator", 
    "Perform mathematical calculations. Parameters: expression (str)",
    calculate_function
)

# To use the tool
result = tool.call(expression="2 + 2")
```

When creating custom tools:
1. Define a function that accepts named parameters
2. Create a Tool instance with a descriptive name and clear description
3. Register it with the agent

## Memory Class

The `Memory` class provides a simple way to store:
- **Conversation history**: Recent exchanges between user and agent
- **Important facts**: Key information that should persist

Key methods:
- `add_interaction(role, content)`: Adds a message to the history
- `remember_fact(key, value)`: Stores important information
- `get_fact(key)`: Retrieves stored information
- `get_context()`: Formats all memory for inclusion in prompts

The memory system has a `max_history` parameter that limits how many past interactions are remembered to prevent context windows from getting too large.

## SimpleAgent Class

The `SimpleAgent` class is the core of the implementation:

1. **Initialization**:
   - Connects to the Anthropic API
   - Sets up the memory system
   - Initializes the tool registry
   - Defines the system prompt

2. **Tool Management**:
   - `register_tool(tool)`: Adds a new tool to the agent's capabilities
   - `_get_tool_descriptions()`: Formats tool information for the prompt

3. **Response Processing**:
   - `_parse_tool_call(response)`: Extracts tool usage from LLM responses
   - `_execute_tool(tool_call)`: Runs the specified tool with parameters
   - `_process_response(response)`: Handles the full response processing flow

4. **Core Functionality**:
   - `process_input(user_input)`: Main method to handle user messages

## System Prompt

The system prompt is crucial for guiding the agent's behavior. It includes:

1. **Role definition**: What the agent should do
2. **Task approach**: How to break down problems
3. **Tool usage format**: How to structure responses when using tools
4. **Available tools**: List of tools with descriptions

The current system prompt instructs the model to:
- Structure tool calls in a specific XML format
- Show reasoning before taking actions
- Break complex tasks into steps

## How to Run the Code

### Prerequisites

1. Python 3.8 or newer
2. Anthropic API key

### Step 1: Install Dependencies

```bash
pip install anthropic
```

### Step 2: Set Up Your API Key

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

### Step 3: Create Your Script

Save the provided code as `anthropic-agent/agent.py`.

### Step 4: Run the Agent

```bash
python anthropic-agent/agent.py
```

The default implementation includes an interactive loop where you can chat with the agent and see it use tools.

## Customizing the Agent

### Adding Custom Tools

1. **Define your function**:
```python
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of text (positive/negative)."""
    # In a real implementation, use a sentiment analysis library
    # This is a simple placeholder
    if "good" in text.lower() or "great" in text.lower():
        return "Positive sentiment detected"
    elif "bad" in text.lower() or "terrible" in text.lower():
        return "Negative sentiment detected"
    return "Neutral sentiment"
```

2. **Register it with the agent**:
```python
agent.register_tool(Tool(
    "analyze_sentiment",
    "Analyze text sentiment. Parameters: text (str)",
    analyze_sentiment
))
```

### Modifying the System Prompt

You can customize the agent's behavior by changing the system prompt:

```python
agent.system_prompt = """
You are a specialized customer service AI. Your goal is to:
1. Help users with their queries
2. Use tools to retrieve information when needed
3. Always maintain a friendly and helpful tone

When you need to use a tool, respond in the following format:
...
"""
```

### Extending the Memory System

To add more sophisticated memory:

```python
class EnhancedMemory(Memory):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        self.conversation_summary = ""
        
    def summarize_conversation(self):
        """Create a summary of the conversation so far."""
        # In a real implementation, you might use an LLM to generate this
        messages = [f"{m['role']}: {m['content']}" for m in self.history]
        summary = "\n".join(messages)
        self.conversation_summary = f"Summary of {len(messages)} messages: {summary}"
        
    def get_context(self):
        """Get enhanced memory context including summary."""
        base_context = super().get_context()
        return f"{base_context}\n\nConversation Summary:\n{self.conversation_summary}"
```



## Explaining Code 

```python
"""
Simple AI Agent Implementation
==============================

This module implements a basic AI agent using the Anthropic API without any AI agent libraries.
It demonstrates core agent concepts like:
1. Task planning
2. Tool usage
3. Memory
4. Reasoning steps

Requirements:
- Python 3.8+
- anthropic package (pip install anthropic)
"""
```

- **Docstring**: Describes the module, its purpose, and requirements.

---

```python
import os
import json
import time
from typing import Dict, List, Any, Callable, Optional, Union
import anthropic
```

- **Imports**:
    - `os`: For environment variables.
    - `json`: For parsing and formatting JSON.
    - `time`: For time-related functions.
    - `typing`: For type hints.
    - `anthropic`: Anthropic API client.

---

### Tool Class

```python
class Tool:
    """
    Represents a function that the agent can use to interact with external systems.
    """
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a new tool.
        
        Args:
            name: The name of the tool
            description: Description of what the tool does and how to use it
            function: The actual function to call when the tool is used
        """
        self.name = name
        self.description = description
        self.function = function
        
    def call(self, **kwargs) -&gt; str:
        """Execute the tool with the given parameters."""
        try:
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
```

- **Purpose**: Encapsulates a callable tool with a name and description.
- `__init__`: Initializes the tool.
- `call`: Executes the tool with given parameters, handles errors.

---

### Memory Class

```python
class Memory:
    """
    Simple memory system to store the agent's conversation history and important information.
    """
    def __init__(self, max_history: int = 10):
        """
        Initialize the memory system.
        
        Args:
            max_history: Maximum number of interactions to remember
        """
        self.history = []
        self.max_history = max_history
        self.important_facts = {}
        
    def add_interaction(self, role: str, content: str):
        """Add an interaction to the history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) &gt; self.max_history:
            self.history.pop(0)
            
    def remember_fact(self, key: str, value: Any):
        """Store an important fact that should be remembered."""
        self.important_facts[key] = value
        
    def get_fact(self, key: str) -&gt; Any:
        """Retrieve a stored fact."""
        return self.important_facts.get(key)
    
    def get_context(self) -&gt; str:
        """Get the memory context for the agent."""
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in self.history])
        facts_text = "\n".join([f"{k}: {v}" for k, v in self.important_facts.items()])
        
        return f"Memory:\n{history_text}\n\nImportant Facts:\n{facts_text}"
```

- **Purpose**: Stores conversation history and important facts.
- `__init__`: Initializes memory with a max history.
- `add_interaction`: Adds a conversation turn, trims if too long.
- `remember_fact`: Stores a key-value fact.
- `get_fact`: Retrieves a fact.
- `get_context`: Returns a formatted string of history and facts.

---

### SimpleAgent Class

```python
class SimpleAgent:
    """
    A simple AI agent that can plan and execute tasks using the Anthropic API.
    """
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the agent.
        
        Args:
            api_key: Anthropic API key
            model: The model to use for generating responses
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.memory = Memory()
        self.tools = {}
        
        # Note: Using double curly braces to escape actual curly braces in the system prompt
        self.system_prompt = """
        You are a helpful AI assistant that is part of an agent system. Your role is to:
        1. Understand the user's goal or query
        2. Break down complex tasks into steps
        3. Use your available tools when needed to accomplish tasks
        4. Remember important information
        5. Always provide your reasoning before taking actions
        
        When you need to use a tool, respond in the following format:
        
        &lt;reasoning&gt;
        Your step-by-step thought process here
        &lt;/reasoning&gt;
        
        &lt;tool&gt;
        {{
            "tool_name": "name_of_tool",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
        &lt;/tool&gt;
        
        Available tools:
        {tool_descriptions}
        """
```

- **Purpose**: Main agent class.
- `__init__`: Sets up the Anthropic client, model, memory, tools, and system prompt.

---

#### Tool Registration

```python
    def register_tool(self, tool: Tool):
        """Register a new tool that the agent can use."""
        self.tools[tool.name] = tool
```

- Adds a tool to the agent's available tools.

---

#### Tool Descriptions

```python
    def _get_tool_descriptions(self) -&gt; str:
        """Get formatted descriptions of all available tools."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
```

- Returns a formatted string listing all tools and their descriptions.

---

#### Tool Call Parsing

```python
    def _parse_tool_call(self, response: str) -&gt; Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response."""
        try:
            # Extract the tool section
            if "&lt;tool&gt;" not in response:
                return None
                
            tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()
            tool_data = json.loads(tool_section)
            
            return {
                "tool_name": tool_data["tool_name"],
                "parameters": tool_data["parameters"]
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing tool call: {e}")
            print(f"Response was: {response}")
            return None
```

- Extracts and parses the tool call from the model's response.

---

#### Tool Execution

```python
    def _execute_tool(self, tool_call: Dict[str, Any]) -&gt; str:
        """Execute a tool based on the parsed tool call."""
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
            
        tool = self.tools[tool_name]
        return tool.call(**parameters)
```

- Executes the specified tool with parameters.

---

#### Response Processing

```python
    def _process_response(self, response: str) -&gt; str:
        """Process the model's response to extract reasoning and potentially execute tools."""
        # Extract reasoning if present
        reasoning = ""
        reasoning_section = ""
        if "&lt;reasoning&gt;" in response:
            try:
                reasoning_section = response.split("&lt;reasoning&gt;")[^1].split("&lt;/reasoning&gt;")[^0].strip()
                reasoning = f"Reasoning:\n{reasoning_section}\n\n"
            except IndexError:
                print("Warning: Could not extract reasoning section properly")
        
        # Check for tool calls
        tool_call = self._parse_tool_call(response)
        if tool_call:
            tool_result = self._execute_tool(tool_call)
            
            # Add the tool usage to memory
            self.memory.add_interaction(
                "agent", 
                f"Used tool: {tool_call['tool_name']} with parameters: {tool_call['parameters']}"
            )
            self.memory.add_interaction("tool_result", tool_result)
            
            # Generate a follow-up response with the tool result
            return self._generate_follow_up(tool_call, tool_result)
        
        # If no tool call, just return the response without the XML tags
        clean_response = response
        if "&lt;reasoning&gt;" in response and reasoning_section:
            clean_response = response.replace(
                f"&lt;reasoning&gt;{reasoning_section}&lt;/reasoning&gt;", 
                f"I thought about this:\n{reasoning_section}\n\n"
            )
        
        return clean_response
```

- Extracts reasoning, executes tools if needed, and cleans up the response.

---

#### Follow-up Generation

```python
    def _generate_follow_up(self, tool_call: Dict[str, Any], tool_result: str) -&gt; str:
        """Generate a follow-up response after executing a tool."""
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create the message with system instructions and context
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt.format(tool_descriptions=self._get_tool_descriptions()) + f"\n\n{memory_context}",
            messages=[
                {
                    "role": "user", 
                    "content": f"I want you to continue helping me after using a tool. You used the tool '{tool_call['tool_name']}' with parameters {json.dumps(tool_call['parameters'])} and got this result:\n\n{tool_result}\n\nPlease analyze this result and continue helping me."
                }
            ],
            max_tokens=1000
        )
        
        return response.content[^0].text
```

- After a tool is used, asks the model to continue based on the tool result.

---

#### User Input Processing

```python
    def process_input(self, user_input: str) -&gt; str:
        """
        Process user input and generate a response, potentially using tools.
        
        Args:
            user_input: The user's message or query
            
        Returns:
            The agent's response
        """
        # Store user input in memory
        self.memory.add_interaction("user", user_input)
        
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create message with system prompt, memory, and user input
        # Note: In Claude's API, system is a separate parameter, not a message role
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt.format(tool_descriptions=self._get_tool_descriptions()) + f"\n\n{memory_context}",
            messages=[
                {
                    "role": "user", 
                    "content": user_input
                }
            ],
            max_tokens=1500
        )
        
        response_text = response.content[^0].text
        
        # Process the response (execute tools if needed)
        final_response = self._process_response(response_text)
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", final_response)
        
        return final_response
```

- Handles user input, updates memory, gets a response, processes it, and stores the result.

---

### Example Usage

```python
def main():
    """Example of how to use the SimpleAgent."""
    # Get API key from environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set the ANTHROPIC_API_KEY environment variable")
        return
        
    # Create the agent
    agent = SimpleAgent(api_key)
    
    # Define and register some example tools
    def search_web(query: str) -&gt; str:
        """Simulated web search tool."""
        # In a real implementation, this would call a search API
        return f"Top results for '{query}':\n1. Example result 1\n2. Example result 2"
    
    def get_current_time() -&gt; str:
        """Tool to get the current time."""
        return f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def calculate(expression: str) -&gt; str:
        """Simple calculator tool."""
        try:
            return f"Result: {eval(expression)}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    # Register the tools
    agent.register_tool(Tool(
        "search", 
        "Search the web for information. Parameters: query (str)",
        search_web
    ))
    
    agent.register_tool(Tool(
        "get_time", 
        "Get the current time. No parameters needed.",
        get_current_time
    ))
    
    agent.register_tool(Tool(
        "calculate", 
        "Evaluate a mathematical expression. Parameters: expression (str)",
        calculate
    ))
    
    # Simple interaction loop
    print("SimpleAgent initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        try:
            response = agent.process_input(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")
```

- **main()**:
    - Gets API key.
    - Creates the agent.
    - Defines three example tools: search, get_time, calculate.
    - Registers the tools.
    - Starts a loop for user interaction.

---

## Summary

- **Tool**: Encapsulates callable functions for the agent.
- **Memory**: Stores conversation and facts.
- **SimpleAgent**: Handles user input, tool use, memory, and interaction with Anthropic API.
- **main()**: Example of how to use the agent and tools interactively.
