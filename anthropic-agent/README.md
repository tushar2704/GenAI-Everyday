# Simple AI Agent Implementation: Code Instructions

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

## Teaching Tips

When presenting this code in your teaching session:

1. **Start with the basics**:
   - Explain what an AI agent is
   - Show how tools extend an LLM's capabilities
   - Demonstrate the importance of memory for context

2. **Build incrementally**:
   - First show a simple agent without tools
   - Add tools one by one
   - Demonstrate how memory enhances capabilities

3. **Interactive exercises**:
   - Have participants create their own tools
   - Challenge them to extend the memory system
   - Ask them to customize the agent for specific use cases

4. **Common pitfalls to discuss**:
   - Tool errors and how to handle them
   - Context limitations and strategies
   - Prompt engineering for reliable tool usage

5. **Advanced topics**:
   - Creating multi-agent systems
   - Implementing planning capabilities
   - Adding state machines for complex workflows

## Final Notes

This implementation is deliberately simple to show the core concepts, but it can be extended in many ways:

- Adding more sophisticated memory systems
- Implementing planning algorithms
- Creating specialized agents for different domains
- Adding authentication and security features
- Implementing web interfaces or API endpoints

The code provides a solid foundation that you can build upon for your specific needs.
