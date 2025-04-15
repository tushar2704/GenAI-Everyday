## Simple AI Agent Implementation with Ollama
#### © 2025 All rights reserved by Author [Tushar Aggarwal](https://www.linkedin.com/in/tusharaggarwalinseec/)

### Overview

This project implements a basic AI agent that interacts with the Ollama language model server. It showcases fundamental agent concepts without relying on external agent libraries. The agent can perform task planning, utilize tools, maintain memory, and reason about actions.

### Features

* **Task Planning:** Breaks down complex tasks into manageable steps.
* **Tool Usage:** Integrates external functions (tools) to enhance capabilities.
* **Memory:** Stores conversation history and important facts for context.
* **Reasoning:** Provides step-by-step thought processes before acting.
* **Ollama Integration:** Leverages Ollama for language model interactions.
* **Streaming Support:** (In `OllamaAgentWithStreaming`) Enables real-time output of the agent's responses.


### Requirements

* Python 3.8+
* `requests` library (`pip install requests`)
* Running Ollama server (e.g., locally on `http://localhost:11434`)


### Setup

1. **Install Dependencies:**

```bash
pip install requests
```

2. **Ensure Ollama is Running:**
    * Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)
    * Run Ollama and ensure it's accessible at the specified URL (default: `http://localhost:11434`). You might need to pull a model first (e.g., `ollama pull llama3`).

### Usage

1. **Define Tools:** Create functions that your agent can use (e.g., a search tool).
2. **Register Tools:**  Add these tools to the agent.
3. **Instantiate Agent:** Create an instance of `OllamaAgent` or `OllamaAgentWithStreaming`.
4. **Process Input:**  Pass user input to the agent's `process_input` or `process_input_streaming` method.

### Code Explanation

```python
import os
import json
import time
import requests
from typing import Dict, List, Any, Callable, Optional, Union

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

* **`Tool` Class:**
    * Defines a tool the agent can use.
    * `__init__`: Initializes the tool with a name, description, and a function to execute.
    * `call`: Executes the tool's function with given keyword arguments and handles potential errors.

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

* **`Memory` Class:**
    * Implements a simple memory system.
    * `__init__`: Initializes the memory with a maximum history size and a dictionary for storing important facts.
    * `add_interaction`: Adds a new interaction (role and content) to the history, maintaining the maximum history size.
    * `remember_fact`: Stores a key-value pair as an important fact.
    * `get_fact`: Retrieves a stored fact by key.
    * `get_context`: Formats the memory history and important facts into a context string for the agent.

```python
class OllamaAgent:
    """
    A simple AI agent that can plan and execute tasks using Ollama.
    """
    def __init__(self, 
                 model: str = "llama3", 
                 ollama_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the agent.
        
        Args:
            model: The Ollama model to use (e.g., "llama3", "mistral", "gemma")
            ollama_url: URL where Ollama API is running
            temperature: Sampling temperature for generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = Memory()
        self.tools = {}
        
        self.system_prompt = """
        You are a helpful AI assistant that is part of an agent system. Your role is to:
        1. Understand the user's goal or query
        2. Break down complex tasks into steps
        3. Use your available tools when needed to accomplish tasks
        4. Remember important information
        5. Always provide your reasoning before taking actions
        
        For most questions, just answer directly and helpfully.
        
        When you need to use a tool, respond in the following format:
        
        &lt;reasoning&gt;
        Your step-by-step thought process here
        &lt;/reasoning&gt;
        
        &lt;tool&gt;
        {
            "tool_name": "name_of_tool",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            }
        }
        &lt;/tool&gt;
        
        Available tools:
        {tool_descriptions}
        """
        
    def register_tool(self, tool: Tool):
        """Register a new tool that the agent can use."""
        self.tools[tool.name] = tool
        
    def _get_tool_descriptions(self) -&gt; str:
        """Get formatted descriptions of all available tools."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
        
    def _call_ollama(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Call the Ollama API to generate a response.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            
        Returns:
            The generated text response
        """
        endpoint = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return f"Error: Failed to communicate with Ollama - {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing Ollama response: {e}")
            return "Error: Failed to parse Ollama response"
        
    def _parse_tool_call(self, response: str) -&gt; Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response."""
        try:
            # Extract the tool section
            if "&lt;tool&gt;" not in response:
                return None
                
            tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()
            # Fix common JSON formatting issues
            tool_section = tool_section.replace("'", '"')
            
            # Handle case where the model might not format JSON correctly
            try:
                tool_data = json.loads(tool_section)
            except json.JSONDecodeError:
                # Fallback for malformed JSON - try to extract tool name and parameters directly
                import re
                tool_name_match = re.search(r'"tool_name":\s*"([^"]+)"', tool_section)
                tool_name = tool_name_match.group(1) if tool_name_match else None
                
                if not tool_name:
                    return None
                    
                # Try to extract parameters with a more flexible approach
                params = {}
                param_pattern = r'"([^"]+)":\s*"([^"]+)"'
                param_matches = re.finditer(param_pattern, tool_section)
                
                for match in param_matches:
                    key, value = match.groups()
                    if key != "tool_name":
                        params[key] = value
                
                return {
                    "tool_name": tool_name,
                    "parameters": params
                }
            
            return {
                "tool_name": tool_data["tool_name"],
                "parameters": tool_data.get("parameters", {})
            }
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error parsing tool call: {e}")
            print(f"Response was: {response}")
            return None
            
    def _execute_tool(self, tool_call: Dict[str, Any]) -&gt; str:
        """Execute a tool based on the parsed tool call."""
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
            
        tool = self.tools[tool_name]
        return tool.call(**parameters)
    
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
        
    def _generate_follow_up(self, tool_call: Dict[str, Any], tool_result: str) -&gt; str:
        """Generate a follow-up response after executing a tool."""
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create a system message with context
        system_message = self.system_prompt.format(tool_descriptions=self._get_tool_descriptions()) + f"\n\n{memory_context}"
        
        # Create the messages for the follow-up
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"I want you to continue helping me after using a tool. You used the tool '{tool_call['tool_name']}' with parameters {json.dumps(tool_call['parameters'])} and got this result:\n\n{tool_result}\n\nPlease analyze this result and continue helping me."}
        ]
        
        # Get response from the LLM
        response = self._call_ollama(messages)
        return response
    
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
        
        # Create a system message with context
        system_message = self.system_prompt.format(tool_descriptions=self._get_tool_descriptions()) + f"\n\n{memory_context}"
        
        # Create the messages for Ollama
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        # Get response from the LLM
        response_text = self._call_ollama(messages)
        
        # Process the response (execute tools if needed)
        final_response = self._process_response(response_text)
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", final_response)
        
        return final_response
```

* **`OllamaAgent` Class:**
    * The main class that represents the AI agent.
    * `__init__`:
        * Initializes the agent with the specified Ollama model, URL, temperature, and maximum tokens.
        * Creates a `Memory` instance and a dictionary to store tools.
        * Defines the `system_prompt` that guides the agent's behavior.  It includes instructions on task planning, tool usage, memory, and reasoning.
    * `register_tool`: Adds a tool to the agent's available tools.
    * `_get_tool_descriptions`: Formats the descriptions of all registered tools into a string.
    * `_call_ollama`:
        * Sends a request to the Ollama API to generate a response.
        * Constructs the payload with the model, messages, and generation options.
        * Handles potential request exceptions and JSON parsing errors.
    * `_parse_tool_call`:
        * Extracts tool call information from the model's response.  It looks for the `&lt;tool&gt;` XML tags.
        * Parses the JSON within the `&lt;tool&gt;` tags to get the `tool_name` and `parameters`.
        * Includes robust error handling for malformed JSON and missing data.
    * `_execute_tool`:
        * Executes a tool based on the parsed tool call.
        * Retrieves the tool from the `self.tools` dictionary.
        * Calls the tool's `call` method with the extracted parameters.
        * Handles the case where the tool is not found.
    * `_process_response`:
        * Processes the model's response to extract reasoning and potentially execute tools.
        * Extracts the reasoning section from the response (if present).
        * Calls `_parse_tool_call` to check if the model wants to use a tool.
        * If a tool call is detected:
            * Executes the tool using `_execute_tool`.
            * Adds the tool usage and result to memory.
            * Calls `_generate_follow_up` to get a follow-up response from the model, incorporating the tool's result.
        * If no tool call is detected, returns the original response (cleaned of XML tags).
    * `_generate_follow_up`:
        * Generates a follow-up response after a tool has been executed.
        * Gets the memory context.
        * Constructs a new system message that informs the model about the tool that was used and its result.
        * Sends the new system message and a user prompt to Ollama to get the follow-up response.
    * `process_input`:
        * The main method for processing user input.
        * Adds the user input to memory.
        * Gets the memory context.
        * Constructs the full system message with tool descriptions and memory context.
        * Sends the system message and user input to Ollama.
        * Processes the response using `_process_response` to potentially execute tools and get a final response.
        * Adds the final response to memory.
        * Returns the final response.

```python
class OllamaAgentWithStreaming(OllamaAgent):
    """
    Extension of OllamaAgent that supports streaming responses.
    """
    def _stream_ollama(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Stream the response from Ollama API.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            
        Returns:
            The complete generated text response
        """
        endpoint = f"{self.ollama_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        full_response = ""
        
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            print("\nAgent: ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            print(content, end="", flush=True)
                            full_response += content
                    except json.JSONDecodeError:
                        pass
            print("\n")  # New line after streaming completes
            
            return full_response
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Failed to communicate with Ollama - {str(e)}"
            print(f"\nAgent: {error_msg}")
            return error_msg
        
    def process_input_streaming(self, user_input: str) -&gt; str:
        """
        Process user input and generate a streaming response.
        This simplified version doesn't attempt to parse tool calls during streaming.
        
        Args:
            user_input: The user's message or query
            
        Returns:
            The agent's complete response after streaming
        """
        # Store user input in memory
        self.memory.add_interaction("user", user_input)
        
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create a system message with context - for simpler inputs, avoid complex tool usage instructions
        system_message = """
        You are a helpful AI assistant. Respond directly to the user's question or request.
        """
        
        # Create the messages for Ollama
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        # Stream the response from the LLM
        response_text = self._stream_ollama(messages)
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", response_text)
        
        return response_text
```

* **`OllamaAgentWithStreaming` Class:**
    * Inherits from `OllamaAgent` to add streaming response functionality.
    * `_stream_ollama`:
        * Similar to `_call_ollama` but uses `stream=True` in the request to Ollama.
        * Iterates through the response lines, decodes the JSON chunks, and prints the content in real-time.
        * Accumulates the full response text for later use.
    * `process_input_streaming`:
        * Processes user input and generates a streaming response.
        * Uses a simplified system message to avoid complex tool usage during streaming.
        * Calls `_stream_ollama` to get the streaming response.
        * Stores the final response in memory.


### Example Usage

(This would require you to define a tool and set up the agent)

```python
# Example Tool (Simple Search)
def search_wikipedia(query: str) -&gt; str:
    """
    Searches Wikipedia for the given query.
    """
    import wikipedia
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.PageError:
        return "No results found."
    except wikipedia.exceptions.DisambiguationError as e:
        return "Disambiguation Error: " + str(e)

search_tool = Tool(
    name="wikipedia_search",
    description="Searches Wikipedia for a given query.",
    function=search_wikipedia
)

# Instantiate the Agent
agent = OllamaAgent(model="llama3")
agent.register_tool(search_tool)

# Process Input
user_input = "What is the capital of France?"
response = agent.process_input(user_input)
print(f"Agent: {response}")

user_input = "Tell me about Tushar Aggarwal"
response = agent.process_input(user_input)
print(f"Agent: {response}")

# Example using Streaming Agent
streaming_agent = OllamaAgentWithStreaming(model="llama3")
user_input = "Write a short poem about spring."
response = streaming_agent.process_input_streaming(user_input)
print(f"Final Response: {response}")
```


### Notes

* The agent relies on the Ollama API being available at the specified URL.
* The quality of the agent's responses depends on the chosen Ollama model and the system prompt.
* The tool parsing logic is relatively simple and might need adjustments for different model behaviors.
* The streaming agent provides real-time output but doesn't support tool usage during streaming in this simplified version.


