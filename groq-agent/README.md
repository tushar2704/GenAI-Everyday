## Simple AI Agent Implementation with GROQ API
#### © 2025 All rights reserved by Author [Tushar Aggarwal](https://www.linkedin.com/in/tusharaggarwalinseec/)
#### Overview

This project implements a simple AI agent that leverages the GROQ API for generating responses, planning tasks, and interacting with external tools. The agent is designed to:

* Understand user goals and queries.
* Break down complex tasks into manageable steps.
* Utilize available tools to accomplish tasks.
* Remember important information across interactions.


#### Features

* **Tool System**: Allows the agent to interact with external functions and services through a flexible tool registration system.
* **Memory System**: Stores conversation history and important facts to maintain context across interactions.
* **Groq API Integration**: Uses the GROQ API for generating text, with support for both standard and streaming responses.
* **Error Handling**: Comprehensive error handling to manage API requests, JSON parsing, and tool execution.
* **Response Parsing**: Enhanced parsing logic to extract tool calls from the agent's responses, even with formatting issues.


#### Getting Started

##### Prerequisites

* A GROQ API key. You can obtain one from the [GROQ website](https://console.groq.com/keys).
* Python 3.6+
* Required Python packages: `requests`


##### Installation

1. Clone the repository:

```bash
git clone https://github.com/tushar2704/GenAI-Everyday.git
cd groq-agent
```

2. Install the required packages:

```bash
pip install requests
```


##### Usage

1. Set up your GROQ API key as an environment variable:

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

2. Instantiate the `GroqAgent` with your API key:

```python
from groq_agent import GroqAgent, Tool
import os

api_key = os.environ.get("GROQ_API_KEY")
agent = GroqAgent(api_key=api_key)
```

3. Define and register tools:

```python
def search_tool(query: str) -&gt; str:
    """A simple search tool (replace with actual implementation)."""
    return f"Search results for: {query}"

search = Tool(name="search", description="Useful for searching the web.", function=search_tool)
agent.register_tool(search)
```

4. Interact with the agent:

```python
response = agent.run("Search for the capital of France.")
print(response)
```


#### Code Explanation

```python
"""Simple AI Agent Implementation with GROQ API #### © 2025 All rights reserved by Author [Tushar Aggarwal](https://www.linkedin.com/in/tusharaggarwalinseec/)"""
import os
import json
import time
import requests
import argparse
import re
import traceback
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

class GroqAgent:
    """
    A simple AI agent that can plan and execute tasks using the Groq API.
    """
    def __init__(self, 
                 api_key: str,
                 model: str = "llama3-8b-8192",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the agent.
        
        Args:
            api_key: Your Groq API key
            model: The model to use (e.g., "llama3-8b-8192", "mixtral-8x7b-32768")
            temperature: Sampling temperature for generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = Memory()
        self.tools = {}
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Improved system prompt with double braces to escape formatting
        self.system_prompt = """
        You are a helpful AI assistant that is part of an agent system. Your role is to:
        1. Understand the user's goal or query
        2. Break down complex tasks into steps
        3. Use your available tools when needed to accomplish tasks
        4. Remember important information
        5. Always provide your reasoning before taking actions
        
        When you need to use a tool, respond EXACTLY in the following format (with no additional spaces or newlines):
        
        &lt;reasoning&gt;
        Your step-by-step thought process here
        &lt;/reasoning&gt;
        
        &lt;tool&gt;
        {{"tool_name":"name_of_tool","parameters":{{"param1":"value1","param2":"value2"}}}}
        &lt;/tool&gt;
        
        IMPORTANT: The tool section must be valid JSON without line breaks between keys and values.
        
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
    
    def diagnose_response(self, response: str) -&gt; str:
        """Diagnostic function to help troubleshoot response parsing issues."""
        print("\n=== DIAGNOSTIC INFORMATION ===")
        print(f"Full response length: {len(response)} characters")
        print(f"Response starts with: {response[:100]}...")
        print(f"Response ends with: ...{response[-100:]}")
        
        # Check for expected XML tags
        reasoning_tag_open = "&lt;reasoning&gt;" in response
        reasoning_tag_close = "&lt;/reasoning&gt;" in response
        tool_tag_open = "&lt;tool&gt;" in response
        tool_tag_close = "&lt;/tool&gt;" in response
        
        print(f"Contains &lt;reasoning&gt; tag: {reasoning_tag_open}")
        print(f"Contains &lt;/reasoning&gt; tag: {reasoning_tag_close}")
        print(f"Contains &lt;tool&gt; tag: {tool_tag_open}")
        print(f"Contains &lt;/tool&gt; tag: {tool_tag_close}")
        
        # If tool tags are present, extract and analyze the tool section
        if tool_tag_open and tool_tag_close:
            try:
                tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()
                print("\nTool section:")
                print(tool_section)
                
                # Check for common JSON issues
                has_single_quotes = "'" in tool_section
                has_newlines = "\n" in tool_section
                has_tool_name = "tool_name" in tool_section
                has_parameters = "parameters" in tool_section
                
                print(f"\nTool section contains single quotes: {has_single_quotes}")
                print(f"Tool section contains newlines: {has_newlines}")
                print(f"Tool section contains 'tool_name': {has_tool_name}")
                print(f"Tool section contains 'parameters': {has_parameters}")
                
                # Try to parse as JSON and report issues
                try:
                    json_data = json.loads(tool_section.replace("'", '"'))
                    print("✓ Successfully parsed tool section as JSON")
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse tool section as JSON: {e}")
                    # Inspect the character at the error position
                    if hasattr(e, 'pos'):
                        error_pos = e.pos
                        context_start = max(0, error_pos - 10)
                        context_end = min(len(tool_section), error_pos + 10)
                        print(f"Error at position {error_pos}: ...{tool_section[context_start:error_pos]}[HERE]{tool_section[error_pos:context_end]}...")
                except Exception as e:
                print(f"Error analyzing tool section: {e}")
        
        print("=== END DIAGNOSTIC INFORMATION ===\n")
        
        return "Diagnostic information has been printed to the console."
        
    def _call_groq(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Call the Groq API to generate a response.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            
        Returns:
            The generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][^0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Groq API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return f"Error: Failed to communicate with Groq API - {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing Groq response: {e}")
            return "Error: Failed to parse Groq response"
        
    def _stream_groq(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Stream the response from Groq API.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            
        Returns:
            The complete generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        full_response = ""
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            
            print("\nAgent: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        if line_text == "data: [DONE]":
                            break
                        try:
                            chunk = json.loads(line_text[6:])  # Skip "data: " prefix
                            if "choices" in chunk and len(chunk["choices"]) &gt; 0:
                                delta = chunk["choices"][^0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            pass
            
            print("\n")  # New line after streaming completes
            return full_response
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Failed to communicate with Groq API - {str(e)}"
            print(f"\nAgent: {error_msg}")
            return error_msg
        
    def _parse_tool_call(self, response: str) -&gt; Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response with enhanced error handling."""
        try:
            # Check if there's a tool section
            if "&lt;tool&gt;" not in response or "&lt;/tool&gt;" not in response:
                return None
                
            # Extract the tool section
            tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()
            
            # Debug the tool section
            print(f"\nDEBUG - Tool section extracted:\n{tool_section}\n")
            
            # Normalize JSON formatting issues
            # Replace single quotes with double quotes
            normalized = tool_section.replace("'", '"')
            # Remove newlines and extra whitespace between keys and values
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = re.sub(r'"\s+:', '":', normalized)
            normalized = re.sub(r':\s+"', ':"', normalized)
            
            # Try to parse the normalized JSON
            try:
                tool_data = json.loads(normalized)
                return {
                    "tool_name": tool_data["tool_name"],
                    "parameters": tool_data.get("parameters", {})
                }
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                
                # Manual parsing as fallback
                # Extract tool name using regex
                tool_name_pattern = r'"tool_name"\s*:\s*"([^"]+)"'
                tool_name_match = re.search(tool_name_pattern, normalized)
                
                if not tool_name_match:
                    print("Could not find tool_name in response")
                    return None
                    
                tool_name = tool_name_match.group(1)
                print(f"Extracted tool name: {tool_name}")
                
                # Extract parameters section
                params = {}
                params_pattern = r'"parameters"\s*:\s*\{(.+?)\}'
                params_match = re.search(params_pattern, normalized, re.DOTALL)
                
                if params_match:
                    params_text = params_match.group(1)
                    # Extract individual parameters
                    param_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
                    for param_match in re.finditer(param_pattern, params_text):
                        key, value = param_match.groups()
                        params[key] = value
                        
                    print(f"Extracted parameters: {params}")
                
                return {
                    "tool_name": tool_name,
                    "parameters": params
                }
                
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            traceback.print_exc()
            print(f"Response was: {response}")
            return None
            
    def _execute_tool(self, tool_call: Dict[str, Any]) -&gt; str:
        """Execute a tool with enhanced error handling."""
        try:
            tool_name = tool_call["tool_name"]
            parameters = tool_call.get("parameters", {})
            
            if not tool_name:
                return "Error: No tool name specified"
                
            if tool_name not in self.tools:
                return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
                
            tool = self.tools[tool_name]
            print(f"Executing tool '{tool_name}' with parameters: {parameters}")
            
            return tool.call(**parameters)
        except Exception as e:
            print(f"Error executing tool: {e}")
            traceback.print_exc()
            return f"Error executing tool: {str(e)}"
    
    def _process_response(self, response: str) -&gt; str:
        """Process the model's response with enhanced error handling."""
        try:
            # Extract reasoning if present
            reasoning = ""
            if "&lt;reasoning&gt;" in response and "&lt;/reasoning&gt;" in response:
                try:
                    reasoning = response.split("&lt;reasoning&gt;")[^1].split("&lt;/reasoning&gt;")[^0].strip()
                except Exception as e:
                    print(f"Error extracting reasoning: {e}")
            
            # Attempt to parse the tool call
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                # Execute the tool and get the result
                tool_result = self._execute_tool(tool_call)
                return tool_result
            else:
                # If no tool call, return the original response
                return response
        except Exception as e:
            print(f"Error processing response: {e}")
            return f"Error processing response: {str(e)}"

    def run(self, user_input: str, stream: bool = False) -&gt; str:
        """Main method to run the agent."""
        self.memory.add_interaction("user", user_input)
        
        # Construct the messages for the API call
        tool_descriptions = self._get_tool_descriptions()
        system_prompt = self.system_prompt.format(tool_descriptions=tool_descriptions)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.memory.get_context() + "\n" + user_input}
        ]
        
        # Call the Groq API
        if stream:
            response = self._stream_groq(messages)
        else:
            response = self._call_groq(messages)
            
        # Process the response
        processed_response = self._process_response(response)
        self.memory.add_interaction("agent", processed_response)
        
        return processed_response
```


### Line-by-line Explanation:

1. **Imports**:

```python
import os
import json
import time
import requests
import argparse
import re
import traceback
from typing import Dict, List, Any, Callable, Optional, Union
```

    * These lines import necessary Python modules for various functionalities like environment variables, JSON handling, HTTP requests, regular expressions, and type hints.
2. **Tool Class**:

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

    * The `Tool` class is a blueprint for creating tool objects that the agent can use.
    * `__init__`: Initializes the tool with a name, description, and the actual function to be executed.
    * `call`: Executes the tool's function with the given parameters and returns the result as a string.
3. **Memory Class**:

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

    * The `Memory` class manages the agent's memory, storing conversation history and important facts.
    * `__init__`: Initializes the memory with a maximum history size and dictionaries for storing history and facts.
    * `add_interaction`: Adds a new interaction to the history, removing the oldest if the history exceeds the maximum size.
    * `remember_fact`: Stores an important fact in the `important_facts` dictionary.
    * `get_fact`: Retrieves a stored fact by key.
    * `get_context`: Formats the memory content into a string for providing context to the agent.
4. **GroqAgent Class**:

```python
class GroqAgent:
    """
    A simple AI agent that can plan and execute tasks using the Groq API.
    """
    def __init__(self,
                 api_key: str,
                 model: str = "llama3-8b-8192",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize the agent.

        Args:
            api_key: Your Groq API key
            model: The model to use (e.g., "llama3-8b-8192", "mixtral-8x7b-32768")
            temperature: Sampling temperature for generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = Memory()
        self.tools = {}
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        # Improved system prompt with double braces to escape formatting
        self.system_prompt = """
        You are a helpful AI assistant that is part of an agent system. Your role is to:
        1. Understand the user's goal or query
        2. Break down complex tasks into steps
        3. Use your available tools when needed to accomplish tasks
        4. Remember important information
        5. Always provide your reasoning before taking actions

        When you need to use a tool, respond EXACTLY in the following format (with no additional spaces or newlines):

        &lt;reasoning&gt;
        Your step-by-step thought process here
        &lt;/reasoning&gt;

        &lt;tool&gt;
        {{"tool_name":"name_of_tool","parameters":{{"param1":"value1","param2":"value2"}}}}
        &lt;/tool&gt;

        IMPORTANT: The tool section must be valid JSON without line breaks between keys and values.

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

    def diagnose_response(self, response: str) -&gt; str:
        """Diagnostic function to help troubleshoot response parsing issues."""
        print("\n=== DIAGNOSTIC INFORMATION ===")
        print(f"Full response length: {len(response)} characters")
        print(f"Response starts with: {response[:100]}...")
        print(f"Response ends with: ...{response[-100:]}")

        # Check for expected XML tags
        reasoning_tag_open = "&lt;reasoning&gt;" in response
        reasoning_tag_close = "&lt;/reasoning&gt;" in response
        tool_tag_open = "&lt;tool&gt;" in response
        tool_tag_close = "&lt;/tool&gt;" in response

        print(f"Contains &lt;reasoning&gt; tag: {reasoning_tag_open}")
        print(f"Contains &lt;/reasoning&gt; tag: {reasoning_tag_close}")
        print(f"Contains &lt;tool&gt; tag: {tool_tag_open}")
        print(f"Contains &lt;/tool&gt; tag: {tool_tag_close}")

        # If tool tags are present, extract and analyze the tool section
        if tool_tag_open and tool_tag_close:
            try:
                tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()
                print("\nTool section:")
                print(tool_section)

                # Check for common JSON issues
                has_single_quotes = "'" in tool_section
                has_newlines = "\n" in tool_section
                has_tool_name = "tool_name" in tool_section
                has_parameters = "parameters" in tool_section

                print(f"\nTool section contains single quotes: {has_single_quotes}")
                print(f"Tool section contains newlines: {has_newlines}")
                print(f"Tool section contains 'tool_name': {has_tool_name}")
                print(f"Tool section contains 'parameters': {has_parameters}")

                # Try to parse as JSON and report issues
                try:
                    json_data = json.loads(tool_section.replace("'", '"'))
                    print("✓ Successfully parsed tool section as JSON")
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse tool section as JSON: {e}")
                    # Inspect the character at the error position
                    if hasattr(e, 'pos'):
                        error_pos = e.pos
                        context_start = max(0, error_pos - 10)
                        context_end = min(len(tool_section), error_pos + 10)
                        print(f"Error at position {error_pos}: ...{tool_section[context_start:error_pos]}[HERE]{tool_section[error_pos:context_end]}...")
                except Exception as e:
                print(f"Error analyzing tool section: {e}")

        print("=== END DIAGNOSTIC INFORMATION ===\n")

        return "Diagnostic information has been printed to the console."

    def _call_groq(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Call the Groq API to generate a response.

        Args:
            messages: List of message objects with 'role' and 'content'

        Returns:
            The generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json["choices"][^0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Groq API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return f"Error: Failed to communicate with Groq API - {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing Groq response: {e}")
            return "Error: Failed to parse Groq response"

    def _stream_groq(self, messages: List[Dict[str, str]]) -&gt; str:
        """
        Stream the response from Groq API.

        Args:
            messages: List of message objects with 'role' and 'content'

        Returns:
            The complete generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        full_response = ""

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            print("\nAgent: ", end="", flush=True)

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        if line_text == "data: [DONE]":
                            break
                        try:
                            chunk = json.loads(line_text[6:])  # Skip "data: " prefix
                            if "choices" in chunk and len(chunk["choices"]) &gt; 0:
                                delta = chunk["choices"][^0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            pass

            print("\n")  # New line after streaming completes
            return full_response
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Failed to communicate with Groq API - {str(e)}"
            print(f"\nAgent: {error_msg}")
            return error_msg

    def _parse_tool_call(self, response: str) -&gt; Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response with enhanced error handling."""
        try:
            # Check if there's a tool section
            if "&lt;tool&gt;" not in response or "&lt;/tool&gt;" not in response:
                return None

            # Extract the tool section
            tool_section = response.split("&lt;tool&gt;")[^1].split("&lt;/tool&gt;")[^0].strip()

            # Debug the tool section
            print(f"\nDEBUG - Tool section extracted:\n{tool_section}\n")

            # Normalize JSON formatting issues
            # Replace single quotes with double quotes
            normalized = tool_section.replace("'", '"')
            # Remove newlines and extra whitespace between keys and values
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = re.sub(r'"\s+:', '":', normalized)
            normalized = re.sub(r':\s+"', ':"', normalized)

            # Try to parse the normalized JSON
            try:
                tool_data = json.loads(normalized)
                return {
                    "tool_name": tool_data["tool_name"],
                    "parameters": tool_data.get("parameters", {})
                }
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")

                # Manual parsing as fallback
                # Extract tool name using regex
                tool_name_pattern = r'"tool_name"\s*:\s*"([^"]+)"'
                tool_name_match = re.search(tool_name_pattern, normalized)

                if not tool_name_match:

```


