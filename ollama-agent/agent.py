"""
Simple AI Agent Implementation with Ollama
=========================================

This module implements a basic AI agent using Ollama without any AI agent libraries.
It demonstrates core agent concepts like:
1. Task planning
2. Tool usage
3. Memory
4. Reasoning steps

Requirements:
- Python 3.8+
- requests package (pip install requests)
"""

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
        
    def call(self, **kwargs) -> str:
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
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def remember_fact(self, key: str, value: Any):
        """Store an important fact that should be remembered."""
        self.important_facts[key] = value
        
    def get_fact(self, key: str) -> Any:
        """Retrieve a stored fact."""
        return self.important_facts.get(key)
    
    def get_context(self) -> str:
        """Get the memory context for the agent."""
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in self.history])
        facts_text = "\n".join([f"{k}: {v}" for k, v in self.important_facts.items()])
        
        return f"Memory:\n{history_text}\n\nImportant Facts:\n{facts_text}"

class OllamaAgent:
    """
    A simple AI agent that can plan and execute tasks using Ollama.
    """
    def __init__(self, 
                 model: str = "lqwen2:0.5b", 
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
        
        When you need to use a tool, respond in the following format:
        
        <reasoning>
        Your step-by-step thought process here
        </reasoning>
        
        <tool>
        {
            "tool_name": "name_of_tool",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            }
        }
        </tool>
        
        Available tools:
        {tool_descriptions}
        """
        
    def register_tool(self, tool: Tool):
        """Register a new tool that the agent can use."""
        self.tools[tool.name] = tool
        
    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
        
    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
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
        
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response."""
        try:
            # Extract the tool section
            if "<tool>" not in response:
                return None
                
            tool_section = response.split("<tool>")[1].split("</tool>")[0].strip()
            tool_data = json.loads(tool_section)
            
            return {
                "tool_name": tool_data["tool_name"],
                "parameters": tool_data["parameters"]
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing tool call: {e}")
            print(f"Response was: {response}")
            return None
            
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool based on the parsed tool call."""
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
            
        tool = self.tools[tool_name]
        return tool.call(**parameters)
    
    def _process_response(self, response: str) -> str:
        """Process the model's response to extract reasoning and potentially execute tools."""
        # Extract reasoning if present
        reasoning = ""
        reasoning_section = ""
        if "<reasoning>" in response:
            try:
                reasoning_section = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
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
        if "<reasoning>" in response and reasoning_section:
            clean_response = response.replace(
                f"<reasoning>{reasoning_section}</reasoning>", 
                f"I thought about this:\n{reasoning_section}\n\n"
            )
        
        return clean_response
        
    def _generate_follow_up(self, tool_call: Dict[str, Any], tool_result: str) -> str:
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
    
    def process_input(self, user_input: str) -> str:
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

class OllamaAgentWithStreaming(OllamaAgent):
    """
    Extension of OllamaAgent that supports streaming responses.
    """
    def _stream_ollama(self, messages: List[Dict[str, str]]) -> str:
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
            print()  # New line after streaming completes
            
            return full_response
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Failed to communicate with Ollama - {str(e)}"
            print(f"\nAgent: {error_msg}")
            return error_msg
        
    def process_input_streaming(self, user_input: str) -> str:
        """
        Process user input and generate a streaming response.
        Note: This method doesn't support tool usage during streaming.
        
        Args:
            user_input: The user's message or query
            
        Returns:
            The agent's complete response after streaming
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
        
        # Stream the response from the LLM
        response_text = self._stream_ollama(messages)
        
        # Process the response (here we don't use tools during streaming for simplicity)
        # You could add tool processing after streaming if needed
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", response_text)
        
        return response_text


# Example usage
def main():
    """Example of how to use the OllamaAgent."""
    # Check if Ollama is running at the default URL
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Ollama API is not responding correctly. Please ensure it's running.")
            return
            
        # Print available models
        models = response.json().get("models", [])
        if models:
            print("Available Ollama models:")
            for model in models:
                print(f"- {model['name']}")
        else:
            print("No models found in Ollama. Please pull a model first (e.g., 'ollama pull llama3')")
            return
    except requests.exceptions.RequestException:
        print("Could not connect to Ollama. Please ensure it's running at http://localhost:11434")
        return
    
    # Ask user which model to use
    selected_model = input("Enter the model name to use (default: llama3): ").strip() or "llama3"
    
    # Create the agent
    agent = OllamaAgentWithStreaming(model=selected_model)
    
    # Define and register some example tools
    def search_web(query: str) -> str:
        """Simulated web search tool."""
        # In a real implementation, this would call a search API
        return f"Top results for '{query}':\n1. Example result 1\n2. Example result 2"
    
    def get_current_time() -> str:
        """Tool to get the current time."""
        return f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def calculate(expression: str) -> str:
        """Simple calculator tool."""
        try:
            # Use a safer evaluation approach
            import ast
            def safe_eval(expr):
                return ast.literal_eval(expr)
            return f"Result: {safe_eval(expression)}"
        except Exception as e:
            return f"Error calculating: {str(e)}"
    
    def get_weather(location: str) -> str:
        """Simulated weather tool."""
        weather_conditions = ["sunny", "cloudy", "rainy", "snowy"]
        temperatures = range(0, 40)
        import random
        condition = random.choice(weather_conditions)
        temp = random.choice(temperatures)
        return f"Weather in {location}: {condition}, {temp}°C"
    
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
    
    agent.register_tool(Tool(
        "get_weather",
        "Get the current weather for a location. Parameters: location (str)",
        get_weather
    ))
    
    # Ask user if they want streaming mode
    use_streaming = input("Do you want to use streaming mode? (y/n, default: y): ").strip().lower() != 'n'
    
    # Simple interaction loop
    print(f"OllamaAgent initialized with model {selected_model}. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        try:
            if use_streaming:
                # Streaming doesn't support tools yet
                response = agent.process_input_streaming(user_input)
            else:
                print("\nAgent: ", end="")
                response = agent.process_input(user_input)
                print(response)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()