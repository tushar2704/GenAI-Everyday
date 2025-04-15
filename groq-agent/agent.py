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
        
        <reasoning>
        Your step-by-step thought process here
        </reasoning>
        
        <tool>
        {{"tool_name":"name_of_tool","parameters":{{"param1":"value1","param2":"value2"}}}}
        </tool>
        
        IMPORTANT: The tool section must be valid JSON without line breaks between keys and values.
        
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
    
    def diagnose_response(self, response: str) -> str:
        """Diagnostic function to help troubleshoot response parsing issues."""
        print("\n=== DIAGNOSTIC INFORMATION ===")
        print(f"Full response length: {len(response)} characters")
        print(f"Response starts with: {response[:100]}...")
        print(f"Response ends with: ...{response[-100:]}")
        
        # Check for expected XML tags
        reasoning_tag_open = "<reasoning>" in response
        reasoning_tag_close = "</reasoning>" in response
        tool_tag_open = "<tool>" in response
        tool_tag_close = "</tool>" in response
        
        print(f"Contains <reasoning> tag: {reasoning_tag_open}")
        print(f"Contains </reasoning> tag: {reasoning_tag_close}")
        print(f"Contains <tool> tag: {tool_tag_open}")
        print(f"Contains </tool> tag: {tool_tag_close}")
        
        # If tool tags are present, extract and analyze the tool section
        if tool_tag_open and tool_tag_close:
            try:
                tool_section = response.split("<tool>")[1].split("</tool>")[0].strip()
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
        
    def _call_groq(self, messages: List[Dict[str, str]]) -> str:
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
            return response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Groq API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return f"Error: Failed to communicate with Groq API - {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing Groq response: {e}")
            return "Error: Failed to parse Groq response"
        
    def _stream_groq(self, messages: List[Dict[str, str]]) -> str:
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
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
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
        
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call information from the model's response with enhanced error handling."""
        try:
            # Check if there's a tool section
            if "<tool>" not in response or "</tool>" not in response:
                return None
                
            # Extract the tool section
            tool_section = response.split("<tool>")[1].split("</tool>")[0].strip()
            
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
            
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
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
    
    def _process_response(self, response: str) -> str:
        """Process the model's response with enhanced error handling."""
        try:
            # Extract reasoning if present
            reasoning = ""
            if "<reasoning>" in response and "</reasoning>" in response:
                try:
                    reasoning_section = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                    reasoning = f"Reasoning:\n{reasoning_section}\n\n"
                except IndexError:
                    print("Warning: Could not extract reasoning section properly")
            
            # Check for tool calls
            tool_call = self._parse_tool_call(response)
            if tool_call:
                print(f"Successfully parsed tool call: {tool_call}")
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
            if "<reasoning>" in response and "</reasoning>" in response:
                reasoning_section = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                clean_response = response.replace(
                    f"<reasoning>{reasoning_section}</reasoning>", 
                    f"I thought about this:\n{reasoning_section}\n\n"
                )
                
            # Remove any tool tags that might be present but couldn't be parsed
            clean_response = re.sub(r'<tool>.*?</tool>', '', clean_response, flags=re.DOTALL)
            
            return clean_response
        except Exception as e:
            print(f"Error in _process_response: {e}")
            traceback.print_exc()
            # Return a simplified response when processing fails
            return "I encountered an error while processing my response. Let me try to answer your question directly: " + response.split("</reasoning>")[-1] if "</reasoning>" in response else response
        
    def _generate_follow_up(self, tool_call: Dict[str, Any], tool_result: str) -> str:
        """Generate a follow-up response after executing a tool."""
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create a system message with context
        system_message = self.system_prompt.format(tool_descriptions=self._get_tool_descriptions())
        
        # Create the messages for the follow-up
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"I want you to continue helping me after using a tool. You used the tool '{tool_call['tool_name']}' with parameters {json.dumps(tool_call['parameters'])} and got this result:\n\n{tool_result}\n\nPlease analyze this result and continue helping me."}
        ]
        
        # Get response from the LLM
        response = self._call_groq(messages)
        return response
    
    def process_input(self, user_input: str, use_streaming: bool = False) -> str:
        """
        Process user input and generate a response, potentially using tools.
        
        Args:
            user_input: The user's message or query
            use_streaming: Whether to stream the response or not
            
        Returns:
            The agent's response
        """
        # Store user input in memory
        self.memory.add_interaction("user", user_input)
        
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Create a system message with context
        system_message = self.system_prompt.format(tool_descriptions=self._get_tool_descriptions())
        
        # Create the messages for Groq
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
        
        # Get response from the LLM (streaming or regular)
        if use_streaming:
            response_text = self._stream_groq(messages)
        else:
            response_text = self._call_groq(messages)
        
        # Add diagnostic information
        self.diagnose_response(response_text)
        
        # Process the response (execute tools if needed)
        # For streaming, we don't process tools yet
        if not use_streaming:
            final_response = self._process_response(response_text)
        else:
            final_response = response_text
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", final_response)
        
        return final_response

# List of available Groq models
AVAILABLE_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama3-1-8b-8192",
    "llama3-1-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
]

def list_available_models():
    """Print a list of available Groq models."""
    print("Available Groq models:")
    for model in AVAILABLE_MODELS:
        print(f"- {model}")

def main():
    """Example of how to use the GroqAgent."""
    parser = argparse.ArgumentParser(description="Run an AI agent using the Groq API")
    parser.add_argument("--api-key", "-k", help="Your Groq API key")
    parser.add_argument("--model", "-m", help="The model to use", default="llama3-8b-8192")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
    parser.add_argument("--streaming", "-s", action="store_true", help="Use streaming mode")
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please provide a Groq API key using --api-key or set the GROQ_API_KEY environment variable")
        return
    
    # Use the model from argument, or prompt if not provided
    model = args.model
    if not model:
        list_available_models()
        model = input("Enter the model name to use (default: llama3-8b-8192): ").strip() or "llama3-8b-8192"
    
    # Create the agent
    agent = GroqAgent(api_key=api_key, model=model)
    
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
    
    # Use streaming mode based on argument, or prompt if not provided
    use_streaming = args.streaming
    if not args.streaming:
        use_streaming_input = input("Do you want to use streaming mode? (y/n, default: y): ").strip().lower()
        use_streaming = use_streaming_input != 'n'
    
    # Simple interaction loop
    print(f"GroqAgent initialized with model {model}. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        try:
            if not use_streaming:
                print("\nAgent: ", end="")
            
            response = agent.process_input(user_input, use_streaming=use_streaming)
            
            if not use_streaming:
                print(response)
        except Exception as e:
            print(f"\nError: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main()