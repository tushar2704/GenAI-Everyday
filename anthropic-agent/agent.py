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

import os
import json
import time
from typing import Dict, List, Any, Callable, Optional, Union
import anthropic

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
        except (KeyError, IndexError, json.JSONDecodeError):
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
        if "<reasoning>" in response:
            reasoning_section = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            reasoning = f"Reasoning:\n{reasoning_section}\n\n"
        
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
        if "<reasoning>" in response:
            clean_response = response.replace(
                f"<reasoning>{reasoning_section}</reasoning>", 
                f"I thought about this:\n{reasoning_section}\n\n"
            )
        
        return clean_response
        
    def _generate_follow_up(self, tool_call: Dict[str, Any], tool_result: str) -> str:
        """Generate a follow-up response after executing a tool."""
        messages = [
            {
                "role": "system", 
                "content": self.system_prompt.format(tool_descriptions=self._get_tool_descriptions())
            },
            {
                "role": "user", 
                "content": f"I want you to continue helping me after using a tool. You used the tool '{tool_call['tool_name']}' with parameters {json.dumps(tool_call['parameters'])} and got this result:\n\n{tool_result}\n\nPlease analyze this result and continue helping me."
            }
        ]
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=1000
        )
        
        return response.content[0].text
    
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
        
        # Create message with system prompt, memory and user input
        messages = [
            {
                "role": "system", 
                "content": self.system_prompt.format(tool_descriptions=self._get_tool_descriptions()) + f"\n\n{memory_context}"
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]
        
        # Call the Anthropic API
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=1500
        )
        
        response_text = response.content[0].text
        
        # Process the response (execute tools if needed)
        final_response = self._process_response(response_text)
        
        # Store the final response in memory
        self.memory.add_interaction("assistant", final_response)
        
        return final_response
        
# Example usage
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
            
        response = agent.process_input(user_input)
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()
