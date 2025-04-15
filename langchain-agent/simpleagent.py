from langchain_ollama import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Initialize the Ollama model
model = Ollama(model="llama3.1")

# Define tools that the agent can use
tools = [
    Tool(
        name="Search",
        func=lambda x: "Search results for: " + x,
        description="Useful for searching information"
    ),
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Useful for performing calculations"
    )
]

# Create a prompt template for the agent
prompt_template = """
You are a helpful AI assistant. Use the following tools to answer the user's question:

{tools}

Use the following format:
Question: the input question
Thought: your reasoning about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original user question

Question: {input}
Thought:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create the agent
agent = create_react_agent(model, tools, prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({"input": "What is the capital of France and what is 25 * 42?"})
print(response["output"])
