# REACT Agents From Scratch

A Python implementation of the REACT (Reasoning and Acting) agent framework, built from the ground up to illustrate the core concepts without heavy reliance on external agent libraries like LangChain or LlamaIndex.

This project aims to provide a clear, understandable, and minimal example of how an LLM can reason about a task, select appropriate tools, act upon them, observe the results, and iterate until the task is complete.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges if relevant: build status, coverage, etc. -->

## Table of Contents

*   [What is REACT?](#what-is-react)
*   [Why "From Scratch"?](#why-from-scratch)
*   [Features](#features)
*   [Architecture Overview](#architecture-overview)
*   [Installation](#installation)
*   [Configuration](#configuration)
*   [Usage](#usage)
*   [Tools](#tools)
    *   [Available Tools](#available-tools)
    *   [Adding New Tools](#adding-new-tools)
*   [How it Works (The REACT Loop)](#how-it-works-the-react-loop)
*   [Extending the Agent](#extending-the-agent)
*   [Contributing](#contributing)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## What is REACT?

REACT stands for **Reasoning and Acting**. It's a framework proposed by Shunyu Yao et al. (Paper: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)) that enables Large Language Models (LLMs) to solve complex tasks by interleaving reasoning steps (generating thoughts about the problem) with action steps (using external tools like search engines, calculators, or APIs).

The core idea is a loop:

1.  **Thought:** The LLM analyzes the current state and the overall goal, deciding the next step.
2.  **Action:** Based on the thought, the LLM decides to use a specific tool with certain inputs.
3.  **Observation:** The result of executing the action (tool output) is fed back to the LLM.
4.  **(Repeat):** The LLM uses the observation to generate the next thought, continuing the cycle until the final answer is reached.

## Why "From Scratch"?

While excellent frameworks like LangChain exist, building REACT from scratch offers several benefits:

*   **Transparency:** Understand the exact mechanics of the agent loop without layers of abstraction.
*   **Learning:** A great way to learn the fundamental principles of agent design.
*   **Customization:** Easily modify prompts, parsing logic, and tool integration specific needs.
*   **Minimalism:** Avoids the overhead and complexity of large libraries if only core REACT functionality is needed.

## Features

*   Clear implementation of the core REACT loop (Thought, Action, Observation).
*   Integration with an LLM (e.g., OpenAI's GPT models - easily adaptable).
*   Modular tool system: Define and add custom tools easily.
*   Example tools included (e.g., Search, Calculator - *you might need to implement these*).
*   Configurable prompts for reasoning and action generation.
*   Step-by-step logging/tracing of the agent's thought process.

## Architecture Overview

*(Describe the main components of your code. Adjust based on your structure)*

*   **`agent.py`:** Contains the main `ReactAgent` class orchestrating the loop.
*   **`llm.py`:** Wrapper/interface for interacting with the chosen LLM API (e.g., OpenAI).
*   **`prompts.py`:** Stores the prompt templates used for reasoning and formatting.
*   **`tools/`:** Directory containing modules for each available tool.
    *   **`base_tool.py`:** (Optional) Base class or definition for tools.
    *   **`search.py`:** Example Search tool implementation.
    *   **`calculator.py`:** Example Calculator tool implementation.
*   **`parser.py`:** Logic to parse the LLM's output to extract Thoughts and Actions/Action Inputs.
*   **`main.py`:** Entry point script to run the agent with a given task.
*   **`config.py` / `.env`:** Handles configuration like API keys and model settings.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/react-from-scratch.git
    cd react-from-scratch
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have a `requirements.txt` file listing packages like `openai`, `python-dotenv`, etc.)*

4.  **Set up environment variables:**
    Create a `.env` file in the project root directory and add your API keys and other configurations:
    ```dotenv
    # .env file
    OPENAI_API_KEY="your_openai_api_key_here"
    # Add other relevant variables, e.g., SERPAPI_API_KEY if using a search tool
    ```

## Configuration

Agent behavior can be configured via:

*   **Environment Variables (`.env`):** Primarily for secrets like API keys.
*   **Constants/Config file (`config.py` or `config.yaml`):** (Optional) For settings like:
    *   LLM model name (e.g., `gpt-3.5-turbo`, `gpt-4`)
    *   LLM temperature
    *   Maximum iterations/steps for the agent
    *   Prompt template paths or strings

*(Specify where users should look to configure these settings in your code)*

## Usage

Run the agent from the command line:

```bash
python main.py "Your question or task for the agent here"
