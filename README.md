# AI-Powered Agentic Workflow for Project Management

## Overview

This project is designed in two phases:

1. Phase 1 focuses on building a reusable library of AI agents.
2. Phase 2 uses that library to implement an agentic workflow for project management.

The goal is to create a flexible, multi-agent system that can collaborate to handle real-world product-development tasks such as story generation, feature planning, engineering task breakdown, and workflow routing.

---

## Phase 1: Building Your Agent Library

In the first phase, you build a library of reusable agents that can be used in Agentic workflows beyond this project. The agent classes are implemented in the `phase_1/workflow_agents/base_agents.py` file and validated through standalone test scripts.

By the end of this phase, you will have:

- Implemented seven agent classes in a single `base_agents.py` file
- Verified each agent using a dedicated test script
- Organized the code into a clean, importable package for reuse in Phase 2

### Directory Structure

```text
phase_1/
├── workflow_agents/
│   ├── __init__.py
│   └── base_agents.py
├── direct_prompt_agent_test.py
├── augmented_prompt_agent_test.py
├── knowledge_augmented_prompt_agent_test.py
├── rag_knowledge_prompt_agent_test.py
├── evaluation_agent_test.py
├── routing_agent_test.py
├── action_planning_agent_test.py
└── README.md
```

### Included Agent Types

#### 1. Direct Prompt Agent
A Direct Prompt Agent sends the user prompt directly to the LLM with no extra context, memory, or tools. It is the most straightforward implementation and returns the LLM's textual response.

Implementation requirements:

- Import `OpenAI` from the OpenAI Python library
- Store the API key in `openai_api_key`
- Use the `gpt-3.5-turbo` model
- Pass the user prompt as a user message
- Return only the response text, not the full JSON payload

#### 2. Augmented Prompt Agent
An Augmented Prompt Agent behaves according to a defined persona. The agent is instructed to act as a specific type of assistant so the response is more targeted and contextually relevant.

Implementation requirements:

- Create a persona attribute
- Call the OpenAI chat completions API
- Include a system prompt telling the agent to adopt that persona and ignore previous context
- Return the text content only

#### 3. Knowledge Augmented Prompt Agent
This agent adds explicit knowledge to the response logic, ensuring the answer is based on supplied information rather than the model's internal knowledge.

Implementation requirements:

- Create a `persona` attribute
- Create a `knowledge` attribute
- In `respond`, build a system message that includes:
  - `You are _persona_ knowledge-based assistant. Forget all previous context.`
  - `Use only the following knowledge to answer, do not use your own knowledge: _knowledge_`
  - `Answer the prompt based on this knowledge, not your own.`
- Append the user prompt as a separate message

#### 4. RAG Knowledge Prompt Agent
This agent uses retrieval-augmented generation for dynamic knowledge sourcing. It is already provided in the starter project and can be explored as a reference for RAG patterns.

#### 5. Evaluation Agent
The Evaluation Agent checks the response of another worker agent against a set of criteria and can iterate to improve the final answer.

Implementation requirements:

- Define class attributes, including `max_interactions`
- Run an evaluation loop limited by `max_interactions`
- Retrieve the worker agent's response
- Build an evaluation prompt and the correction prompt
- Use OpenAI with `temperature=0`
- Return a dictionary containing the final response, the evaluation result, and the iterations performed

#### 6. Routing Agent
The Routing Agent chooses the best specialized agent based on semantic similarity between the user prompt and agent descriptions.

Implementation requirements:

- Store `agents` with descriptions and callable functions
- Implement `get_embedding` using the `text-embedding-3-large` model
- For each agent, compute embeddings for both the prompt and the agent description
- Measure cosine similarity
- Route the prompt to the most relevant agent and return that response

#### 7. Action Planning Agent
The Action Planning Agent extracts the sequence of actions needed for a task based on a user prompt and relevant knowledge.

Implementation requirements:

- Initialize the API key and agent knowledge
- Instantiate the OpenAI client
- Send a request to `gpt-3.5-turbo` with a system prompt that frames the agent as an action planner
- Pass the user prompt to the model
- Extract readable action steps from the response and remove empty or irrelevant lines

### Environment Setup

Create a `.env` file containing your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

This is used by the agent test scripts to authenticate with OpenAI.

---

## Phase 2: Implement an Agentic Workflow Using the Agent Library

Once the agent library is tested and complete, Phase 2 builds a practical agentic workflow for project management. The focus is on using specialized agents to support product management tasks and build feedback-driven, structured outputs.

This phase is not a chatbot; it is a workflow system that processes a prompt and produces a structured output. The system is tested with realistic product-management prompts, sometimes called "golden prompts."

### Workflow Goal

The workflow models a Technical Program Manager (TPM) using specialized agents to:

- Interpret product specifications
- Define user stories
- Define product features
- Break down engineering tasks
- Route tasks to the most appropriate agent
- Evaluate results before finalizing output

### Phase 2 Project Structure

```text
phase_2/
├── agentic_workflow.py
├── Product-Spec-Email-Router.txt
├── README.md
├── workflow_output.txt
├── workflow_agents/
│   └── __init__.py
```

The workflow is built in `phase_2/agentic_workflow.py` and uses the agent classes from `workflow_agents.base_agents` developed in Phase 1.

### Workflow Implementation Steps

Follow the TODO comments in `agentic_workflow.py` and complete the workflow in the following order:

1. Import `ActionPlanningAgent`, `KnowledgeAugmentedPromptAgent`, `EvaluationAgent`, and `RoutingAgent`
2. Load the OpenAI API key from environment variables
3. Load the product specification from `Product-Spec-Email-Router.txt` into a variable named `product_spec`
4. Instantiate the `ActionPlanningAgent` using the provided knowledge string
5. Complete `knowledge_product_manager` by appending the product specification
6. Instantiate the Product Manager knowledge agent with `persona_product_manager`
7. Instantiate the Product Manager evaluation agent using the evaluation persona and criteria
8. Instantiate the Program Manager knowledge and evaluation agents
9. Instantiate the Development Engineer knowledge and evaluation agents
10. Instantiate the `RoutingAgent` with routes for Product Manager, Program Manager, and Development Engineer
11. Define support functions that call each knowledge agent and then evaluate the output
12. Run the workflow by:
    - extracting workflow steps through `action_planning_agent.extract_steps_from_prompt()`
    - iterating over each step
    - routing each step to the correct agent
    - appending results to `completed_steps`
    - printing the workflow output

### Sample Workflow Pattern

The workflow is designed to process a prompt like a real TPM task and then break it into a plan. Example behavior:

- Action planning identifies the steps required to complete the request
- Product Manager agent defines user stories
- Program Manager agent identifies features
- Development Engineer agent creates engineering tasks
- Routing decides which agent should handle each step
- Evaluation agents validate the output against defined criteria

### Evaluation and Validation

The phase 2 workflow checks that generated outputs follow expected structures, for example:

- User story format: `As a [type of user], I want [an action or feature] so that [benefit/value].`
- Feature format: name, description, key functionality, and user benefit
- Engineering task format: task ID, title, related user story, description, acceptance criteria, effort, and dependencies

---

## Project Summary

This repository combines two complementary stages:

- Phase 1 builds the reusable agent library.
- Phase 2 uses that library to create a practical project-management workflow.

Together, they illustrate how agentic systems can be structured, validated, and routed to solve realistic business problems with LLM-based multi-agent collaboration.

---

## Quick Setup

1. Ensure Python dependencies for OpenAI and dotenv are installed.
2. Create a `.env` file in the project root or relevant test folder.
3. Add your API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

4. Complete Phase 1 agent implementations and verify them with the provided test scripts.
5. Use the Phase 2 workflow script to run the end-to-end project management flow.

---

## Notes

- This project is meant to be extended and reused for other agentic workflows beyond project management.
- The reusable agent library is intentionally modular so it can support new workflows with minimal changes.
- The final workflow should output a structured set of deliverables derived from the product specification and action plan.
