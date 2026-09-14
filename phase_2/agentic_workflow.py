# agentic_workflow.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase_1.workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
product_spec = ""
with open("Product-Spec-Email-Router.txt", "r") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"\nProduct Spec:\n{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent
)
# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

program_manager_evaluation_criteria = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=program_manager_evaluation_criteria,
    worker_agent=program_manager_knowledge_agent
)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

development_engineer_evaluation_criteria = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=development_engineer_evaluation_criteria,
    worker_agent=development_engineer_knowledge_agent
)


# Job function persona support functions
 
def product_manager_support_function(query):
    knowledge_response = product_manager_knowledge_agent.respond(query)
    res = product_manager_evaluation_agent.evaluate(knowledge_response)
    return res["final_response"] if isinstance(res, dict) else res

def program_manager_support_function(query):
    knowledge_response = program_manager_knowledge_agent.respond(query)
    res = program_manager_evaluation_agent.evaluate(knowledge_response)
    return res["final_response"] if isinstance(res, dict) else res

def development_engineer_support_function(query):
    knowledge_response = development_engineer_knowledge_agent.respond(query)
    res = development_engineer_evaluation_agent.evaluate(knowledge_response)
    return res["final_response"] if isinstance(res, dict) else res

# Routing Agent
routes = [
    {
        "name": "Product Manager",
        "description": "Handles product-related queries and tasks.",
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": "Handles program management-related queries and tasks.",
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": "Handles development-related queries and tasks.",
        "func": development_engineer_support_function
    }
]


routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=routes
)

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

workflow_steps = action_planning_agent.generate_response(workflow_prompt)
completed_steps = []
for step in workflow_steps:
    print(f"\nExecuting step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"Result of step: {result}")

if completed_steps:
    final_output = completed_steps[-1]
    print(f"\nFinal output of the workflow: {final_output}")
    
    # Save output to a text file for easy viewing
    output_filepath = os.path.join(os.path.dirname(__file__), "workflow_output.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(str(final_output))
    print(f"\nSaved final output to {output_filepath}")

print("\n*** Workflow execution finished ***\n")