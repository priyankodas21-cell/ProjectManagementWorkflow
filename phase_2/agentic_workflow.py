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
    "Return exactly three steps, one per line and in this order: "
    "Product Manager creates user stories from the product specification; "
    "Program Manager groups those user stories into product features; "
    "Development Engineer creates engineering tasks from the product specification, "
    "user stories, and features. Each step must name its role and artifact."
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
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups."
)
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
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story."
)
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
    res = product_manager_evaluation_agent.evaluate(query)
    return res["final_response"] if isinstance(res, dict) else res

def program_manager_support_function(query):
    res = program_manager_evaluation_agent.evaluate(query)
    return res["final_response"] if isinstance(res, dict) else res

def development_engineer_support_function(query):
    res = development_engineer_evaluation_agent.evaluate(query)
    return res["final_response"] if isinstance(res, dict) else res

# Routing Agent
routes = [
    {
        "name": "Product Manager",
        "description": "Creates user stories from a product specification, using user personas, actions, and benefits. Does not group stories into features or create engineering tasks.",
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": "Groups supplied user stories into product features, each with a feature name, description, key functionality, and user benefit. Does not create user stories or engineering tasks.",
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": "Creates engineering implementation tasks from the product specification, user stories, and features, including acceptance criteria, effort estimates, and dependencies. Does not create user stories or group features.",
        "func": development_engineer_support_function
    }
]


routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=routes
)

# Run the planned and routed workflow

print("\n*** Workflow execution started ***\n")

print("\nGenerating the Email Router development plan from the product specification")

workflow_prompt = (
    "Create exactly three ordered workflow steps for a complete product development plan. "
    "Return one step per line: first, Product Manager creates user stories; second, "
    "Program Manager creates product features from those stories; third, Development "
    "Engineer creates engineering tasks from the product specification, stories, and features. "
    "Do not make one step per product requirement.\n\n"
    f"Product specification:\n{product_spec}"
)
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step_number, step in enumerate(workflow_steps, start=1):
    print(f"\n--- Workflow step {step_number}: {step} ---")

    routed_query = (
        f"Current workflow step: {step}\n\n"
        f"Product Specification:\n{product_spec}"
    )
    if completed_steps:
        prior_results = "\n\n".join(
            f"{completed_step['step']}\n{completed_step['result']}"
            for completed_step in completed_steps
        )
        routed_query += f"\n\nValidated results from prior steps:\n{prior_results}"

    step_result = routing_agent.route(routed_query, routing_input=step)
    completed_steps.append({"step": step, "result": step_result})
    print(f"\nResult for step {step_number}:\n{step_result}")

final_output = "Email Router Development Plan\n============================\n\n" + "\n\n".join(
    f"{item['step']}\n{'-' * len(item['step'])}\n{item['result']}"
    for item in completed_steps
)

print(f"\nFinal output of the workflow:\n{final_output}")

output_filepath = os.path.join(os.path.dirname(__file__), "workflow_output.txt")
with open(output_filepath, "w", encoding="utf-8") as f:
    f.write(final_output)
print(f"\nSaved final output to {output_filepath}")

print("\n*** Workflow execution finished ***\n")