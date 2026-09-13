from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# The agent likely used its internal knowledge and the OpenAI API to answer the prompt.
# The system prompt specifying the persona influenced the agent to respond in the style of a college professor, starting with "Dear students,".
