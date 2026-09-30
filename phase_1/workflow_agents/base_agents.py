import os
from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


def _get_openai_client(api_key):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and api_key and str(api_key).startswith("voc-"):
        base_url = "https://openai.vocareum.com/v1"
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


# DirectPromptAgent class definition 
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initialize the agent
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = _get_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    
        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = _get_openai_client(self.openai_api_key)
        system_prompt = f"{self.persona}. Forget all previous context."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content 



# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = _get_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{self.persona} knowledge-based assistant. Use the following knowledge and the task context supplied in the request. Do not rely on outside knowledge or invent product-specific facts; state when the supplied context is insufficient.\nKnowledge: {self.knowledge}"},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = _get_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            start = end - self.chunk_overlap
            chunk_id += 1
            if end==len(text):
                break

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = _get_openai_client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent=None, max_interactions=5):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt, validator=None, normalizer=None):
        # This method manages interactions between agents to achieve a solution.
        client = _get_openai_client(self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        accepted = False

        for i in range(self.max_interactions):  
            print(f"\n--- Interaction {i+1} ---")
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            if normalizer:
                response_from_worker = normalizer(response_from_worker)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Validate and evaluate the response")
            validation_error = validator(response_from_worker) if validator else None
            if validation_error:
                evaluation = f"No: deterministic validation failed. {validation_error}"
            elif validator:
                evaluation = "Yes: deterministic validation passed."
            else:
                eval_prompt = (
                    f"Original task and supplied context:\n{initial_prompt}\n\n"
                    f"Does the following answer: {response_from_worker}\n"
                    f"Meet this criteria: {self.evaluation_criteria}\n"
                    f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are {self.persona}, an evaluator agent. Forget previous context."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    temperature=0
                )
                evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                accepted = True
                break
            else:
                print(" Step 4: Prepare corrections")
                if validation_error:
                    instructions = validation_error
                else:
                    instruction_prompt = (
                        f"Provide concise, specific corrections for this failed answer: {evaluation}"
                    )
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are {self.persona}, an evaluator agent. Forget previous context."},
                            {"role": "user", "content": instruction_prompt}
                        ],
                        temperature=0
                    )
                    instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Now produce the complete corrected answer to the original prompt. "
                    f"Do not return advice, instructions, or an explanation of the corrections. "
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
                "final_response": response_from_worker,
                "evaluation": evaluation,
                "iterations": i + 1,
                "accepted": accepted
        }   



class RoutingAgent():

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        client = _get_openai_client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        embedding = response.data[0].embedding
        return embedding 

    def route_prompt(self, user_input, routing_input=None):
        if routing_input:
            planned_role = next(
                (agent for agent in self.agents if agent["name"].casefold() in routing_input.casefold()),
                None
            )
            if planned_role is not None:
                print(f"[Router] Following planned role: {planned_role['name']}")
                return planned_role["func"](user_input)

        input_emb = self.get_embedding(routing_input or user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

    route = route_prompt




class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        client = _get_openai_client(self.openai_api_key)
        system_prompt = f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.choices[0].message.content

        steps = [step.strip() for step in response_text.split("\n") if step.strip()]

        return steps

    generate_response = extract_steps_from_prompt
