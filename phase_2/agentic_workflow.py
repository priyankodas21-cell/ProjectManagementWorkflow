# agentic_workflow.py

import os
import re
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
    "Create a complete set of user stories covering the product's in-scope functional requirements, non-functional requirements, and measurable objectives. "
    "Give every story a unique ID in order, starting with US-001. "
    "Write every story in exactly this form: As a [type of user], I want [action or feature] so that [benefit/value]. "
    "Include applicable requirements and targets for processing time, uptime, email throughput, growth, encryption, TLS, access control, authentication, and privacy compliance. "
    "Use only user classes, targets, and product behavior supported by the supplied specification. "
    "Do not use alternate openings such as 'I need' or 'I expect'. Output each ID on its own line immediately before its story. "
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
    "Accept only a complete set of user stories covering the in-scope functional requirements, non-functional requirements, and measurable objectives in the supplied product specification. "
    "Every story must have a unique US-nnn ID and follow exactly this structure: As a [type of user], I want [an action or feature] so that [benefit/value]. "
    "Check for stories covering applicable processing-time, uptime, throughput, growth, encryption, TLS, access-control, authentication, and privacy targets. "
    "Reject stories with alternate openings, missing persona/action/benefit, unsupported product facts, or substantial requirement gaps. "
    "Request corrections for every defect found."
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
    "Create a complete set of product features by grouping the supplied user stories into cohesive product capabilities. "
    "For every feature, write these literal labels, each on its own line and in this order: Feature Name: [title], Description: [purpose], Key Functionality: [capabilities], User Benefit: [value], Related User Stories: [US-nnn IDs]. "
    "Do not use a heading such as 'Feature 1: ...' instead of the literal Feature Name: field. "
    "Under Related User Stories, list the US-nnn IDs that the feature groups. "
    "Cover the supplied stories without inventing product capabilities."
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

program_manager_evaluation_criteria = (
    "Accept only a complete, cohesive grouping of the supplied user stories into product features grounded in the supplied specification. "
    "Every feature must include these literal fields, each on its own line and in this order: "
    "Feature Name:, Description:, Key Functionality:, User Benefit:, Related User Stories:. "
    "A heading such as 'Feature 1: ...' does not replace the Feature Name: field. "
    "Related User Stories must cite valid US-nnn IDs from the supplied stories. "
    "Reject missing/reordered labels, omitted major story groups, duplicated features, or unsupported capabilities. "
    "Request corrections for every defect found."
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
    "Create a complete engineering task breakdown covering the supplied user stories and product features. "
    "Write enough tasks to cover every feature and all applicable functional and non-functional requirements; do not return only representative tasks. "
    "For every task, write these literal labels, each on its own line and in this order: Task ID: TASK-nnn, Task Title:, Related User Story:, Description:, Acceptance Criteria:, Estimated Effort:, Dependencies:. "
    "Related User Story must cite one or more valid US-nnn IDs, not a feature name or N/A. Ensure every story and feature is covered by at least one task. "
    "Include modest, testable work for the stated performance, reliability, security/privacy, and scalability requirements. For scalability, limit the plan to a basic test against the specified email-volume targets; do not add production infrastructure or extensive capacity-planning work. "
    "Use the supplied product specification and prior workflow results. Do not invent requirements."
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

development_engineer_evaluation_criteria = (
    "Accept only a complete engineering task breakdown that covers the supplied user stories and features; do not accept one sample task when multiple tasks are needed. "
    "The tasks must cover applicable performance, reliability, security/privacy, and scalability requirements from the supplied specification. "
    "Every task must contain these literal labels, each on its own line and in this order: Task ID:, Task Title:, Related User Story:, Description:, Acceptance Criteria:, Estimated Effort:, Dependencies:. "
    "Task ID must have a unique value such as TASK-001. Related User Story must cite valid US-nnn IDs, never a feature name or N/A. "
    "Every story and feature must be covered by at least one task. Keep scalability work to a basic test against the specification's email-volume target; do not require production infrastructure or extensive capacity planning. "
    "Reject missing/reordered labels, tasks without a related story, incomplete coverage, or unsupported requirements. Request corrections for every defect found."
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=development_engineer_evaluation_criteria,
    worker_agent=development_engineer_knowledge_agent
)


# Job function persona support functions

STORY_REQUIREMENT_CHECKS = {
    "real-time email retrieval": ("real-time", "retriev"),
    "email metadata preprocessing": ("metadata", "preprocess"),
    "message classification and intent": ("classif", "intent"),
    "classification confidence score": ("confidence",),
    "knowledge base retrieval": ("knowledge base", "retriev"),
    "knowledge base updates": ("knowledge base", "updat"),
    "RAG response generation": ("rag", "response"),
    "human response approval": ("approval", "review"),
    "SME email routing": ("rout", "sme"),
    "routing context and metadata": ("context", "metadata"),
    "performance dashboard": ("dashboard", "metric"),
    "system configuration": ("configuration",),
    "manual override": ("manual", "override"),
    "five-second processing target": ("5 seconds",),
    "99.9% uptime target": ("99.9%",),
    "10,000 emails per hour target": ("10,000", "hour"),
    "200% email-volume growth target": ("200%",),
    "AES-256 encryption": ("aes-256",),
    "TLS 1.2 transport security": ("tls 1.2",),
    "role-based access control": ("rbac",),
    "administrative multi-factor authentication": ("mfa",),
    "GDPR and CCPA compliance": ("gdpr", "ccpa"),
    "PII masking": ("pii", "mask"),
    "90% routing accuracy objective": ("90%", "rout"),
    "60% response-time reduction objective": ("60%", "response"),
    "40% routine-response automation objective": ("40%", "routine"),
    "standardized responses objective": ("standard", "response"),
    "70% email-triage reduction objective": ("70%", "triage"),
    "customer satisfaction objective": ("30%", "satisfaction"),
    "communication analytics objective": ("analytics", "trend"),
    "knowledge-gap identification objective": ("knowledge", "gap"),
    "feedback and model-training capability": ("feedback", "training"),
}


def _clean_output_line(line):
    line = line.replace("**", "").replace("__", "").strip()
    return re.sub(r"^(?:(?:[-*])|(?:\d+[.)]))\s*", "", line).strip()


def _story_blocks(response):
    matches = list(re.finditer(r"(?im)^\s*\**(US-\d{3})\**\s*$", response))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
        blocks.append((match.group(1), response[match.end():end]))
    return blocks


def normalize_user_stories(response):
    normalized_lines = []
    benefit_markers = re.compile(
        r"\s+(to\s+(?:improve|ensure|streamline|provide|enhance|support|enable|facilitate|reduce|increase|maintain|achieve|allow|help|promote|protect|guide|meet|handle|address|deliver|monitor|identify|automate|classify|route|retrieve|process|secure|personalize|communicate|comply|establish|implement|use|preserve|prevent|measure|scale|maintain)\b|for\s+(?![a-z]+ing\b)[a-z][\w-]*\b)",
        re.IGNORECASE,
    )
    for line in response.splitlines():
        clean_line = _clean_output_line(line)
        if clean_line.lower().startswith(("as a ", "as an ")) and "i want" in clean_line.lower() and " so that " not in clean_line.lower():
            purposes = list(benefit_markers.finditer(clean_line))
            if purposes:
                purpose = purposes[-1]
                action = clean_line[:purpose.start()].rstrip(" ,.;")
                benefit_clause = clean_line[purpose.start():].strip().rstrip(". ")
                if benefit_clause.lower().startswith("to "):
                    benefit = "so that I can " + benefit_clause[3:]
                else:
                    benefit = "so that I can achieve " + benefit_clause[4:]
                clean_line = f"{action} {benefit}."
        normalized_lines.append(clean_line)
    return "\n".join(normalized_lines)


def _story_lines(response):
    result = []
    for story_id, block in _story_blocks(response):
        lines = [_clean_output_line(line) for line in block.splitlines() if _clean_output_line(line)]
        story_line = next((line for line in lines if re.match(r"^As an? .+?, I want .+? so that .+", line, re.IGNORECASE)), None)
        if story_line:
            result.append((story_id, story_line))
    return result


def validate_story_format(response):
    blocks = _story_blocks(response)
    if not blocks:
        return "No US-nnn user stories were found. Return the complete corrected user-story list."

    story_ids = [story_id for story_id, _ in blocks]
    if len(story_ids) != len(set(story_ids)):
        return "User-story IDs must be unique. Correct duplicate IDs."

    malformed_story_ids = [
        story_id for story_id, block in blocks
        if not any(
            re.match(r"^As an? .+?, I want .+? so that .+", _clean_output_line(line), re.IGNORECASE)
            for line in block.splitlines()
        )
    ]
    if malformed_story_ids:
        return (
            "These stories do not match the required format: " + ", ".join(malformed_story_ids) + ". "
            "Rewrite every listed story using 'As a/an [user], I want [action] so that [benefit]'. "
            "Keep their IDs and return the complete corrected story list, not advice."
        )
    return None


def _missing_story_requirements(story_text):
    normalized = story_text.lower()
    normalized = re.sub(r"role[ -]+based access control", "rbac", normalized)
    normalized = re.sub(r"multi[ -]+factor authentication", "mfa", normalized)
    return [
        requirement for requirement, terms in STORY_REQUIREMENT_CHECKS.items()
        if not all(term in normalized for term in terms)
    ]


def validate_user_stories(response):
    format_error = validate_story_format(response)
    if format_error:
        return format_error

    story_text = "\n".join(story_line.lower() for _, story_line in _story_lines(response))
    missing = _missing_story_requirements(story_text)
    if missing:
        return "Stories do not explicitly cover these specified requirements/objectives: " + ", ".join(missing) + ". Add or correct stories for them."
    return None


def _format_story_id(story_id, story_line):
    return f"{story_id}\n{story_line}"


def _supplemental_story_ids(response, next_number):
    additions = []
    seen_lines = set()
    for _, story_line in _story_lines(response):
        normalized_line = story_line.casefold()
        if normalized_line in seen_lines:
            continue
        seen_lines.add(normalized_line)
        additions.append(_format_story_id(f"US-{next_number:03d}", story_line))
        next_number += 1
    return additions, next_number


def _extract_story_ids(text):
    return set(re.findall(r"(?im)^\s*(?:[-*]\s*)?\**(US-\d{3})\**\s*$", text))


def _feature_blocks(response):
    lines = [_clean_output_line(line) for line in response.splitlines()]
    starts = [index for index, line in enumerate(lines) if line.startswith("Feature Name:")]
    return [
        "\n".join(line for line in lines[start:(starts[index + 1] if index + 1 < len(starts) else len(lines))] if line)
        for index, start in enumerate(starts)
    ]


def normalize_features(response):
    normalized_blocks = []
    for block in _feature_blocks(response):
        lines = [_clean_output_line(line) for line in block.splitlines() if _clean_output_line(line)]
        description = next(
            (line[len("Description:"):].strip() for line in lines if line.startswith("Description:") and line[len("Description:"):].strip()),
            None,
        )
        functionality_indexes = [index for index, line in enumerate(lines) if line.startswith("Key Functionality:")]
        if description and functionality_indexes:
            index = functionality_indexes[0]
            if not lines[index][len("Key Functionality:"):].strip():
                lines[index] = f"Key Functionality: {description}"
        elif description:
            benefit_index = next((index for index, line in enumerate(lines) if line.startswith("User Benefit:")), None)
            if benefit_index is not None:
                lines.insert(benefit_index, f"Key Functionality: {description}")
        normalized_blocks.append("\n".join(lines))
    return "\n\n".join(normalized_blocks) if normalized_blocks else response


def validate_features(response, source_query):
    story_ids = _extract_story_ids(source_query)
    lines = [_clean_output_line(line) for line in response.splitlines()]
    starts = [index for index, line in enumerate(lines) if line.startswith("Feature Name:")]
    if not starts:
        return "No literal Feature Name: fields were found. Return features using the required fields, not headings alone."

    linked_story_ids = set()
    required_labels = ("Feature Name:", "Description:", "Key Functionality:", "User Benefit:", "Related User Stories:")
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        feature_lines = [line for line in lines[start:end] if line]
        label_positions = []
        for label in required_labels:
            if not any(line.startswith(label) and line[len(label):].strip() for line in feature_lines):
                return f"Each feature must include a non-empty '{label}' field. Correct the complete feature list."
            label_positions.append(next(index for index, line in enumerate(feature_lines) if line.startswith(label)))
        if label_positions != sorted(label_positions):
            return "Feature fields must appear in this order: Feature Name, Description, Key Functionality, User Benefit, Related User Stories."
        relation_line = next(line for line in feature_lines if line.startswith("Related User Stories:"))
        linked_story_ids.update(re.findall(r"US-\d{3}", relation_line))

    unknown_ids = linked_story_ids - story_ids
    missing_ids = story_ids - linked_story_ids
    if unknown_ids or missing_ids:
        details = []
        if unknown_ids:
            details.append("unknown story IDs: " + ", ".join(sorted(unknown_ids)))
        if missing_ids:
            details.append("stories not linked to a feature: " + ", ".join(sorted(missing_ids)))
        return "Feature story links are invalid (" + "; ".join(details) + "). Correct the feature list."
    return None


def _collect_valid_features(response, allowed_story_ids):
    valid_blocks = []
    linked_ids = set()
    for feature_block in _feature_blocks(response):
        relation_lines = [
            _clean_output_line(line)
            for line in feature_block.splitlines()
            if _clean_output_line(line).startswith("Related User Stories:")
        ]
        feature_story_ids = set().union(*(
            set(re.findall(r"US-\d{3}", line)) for line in relation_lines
        )) if relation_lines else set()
        if not feature_story_ids or not feature_story_ids.issubset(allowed_story_ids):
            continue
        if validate_features(feature_block, "\n".join(sorted(feature_story_ids))) is None:
            valid_blocks.append(feature_block)
            linked_ids.update(feature_story_ids)
    return valid_blocks, linked_ids


def validate_engineering_tasks(response, source_query):
    story_ids = _extract_story_ids(source_query)
    lines = [_clean_output_line(line) for line in response.splitlines()]
    starts = [index for index, line in enumerate(lines) if re.match(r"^Task ID:\s*TASK-\d+\s*$", line)]
    if not starts:
        return "No correctly formatted Task ID: TASK-nnn fields were found. Return the complete task breakdown."

    task_ids = [re.search(r"TASK-\d+", lines[index]).group(0) for index in starts]
    if len(task_ids) != len(set(task_ids)):
        return "Task IDs must be unique. Correct duplicate task IDs."

    required_labels = ("Task ID:", "Task Title:", "Related User Story:", "Description:", "Acceptance Criteria:", "Estimated Effort:", "Dependencies:")
    linked_story_ids = set()
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        task_lines = [line for line in lines[start:end] if line]
        label_positions = []
        for label in required_labels:
            if not any(line.startswith(label) and line[len(label):].strip() for line in task_lines):
                return f"Each task must include a non-empty '{label}' field. Correct the complete task breakdown."
            label_positions.append(next(index for index, line in enumerate(task_lines) if line.startswith(label)))
        if label_positions != sorted(label_positions):
            return "Task fields must appear in this order: Task ID, Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort, Dependencies."
        relation_line = next(line for line in task_lines if line.startswith("Related User Story:"))
        if re.search(r"\b(?:N/A|None)\b", relation_line, re.IGNORECASE):
            return "Every task must reference one or more valid US-nnn IDs, not N/A or None."
        linked_story_ids.update(re.findall(r"US-\d{3}", relation_line))

    unknown_ids = linked_story_ids - story_ids
    missing_ids = story_ids - linked_story_ids
    if unknown_ids or missing_ids:
        details = []
        if unknown_ids:
            details.append("unknown story IDs: " + ", ".join(sorted(unknown_ids)))
        if missing_ids:
            details.append("stories without an engineering task: " + ", ".join(sorted(missing_ids)))
        return "Task story links are invalid (" + "; ".join(details) + "). Correct the task breakdown."
    return None
 
def product_manager_support_function(query):
    res = product_manager_evaluation_agent.evaluate(
        query,
        validator=validate_user_stories,
        normalizer=normalize_user_stories,
    )
    if isinstance(res, dict) and res.get("accepted"):
        return res["final_response"]

    initial_response = normalize_user_stories(res.get("final_response", "")) if isinstance(res, dict) else ""
    format_error = validate_story_format(initial_response)
    if format_error:
        raise RuntimeError(f"User-story stage failed format validation: {format_error}")

    existing_stories = [story_line for _, story_line in _story_lines(initial_response)]
    existing_story_ids = [int(story_id.split("-")[1]) for story_id, _ in _story_lines(initial_response)]
    next_story_number = max(existing_story_ids, default=0) + 1
    missing_requirements = _missing_story_requirements("\n".join(existing_stories))

    recovery_rounds = 0
    while missing_requirements and recovery_rounds < 3:
        recovery_rounds += 1
        missing_before_round = set(missing_requirements)
        for batch_start in range(0, len(missing_requirements), 4):
            requirement_batch = missing_requirements[batch_start:batch_start + 4]
            requirements_text = "\n".join(
                f"- {requirement}; include every exact term from this list in the story: {list(STORY_REQUIREMENT_CHECKS[requirement])!r}"
                for requirement in requirement_batch
            )
            supplemental_prompt = (
                "Write one new, distinct user story for each listed requirement. Include every listed term verbatim "
                "in the corresponding story. Each story must follow exactly: As a [type of user], I want [action] "
                "so that [benefit]. Use unique placeholder IDs beginning at US-901, one ID per story. "
                "Return only the stories, not explanations.\n\n"
                f"Requirements still uncovered:\n{requirements_text}\n\n"
                f"Product specification:\n{product_spec}"
            )
            supplemental_result = product_manager_evaluation_agent.evaluate(
                supplemental_prompt,
                validator=validate_story_format,
                normalizer=normalize_user_stories,
            )
            if not supplemental_result.get("accepted"):
                continue
            additions, next_story_number = _supplemental_story_ids(
                supplemental_result["final_response"], next_story_number
            )
            existing_stories.extend(
                story_line for _, story_line in _story_lines("\n\n".join(additions))
            )

        missing_requirements = _missing_story_requirements("\n".join(existing_stories))
        if set(missing_requirements) == missing_before_round:
            break

    if not missing_requirements:
        return "\n\n".join(
            _format_story_id(f"US-{index:03d}", story_line)
            for index, story_line in enumerate(existing_stories, start=1)
        )

    raise RuntimeError(
        "User-story stage still lacks coverage after targeted recovery: "
        + ", ".join(missing_requirements)
    )

def program_manager_support_function(query):
    res = program_manager_evaluation_agent.evaluate(
        query,
        validator=lambda response: validate_features(response, query),
        normalizer=normalize_features,
    )
    if isinstance(res, dict) and res.get("accepted"):
        return res["final_response"]

    initial_features = res.get("final_response", "") if isinstance(res, dict) else ""
    story_blocks = _story_blocks(query)
    all_story_ids = {story_id for story_id, _ in story_blocks}
    if not all_story_ids:
        raise RuntimeError("Feature stage could not find user-story IDs in its input.")

    valid_features = []
    linked_story_ids = set()
    valid_features, linked_story_ids = _collect_valid_features(initial_features, all_story_ids)
    recovery_rounds = 0
    max_recovery_rounds = 3
    batch_size = 4

    while linked_story_ids != all_story_ids and recovery_rounds < max_recovery_rounds:
        recovery_rounds += 1
        missing_story_blocks = [
            (story_id, block)
            for story_id, block in story_blocks
            if story_id not in linked_story_ids
        ]
        linked_before_round = linked_story_ids.copy()

        for batch_start in range(0, len(missing_story_blocks), batch_size):
            batch = missing_story_blocks[batch_start:batch_start + batch_size]
            batch_story_ids = {story_id for story_id, _ in batch}
            batch_query = (
                "Create one or more cohesive product features covering every supplied user story. "
                "Return only valid features using all five required fields, including a non-empty Key Functionality, "
                "and list the exact US-nnn IDs covered by each feature. Do not omit any supplied story.\n\n"
                f"Product specification:\n{product_spec}\n\n"
                "Stories still missing feature coverage:\n"
                + "\n\n".join(f"{story_id}\n{block.strip()}" for story_id, block in batch)
            )
            batch_result = program_manager_evaluation_agent.evaluate(
                batch_query,
                validator=lambda response, prompt=batch_query: validate_features(response, prompt),
                normalizer=normalize_features,
            )
            recovered_features, recovered_ids = _collect_valid_features(
                batch_result.get("final_response", ""), batch_story_ids
            )
            valid_features.extend(recovered_features)
            linked_story_ids.update(recovered_ids)

        if linked_story_ids == linked_before_round:
            break

    combined_features = "\n\n".join(valid_features).strip()
    combined_error = validate_features(combined_features, query)
    if combined_error is None:
        return combined_features

    raise RuntimeError(f"Feature stage failed validation after targeted recovery: {combined_error}")

def development_engineer_support_function(query):
    res = development_engineer_evaluation_agent.evaluate(query, validator=lambda response: validate_engineering_tasks(response, query))
    if isinstance(res, dict) and res.get("accepted"):
        return res["final_response"]
    raise RuntimeError(f"Engineering-task stage failed validation: {res.get('evaluation', 'No accepted result')}")

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
planned_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
expected_stages = (
    ("Product Manager", "Product Manager: Create user stories from the product specification."),
    ("Program Manager", "Program Manager: Group the completed user stories into product features."),
    ("Development Engineer", "Development Engineer: Create engineering tasks from the specification, user stories, and features."),
)
planned_by_role = {}
for planned_step in planned_steps:
    cleaned_step = re.sub(r"^\s*(?:(?:[-*])|(?:\d+[.)])|(?:step\s+\d+[:.)]?))\s*", "", planned_step, flags=re.IGNORECASE).strip()
    matching_roles = [role for role, _ in expected_stages if role.casefold() in cleaned_step.casefold()]
    if len(matching_roles) == 1:
        planned_by_role.setdefault(matching_roles[0], cleaned_step)

workflow_steps = [
    planned_by_role.get(role, fallback)
    for role, fallback in expected_stages
]
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