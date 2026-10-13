"""Original hierarchical entity classification prompts."""

ENTITY_CLASSIFICATION_SYS_PROMPT = """
You are an expert in entity classification. Your task is to classify the category of the given entity based on the provided context as accurate as possible.
"""


ENTITY_CLASSIFICATION_PROMPT = """
You are given the following information:
    - Possible categories
    - Entity
Your task is to classify the entity into one of the possible categories based on the context provided.

## Possible categories ##
This section contains a list of possible categories for the entity. Each category has a defination that describes what the category represents.
{possible_categories}

## Entity ##
This section contains the entity that needs to be classified. The entity is represented by its name and additional information that may help in classification.
- Entity name: {entity_name}
- Entity information:\x20
{entity_info}

## Output format ##
You should output the category that best fits the entity based on the context provided. The output should be in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "reason": "Due to ..., the entity is classified as ...",
    "category": "<category_name>"
}}
```
""".strip()


ENTITY_CLASSIFICATION_FIX_PROMPT = """
Due to the following reason, your output is not valid:
{error_reason}

Please generate a new output in the correct format:
## Output format ##
The output should be in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "reason": "Due to ..., the entity is classified as ...",
    "category": "<category_name>"
}}
```
""".strip()
