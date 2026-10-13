"""Original candidate clustering and entity-equivalence prompts."""

PRE_DEDUPLICATE_SYS_PROMPT = """
You are an expert in deduplicating entities. Your task is to identify and cluster duplicate entities based on the provided context as strictly as possible.
""".strip()


PRE_DEDUPLICATE_PROMPT = """
You are given the following information:
    - Entities
Your task is to identify and cluster duplicate entities based on the context provided.
If you think some entities may de duplicates, do not hesitate to cluster them together.
You should consider the following aspects:
1. Some entities may have different names but refer to the same thing, so you should cluster them together.
2. Do not hesitate to cluster entities that are similar, you should be strict in clustering.
3. If the given entities are not duplicates, you should return an empty dictionary.

## Entities ##
This section contains the entities that may have duplicates.
{entities}

## Output format ##
If there are no duplicates, you should return an empty dictionary.
You should output the deduplicated entities in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "cluster1": ["entity_i", "entity_j", ...],
    "cluster2": ["entity_m", "entity_n", ...],
    ...
}}
"""


PRE_DEDUPLICATE_FIX_PROMPT = """
Due to the following reason, your output is not valid:
{error_reason}

Please generate a new output in the correct format:
## Output format ##
If there are no duplicates, you should return an empty dictionary.
You should output the deduplicated entities in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "cluster1": ["entity_i", "entity_j", ...],
    "cluster2": ["entity_m", "entity_n", ...],
    ...
}}
"""


DEDUPLICATE_SYS_PROMPT = """
You are an expert in checking entity duplicates. Your task is to check whether the given entities are same based on the provided context as accurate as possible.
""".strip()


DEDUPLICATE_PROMPT = """
You are given the following information:
    - Entities
Your task is to check whether the given entities are same based on the context provided.
You should determine if the entities are same or not as accurate as possible.
You should consider the following aspects:
1. Carefully read the information provided for each entity.
2. Determine if the entities are same or not based on the information provided.

## Entities ##
This section contains the entities that need to determine whether they are the same or not.
{entities_info}

## Output format ##
You should output the deduplicated entities in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "judgement": "same" or "different",
    "reason": "Due to ..., the entities are classified as same or different."
}}
""".strip()


DEDUPLICATE_FIX_PROMPT = """
Due to the following reason, your output is not valid:
{error_reason}

Please generate a new output in the correct format:
## Output format ##
You should output the deduplicated entities in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "judgement": "same" or "different",
    "reason": "Due to ..., the entities are classified as same or different."
}}
""".strip()
