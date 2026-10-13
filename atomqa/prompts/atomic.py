"""Paper prompts with normalized Python placeholders. OUTPUT_GENERATION_PROMPT is an explicit integration prompt, not a verbatim appendix prompt."""

# Integration messages around the verbatim appendix templates below.
OPERAND_GENERATION_SYSTEM_PROMPT = "You generate valid operands for a specified atomic operator."
OUTPUT_GENERATION_SYSTEM_PROMPT = "You compute one atomic operation and express it as one QA pair."
VERIFICATION_SYSTEM_PROMPT = "You verify atomic operations."
PERTURBER_SYSTEM_PROMPT = "You perturb one atomic output while keeping its operand fixed."

OPERAND_REPAIR_PROMPT = """Invalid output: {error_reason}.
Return a non-empty JSON object mapping distinct valid operands for the given operator to their descriptions, wrapped in a json code block."""
OUTPUT_REPAIR_PROMPT = """Invalid output: {error_reason}.
Return exactly one QA for the given operator. Keep the operand {entity_name} explicitly in the question:
<information>
- Question: <question for the given operator and operand> - Answer: <concise operator output>
</information>"""
VERIFICATION_REPAIR_PROMPT = "Invalid output: {error_reason}. Return JSON with judgement (pass/fail) and reason."
PERTURBER_REPAIR_PROMPT = "Invalid output: {error_reason}. Return a JSON object with output and reason."

OPERAND_GENERATION_PROMPT = """
You are given the following information:
    - Category
    - Operator
    - Existing Operand

Your task is to generate new operands that are valid inputs for the given operator under the specified category.
Your generation should consider the following aspects:
1. The generated operands should be specific and fall under the given category.
2. The generated operands should be valid inputs for the given operator.
3. The generated operands should not be similar to the existing operands.
4. Each generated operand should have a brief description that help understand the operands.

## Category ##
This section contains the category that you need to generate operands for.
- Category Name: {category}
- Category Definition: {definition}

## Operator ##
- Operator: {operator}
- Operator Definition: {operator_definition}

## Existing operands ##
This section contains the existing operands that are already classified under the category.\x20
You should avoid generating operands that are similar to the existing ones.
- Existing operands:
{existing_entities}

## Output format ##
You should output the generated operands in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "operand_name1": "Description of operand 1",
    "operand_name2": "Description of operands 2",
    ...
}}
```
""".strip()

VERIFICATION_PROMPT = """
You are given the following information:
    - Function: {function}
    - Operand: {operand}
    - Output: {output}

Your task is to verify whether the function is logically consistent, that is, whether **Function(Operand)** equals the **Output**.

## Output format ##
You should output the judgement in the JSON format, wrapped in a proper Json code block using triple backticks like this:
```json
{{
    "judgement": "pass"/"fail",
    "reason": "reason for your judgement"
}}
```
""".strip()

OUTPUT_GENERATION_PROMPT = """
Given the atomic operator and operand below, compute its output and express the atomic operation as one concise question-answer pair.

Category: {category}
Category definition: {definition}
Operator: {operator}
Operator definition: {operator_definition}
Operand: {entity_name}
Operand description: {entity_description}

Apply only this operator. The question must explicitly contain the operand exactly as written. The answer must be the concise output, not an explanation. Do not introduce other operators.
Output one line inside the information tags:
<information>
- Question: <question containing the operand> - Answer: <operator output>
</information>
""".strip()

PERTURBER_PROMPT = """
You are the PERTURBER agent. Your job is to apply controlled edits to specific atomic functions in the Sample Output, guided by the provided (subject, relation, object) triple and payload location.

A *controlled edit* is a minimal, deterministic change that makes the function incorrect while preserving overall coherence and fluency.

Return **exactly one JSON object** describing the edit.

Use the Sample Input only to preserve coherence (e.g., pronouns); all edits **must** be grounded in the Sample Output and the given function.


## Sample Input
This section provides the original input of the data sample (e.g., question/instruction/code comment).
{input_sample}

## Sample Output
This section provides the original answer of the data sample (sentences, equations, or code).
{output_sample}

## Atomic function to be Perturbed (from ENCODER)
This section provides the function to be perturbed, along with its unique fid and the corresponding payload location in the Sample Output.
function: {triple}
Payload: {payload}

## Output Format
Your answer should wrapped in a proper Json code block using triple backticks like this:
## Output Format
Your answer should wrapped in a proper Json code block using triple backticks like this:
```json
{example}
```
""".strip()
