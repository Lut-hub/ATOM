import argparse
import json
import logging
import os
import re

from atomqa.client import call_gpt
from atomqa.io import read_json, write_json
from atomqa.llm import extract_json, extract_solution, get_gpt_output
from atomqa.prompts.atomic import (
    OPERAND_GENERATION_PROMPT, OUTPUT_GENERATION_PROMPT, VERIFICATION_PROMPT,
    OPERAND_GENERATION_SYSTEM_PROMPT, OUTPUT_GENERATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT, OPERAND_REPAIR_PROMPT, OUTPUT_REPAIR_PROMPT,
    VERIFICATION_REPAIR_PROMPT,
)


def gen_entity_name(category, definition, existing_entities, operator, operator_definition):
    """
    Generate a new entity name based on the category and existing entities.
    """
    existing_entities_str = "\n".join(existing_entities) if existing_entities else "No existing entities, you should generate based on the category and definition."
    if not operator or not operator_definition:
        raise ValueError("An explicit operator and operator definition are required")
    template = OPERAND_GENERATION_PROMPT
    user_prompt = template.format(category=category, definition=definition, existing_entities=existing_entities_str,
                                  operator=operator, operator_definition=operator_definition)
    sys_prompt = OPERAND_GENERATION_SYSTEM_PROMPT
    error_fix_prompt = OPERAND_REPAIR_PROMPT
    all_design_traj = []
    all_design_traj.append({"role": "system", "content": sys_prompt})
    all_design_traj.append({"role": "user", "content": user_prompt})
    logging.info(f"Calling GPT with prompt: \n{user_prompt}")

    max_it = 10
    iteration = 0
    test_json = None
    while iteration < max_it:
        response = call_gpt(all_design_traj, model="o3-mini")
        if response is None or len(response) == 0:
            logging.error("GPT response is None or empty.")
            iteration += 1
            continue
        response_text = response[0]
        if response_text is None:
            logging.error("GPT response text is None.")
            iteration += 1
            continue
        all_design_traj.append({"role": "assistant", "content": response_text})
        logging.info(f"GPT response: {response_text}")
        try:
            test_code = extract_json(response_text.strip())
            if len(test_code) == 0:
                logging.error("The extracted code is empty.")
                reason = "The output is not wrapped in a proper Json code block using triple backticks."
                feedback = error_fix_prompt.format(error_reason=reason)
                all_design_traj.append({"role": "user", "content": feedback})
                iteration += 1
                continue
            else:
                test_json = json.loads(test_code[0].replace("\n", "").replace("\t", ""))
        except Exception as e:
            logging.error(f"Error extracting code from model output: {e}")
            reason = f"Error extracting json from output: {e}"
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue

        if not isinstance(test_json, dict) or len(test_json) == 0:
            logging.error("The extracted json is None.")
            reason = "The extracted json is None."
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue
        if not isinstance(test_json, dict):
            logging.error("The extracted json is not a dictionary.")
            reason = "The extracted json is not a dictionary."
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue
        else:
            if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) for k, v in test_json.items()):
                iteration += 1
                continue
            logging.info(f"Generated entities: {test_json}")
            return test_json

    return None

def gen_entity_information(category, definition, entity_name, entity_desc, operator, operator_definition):
    """
    Generate entity information based on the entity name and its information.
    """
    if not operator or not operator_definition:
        raise ValueError("An explicit operator and operator definition are required")
    template = OUTPUT_GENERATION_PROMPT
    user_prompt = template.format(category=category, definition=definition, entity_name=entity_name,
                                  entity_description=entity_desc, operator=operator, operator_definition=operator_definition)
    sys_prompt = OUTPUT_GENERATION_SYSTEM_PROMPT
    error_fix_prompt = OUTPUT_REPAIR_PROMPT
    all_design_traj = []
    all_design_traj.append({"role": "system", "content": sys_prompt})
    all_design_traj.append({"role": "user", "content": user_prompt})
    logging.info(f"Calling GPT with prompt: \n{user_prompt}")

    max_it = 10
    iteration = 0
    test_json = None
    while iteration < max_it:
        response = call_gpt(all_design_traj, model="o3-mini")
        if response is None or len(response) == 0:
            logging.error("GPT response is None or empty.")
            iteration += 1
            continue
        response_text = response[0]
        if response_text is None:
            logging.error("GPT response text is None.")
            iteration += 1
            continue
        all_design_traj.append({"role": "assistant", "content": response_text})
        logging.info(f"GPT response: {response_text}")

        test_answer = extract_solution(response_text.strip())
        if len(test_answer) == 0:
            logging.error("The extracted solution is empty.")
            reason = "The output is not wrapped in a proper <information>...</information> block."
            feedback = error_fix_prompt.format(error_reason=reason, entity_name=entity_name)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue

        test_answer = test_answer[0]
        # - Question: What is the XXX of {entity_name}? - Answer: AAA
        pattern = r'^\s*- Question: (.+?) - Answer: (.+?)\s*$'
        matches = re.findall(pattern, test_answer, re.MULTILINE)
        if len(matches) != 1 or entity_name not in matches[0][0]:
            logging.error("No valid question-answer pairs found in the response.")
            reason = "No valid question-answer pairs found in the response."
            feedback = error_fix_prompt.format(error_reason=reason, entity_name=entity_name)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue
        else:
            test_json = [{"prompt": q.strip(), "template": q.strip().replace(entity_name, "{}"), "attribute": a.strip()} for (q, a) in matches]
            break

    return test_json

def gen_entity(entity_data_path, leave_data_path, max_entites_per_leave=25, ckpt_step=1000, max_rounds=10):
    if os.path.exists(entity_data_path):
        with open(entity_data_path, 'r', encoding='utf-8') as f:
            entity_data = json.load(f)
    else:
        entity_data = {}

    with open(leave_data_path, 'r', encoding='utf-8') as f:
        leaves = json.load(f)

    missing = [key for key, leaf in leaves.items() if not leaf.get("operator") or not leaf.get("operator_definition")]
    if missing:
        raise ValueError(f"Generation requires explicit operator bindings for every leaf; missing {len(missing)}. "
                         "Supply --operators when extracting leaves.")
    for name, entity in entity_data.items():
        for item in entity["items"]:
            if (item.get("operand") != name or not item.get("operator")
                    or item.get("output") != item.get("attribute")
                    or item.get("verification", {}).get("judgement") != "pass"):
                raise ValueError(f"Generation cannot carry unverified input records for {name!r}")
    for leaf in leaves.values():
        for name in leaf["entities"]:
            matching = [item for item in entity_data.get(name, {}).get("items", [])
                        if item.get("operator") == leaf["operator"]
                        and item.get("verification", {}).get("judgement") == "pass"]
            if not matching:
                raise ValueError(f"Unverified seed/checkpoint operand {name!r}")
    logging.info(f"Total leaves: {len(leaves)}")

    ind = 0

    for round_index in range(max_rounds):
        max_count = 0
        for category, data in leaves.items():
            if len(data["entities"]) >= max_entites_per_leave:
                logging.info(f"Category {category} has {len(data['entities'])} entities, which is more than the limit of {max_entites_per_leave}.")
                # ind += 1
                max_count += 1
                continue
            logging.info(f"Processing category {category} with {len(data['entities'])}/{max_entites_per_leave} entities.")
            existing_entities = data["entities"]
            definition = data["definition"]
            operator = data["operator"]
            operator_definition = data["operator_definition"]
            category_name = data.get("name", category)
            new_entities = gen_entity_name(category_name, definition, existing_entities, operator, operator_definition)

            success_entity = []
            if new_entities is None or len(new_entities) == 0:
                logging.error(f"Failed to generate new entities for category {category}.")
                ind += 1
                continue
            for entity_name, entity_desc in new_entities.items():
                if len(data["entities"]) + len(success_entity) >= max_entites_per_leave:
                    break
                if entity_name in existing_entities:
                    continue
                items = gen_entity_information(category_name, definition, entity_name, entity_desc, operator, operator_definition)
                if items is None or len(items) == 0:
                    logging.error(f"Failed to generate information for entity {entity_name} in category {category}.")
                    continue
                verified = []
                for item in items:
                    judgment = get_gpt_output(
                        VERIFICATION_SYSTEM_PROMPT,
                        VERIFICATION_PROMPT.format(function=f"{operator}: {operator_definition}",
                                                   operand=entity_name, output=item["attribute"]),
                        VERIFICATION_REPAIR_PROMPT,
                        {"judgement": "", "reason": ""})
                    if judgment and judgment.get("judgement") == "pass":
                        item.update({"operand": entity_name, "operator": operator,
                                     "operator_definition": operator_definition,
                                     "output": item["attribute"], "verification": judgment})
                        verified.append(item)
                items = verified
                if not items:
                    continue
                success_entity.append(entity_name)
                if entity_name in entity_data:
                    entity_data[entity_name]["items"].extend(items)
                else:
                    entity_info = {"subject": entity_name, "items": items, "category": category}
                    entity_data[entity_name] = entity_info
            data["entities"].extend(success_entity)
            logging.info(f"Generated {len(success_entity)} new entities for category {category}.")

            if ind % ckpt_step == 0:
                with open(entity_data_path.replace(".json", f"_ckpt_{ind}.json"), "w", encoding='utf-8') as f:
                    json.dump(entity_data, f, ensure_ascii=False, indent=4)
                logging.info(f"Checkpoint saved at {entity_data_path.replace('.json', f'_ckpt_{ind}.json')}")

                with open(leave_data_path.replace(".json", f"_ckpt_{ind}.json"), "w", encoding='utf-8') as f:
                    json.dump(leaves, f, ensure_ascii=False, indent=4)
                logging.info(f"Leaves checkpoint saved at {leave_data_path.replace('.json', f'_ckpt_{ind}.json')}")
            ind += 1
        logging.info(f"Now leaves with enough entities: {max_count}, total leaves: {len(leaves)}")
        write_json(entity_data_path, entity_data)
        write_json(leave_data_path, leaves)
        if all(len(leaf["entities"]) >= max_entites_per_leave for leaf in leaves.values()):
            logging.info("No more leaves to process, exiting.")
            break

    if any(len(leaf["entities"]) < max_entites_per_leave for leaf in leaves.values()):
        raise RuntimeError("Generation reached --max-rounds before filling every leaf; resume from the output files")
    return entity_data

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Generate and verify Atomic-QA for explicit operators")
    parser.add_argument("--entities", help="Verified atomic checkpoint dictionary for resuming generation")
    parser.add_argument("--leaves", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-entities", type=int, default=25)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    if min(args.max_entities, args.max_rounds, args.checkpoint_every) < 1:
        parser.error("Generation limits must be positive")
    output = Path(args.output_dir)
    entities = read_json(args.entities) if args.entities else {}
    leaves = read_json(args.leaves)
    referenced = {name for leaf in leaves.values() for name in leaf["entities"]}
    if referenced - entities.keys():
        parser.error("Leaf entity lists refer to missing entities; supply the matching --entities file")
    write_json(output / "entities.json", entities)
    write_json(output / "leaves.json", leaves)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gen_entity(str(output / "entities.json"), str(output / "leaves.json"),
               args.max_entities, args.checkpoint_every, args.max_rounds)
    write_json(output / "run.json", {"model": os.getenv("ATOM_MODEL", "o3-mini"),
                                    "max_entities": args.max_entities, "max_rounds": args.max_rounds})


if __name__ == "__main__":
    main()
