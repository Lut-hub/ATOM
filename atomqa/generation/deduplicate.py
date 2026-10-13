from atomqa.prompts.deduplication import (PRE_DEDUPLICATE_SYS_PROMPT, PRE_DEDUPLICATE_PROMPT, PRE_DEDUPLICATE_FIX_PROMPT,
                        DEDUPLICATE_SYS_PROMPT, DEDUPLICATE_PROMPT, DEDUPLICATE_FIX_PROMPT)
from atomqa.client import call_gpt
from atomqa.llm import get_gpt_output, extract_json
import logging
import json
from copy import deepcopy
from atomqa.io import read_json, write_json

def pre_dedup_entity(entities: list):
    user_prompt = PRE_DEDUPLICATE_PROMPT.format(entities="\n".join(entities))
    sys_prompt = PRE_DEDUPLICATE_SYS_PROMPT
    error_fix_prompt = PRE_DEDUPLICATE_FIX_PROMPT
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

        if test_json is None:
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
        if any([not isinstance(v, list) for v in test_json.values()]):
            logging.error("The extracted json values are not lists.")
            reason = "The extracted json values are not lists."
            feedback = error_fix_prompt.format(error_reason=reason)
            all_design_traj.append({"role": "user", "content": feedback})
            iteration += 1
            continue
        else:
            logging.info(f"Generated entities: {test_json}")
            return test_json

    return None


def dedup_entity(entity_data_path, leave_data_path, ckpt_step=1000):
    with open(entity_data_path, 'r', encoding='utf-8') as f:
        entity_data: dict = json.load(f)

    with open(leave_data_path, 'r', encoding='utf-8') as f:
        leave_data: dict = json.load(f)

    template_dic = {"judgement": "same", "reason": "Due to ..., the entities are classified as same or different."}
    ind_step = 0
    for category, cat_info in leave_data.items():
        logging.info(f"Processing category: {category}, step: {ind_step}")
        visited = cat_info.get("visited", False)
        if visited:
            ind_step += 1
            logging.info(f"Category {category} already visited, skipping.")
            continue
        siblings = cat_info.get('siblings', [])
        all_entities = [deepcopy(cat_info['entities'])]

        for sib in siblings:
            if sib not in leave_data:
                logging.warning(f"Sibling {sib} not found in entity data, skipping.")
                continue
            all_entities.append(deepcopy(leave_data[sib]['entities']))

        first = True

        while first or len(all_entities) > 1:
            log_str = "\n".join([f"{ent}" for ent in all_entities])
            logging.info(f"Current entities to process:\n {log_str}")
            first = False
            entity_set: list = deepcopy(all_entities.pop(0))
            entity_set += deepcopy(all_entities.pop(0)) if len(all_entities) > 0 else []
            entity_set = list(dict.fromkeys(entity_set))
            if len(entity_set) < 2:
                logging.info(f"Not enough entities to deduplicate for category {category}, skipping.")
                continue

            logging.info(f"Processing category {category} with entities: {entity_set}")
            cluster_data = pre_dedup_entity(entity_set)
            if cluster_data is None:
                raise RuntimeError(f"Deduplication failed for {category}; category remains unvisited")
            if len(cluster_data) == 0:
                logging.error(f"Cluster data is None or empty for category {category}.")
                all_entities.append(entity_set)
                continue

            for cluster_name, cluster_entities in cluster_data.items():
                entities_info = ""

                need_dep = []
                for entity in cluster_entities:
                    if entity not in entity_data:
                        logging.warning(f"Entity {entity} not found in entity data, skipping.")
                        continue
                    if entity in need_dep:
                        logging.info(f"Entity {entity} already in need_dep, skipping.")
                        continue
                    if entity not in entity_set:
                        logging.info(f"Entity {entity} not in current entity set, skipping.")
                        continue
                    need_dep.append(entity)
                    entity_info = entity_data[entity]
                    entities_info += f"-Name: {entity}\n" + "".join([f"\n  -{item['prompt']} {item['attribute']}" for item in entity_info["items"]]) + "\n"

                if len(need_dep) < 2:
                    logging.info(f"Not enough entities to deduplicate for cluster {cluster_name}, skipping.")
                    continue

                user_prompt = DEDUPLICATE_PROMPT.format(entities_info=entities_info)
                sys_prompt = DEDUPLICATE_SYS_PROMPT
                error_fix_prompt = DEDUPLICATE_FIX_PROMPT
                judge_data = get_gpt_output(sys_prompt=sys_prompt, user_prompt=user_prompt, error_fix_prompt=error_fix_prompt, desired_outpot_dict=template_dic)
                if judge_data is None or len(judge_data) == 0:
                    raise RuntimeError(f"Could not verify duplicate cluster {cluster_name}")
                if judge_data.get("judgement") not in ("same", "different"):
                    raise ValueError("Duplicate judgement must be same or different")

                if judge_data.get("judgement", "") == "same":
                    base_entity = need_dep[0]
                    existing_template = [item["template"] for item in entity_data[base_entity]["items"]]
                    entity_data[base_entity].setdefault("aliases", [])
                    b_len = len(entity_set)
                    for entity in need_dep[1:]:
                        for item in entity_data[entity]["items"]:
                            if item["template"] not in existing_template:
                                if "operator" in item:
                                    item = deepcopy(item)
                                    item["prompt"] = item["prompt"].replace(entity, base_entity)
                                    item["operand"] = base_entity
                                entity_data[base_entity]["items"].append(item)
                                existing_template.append(item["template"])
                        entity_data[base_entity]["aliases"].append(entity)
                        cat = entity_data[entity]["category"]
                        entity_data.pop(entity, None)
                        for leaf in leave_data.values():
                            leaf["entities"] = list(dict.fromkeys(
                                base_entity if name == entity else name for name in leaf["entities"]))
                        entity_set.remove(entity)
                    logging.info(f"Entity {entity} merged into {base_entity}. Length of entity_set: {len(entity_set)}/{b_len}")
                    logging.info(f"Entities {need_dep} are classified as same, merged into {base_entity}.")
                else:
                    logging.info(f"Entities {need_dep} are classified as different, no action taken.")
                    continue

            all_entities.append(entity_set)

        cat_info["visited"] = True
        for sib in siblings:
            if sib in leave_data:
                leave_data[sib]["visited"] = True
        if ind_step % ckpt_step == 0:
            logging.info(f"Checkpoint at step {ind_step}, saving entity data.")
            with open(entity_data_path.replace(".json", f"_ckpt_{ind_step}.json"), 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, ensure_ascii=False, indent=4)
            with open(leave_data_path.replace(".json", f"_ckpt_{ind_step}.json"), 'w', encoding='utf-8') as f:
                json.dump(leave_data, f, ensure_ascii=False, indent=4)
        ind_step += 1
    return entity_data, leave_data


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Deduplicate entities across sibling leaves")
    parser.add_argument("--entities", required=True)
    parser.add_argument("--leaves", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    output = Path(args.output_dir)
    write_json(output / "entities.json", read_json(args.entities))
    write_json(output / "leaves.json", read_json(args.leaves))
    entities, leaves = dedup_entity(str(output / "entities.json"), str(output / "leaves.json"), args.checkpoint_every)
    write_json(output / "entities.json", entities)
    write_json(output / "leaves.json", leaves)


if __name__ == "__main__":
    main()
