from atomqa.client import call_gpt
import json
from atomqa.prompts.classification import ENTITY_CLASSIFICATION_SYS_PROMPT, ENTITY_CLASSIFICATION_PROMPT, ENTITY_CLASSIFICATION_FIX_PROMPT
from atomqa.llm import get_gpt_output
import logging

POST_CHECK = True

from .leaves import expand_tree
from atomqa.io import read_json, write_json

def classify_entity(entity_data_path, cat_tree, ckpt_step=1000):
    with open(entity_data_path, 'r', encoding='utf-8') as f:
        entity_data = json.load(f)

    ind = 0
    desired_outpot_dict = {
        "reason": "Due to ..., the entity is classified as ...",
        "category": "<category_name>"
    }
    leaves = expand_tree({}, cat_tree)

    for entity_name, entity_info in entity_data.items():
        logging.info(f"Processing entity: ({ind}/{len(entity_data)-1})")
        if "category" in entity_info:
            if POST_CHECK and (entity_info["category"] is None or entity_info["category"] not in leaves):
                logging.warning(f"Entity {entity_name} has no category, reclassifying.")
            else:
                logging.info(f"Entity {entity_name} already classified as {entity_info['category']}, skipping.")
                ind += 1
                continue
        logging.info(f"Processing entity: {entity_name} ({ind}/{len(entity_data)-1})")
        root = next(iter(cat_tree))
        path = [root]
        children = cat_tree[root].get("children", {})
        entity_str = "\n".join([f"-{it['prompt']} {it['attribute']}" for it in entity_info["items"]])
        logging.info(f"Current root: {root}, children: {list(children.keys())}")

        while children:
            children_str = "\n".join([f"- {child_n}: {c_it['definition']}" for child_n, c_it in children.items()])

            user_prompt = ENTITY_CLASSIFICATION_PROMPT.format(
                possible_categories=children_str,
                entity_name=entity_name,
                entity_info=entity_str
            )

            sys_prompt = ENTITY_CLASSIFICATION_SYS_PROMPT
            data = get_gpt_output(sys_prompt=sys_prompt, user_prompt=user_prompt, error_fix_prompt=ENTITY_CLASSIFICATION_FIX_PROMPT, desired_outpot_dict=desired_outpot_dict)

            if data is None:
                logging.error("GPT returned None, skipping this entity.")
                entity_data[entity_name]["category"] = None
                raise RuntimeError(f"Could not classify {entity_name}")

            root = data["category"]
            if root not in children:
                raise ValueError(f"Model selected an unknown category: {root}")
            path.append(root)
            children = children[root].get("children", {})
            logging.info(f"Current root: {root}, children: {list(children.keys())}")

            entity_data[entity_name]["category"] = "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path)
            entity_data[entity_name]["category_path"] = path.copy()
        if ind % ckpt_step == 0:
            with open(entity_data_path.replace(".json", f"_ckpt{ind}.json"), 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Checkpoint saved at index {ind}.")
        logging.info(f"Entity {entity_name} classified as {entity_data[entity_name]['category']}.")
        ind += 1

    return entity_data

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Classify seed entities down the taxonomy")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--entities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    # Checkpoint beside the output, without modifying the input file.
    write_json(args.output, read_json(args.entities))
    result = classify_entity(args.output, read_json(args.tree), args.checkpoint_every)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
