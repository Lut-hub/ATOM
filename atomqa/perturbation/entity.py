"""Apply the paper's consistent, equal-length relabeling of atomic x/y values."""

import argparse
from copy import deepcopy
import random
import re
import string

from atomqa.io import read_json, write_json


def generate_random_string(length: int, preserve_case_pattern: str = None) -> str:
    """
    Generate a random string with the same case pattern as the original entity.

    Args:
        length: Length of the random string to generate
        preserve_case_pattern: Original string to preserve case pattern from

    Returns:
        Random string with preserved case pattern
    """
    # Generate base random string
    chars = string.ascii_letters
    random_str = ''.join(random.choice(chars) for _ in range(length))

    if preserve_case_pattern is None:
        return random_str

    # Apply case pattern from original string
    result = []
    for i, char in enumerate(random_str):
        if i < len(preserve_case_pattern):
            if preserve_case_pattern[i].isupper():
                result.append(char.upper())
            else:
                result.append(char.lower())
        else:
            # For extra characters, keep them lowercase
            result.append(char.lower())

    return ''.join(result)


def process_entity_data(input_file: str, output_file: str):
    """Relabel verified operands and outputs with one shared random mapping."""
    data = read_json(input_file)
    values = list(data)
    for subject, entity in data.items():
        for item in entity["items"]:
            if (item.get("operand") != subject or not item.get("operator")
                    or item.get("output") != item.get("attribute")
                    or item.get("verification", {}).get("judgement") != "pass"):
                raise ValueError("Operand perturbation requires verified atomic records")
            if not re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", item["prompt"]):
                raise ValueError(f"Operand {subject!r} must occur explicitly in its question")
            values.append(item["attribute"])
    values = list(dict.fromkeys(values))

    entity_mapping = {}
    used_random_strings = set()
    for original_entity in values:
        entity_length = len(original_entity)
        for attempt in range(10000):
            random_entity = generate_random_string(entity_length, original_entity)
            if random_entity not in used_random_strings and random_entity != original_entity:
                used_random_strings.add(random_entity)
                entity_mapping[original_entity] = random_entity
                break
        else:
            raise ValueError(f"Cannot create a unique replacement of length {entity_length}")

    new_data = {}
    total_items = 0
    for original_entity, entity_data in data.items():
        replacement_entity = entity_mapping[original_entity]
        new_entity_data = deepcopy(entity_data)
        new_entity_data["subject"] = replacement_entity
        new_entity_data["items"] = []
        for item in entity_data["items"]:
            new_item = deepcopy(item)
            replacements = {original_entity: replacement_entity,
                            item["attribute"]: entity_mapping[item["attribute"]]}
            pattern = re.compile(r"(?<!\w)(?:" + "|".join(
                re.escape(value) for value in sorted(replacements, key=len, reverse=True)) + r")(?!\w)")
            new_item["prompt"] = pattern.sub(lambda match: replacements[match.group()], item["prompt"])
            new_item["attribute"] = entity_mapping[item["attribute"]]
            new_item["operand"] = replacement_entity
            new_item["output"] = new_item["attribute"]
            new_item["template"] = new_item["prompt"].replace(replacement_entity, "{}")
            new_item["source_verification"] = new_item.pop("verification")
            new_item["perturbation"] = "operand_relabeling"
            new_entity_data["items"].append(new_item)
            total_items += 1
        new_data[replacement_entity] = new_entity_data

    write_json(output_file + ".mapping.json", entity_mapping)
    write_json(output_file, new_data)
    print(f"Relabeled {total_items} atomic QA pairs with {len(entity_mapping)} consistent value replacements")


def main():
    parser = argparse.ArgumentParser(description="Relabel atomic operands and outputs with equal-length strings")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    process_entity_data(args.input, args.output)


if __name__ == "__main__":
    main()
