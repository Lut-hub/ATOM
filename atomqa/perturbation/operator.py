"""Corrupt atomic QA outputs with the shared Perturber and Verification prompts."""

import argparse
from copy import deepcopy
import json

from atomqa.prompts.atomic import (
    PERTURBER_PROMPT, VERIFICATION_PROMPT, PERTURBER_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT, PERTURBER_REPAIR_PROMPT, VERIFICATION_REPAIR_PROMPT,
)
from atomqa.llm import get_gpt_output
from atomqa.io import read_json, write_json


def perturb_item(item, attempts=3):
    if (not item.get("operand") or not item.get("operator")
            or item.get("output") != item.get("attribute")
            or item.get("verification", {}).get("judgement") != "pass"):
        raise ValueError("Operator perturbation requires a verified atomic QA record")
    for _ in range(attempts):
        prompt = PERTURBER_PROMPT.format(
            input_sample=item["prompt"], output_sample=item["attribute"],
            triple=json.dumps({"operand": item["operand"], "operator": item["operator"],
                               "output": item["attribute"]}, ensure_ascii=False),
            payload=item["attribute"],
            example='{"output": "<incorrect output of this operator>", "reason": "<edit explanation>"}')
        candidate = get_gpt_output(
            PERTURBER_SYSTEM_PROMPT, prompt,
            PERTURBER_REPAIR_PROMPT,
            {"output": "", "reason": ""})
        if not candidate or not isinstance(candidate["output"], str):
            continue
        answer = candidate["output"].strip()
        if not answer or answer == item["attribute"]:
            continue
        judgment = get_gpt_output(
            VERIFICATION_SYSTEM_PROMPT,
            VERIFICATION_PROMPT.format(
                function=f"{item['operator']}: {item.get('operator_definition', '')}",
                operand=item["operand"], output=answer),
            VERIFICATION_REPAIR_PROMPT,
            {"judgement": "", "reason": ""})
        if judgment and judgment.get("judgement") == "fail":
            result = deepcopy(item)
            result["source_verification"] = result.pop("verification")
            result.update({"attribute": answer, "output": answer,
                           "perturbation": "operator_output", "corruption_verification": judgment,
                           "perturbation_reason": candidate["reason"]})
            return result
    raise RuntimeError("No verified incorrect output was obtained; original QA was not relabeled as corrupted")


def main():
    parser = argparse.ArgumentParser(description="Rewrite and verify incorrect atomic outputs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    data = read_json(args.input)
    result = {}
    for name, entity in data.items():
        revised = deepcopy(entity)
        revised["items"] = [perturb_item(item, args.attempts) for item in entity["items"]]
        result[name] = revised
        write_json(args.output + ".checkpoint.json", result)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
