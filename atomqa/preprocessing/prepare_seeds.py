"""Normalize FB15k, ZsRE, and wiki_recent seeds into subject-grouped QA data."""

import argparse
from atomqa.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Normalize knowledge-editing seeds by subject")
    parser.add_argument("--wiki-recent", nargs="*")
    parser.add_argument("--zsre")
    parser.add_argument("--fb15k")
    parser.add_argument("--wiki-recent-answer", choices=["target_new", "ground_truth"], default="target_new")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = []
    sources.extend((path, args.wiki_recent_answer) for path in (args.wiki_recent or []))
    if args.zsre:
        sources.append((args.zsre, "ground_truth"))
    if args.fb15k:
        sources.append((args.fb15k, "attribute"))
    if not sources:
        parser.error("At least one seed input is required")
    fn_data = []
    for path, answer_field in sources:
        for item in read_json(path):
            answer = item[answer_field]
            if path == args.zsre:
                answer = answer[0]
            if not isinstance(answer, str):
                raise ValueError(f"Expected a string answer in {path}: {answer_field}")
            fn_data.append({"prompt": item["prompt"], "subject": item["subject"],
                            "template": item.get("template", item["prompt"].replace(item["subject"], "{}")),
                            "attribute": answer})
    entity_data = {}
    for item in fn_data:
        subject = item["subject"]
        entity_data.setdefault(subject, {"subject": subject, "items": []})
        entity_data[subject]["items"].append({key: item[key] for key in ("prompt", "template", "attribute")})
    from pathlib import Path
    write_json(Path(args.output_dir) / "base_data.json", fn_data)
    write_json(Path(args.output_dir) / "entities.json", entity_data)
    print(f"Prepared {len(fn_data)} QA pairs for {len(entity_data)} entities")


if __name__ == "__main__":
    main()
