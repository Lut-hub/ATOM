"""Merge and shuffle Atomic-QA records for export."""
import random

from atomqa.io import read_json, write_json


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge and shuffle Atomic-QA records")
    parser.add_argument("--entities", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["raw", "alpaca"], default="raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    combined_data = []
    for path in args.entities:
        all_data = read_json(path)
        if isinstance(all_data, dict):
            for value in all_data.values():
                combined_data.extend(value["items"])
        elif isinstance(all_data, list):
            combined_data.extend(all_data)
        else:
            raise ValueError("Input must be an entity dictionary or a flat QA list")
    random.Random(args.seed).shuffle(combined_data)
    if args.format == "alpaca":
        combined_data = [{"instruction": row["prompt"], "input": "", "output": row["attribute"]}
                         for row in combined_data]
    write_json(args.output, combined_data)
    print(f"Exported {len(combined_data)} QA pairs")


if __name__ == "__main__":
    main()
