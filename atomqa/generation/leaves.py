import logging
import json

def get_subtree(cat_tree: dict, path: list):
    """
    Get the subtree from the category tree based on the given path.
    """
    subtree = cat_tree[path[0]]["children"]
    for p in path[1:-1]:
        subtree = subtree[p]["children"]
    subtree = subtree[path[-1]] if len(path) > 1 else cat_tree[path[0]]
    return list(subtree["children"].keys()), subtree["definition"]


def expand_tree(entity_data, cat_tree):
    """Flatten the original taxonomy without merging equal names on different paths."""
    leaves = {}

    def visit(nodes, path):
        for name, node in nodes.items():
            current = path + [name]
            if node.get("children"):
                visit(node["children"], current)
            else:
                key = "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in current)
                leaves[key] = {"name": name, "path": current,
                               "definition": node["definition"], "entities": [], "siblings": []}
                for field in ("operator", "operator_definition"):
                    if field in node:
                        leaves[key][field] = node[field]
    visit(cat_tree, [])
    sibling_groups = {}
    by_name = {}
    for key, leaf in leaves.items():
        sibling_groups.setdefault(tuple(leaf["path"][:-1]), []).append(key)
        by_name.setdefault(leaf["name"], []).append(key)
    for keys in sibling_groups.values():
        for key in keys:
            leaves[key]["siblings"] = [other for other in keys if other != key]
    for name, entity in entity_data.items():
        category = entity.get("category")
        if category in leaves:
            key = category
        elif len(by_name.get(category, [])) == 1:
            key = by_name[category][0]
        else:
            raise ValueError(f"Entity {name!r} needs an unambiguous leaf category; run classify first")
        leaves[key]["entities"].append(name)
    return leaves


def main():
    import argparse
    from atomqa.io import read_json, write_json
    parser = argparse.ArgumentParser(description="Extract and split taxonomy leaves")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--entities", help="Classified seed entities; optional for an empty tree")
    parser.add_argument("--operators", help="JSON mapping full leaf paths to operator and operator_definition")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args()
    if args.splits < 1:
        parser.error("--splits must be positive")
    entities = read_json(args.entities) if args.entities else {}
    leaves = expand_tree(entities, read_json(args.tree))
    if args.operators:
        for key, operator in read_json(args.operators).items():
            if key not in leaves or any(not isinstance(operator.get(f), str) or not operator[f].strip()
                                        for f in ("operator", "operator_definition")):
                raise ValueError(f"Invalid operator binding: {key}")
            leaves[key].update({f: operator[f] for f in ("operator", "operator_definition")})
    from pathlib import Path
    keys = list(leaves)
    batch_size = (len(keys) + args.splits - 1) // args.splits
    for i in range(args.splits):
        part = {key: leaves[key] for key in keys[i * batch_size:(i + 1) * batch_size]}
        write_json(Path(args.output_dir) / f"leaves_{i}.json", part)
    print(f"Wrote {len(leaves)} leaves in {args.splits} splits")


if __name__ == "__main__":
    main()
