"""Shared JSON handling; reject unavailable Git LFS payloads explicitly."""

import json
from pathlib import Path


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        if handle.readline().startswith("version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer, not a downloaded dataset")
        handle.seek(0)
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
