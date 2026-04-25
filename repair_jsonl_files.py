import json
from pathlib import Path


def recover_json_objects(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i = 0
    objs = []

    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break

        # Previous script mistakenly wrote literal "\n" between objects.
        if i + 1 < len(text) and text[i] == "\\" and text[i + 1] == "n":
            i += 2
            continue

        obj, j = decoder.raw_decode(text, i)
        objs.append(obj)
        i = j

    fixed = "".join(json.dumps(obj, ensure_ascii=True) + "\n" for obj in objs)
    path.write_text(fixed, encoding="utf-8")
    return len(objs)


def main() -> None:
    for name in ("grid_expert_sft_train.jsonl", "grid_expert_sft_val.jsonl"):
        p = Path(name)
        if p.exists():
            n = recover_json_objects(p)
            print(f"{name}: recovered {n} JSON rows")
        else:
            print(f"{name}: file not found")


if __name__ == "__main__":
    main()
