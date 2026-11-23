import hashlib
import json
import math
from pathlib import Path
import re


BUDGETS = {"place_bottle": 400, "shake_bottle": 700, "place_fan": 400, "turn_switch": 400,
           "press_buttons": 600, "place_via_divider": 400, "open_laptop": 700, "stack_blocks": 1200}
DOMAINS = ("val", "test", "test_random")
RELEASE_DIGESTS = {
    "test-cases.jsonl": "9a88101906acce3ecc55011121bd012e85047ba4a9b75f9e7a670309d7d0951b",
    "test-scenes.jsonl": "597f6f1bcabe7630cf3bf767c0ddeee44513ab64dc4f527ef0772941e135421c",
    "val-cases.jsonl": "520b60edcb73abb623e865b1324b32d4d9a9b63265097394ef55c21f10edb9a1",
    "val-scenes.jsonl": "1f18aa23b15b0a9180d9bc0adf0b4cf4a94a9173dfb66a4eb240cd7dbdff0504",
}


def _unique_pairs(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON field")
    return result


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                yield json.loads(line, object_pairs_hook=_unique_pairs)
            except ValueError as exc:
                raise ValueError(f"{path}, line {line_no}: {exc}") from exc


def validate_appearance(value):
    if not isinstance(value, dict) or set(value) != {
            "appearance_seed", "table_texture", "wall_texture", "ambient_light", "point_lights"}:
        raise ValueError("Invalid appearance fields")
    seed = value["appearance_seed"]
    if not isinstance(seed, str) or not re.fullmatch(r"(0|[1-9][0-9]{0,19})", seed) or int(seed) >= 2**64:
        raise ValueError("Appearance seed must be uint64 in decimal string form")
    if any(not isinstance(value[key], str) or not re.fullmatch(r"unseen/[0-9]+", value[key])
           for key in ("table_texture", "wall_texture")):
        raise ValueError("Invalid texture ID")

    def vector(v):
        return isinstance(v, list) and len(v) == 3 and all(type(n) in (int, float) and math.isfinite(n) for n in v)

    lights = value["point_lights"]
    if (not vector(value["ambient_light"]) or min(value["ambient_light"]) < 0 or not isinstance(lights, list)
            or len(lights) != 2 or any(not isinstance(light, list) or len(light) != 2
            or not all(vector(v) for v in light) or min(light[1]) < 0 for light in lights)):
        raise ValueError("Invalid lighting parameters")


def load_suite(root, domain, *, verify_release=False):
    if domain not in DOMAINS:
        raise ValueError("Unknown evaluation domain")
    split = "val" if domain == "val" else "test"
    directory = Path(root) / "evaluation"
    if verify_release:
        for suffix in ("cases", "scenes"):
            name = f"{split}-{suffix}.jsonl"
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != RELEASE_DIGESTS[name]:
                raise ValueError(f"{name} does not match the RoboInF-4K v1.0.0 evaluation files")
    scenes, cases = {}, {}
    fields = {"task", "scene_id", "scene_seed"}
    if split == "test":
        fields.add("test_random_appearance")
    for row in read_jsonl(directory / f"{split}-scenes.jsonl"):
        if (not isinstance(row, dict) or set(row) != fields or row["task"] not in BUDGETS
                or type(row["scene_id"]) is not int or row["scene_id"] < 0
                or type(row["scene_seed"]) is not int or not 0 <= row["scene_seed"] < 2**32):
            raise ValueError("Invalid public scene record")
        key = row["task"], row["scene_id"]
        if key in scenes:
            raise ValueError("Duplicate scene")
        if split == "test":
            validate_appearance(row["test_random_appearance"])
        scenes[key] = row
    for row in read_jsonl(directory / f"{split}-cases.jsonl"):
        if (not isinstance(row, dict) or set(row) != {"case_id", "instruction", "task", "scene_id"}
                or not isinstance(row["case_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", row["case_id"])
                or not isinstance(row["instruction"], str) or not row["instruction"].strip()
                or type(row["scene_id"]) is not int or (row["task"], row["scene_id"]) not in scenes
                or row["case_id"] in cases):
            raise ValueError("Invalid, duplicate or unpaired public case")
        cases[row["case_id"]] = row
    if not cases or {(row["task"], row["scene_id"]) for row in cases.values()} != scenes.keys():
        raise ValueError("Expected at least one case for each scene")
    return scenes, cases
