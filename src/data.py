"""Deterministic synthetic Chinese evidence QA data for post-training experiments.

The generator uses only standard-library randomness.  All records are fabricated
from a small symbolic world; they are not sampled from real documents.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 20261028
GENERATOR_VERSION = "synthetic-evidence-qa-v1"
TASK_TYPES = ("lookup", "two_hop", "unanswerable", "negation")
SPLIT_SIZES = {
    "train_sft": 512,
    "train_pref": 256,
    "validation": 96,
    "test_id": 160,
    "test_ood": 160,
    "test_counterfactual": 160,
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_ID_PREFIXES = ("青匣", "绒团", "铜雀", "白榆", "雾铃", "石榴", "银扣", "竹筒")
_OOD_PREFIXES = ("霜岛", "烛岛", "砾环", "汐屿", "暮钟", "珊瑚", "云礁", "星槎")
_COLORS = ("赤色", "青色", "琥珀色", "灰色", "紫色", "银白色")
_PEOPLE = ("灰鸥", "青鹿", "白泉", "铜蛙", "银鹭", "雾狐", "苔雀", "蓝獭")
_PLACES = ("北柜", "南架", "石室", "东箱", "云仓", "浅坑", "木龛", "蓝屋")
_OOD_PEOPLE = ("雾松", "珊羽", "岩狐", "霜貂", "瓷鹤", "霞獭", "苔鸮", "雾鲤")
_OOD_PLACES = ("远屿", "岚坞", "盐穴", "星廊", "落岩", "杉坞", "素堂", "雨庭")
_INSTRUCTION_ID = (
    '仅依据编号材料；只输出JSON，evidence给出全部必要材料编号，不列无关材料。例'
    '{"answer":"17","evidence":[2,4]}；不可答用'
    '{"answer":"未知","evidence":[]}。'
)
_INSTRUCTION_OOD = (
    '编号材料为唯一依据；只输出JSON，证据列全必要材料编号，不添无关项。例'
    '{"answer":"17","evidence":[2,4]}；不可答用'
    '{"answer":"未知","evidence":[]}。'
)


def _fact(subject: str, relation: str, value: Any, text: str) -> dict[str, Any]:
    return {"subject": subject, "relation": relation, "value": value, "text": text}


def _tag(prefixes: tuple[str, ...], serial: int, role: str = "") -> str:
    prefix = prefixes[(serial * 7 + len(role)) % len(prefixes)]
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = serial
    digits = "0" if value == 0 else ""
    while value:
        value, remainder = divmod(value, len(alphabet))
        digits = alphabet[remainder] + digits
    suffix = digits.rjust(3, "0")
    return f"{prefix}{role}{suffix}"


def _reindex_and_render(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for index, source in enumerate(facts, start=1):
        item = dict(source)
        item["index"] = index
        indexed.append(item)
    return indexed


def _question(task_type: str, query: dict[str, Any], style: str) -> str:
    subject = query.get("subject", query.get("item", ""))
    if style == "ood":
        templates = {
            "lookup": f"材料记录了{subject}呈现何种色泽？",
            "two_hop_relation": f"{subject}最终被放置在何处？",
            "two_hop_sum": "两包物品合计有多少枚？",
            "unanswerable": f"记录能够确定{subject}由谁保管吗？",
            "negation": f"谁保管{subject}？",
        }
    else:
        templates = {
            "lookup": f"{subject}登记的颜色是什么？",
            "two_hop_relation": f"根据记录，{subject}最后存在哪里？",
            "two_hop_sum": "两袋合计多少枚？",
            "unanswerable": f"{subject}由谁保管？",
            "negation": f"谁保管{subject}？",
        }
    kind = query.get("kind", task_type)
    if task_type == "two_hop":
        kind = "two_hop_sum" if query.get("mode") == "sum" else "two_hop_relation"
    return templates[kind]


def _make_row(
    *,
    row_id: str,
    split: str,
    task_type: str,
    facts: list[dict[str, Any]],
    query: dict[str, Any],
    style: str,
    rng: random.Random,
    pair_id: str | None = None,
    shuffle_facts: bool = True,
) -> dict[str, Any]:
    rendered_facts = copy.deepcopy(facts)
    if shuffle_facts:
        rng.shuffle(rendered_facts)
    indexed = _reindex_and_render(rendered_facts)
    instruction = _INSTRUCTION_OOD if style == "ood" else _INSTRUCTION_ID
    material_text = "\n".join(f'[{f["index"]}] {f["text"]}' for f in indexed)
    question = _question(task_type, query, style)
    prompt = f"{instruction}\n{material_text}\n{question}"
    latent = {"facts": indexed, "query": copy.deepcopy(query)}
    row: dict[str, Any] = {
        "id": row_id,
        "split": split,
        "task_type": task_type,
        "prompt": prompt,
        "answer": "",
        "evidence": [],
        "target": "",
        "metadata": {
            "generator": GENERATOR_VERSION,
            "template_family": style,
            "latent": latent,
        },
    }
    if pair_id is not None:
        row["pair_id"] = pair_id
    label = derive_label(row)
    row["answer"] = label["answer"]
    row["evidence"] = label["evidence"]
    row["target"] = compact_target(row["answer"], row["evidence"])
    return row


def compact_target(answer: str, evidence: list[int]) -> str:
    """Serialize the required response using compact, stable JSON."""
    return json.dumps(
        {"answer": answer, "evidence": evidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def derive_label(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute a row's label solely from its symbolic latent facts and query."""
    task_type = row["task_type"]
    latent = row["metadata"]["latent"]
    facts: list[dict[str, Any]] = latent["facts"]
    query: dict[str, Any] = latent["query"]

    if task_type == "lookup":
        matches = [
            f for f in facts
            if f["subject"] == query["subject"] and f["relation"] == query["relation"]
        ]
        if len(matches) != 1:
            raise ValueError("lookup query must have exactly one supporting fact")
        return {"answer": str(matches[0]["value"]), "evidence": [matches[0]["index"]]}

    if task_type == "two_hop":
        if query["mode"] == "sum":
            values: list[dict[str, Any]] = []
            for subject in (query["left"], query["right"]):
                matches = [
                    f for f in facts
                    if f["subject"] == subject and f["relation"] == "quantity"
                ]
                if len(matches) != 1 or not isinstance(matches[0]["value"], int):
                    raise ValueError("sum query must have one integer fact per operand")
                values.extend(matches)
            return {
                "answer": str(sum(int(f["value"]) for f in values)),
                "evidence": sorted(f["index"] for f in values),
            }
        first = [
            f for f in facts
            if f["subject"] == query["item"] and f["relation"] == "held_by"
        ]
        if len(first) != 1:
            raise ValueError("two-hop query must have one first-hop fact")
        second = [
            f for f in facts
            if f["subject"] == first[0]["value"] and f["relation"] == "stored_at"
        ]
        if len(second) != 1:
            raise ValueError("two-hop query must have one second-hop fact")
        return {
            "answer": str(second[0]["value"]),
            "evidence": sorted((first[0]["index"], second[0]["index"])),
        }

    if task_type == "unanswerable":
        matches = [
            f for f in facts
            if f["subject"] == query["subject"] and f["relation"] == query["relation"]
        ]
        if len(matches) > 1:
            raise ValueError("unanswerable query must have at most one supporting fact")
        if matches:
            # Counterfactual partners may replace a missing relation with the
            # requested relation, making the same question answerable.
            return {"answer": str(matches[0]["value"]), "evidence": [matches[0]["index"]]}
        return {"answer": "未知", "evidence": []}

    if task_type == "negation":
        candidate_facts = [
            f for f in facts
            if f["subject"] == query["item"] and f["relation"] == "candidate_set"
        ]
        uniqueness_facts = [
            f for f in facts
            if f["subject"] == query["item"] and f["relation"] == "exactly_one"
        ]
        if len(candidate_facts) != 1 or len(uniqueness_facts) != 1:
            raise ValueError("negation query needs a candidate set and uniqueness fact")
        candidates = candidate_facts[0]["value"]
        excluded = [
            f for f in facts
            if f["subject"] == query["item"] and f["relation"] == "not_holder"
        ]
        remaining = [candidate for candidate in candidates if candidate not in {f["value"] for f in excluded}]
        if len(remaining) != 1:
            raise ValueError("negation query must leave exactly one candidate")
        supporting = [candidate_facts[0], uniqueness_facts[0], *excluded]
        return {
            "answer": str(remaining[0]),
            "evidence": sorted(f["index"] for f in supporting),
        }

    raise ValueError(f"unknown task_type: {task_type}")


def _sample_facts(
    task_type: str, serial: int, rng: random.Random, prefixes: tuple[str, ...], style: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Return facts, query, and the chosen fact values for one synthetic item."""
    item = _tag(prefixes, serial, "物")
    person_prefixes = _OOD_PEOPLE if style == "ood" else _PEOPLE
    place_prefixes = _OOD_PLACES if style == "ood" else _PLACES
    owner1 = _tag(person_prefixes, serial + 31, "")
    owner2 = _tag(person_prefixes, serial + 67, "")
    place1 = _tag(place_prefixes, serial + 13, "")
    place2 = _tag(place_prefixes, serial + 49, "")

    if task_type == "lookup":
        color = rng.choice(_COLORS)
        shape = rng.choice(("圆形", "方形", "弧形"))
        other_color = rng.choice(_COLORS)
        other = _tag(prefixes, serial + 97, "物")
        facts = [
            _fact(item, "color", color, f"{item}登记为{color}。"),
            _fact(item, "shape", shape, f"{item}外形为{shape}。"),
            _fact(other, "color", other_color, f"{other}登记为{other_color}。"),
            _fact(owner1, "holds", _tag(prefixes, serial + 41, "扣"), f"{owner1}持有一枚扣件。"),
            _fact(place1, "near", place2, f"{place1}邻近{place2}。"),
        ]
        query = {"subject": item, "relation": "color"}
        return facts, query, {"color": color}

    if task_type == "two_hop":
        # The serial advances by four for a fixed task in ordinary splits and
        # by one across the counterfactual pairs, so an eight-step cycle keeps
        # relation chains and arithmetic cases balanced in every split.
        if serial % 8 < 4:
            shelf = place1
            facts = [
                _fact(item, "held_by", owner1, f"{item}交由{owner1}保管。"),
                _fact(owner1, "stored_at", shelf, f"{owner1}将所管物品存入{shelf}。"),
                _fact(owner2, "stored_at", place2, f"{owner2}把物品放在{place2}。"),
                _fact(owner2, "held_by", _tag(prefixes, serial + 83, "物"), f"{owner2}还保管另一件物品。"),
            ]
            query = {"mode": "relation", "item": item}
            return facts, query, {"shelf": shelf}
        left = _tag(prefixes, serial + 17, "袋")
        right = _tag(prefixes, serial + 29, "袋")
        left_count, right_count = rng.randint(2, 18), rng.randint(2, 18)
        facts = [
            _fact(left, "quantity", left_count, f"{left}装有{left_count}枚棋子。"),
            _fact(right, "quantity", right_count, f"{right}装有{right_count}枚棋子。"),
            _fact(owner1, "quantity", rng.randint(1, 9), f"{owner1}另收有若干棋子。"),
            _fact(place1, "near", place2, f"{place1}邻近{place2}。"),
        ]
        query = {"mode": "sum", "left": left, "right": right}
        return facts, query, {"left": left_count, "right": right_count}

    if task_type == "unanswerable":
        alias = _tag(_PEOPLE, serial + 31, "")
        facts = [
            _fact(item, "designation", alias, f"{item}登记的代号是{alias}。"),
            _fact(item, "stored_at", place1, f"{item}存放在{place1}。"),
            _fact(owner1, "holds", place2, f"{owner1}保管着一枚石片。"),
            _fact(place1, "near", place2, f"{place1}与{place2}相邻。"),
        ]
        query = {"subject": item, "relation": "held_by"}
        return facts, query, {"absent_relation": "held_by"}

    if task_type == "negation":
        candidates = [owner1, owner2, _tag(person_prefixes, serial + 101, "")]
        rng.shuffle(candidates)
        item = _tag(prefixes, serial, "物")
        if style == "ood":
            candidate_text = (
                f"{item}候选名单列有{candidates[0]}、{candidates[1]}、{candidates[2]}，"
                "未列者不在候选内。"
            )
        else:
            candidate_text = (
                f"{item}候选名单为{candidates[0]}/{candidates[1]}/{candidates[2]}，"
                "不含其他人。"
            )
        facts = [
            _fact(item, "candidate_set", candidates, candidate_text),
            _fact(item, "not_holder", candidates[0], f"{candidates[0]}未保管{item}。"),
            _fact(item, "not_holder", candidates[1], f"{candidates[1]}未保管{item}。"),
            _fact(item, "exactly_one", 1, f"{item}保管者仅一人。"),
        ]
        query = {"item": item}
        return facts, query, {"excluded": candidates[:2]}

    raise ValueError(f"unknown task_type: {task_type}")


def _build_split(
    name: str,
    size: int,
    rng: random.Random,
    serial_start: int,
    style: str = "id",
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    prefixes = _OOD_PREFIXES if style == "ood" else _ID_PREFIXES
    for index in range(size):
        task_type = TASK_TYPES[index % len(TASK_TYPES)]
        serial = serial_start + index
        facts, query, _ = _sample_facts(task_type, serial, rng, prefixes, style)
        rows.append(
            _make_row(
                row_id=f"{name}-{index:04d}",
                split=name,
                task_type=task_type,
                facts=facts,
                query=query,
                style=style,
                rng=rng,
            )
        )
    return rows, serial_start + size


def _counterfactual_pair(
    pair_number: int, task_type: str, rng: random.Random
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Keep counterfactual codes in their own range, disjoint from regular rows,
    # while retaining compact three-character base-36 identifiers.
    serial = 2_000 + pair_number * 4
    prefixes = _ID_PREFIXES
    facts, query, _ = _sample_facts(task_type, serial, rng, prefixes, "id")
    changed_facts = copy.deepcopy(facts)

    if task_type == "lookup":
        fact = next(f for f in changed_facts if f["subject"] == query["subject"] and f["relation"] == "color")
        old = fact["value"]
        fact["value"] = next(color for color in _COLORS if color != old)
        fact["text"] = f"{fact['subject']}登记为{fact['value']}。"
    elif task_type == "two_hop":
        if query["mode"] == "sum":
            fact = next(f for f in changed_facts if f["subject"] == query["right"] and f["relation"] == "quantity")
            fact["value"] = int(fact["value"]) + 3
            fact["text"] = f"{fact['subject']}装有{fact['value']}枚棋子。"
        else:
            fact = next(f for f in changed_facts if f["relation"] == "stored_at" and f["subject"] == next(x["value"] for x in facts if x["relation"] == "held_by" and x["subject"] == query["item"]))
            fact["value"] = next(place for place in _PLACES if place != fact["value"])
            fact["text"] = f"{fact['subject']}将所管物品存入{fact['value']}。"
    elif task_type == "unanswerable":
        fact = next(f for f in changed_facts if f["subject"] == query["subject"] and f["relation"] == "designation")
        fact["relation"] = "held_by"
        fact["text"] = f"{fact['subject']}由{fact['value']}保管。"
    elif task_type == "negation":
        candidates = next(f["value"] for f in changed_facts if f["relation"] == "candidate_set")
        not_facts = [f for f in changed_facts if f["relation"] == "not_holder"]
        original_excluded = {f["value"] for f in not_facts}
        remaining = next(candidate for candidate in candidates if candidate not in original_excluded)
        not_facts[1]["value"] = remaining
        not_facts[1]["text"] = f"{remaining}未保管{not_facts[1]['subject']}。"

    pair_id = f"cf-pair-{pair_number:04d}"
    base = _make_row(
        row_id=f"test_counterfactual-{(pair_number - 1) * 2:04d}",
        split="test_counterfactual",
        task_type=task_type,
        facts=facts,
        query=query,
        style="id",
        rng=rng,
        pair_id=pair_id,
        shuffle_facts=False,
    )
    altered = _make_row(
        row_id=f"test_counterfactual-{(pair_number - 1) * 2 + 1:04d}",
        split="test_counterfactual",
        task_type=task_type,
        facts=changed_facts,
        query=query,
        style="id",
        rng=rng,
        pair_id=pair_id,
        shuffle_facts=False,
    )
    if base["answer"] == altered["answer"]:
        raise AssertionError(f"counterfactual failed to change answer: {pair_id}")
    return base, altered


def build_dataset(seed: int = SEED) -> dict[str, list[dict[str, Any]]]:
    """Build all canonical rows in memory; no filesystem writes are performed."""
    rng = random.Random(seed)
    result: dict[str, list[dict[str, Any]]] = {}
    serial = 1
    for split in ("train_sft", "train_pref", "validation", "test_id"):
        result[split], serial = _build_split(split, SPLIT_SIZES[split], rng, serial)
    result["test_ood"], serial = _build_split(
        "test_ood", SPLIT_SIZES["test_ood"], rng, serial, style="ood"
    )
    pairs: list[dict[str, Any]] = []
    pair_number = 1
    for task_type in TASK_TYPES:
        for _ in range(20):
            pairs.extend(_counterfactual_pair(pair_number, task_type, rng))
            pair_number += 1
    result["test_counterfactual"] = pairs
    return result


def _fact_count(row: dict[str, Any]) -> int:
    return len(row["metadata"]["latent"]["facts"])


def _gold_wrong_answer(answer: str) -> str:
    if answer.isdecimal():
        return str(int(answer) + 1)
    for candidate in ("赤色", "青色", "紫色", "灰鸥", "青鹿", "白泉", "未知之外"):
        if candidate != answer:
            return candidate
    return "另一答案"


def make_preference(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Construct one preference pair with a controlled, labeled rejection error.

    The four error classes isolate wrong answer, wrong citation, unsupported
    guess, and over-refusal.  Over-refusal is omitted for unanswerable rows
    because abstaining is correct for those examples.
    """
    answer = row["answer"]
    gold_evidence = list(row["evidence"])
    wrong_answer = _gold_wrong_answer(answer)
    error_types = ["wrong_answer", "wrong_citation", "unsupported_answer"]
    if answer != "未知":
        error_types.append("over_refusal")
    error_type = rng.choice(error_types)

    if error_type == "wrong_answer":
        rejected = compact_target(wrong_answer, gold_evidence)
    elif error_type == "wrong_citation":
        if not gold_evidence:
            evidence = [1]
        else:
            unrelated = next(
                (index for index in range(1, _fact_count(row) + 1) if index not in gold_evidence),
                None,
            )
            # If every material is required, produce an incomplete citation
            # rather than inventing an out-of-range evidence ID.
            evidence = (
                sorted([*gold_evidence, unrelated])
                if unrelated is not None
                else gold_evidence[:-1]
            )
        rejected = compact_target(answer, evidence)
    elif error_type == "unsupported_answer":
        rejected = compact_target(wrong_answer, [])
    else:
        rejected = compact_target("未知", [])

    preference: dict[str, Any] = {
        "id": row["id"],
        "split": row["split"],
        "task_type": row["task_type"],
        "prompt": row["prompt"],
        "chosen": row["target"],
        "rejected": rejected,
        "error_type": error_type,
    }
    if "pair_id" in row:
        preference["pair_id"] = row["pair_id"]
    return preference


def dataset_stats(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return a JSON-serializable summary useful for audit and reporting."""
    return {
        split: {
            "rows": len(rows),
            "task_types": dict(sorted(Counter(row["task_type"] for row in rows).items())),
            "two_hop_modes": dict(
                sorted(
                    Counter(
                        row["metadata"]["latent"]["query"].get("mode", "")
                        for row in rows
                        if row["task_type"] == "two_hop"
                    ).items()
                )
            ),
            "unique_prompts": len({row["prompt"] for row in rows}),
        }
        for split, rows in dataset.items()
    }


def write_dataset(
    dataset: dict[str, list[dict[str, Any]]], output_dir: Path = DATA_DIR, seed: int = SEED
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in dataset.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    pref_rng = random.Random(seed + 1)
    preferences = [make_preference(row, pref_rng) for row in dataset["train_pref"]]
    with (output_dir / "preferences_train.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for pair in preferences:
            stream.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "preference_seed": seed + 1,
        "split_sizes": {split: len(rows) for split, rows in dataset.items()},
        "statistics": dataset_stats(dataset),
        "preference_rows": len(preferences),
        "preference_error_types": dict(sorted(Counter(p["error_type"] for p in preferences).items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def generate_dataset(seed: int = SEED, output_dir: Path = DATA_DIR) -> dict[str, list[dict[str, Any]]]:
    """Generate and persist the complete experiment dataset."""
    dataset = build_dataset(seed)
    write_dataset(dataset, Path(output_dir), seed)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    dataset = generate_dataset(seed=args.seed, output_dir=args.output_dir)
    print(json.dumps(dataset_stats(dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
