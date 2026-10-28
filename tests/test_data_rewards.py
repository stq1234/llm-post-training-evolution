from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    SEED,
    SPLIT_SIZES,
    TASK_TYPES,
    build_dataset,
    derive_label,
    make_preference,
)
from src.rewards import format_only_reward, lenient_answer, score_response  # noqa: E402


class DataGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_dataset(SEED)

    def test_sizes_balance_and_prompt_separation(self) -> None:
        self.assertEqual({k: len(v) for k, v in self.dataset.items()}, SPLIT_SIZES)
        all_prompts: list[str] = []
        for split, rows in self.dataset.items():
            self.assertEqual(len({row["prompt"] for row in rows}), len(rows), split)
            all_prompts.extend(row["prompt"] for row in rows)
            counts = Counter(row["task_type"] for row in rows)
            self.assertEqual(set(counts), set(TASK_TYPES), split)
            self.assertEqual(len(set(counts.values())), 1, split)
            two_hop_modes = Counter(
                row["metadata"]["latent"]["query"]["mode"]
                for row in rows
                if row["task_type"] == "two_hop"
            )
            self.assertEqual(len(set(two_hop_modes.values())), 1, split)
        self.assertEqual(len(all_prompts), len(set(all_prompts)))
        sft = {row["prompt"] for row in self.dataset["train_sft"]}
        pref = {row["prompt"] for row in self.dataset["train_pref"]}
        self.assertFalse(sft & pref)

    def test_labels_are_recomputable_and_materials_are_bounded(self) -> None:
        for rows in self.dataset.values():
            for row in rows:
                with self.subTest(row=row["id"]):
                    self.assertEqual(derive_label(row), {"answer": row["answer"], "evidence": row["evidence"]})
                    self.assertEqual(
                        row["target"],
                        json.dumps(
                            {"answer": row["answer"], "evidence": row["evidence"]},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                    self.assertGreaterEqual(len(row["metadata"]["latent"]["facts"]), 4)
                    self.assertLessEqual(len(row["metadata"]["latent"]["facts"]), 6)
                    self.assertLessEqual(len(row["prompt"]), 280)
                    for evidence_id in row["evidence"]:
                        self.assertGreaterEqual(evidence_id, 1)
                        self.assertLessEqual(evidence_id, len(row["metadata"]["latent"]["facts"]))
        self.assertTrue(
            all(row["answer"] == "未知" and row["evidence"] == [] for row in self.dataset["test_id"] if row["task_type"] == "unanswerable")
        )

    def test_counterfactual_pairs_change_one_latent_relation_or_value(self) -> None:
        pairs: dict[str, list[dict]] = defaultdict(list)
        for row in self.dataset["test_counterfactual"]:
            pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 80)
        for pair_id, rows in pairs.items():
            self.assertEqual(len(rows), 2, pair_id)
            self.assertNotEqual(rows[0]["answer"], rows[1]["answer"], pair_id)
            before = rows[0]["metadata"]["latent"]["facts"]
            after = rows[1]["metadata"]["latent"]["facts"]
            self.assertEqual(len(before), len(after))
            differences = []
            for left, right in zip(before, after):
                changed = {key for key in ("subject", "relation", "value") if left[key] != right[key]}
                if changed:
                    differences.append(changed)
            self.assertEqual(len(differences), 1, pair_id)
            self.assertIn(differences[0], ({"relation"}, {"value"}), pair_id)

    def test_negation_candidate_set_does_not_assert_presence_or_uniqueness(self) -> None:
        negation_rows = [
            row
            for rows in self.dataset.values()
            for row in rows
            if row["task_type"] == "negation"
        ]
        for row in negation_rows:
            facts = row["metadata"]["latent"]["facts"]
            candidate_fact = next(f for f in facts if f["relation"] == "candidate_set")
            uniqueness_fact = next(f for f in facts if f["relation"] == "exactly_one")
            text = candidate_fact["text"]
            self.assertIn(row["metadata"]["latent"]["query"]["item"], text)
            self.assertFalse(any(token in text for token in ("之一", "唯一", "恰有", "仅有")))
            self.assertTrue(any(token in uniqueness_fact["text"] for token in ("仅一人", "恰有一位")))

    def test_same_seed_is_deterministic(self) -> None:
        second = build_dataset(SEED)
        self.assertEqual(self.dataset, second)

    def test_ood_words_and_instruction_are_held_out(self) -> None:
        held_out_stems = (
            "霜岛", "烛岛", "砾环", "汐屿", "暮钟", "珊瑚", "云礁", "星槎",
            "雾松", "珊羽", "岩狐", "霜貂", "瓷鹤", "霞獭", "苔鸮", "雾鲤",
            "远屿", "岚坞", "盐穴", "星廊", "落岩", "杉坞", "素堂", "雨庭",
        )
        training_and_id = [
            row
            for split in ("train_sft", "train_pref", "validation", "test_id")
            for row in self.dataset[split]
        ]
        self.assertFalse(
            any(stem in row["prompt"] for row in training_and_id for stem in held_out_stems)
        )
        self.assertTrue(
            all(any(stem in row["prompt"] for stem in held_out_stems) for row in self.dataset["test_ood"])
        )
        self.assertTrue(all(row["prompt"].startswith("编号材料为唯一依据") for row in self.dataset["test_ood"]))
        self.assertTrue(all(row["prompt"].startswith("仅依据编号材料") for row in self.dataset["train_sft"]))


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_dataset(SEED)["validation"]

    def test_gold_and_wrong_evidence_reward(self) -> None:
        row = next(r for r in self.rows if r["task_type"] == "two_hop")
        gold = score_response(row["target"], row)
        self.assertEqual(gold["reward"], 1.0)
        self.assertTrue(gold["format_valid"])
        self.assertTrue(gold["answer_correct"])
        self.assertTrue(gold["evidence_exact"])
        self.assertTrue(gold["joint_correct"])

        fact_count = len(row["metadata"]["latent"]["facts"])
        extra = next(i for i in range(1, fact_count + 1) if i not in row["evidence"])
        wrong_citation = json.dumps(
            {"answer": row["answer"], "evidence": sorted([*row["evidence"], extra])},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        scored = score_response(wrong_citation, row)
        self.assertTrue(scored["answer_correct"])
        self.assertFalse(scored["evidence_exact"])
        self.assertFalse(scored["joint_correct"])
        self.assertAlmostEqual(scored["reward"], 0.55)

        all_materials = json.dumps(
            {
                "answer": row["answer"],
                "evidence": list(range(1, fact_count + 1)),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        exhaustive = score_response(all_materials, row)
        self.assertTrue(exhaustive["answer_correct"])
        self.assertFalse(exhaustive["evidence_exact"])
        self.assertAlmostEqual(exhaustive["reward"], 0.55)

    def test_json_failures_are_strict_and_lenient_is_diagnostic(self) -> None:
        row = self.rows[0]
        wrapped = f"```json\n{row['target']}\n```"
        result = score_response(wrapped, row)
        self.assertFalse(result["format_valid"])
        self.assertFalse(result["answer_correct"])
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(lenient_answer(wrapped), row["answer"])
        self.assertEqual(format_only_reward(row["target"]), 1.0)
        self.assertEqual(format_only_reward(wrapped), 0.0)

        spaced = json.dumps({"answer": row["answer"], "evidence": row["evidence"]}, ensure_ascii=False)
        spaced_result = score_response(spaced, row)
        self.assertTrue(spaced_result["format_valid"])
        self.assertTrue(spaced_result["answer_correct"])
        self.assertTrue(spaced_result["evidence_exact"])
        self.assertAlmostEqual(spaced_result["reward"], 1.0)

        reordered = json.dumps(
            {"evidence": row["evidence"], "answer": row["answer"]},
            ensure_ascii=False,
            indent=2,
        )
        reordered_result = score_response(reordered, row)
        self.assertTrue(reordered_result["format_valid"])
        self.assertEqual(reordered_result["reward"], 1.0)
        self.assertEqual(format_only_reward(reordered), 1.0)

        fact_count = len(row["metadata"]["latent"]["facts"])
        out_of_range = json.dumps(
            {"answer": row["answer"], "evidence": [fact_count + 1]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bounded_result = score_response(out_of_range, row)
        self.assertFalse(bounded_result["format_valid"])
        self.assertTrue(bounded_result["answer_correct"])
        self.assertFalse(bounded_result["evidence_exact"])
        self.assertAlmostEqual(bounded_result["reward"], 0.45)
        self.assertEqual(format_only_reward(out_of_range), 1.0)

        duplicated_citation = json.dumps(
            {
                "answer": row["answer"],
                "evidence": [*row["evidence"], row["evidence"][0]],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        duplicate_result = score_response(duplicated_citation, row)
        self.assertFalse(duplicate_result["format_valid"])
        self.assertTrue(duplicate_result["answer_correct"])
        self.assertFalse(duplicate_result["evidence_exact"])
        self.assertEqual(format_only_reward(duplicated_citation), 0.0)

        duplicate_key = (
            '{"answer":"not-the-gold","answer":'
            + json.dumps(row["answer"], ensure_ascii=False)
            + ',"evidence":'
            + json.dumps(row["evidence"])
            + "}"
        )
        duplicate_key_result = score_response(duplicate_key, row)
        self.assertFalse(duplicate_key_result["format_valid"])
        self.assertFalse(duplicate_key_result["answer_correct"])
        self.assertEqual(format_only_reward(duplicate_key), 0.0)

        extra_field = json.dumps(
            {"answer": row["answer"], "evidence": row["evidence"], "note": "extra"},
            ensure_ascii=False,
        )
        extra_field_result = score_response(extra_field, row)
        self.assertFalse(extra_field_result["format_valid"])
        self.assertFalse(extra_field_result["answer_correct"])
        self.assertEqual(format_only_reward(extra_field), 0.0)

        malformed = score_response('{"answer":"x","evidence":[true]}', row)
        self.assertFalse(malformed["format_valid"])
        self.assertFalse(malformed["answer_correct"])
        self.assertFalse(malformed["evidence_exact"])
        self.assertEqual(malformed["reward"], 0.0)

    def test_preference_rejections_are_labeled_and_not_gold(self) -> None:
        rng = random.Random(20261028)
        error_types: set[str] = set()
        rows = build_dataset(SEED)["train_pref"]
        for row in rows:
            pair = make_preference(row, rng)
            self.assertEqual(pair["chosen"], row["target"])
            self.assertNotEqual(pair["rejected"], pair["chosen"])
            error_types.add(pair["error_type"])
            self.assertLess(score_response(pair["rejected"], row)["reward"], 1.0)
        self.assertEqual(
            error_types,
            {"wrong_answer", "wrong_citation", "unsupported_answer", "over_refusal"},
        )


if __name__ == "__main__":
    unittest.main()
