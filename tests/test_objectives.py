import math
import unittest

import torch

from src.modeling import (
    collate_features,
    dpo_loss,
    grpo_loss,
    group_advantages,
    sequence_logps,
    sft_feature,
    token_logps_from_logits,
)
from src.train import greedy_responses


class ToyTokenizer:
    chat_template = "toy"
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assertion = (tokenize, add_generation_prompt)
        return messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text) + 1))}

    def decode(self, token_ids, skip_special_tokens=True):
        return ",".join(str(token_id) for token_id in token_ids)


class ToyGreedyModel:
    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.input = input_ids
        self.mask = attention_mask
        suffixes = torch.tensor([[7, 99, 99], [8, 9, 99]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, suffixes), dim=1)


class ObjectiveTests(unittest.TestCase):
    def test_sft_builder_masks_prompt_and_keeps_eos(self):
        feature = sft_feature(
            ToyTokenizer(),
            {"id": "toy", "prompt": "Q", "target": {"answer": "A", "evidence": [1]}},
            max_prompt_length=32,
            max_completion_length=32,
        )
        prompt_len = feature["response_mask"].index(1)
        self.assertTrue(all(label == -100 for label in feature["labels"][:prompt_len]))
        self.assertTrue(all(label >= 0 for label in feature["labels"][prompt_len:]))
        self.assertEqual(feature["input_ids"][-1], 99)
        self.assertEqual(feature["response_mask"][-1], 1)

    def test_overlong_prompt_or_target_is_rejected_without_truncation(self):
        tokenizer = ToyTokenizer()
        with self.assertRaisesRegex(ValueError, "exceeding max_prompt_length"):
            sft_feature(
                tokenizer,
                {"id": "long-prompt", "prompt": "long", "target": {"answer": "A", "evidence": []}},
                max_prompt_length=2,
                max_completion_length=32,
            )
        with self.assertRaisesRegex(ValueError, "exceeding max_completion_length"):
            sft_feature(
                tokenizer,
                {"id": "long-target", "prompt": "Q", "target": {"answer": "A", "evidence": []}},
                max_prompt_length=32,
                max_completion_length=4,
            )

    def test_greedy_batch_left_padding_and_per_row_eos_crop(self):
        tokenizer = ToyTokenizer()
        model = ToyGreedyModel()
        results = greedy_responses(
            model,
            tokenizer,
            [{"id": "short", "prompt": "x"}, {"id": "long", "prompt": "xyz"}],
            torch.device("cpu"),
            max_prompt_length=32,
            max_new_tokens=3,
        )
        self.assertTrue(torch.equal(model.mask[0], torch.tensor([0, 0] + [1] * 12)))
        self.assertEqual(results[0]["text"], "7")
        self.assertEqual(results[0]["generated_tokens"], 2)
        self.assertEqual(results[1]["text"], "8,9")
        self.assertEqual(results[1]["generated_tokens"], 3)
        self.assertFalse(results[0]["truncated"])

    def test_shift_mask_scores_completion_and_eos_but_not_prompt_or_padding(self):
        features = [
            {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4], "response_mask": [0, 0, 1, 1]},
            {"input_ids": [1, 5, 6], "labels": [-100, 6, 6], "response_mask": [0, 1, 1]},
        ]
        batch = collate_features(features, pad_token_id=0)
        logits = torch.zeros((2, 4, 8), dtype=torch.float32)
        # The first completion token is predicted at the last prompt position.
        logits[0, 1, 3] = 4.0
        logits[0, 2, 4] = 4.0  # EOS in this toy sequence.
        logits[1, 0, 5] = 4.0
        logits[1, 1, 6] = 4.0
        token_logps = token_logps_from_logits(
            logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"], sequence_chunk_size=2
        )
        self.assertEqual(tuple(token_logps.shape), (2, 3))
        self.assertEqual(float(token_logps[0, 0]), 0.0)  # prompt target is masked
        self.assertLess(float(token_logps[0, 1]), 0.0)
        self.assertLess(float(token_logps[0, 2]), 0.0)  # EOS target is counted
        self.assertEqual(float(token_logps[1, 2]), 0.0)  # right padding is excluded
        sums, _ = sequence_logps(logits, batch["input_ids"], batch["response_mask"], batch["attention_mask"])
        self.assertTrue(torch.allclose(sums, token_logps.sum(dim=-1)))

    def test_dpo_gradient_increases_chosen_and_decreases_rejected_logp(self):
        chosen = torch.tensor([-2.0], requires_grad=True)
        rejected = torch.tensor([-1.0], requires_grad=True)
        ref_chosen = torch.tensor([-2.0])
        ref_rejected = torch.tensor([-1.0])
        loss, metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        loss.backward()
        self.assertLess(float(chosen.grad), 0.0)
        self.assertGreater(float(rejected.grad), 0.0)
        self.assertAlmostEqual(float(metrics["preference_margin"][0]), 0.0, places=7)
        self.assertAlmostEqual(float(loss.detach()), math.log(2.0), places=6)

    def test_grpo_advantages_are_groupwise_population_normalized(self):
        values = group_advantages(torch.tensor([0.0, 1.0, 2.0, 7.0, 7.0, 7.0]), group_size=3, eps=1e-4)
        self.assertAlmostEqual(float(values[:3].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(values[0]), -math.sqrt(1.5), places=3)
        self.assertTrue(torch.allclose(values[3:], torch.zeros(3)))

    def test_grpo_gradient_sign_and_detached_old_reference(self):
        current = torch.zeros((2, 1), requires_grad=True)
        old = torch.zeros((2, 1), requires_grad=True)
        reference = torch.zeros((2, 1), requires_grad=True)
        loss, stats = grpo_loss(
            current,
            old,
            reference,
            torch.ones_like(current, dtype=torch.bool),
            torch.tensor([1.0, -1.0]),
            clip_epsilon=0.2,
            beta=0.02,
        )
        loss.backward()
        self.assertLess(float(current.grad[0, 0]), 0.0)
        self.assertGreater(float(current.grad[1, 0]), 0.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        self.assertAlmostEqual(float(stats["mean_ratio"]), 1.0, places=6)

    def test_grpo_reverse_kl_estimator(self):
        current = torch.tensor([[0.0]], requires_grad=True)
        old = torch.tensor([[0.0]])
        reference = torch.tensor([[math.log(2.0)]])
        loss, stats = grpo_loss(
            current, old, reference, torch.ones_like(current, dtype=torch.bool),
            torch.zeros(1), clip_epsilon=0.2, beta=0.02,
        )
        expected_kl = 1.0 - math.log(2.0)
        self.assertAlmostEqual(float(stats["kl"]), expected_kl, places=6)
        self.assertAlmostEqual(float(loss.detach()), 0.02 * expected_kl, places=6)


if __name__ == "__main__":
    unittest.main()
