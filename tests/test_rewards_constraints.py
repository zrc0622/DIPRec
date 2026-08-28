import json
import tempfile
import unittest
from pathlib import Path

from diprec.constraints import SIDTrie, interest_prefix_allowed_fn
from diprec.grpo import group_layout
from diprec.rewards import hierarchical_advantages, score_plan, token_advantage_mask
from diprec.sidreasoner_reward import compute_score, parse_response


class RewardAndConstraintTest(unittest.TestCase):
    def test_g_times_b_layout(self):
        rows = group_layout(2, 3, 4)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({(row["prompt_index"], row["plan_index"]) for row in rows}), 6)
        self.assertTrue(all(sum(r["candidate_index"] == b for r in rows) == 6 for b in range(4)))

    def test_plan_advantage_is_across_plans_and_candidate_within_plan(self):
        plan_adv, candidate_adv = hierarchical_advantages(
            [1.0, 3.0], [[0.0, 2.0], [100.0, 100.0]], mode="plan_grpo"
        )
        self.assertLess(plan_adv[0], 0)
        self.assertGreater(plan_adv[1], 0)
        self.assertLess(candidate_adv[0][0], 0)
        self.assertGreater(candidate_adv[0][1], 0)
        self.assertEqual(candidate_adv[1], [0.0, 0.0])

    def test_zero_variance_reward_is_finite_zero(self):
        plan_adv, candidate_adv = hierarchical_advantages([2.0, 2.0], [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(plan_adv, [0.0, 0.0])
        self.assertEqual(candidate_adv, [[0.0, 0.0], [0.0, 0.0]])

    def test_token_masks_are_disjoint_and_weighted(self):
        values = token_advantage_mask([9, 10, 20, 30], [10], [20], 2.0, -3.0, 0.5, 2.0)
        self.assertEqual(values, [0.0, 1.0, -6.0, 0.0])
        with self.assertRaises(AssertionError):
            token_advantage_mask([1], [1], [1], 1, 1, 1, 1)

    def test_sid_trie_only_allows_catalog_paths(self):
        trie = SIDTrie.from_sequences([[1, 2, 3], [1, 4, 5]])
        self.assertEqual(trie.allowed([], 99), [1])
        self.assertEqual(trie.allowed([1], 99), [2, 4])
        self.assertEqual(trie.allowed([1, 2], 99), [3])
        self.assertEqual(trie.allowed([1, 2, 3], 99), [99])
        self.assertEqual(trie.allowed([7], 99), [99])
        self.assertTrue(trie.contains([1, 4, 5]))
        self.assertFalse(trie.contains([1, 4, 9]))

    def test_interest_stage_allows_only_interest_vocab_then_suffix(self):
        allowed = interest_prefix_allowed_fn(
            interest_ids=[10, 11],
            pad_id=12,
            end_id=13,
            end_think_ids=[14, 15],
            prompt_length=2,
            k=2,
            eos_token_id=99,
        )
        self.assertEqual(allowed(0, [7, 8]), [10, 11, 12])
        self.assertEqual(allowed(0, [7, 8, 10]), [11, 12])
        self.assertEqual(allowed(0, [7, 8, 12]), [12])
        self.assertEqual(allowed(0, [7, 8, 10, 12]), [13])
        self.assertEqual(allowed(0, [7, 8, 10, 11, 13]), [14])
        self.assertEqual(allowed(0, [7, 8, 10, 11, 13, 14]), [15])
        self.assertEqual(allowed(0, [7, 8, 10, 11, 13, 14, 15]), [99])

    def test_plan_reward_uses_rank_validity_and_duplicates(self):
        candidates = [["a", "b", "x"], ["a", "b", "c"], ["a", "b", "c"]]
        reward, candidate_rewards, details = score_plan(candidates, ["a", "b", "c"], [True, True, True])
        self.assertGreater(reward, 0)
        self.assertGreater(candidate_rewards[1], candidate_rewards[0])
        self.assertAlmostEqual(details["NDCG@5"], 1.0 / __import__("math").log2(3))
        self.assertGreater(details["duplicate_rate"], 0)

    def test_sidreasoner_reward_requires_completed_reasoning(self):
        sid = "<a_1><b_2><c_3>"
        self.assertIsNone(parse_response(sid))
        self.assertEqual(parse_response(f"<think>reason</think>{sid}"), ("<a_1>", "<b_2>", "<c_3>"))

    def test_sidreasoner_reward_accepts_explicit_catalog_path(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.json"
            index_path.write_text(
                json.dumps({"item": ["<a_1>", "<b_2>", "<c_3>"]}), encoding="utf-8"
            )
            result = compute_score(
                "diprec/sidreasoner",
                "<think>reason</think><a_1><b_2><c_3>",
                "<a_1><b_2><c_3>",
                sid_index=str(index_path),
            )
            self.assertEqual(result["valid"], 1.0)
            self.assertGreater(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
