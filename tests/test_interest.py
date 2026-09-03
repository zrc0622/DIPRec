import copy
import unittest

from diprec.interest import (
    TokenRegistry,
    assert_prefix_only_label,
    interest_activation_plan_pool,
    interest_plans_from_history,
    interest_tokens_from_history,
    select_interest_activation_plan,
    topk_interest_indices,
)
from diprec.prompts import sid_prompt


class InterestLabelTest(unittest.TestCase):
    def setUp(self):
        self.history = [
            ["<a_17>", "<b_1>", "<c_1>"],
            ["<a_42>", "<b_2>", "<c_2>"],
            ["<a_17>", "<b_3>", "<c_3>"],
        ]
        self.record = {
            "sample_id": "u:3",
            "dataset": "Video_Games",
            "history_sid_levels": self.history,
            "history_item_sid": ["<a_17><b_1><c_1>", "<a_42><b_2><c_2>", "<a_17><b_3><c_3>"],
            "history_len": 3,
            "target_sid_levels": ["<a_99>", "<b_0>", "<c_0>"],
            "target_item_sid": "<a_99><b_0><c_0>",
        }

    def test_frequency_and_padding_example(self):
        self.assertEqual(
            interest_tokens_from_history(self.history, 3, "frequency"),
            ["<INT_017>", "<INT_042>", "<INT_PAD>"],
        )

    def test_tie_break_is_numeric_sid_order(self):
        history = [["<a_42>", "<b_1>", "<c_1>"], ["<a_17>", "<b_2>", "<c_2>"]]
        self.assertEqual(topk_interest_indices(history, 2), [17, 42])

    def test_time_decay_prefers_recent_cluster(self):
        history = [
            ["<a_1>", "<b_1>", "<c_1>"],
            ["<a_1>", "<b_2>", "<c_2>"],
            ["<a_2>", "<b_3>", "<c_3>"],
        ]
        self.assertEqual(topk_interest_indices(history, 2, "time_decay", time_decay=1.0)[0], 2)

    def test_target_change_cannot_change_label(self):
        label = interest_tokens_from_history(self.history, 3)
        changed = copy.deepcopy(self.record)
        changed["target_sid_levels"] = ["<a_17>", "<b_9>", "<c_9>"]
        changed["target_item_sid"] = "<a_17><b_9><c_9>"
        self.assertEqual(label, interest_tokens_from_history(changed["history_sid_levels"], 3))
        assert_prefix_only_label(changed, label)

    def test_diverse_plans_change_content_not_just_order(self):
        history = [
            ["<a_1>", "<b_0>", "<c_0>"],
            ["<a_1>", "<b_1>", "<c_1>"],
            ["<a_2>", "<b_2>", "<c_2>"],
            ["<a_3>", "<b_3>", "<c_3>"],
            ["<a_4>", "<b_4>", "<c_4>"],
        ]
        plans = interest_plans_from_history(history, 3, "diverse", 4)
        self.assertEqual(plans[0], ["<INT_001>", "<INT_002>", "<INT_003>"])
        self.assertEqual(len(plans), 4)
        self.assertEqual(len({tuple(sorted(plan)) for plan in plans}), 4)

    def test_diverse_plans_are_deterministic_and_prefix_only(self):
        first = interest_plans_from_history(self.history, 3, "diverse", 8)
        changed = copy.deepcopy(self.record)
        changed["target_sid_levels"] = ["<a_17>", "<b_9>", "<c_9>"]
        changed["target_item_sid"] = "<a_17><b_9><c_9>"
        second = interest_plans_from_history(
            changed["history_sid_levels"], 3, "diverse", 8
        )
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)

    def test_single_plan_mode_preserves_legacy_label(self):
        self.assertEqual(
            interest_plans_from_history(self.history, 3, "single", 8),
            [["<INT_017>", "<INT_042>", "<INT_PAD>"]],
        )

    def test_activation_pool_is_compact_deduplicated_and_history_only(self):
        history = [
            ["<a_1>", "<b_0>", "<c_0>"],
            ["<a_1>", "<b_1>", "<c_1>"],
            ["<a_2>", "<b_2>", "<c_2>"],
            ["<a_3>", "<b_3>", "<c_3>"],
            ["<a_4>", "<b_4>", "<c_4>"],
        ]
        plans = interest_activation_plan_pool(history, 3, max_plans=8)
        self.assertEqual(plans[0], ["<INT_001>", "<INT_002>", "<INT_003>"])
        self.assertLessEqual(len(plans), 8)
        content = [tuple(sorted(token for token in plan if token != "<INT_PAD>")) for plan in plans]
        self.assertEqual(len(content), len(set(content)))
        # Aggregate + recent + four singleton interests: no combinatorial
        # short-plan expansion merely to force exactly eight labels.
        self.assertEqual(len(plans), 6)

    def test_activation_pool_does_not_force_multiple_plans_from_one_interest(self):
        history = [["<a_7>", f"<b_{index}>", "<c_0>"] for index in range(4)]
        self.assertEqual(
            interest_activation_plan_pool(history, 3, max_plans=8),
            [["<INT_007>", "<INT_PAD>", "<INT_PAD>"]],
        )

    def test_diverse_activation_selection_visits_pool_without_replacement(self):
        plans = [
            ["<INT_001>", "<INT_PAD>"],
            ["<INT_002>", "<INT_PAD>"],
            ["<INT_003>", "<INT_PAD>"],
        ]
        first_cycle = [
            select_interest_activation_plan(plans, "diverse", epoch, 42, "sample")[0]
            for epoch in range(len(plans))
        ]
        repeated = [
            select_interest_activation_plan(plans, "diverse", epoch, 42, "sample")[0]
            for epoch in range(len(plans))
        ]
        self.assertEqual(sorted(first_cycle), [0, 1, 2])
        self.assertEqual(first_cycle, repeated)
        self.assertEqual(
            select_interest_activation_plan(plans, "single", 99, 42, "sample"),
            (0, plans[0]),
        )

    def test_strict_bottleneck_hides_history(self):
        prompt = sid_prompt(self.record, ["<INT_017>", "<INT_042>", "<INT_PAD>"], 50, "interest_bottleneck")
        for sid in self.record["history_item_sid"]:
            self.assertNotIn(sid, prompt)

    def test_interest_control_and_code_ids_are_disjoint_from_sids(self):
        registry = TokenRegistry(("<a_1>",), ("<INT_001>",), (7,), (11,), 8, 9, 10)
        registry.assert_disjoint()
        with self.assertRaises(AssertionError):
            TokenRegistry(("<a_1>",), ("<INT_001>",), (7,), (11,), 7, 9, 10).assert_disjoint()


if __name__ == "__main__":
    unittest.main()
