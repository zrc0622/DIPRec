import copy
import unittest

from diprec.interest import TokenRegistry, assert_prefix_only_label, interest_tokens_from_history, topk_interest_indices
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
