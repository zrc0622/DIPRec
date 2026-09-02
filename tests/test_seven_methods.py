import json
import tempfile
import unittest
from pathlib import Path

from diprec.baseline_grpo import (
    _description_text,
    baseline_batch_contract,
    build_baseline_rl_rows,
    catalog_training_generation_kwargs,
    exact_match_reward,
    make_rank_aware_reward,
)
from diprec.data import load_item_metadata, resolve_item_metadata
from diprec.prompts import history_prompt
from diprec.sft import catalog_alignment_maps, encode_catalog_sft_records, encode_sft_records


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return [len(messages[-1]["content"])]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) % 89 + 1 for character in text]

    def decode(self, ids, skip_special_tokens=False):
        del ids, skip_special_tokens
        return "assistant\n"


class SevenMethodDataContractTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "sample_id": "Video_Games:u:2",
            "dataset": "Video_Games",
            "history_item_id": ["0", "1"],
            "history_item_sid": ["<a_0><b_0><c_0>", "<a_1><b_1><c_1>"],
            "history_sid_levels": [
                ["<a_0>", "<b_0>", "<c_0>"],
                ["<a_1>", "<b_1>", "<c_1>"],
            ],
            "history_len": 2,
            "target_item_id": "2",
            "target_item_sid": "<a_2><b_2><c_2>",
            "target_sid_levels": ["<a_2>", "<b_2>", "<c_2>"],
        }
        self.sid_map = {
            "0": ("<a_0>", "<b_0>", "<c_0>"),
            "1": ("<a_1>", "<b_1>", "<c_1>"),
            "2": ("<a_2>", "<b_2>", "<c_2>"),
        }
        self.item_metadata = {
            "0": {"title": "Zero", "description": "Description zero"},
            "1": {"title": "One", "description": "Description one"},
            "2": {"title": "Two", "description": "Description two"},
        }

    def test_minionerec_sft_matches_four_official_task_families(self):
        tokenizer = FakeTokenizer()
        sequential = encode_sft_records(
            tokenizer,
            self.record,
            "minionerec_sft",
            50,
            4096,
            3,
            "frequency",
            0.1,
            "interest_bottleneck",
            self.item_metadata,
        )
        catalog = encode_catalog_sft_records(tokenizer, self.item_metadata, self.sid_map, 4096)
        self.assertEqual(
            {row["stage"] for row in sequential + catalog},
            {"history_sid_to_sid", "history_sid_to_title", "title_to_sid", "sid_to_title"},
        )
        self.assertEqual(len(sequential), 2)
        self.assertEqual(len(catalog), 2 * len(self.sid_map))

    def test_minionerec_catalog_alignment_matches_official_deduplication(self):
        sid_map = self.sid_map | {"3": ("<a_2>", "<b_2>", "<c_2>")}
        metadata = self.item_metadata | {
            "3": {"title": "Zero", "description": "another description"}
        }
        sid_to_title, title_to_sid = catalog_alignment_maps(metadata, sid_map)
        self.assertEqual(len(sid_to_title), 3)
        self.assertEqual(len(title_to_sid), 3)
        self.assertEqual(sid_to_title["<a_2><b_2><c_2>"], "Zero")
        self.assertEqual(title_to_sid["Zero"], "<a_2><b_2><c_2>")

    def test_direct_and_diprec_sft_cardinality(self):
        tokenizer = FakeTokenizer()
        direct = encode_sft_records(
            tokenizer,
            self.record,
            "direct_sft",
            50,
            4096,
            3,
            "frequency",
            0.1,
            "interest_bottleneck",
        )
        diprec = encode_sft_records(
            tokenizer,
            self.record,
            "diprec_sft",
            50,
            4096,
            3,
            "frequency",
            0.1,
            "interest_bottleneck",
        )
        self.assertEqual([row["stage"] for row in direct], ["direct_sft"])
        self.assertEqual([row["stage"] for row in diprec], ["interest_plan", "sid_prediction"])

    def test_diverse_diprec_sft_expands_plan_labels_but_keeps_one_sid_task(self):
        tokenizer = FakeTokenizer()
        record = dict(self.record)
        record["history_sid_levels"] = [
            ["<a_0>", "<b_0>", "<c_0>"],
            ["<a_1>", "<b_1>", "<c_1>"],
            ["<a_2>", "<b_2>", "<c_2>"],
            ["<a_3>", "<b_3>", "<c_3>"],
        ]
        rows = encode_sft_records(
            tokenizer, record, "diprec_sft", 50, 4096, 3,
            "frequency", 0.1, "interest_bottleneck",
            sft_plan_mode="diverse", sft_num_plans=4,
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["stage"] for row in rows].count("interest_plan"), 4)
        self.assertEqual([row["stage"] for row in rows].count("sid_prediction"), 1)
        self.assertEqual({row["plan_count"] for row in rows}, {4})

    def test_minionerec_rl_matches_enabled_official_tasks(self):
        rows, counts = build_baseline_rl_rows(
            "minionerec_rl",
            [self.record],
            self.sid_map,
            self.item_metadata,
            50,
            title_sequence_limit=10_000,
        )
        self.assertEqual(
            counts,
            {
                "history_sid_to_sid": 1,
                "title_to_sid": 3,
                "description_to_sid": 3,
                "title_history_to_sid": 1,
            },
        )
        self.assertEqual(len(rows), 8)
        direct, direct_counts = build_baseline_rl_rows(
            "direct_rl", [self.record], self.sid_map, None, 50
        )
        self.assertEqual(len(direct), 1)
        self.assertEqual(direct_counts, {"history_sid_to_sid": 1})

    def test_minionerec_title_history_sampling_matches_pandas_random_state(self):
        records = [
            self.record | {"sample_id": str(index)}
            for index in range(12)
        ]
        rows, counts = build_baseline_rl_rows(
            "minionerec_rl",
            records,
            self.sid_map,
            self.item_metadata,
            50,
            title_sequence_limit=5,
            seed=42,
        )
        selected = [
            row["sample_id"]
            for row in rows
            if row["task"] == "title_history_to_sid"
        ]
        self.assertEqual(counts["title_history_to_sid"], 5)
        self.assertEqual(selected, ["10", "9", "0", "8", "5"])

    def test_minionerec_rl_description_parsing_matches_upstream(self):
        self.assertEqual(_description_text("[' first ', 'second']"), " first ")
        self.assertEqual(_description_text('["first", "second"]'), '["first", "second"]')
        self.assertEqual(_description_text(""), "")
        with self.assertRaisesRegex(ValueError, "descriptions to be strings"):
            _description_text(["first"])

    def test_baseline_rewards_are_group_aware(self):
        target = "<a_2><b_2><c_2>"
        completions = [target, "<a_0><b_0><c_0>", "<a_1><b_1><c_1>"]
        targets = [target] * 3
        self.assertEqual(exact_match_reward(completions, targets), [1.0, 0.0, 0.0])
        ranked = make_rank_aware_reward(3)(completions, targets)
        self.assertEqual(ranked[0], 0.0)
        self.assertLess(ranked[1], 0.0)
        self.assertLess(ranked[2], 0.0)
        self.assertEqual(
            exact_match_reward([f'  "{target}"\n'], [target]),
            [1.0],
        )
        self.assertEqual(
            exact_match_reward(["<a_2> <b_2> <c_2>"], [target]),
            [1.0],
        )

    def test_baseline_rl_uses_official_train_time_beam_sampling(self):
        generation = catalog_training_generation_kwargs(16, 1.0)
        self.assertTrue(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 16)
        self.assertEqual(generation["num_return_sequences"], 16)
        self.assertEqual(generation["top_k"], None)
        self.assertEqual(generation["top_p"], None)

    def test_small_micro_batch_still_forms_one_complete_grpo_group(self):
        contract = baseline_batch_contract(16, 1, 16, 16, 1)
        self.assertEqual(contract["global_micro_batch"], 1)
        self.assertEqual(contract["generation_batch_size"], 16)
        self.assertEqual(contract["steps_per_generation"], 16)
        self.assertEqual(contract["effective_update_batch"], 16)
        self.assertEqual(contract["unique_prompts_per_generation"], 1)
        self.assertEqual(contract["optimizer_updates_per_rollout"], 1)
        self.assertEqual(contract["sampler_repeat_count"], 16)
        with self.assertRaisesRegex(ValueError, "generation_batch_size"):
            baseline_batch_contract(16, 1, 8, 16, 1)
        with self.assertRaisesRegex(ValueError, "one optimizer update"):
            baseline_batch_contract(4, 1, 4, 8, 1)
        with self.assertRaisesRegex(ValueError, "steps_per_generation"):
            baseline_batch_contract(4, 1, 4, 4, 1, steps_per_generation=2)

    def test_two_rank_batch_contract_places_one_duplicate_on_each_rank(self):
        contract = baseline_batch_contract(2, 1, 2, 1, 2, num_iterations=2)
        self.assertEqual(contract["global_micro_batch"], 2)
        self.assertEqual(contract["steps_per_generation"], 1)
        self.assertEqual(contract["effective_update_batch"], 2)
        self.assertEqual(contract["unique_prompts_per_generation"], 1)
        self.assertEqual(contract["optimizer_updates_per_rollout"], 2)
        self.assertEqual(contract["sampler_repeat_count"], 2)

    def test_direct_and_minionerec_rl_share_the_evaluation_history_prompt(self):
        direct, _ = build_baseline_rl_rows(
            "direct_rl", [self.record], self.sid_map, None, 50
        )
        minionerec, _ = build_baseline_rl_rows(
            "minionerec_rl",
            [self.record],
            self.sid_map,
            self.item_metadata,
            50,
            title_sequence_limit=0,
        )
        expected = history_prompt(self.record, 50, reasoning=False)
        self.assertEqual(direct[0]["prompt"], expected)
        self.assertEqual(minionerec[0]["prompt"], expected)

    def test_item_metadata_resolution_and_namespace_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index"
            index.mkdir()
            path = index / "Video_Games.item.json"
            path.write_text(json.dumps(self.item_metadata), encoding="utf-8")
            self.assertEqual(resolve_item_metadata("Games", root), path)
            loaded = load_item_metadata(path, self.sid_map)
            self.assertEqual(loaded["2"]["title"], "Two")
            with self.assertRaisesRegex(ValueError, "missing 1 SID-index items"):
                load_item_metadata(path, self.sid_map | {"3": ("<a_3>", "<b_3>", "<c_3>")})


if __name__ == "__main__":
    unittest.main()
