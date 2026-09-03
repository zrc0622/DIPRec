import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class JointSFTSmokeTest(unittest.TestCase):
    def test_joint_sft_trains_validates_and_saves_best_checkpoint(self):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import WhitespaceSplit
            from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"tiny Qwen runtime is unavailable: {exc}")

        from diprec.data import processed_data_fingerprint, sha256_file
        from diprec.sft import build_parser, train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            data = root / "data"
            debug = root / "debug"
            best = root / "best_checkpoint"
            final = root / "final_checkpoint"
            data.mkdir()

            vocabulary = {
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "<think>": 3,
                "</think>": 4,
                "system": 5,
                "user": 6,
                "assistant": 7,
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                unk_token="<unk>",
                pad_token="<pad>",
                eos_token="<eos>",
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] }} "
                "{{ message['content'] }} {% endfor %}assistant"
            )
            model = Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=len(tokenizer),
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=256,
                )
            )
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)

            sid_map = {
                "0": ["<a_0>", "<b_0>", "<c_0>"],
                "1": ["<a_1>", "<b_1>", "<c_1>"],
                "2": ["<a_2>", "<b_2>", "<c_2>"],
            }
            item_metadata = {
                item_id: {"title": f"Item {item_id}"} for item_id in sid_map
            }
            sid_path = root / "index.json"
            item_path = root / "items.json"
            sid_path.write_text(json.dumps(sid_map), encoding="utf-8")
            item_path.write_text(json.dumps(item_metadata), encoding="utf-8")
            manifest = {
                "schema_version": "diprec.long_history.v1",
                "dataset": "Office_Products",
                "source_kind": "raw_event_interactions",
                "source_sha256": {"raw": "tiny-fixture"},
                "sid_index_sha256": hashlib.sha256(sid_path.read_bytes()).hexdigest(),
                "split_strategy": "leave_last_two_out",
                "max_history_len": 50,
            }
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            def record(sample_id, split):
                return {
                    "schema_version": "diprec.long_history.v1",
                    "sample_id": sample_id,
                    "dataset": "Office_Products",
                    "user_id": sample_id,
                    "split": split,
                    "target_position": 2,
                    "history_item_id": ["0", "1"],
                    "history_item_sid": ["<a_0><b_0><c_0>", "<a_1><b_1><c_1>"],
                    "history_sid_levels": [sid_map["0"], sid_map["1"]],
                    "target_item_id": "2",
                    "target_item_sid": "<a_2><b_2><c_2>",
                    "target_sid_levels": sid_map["2"],
                    "history_len_before_truncation": 2,
                    "history_len": 2,
                    "max_history_len": 50,
                }

            for split in ("train", "valid"):
                rows = [record(f"Office_Products:{split}:0", split)]
                (data / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            (base / "training_config.json").write_text(
                json.dumps(
                    {
                        "method": "minionerec_sft",
                        "model": "tiny-qwen",
                        "item_meta_sha256": sha256_file(item_path),
                        "data_manifest": processed_data_fingerprint(manifest),
                        "checkpoint_role": "best_validation",
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "--method", "diprec_sft",
                    "--model", str(base),
                    "--train_file", str(data / "train.jsonl"),
                    "--valid_file", str(data / "valid.jsonl"),
                    "--sid_index", str(sid_path),
                    "--item_meta", str(item_path),
                    "--output_dir", str(final),
                    "--best_output_dir", str(best),
                    "--training_metrics_file", str(debug / "sft_training_metrics.json"),
                    "--sft_objective", "joint_interest_activation",
                    "--conditioning", "history_visible",
                    "--sft_plan_mode", "diverse",
                    "--sft_num_plans", "4",
                    "--num_epochs", "1",
                    "--micro_batch_size", "1",
                    "--gradient_accumulation_steps", "1",
                    "--max_seq_len", "256",
                    "--log_every", "100",
                    "--plan_example_limit", "1",
                ]
            )
            train(args)

            metrics = json.loads(
                (debug / "sft_training_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["status"], "complete")
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertEqual(metrics["checkpoint_selection_metric"], "valid_sid_loss")
            self.assertIn("valid_plan_loss", metrics["epochs"][0])
            self.assertIn("valid_sid_loss", metrics["epochs"][0])
            self.assertTrue((debug / "plan_pool_statistics.json").is_file())
            self.assertTrue((debug / "sampled_plan_examples.jsonl").is_file())
            for checkpoint in (best, final):
                self.assertTrue((checkpoint / "training_config.json").is_file())
                self.assertTrue((checkpoint / "diprec_interest_adapter.pt").is_file())


if __name__ == "__main__":
    unittest.main()
