import json
import re
import unittest
from pathlib import Path

from diprec.baseline_grpo import build_parser as build_baseline_parser
from diprec.evaluation import build_parser as build_evaluation_parser
from diprec.grpo import build_parser as build_diprec_rl_parser
from diprec.sft import build_parser as build_sft_parser


ROOT = Path(__file__).resolve().parents[1]


def parser_defaults(parser):
    return {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }


def shell_defaults(path):
    values = {}
    pattern = re.compile(r'^([A-Z][A-Z0-9_]*)=(?:"([^"]*)"|([^\s#]+))$')
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(line)
        if match:
            values[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
    return values


def scalar(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class DefaultConfigurationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (ROOT / "configs/diprec_defaults.json").read_text(encoding="utf-8")
        )
        cls.runner = {
            key: scalar(value)
            for key, value in shell_defaults(ROOT / "scripts/run_experiment.sh").items()
        }
        cls.batch_runner = {
            key: scalar(value)
            for key, value in shell_defaults(ROOT / "scripts/run_all_comparisons.sh").items()
        }

    def test_shared_json_and_shell_defaults_match(self):
        mapping = {
            "model": "MODEL",
            "interest_topk": "INTEREST_TOPK",
            "interest_strategy": "INTEREST_STRATEGY",
            "time_decay": "TIME_DECAY",
            "interest_parameterization": "INTEREST_PARAMETERIZATION",
            "conditioning": "CONDITIONING",
            "num_plans": "NUM_PLANS",
            "sft_micro_batch_size": "SFT_MICRO_BATCH_SIZE",
            "sft_gradient_accumulation_steps": "SFT_GRADIENT_ACCUMULATION_STEPS",
            "baseline_rl_per_device_batch_size": "BASELINE_RL_PER_DEVICE_BATCH_SIZE",
            "baseline_rl_gradient_accumulation_steps": "BASELINE_RL_GRADIENT_ACCUMULATION_STEPS",
            "diprec_rl_per_device_batch_size": "DIPREC_RL_PER_DEVICE_BATCH_SIZE",
            "diprec_rl_gradient_accumulation_steps": "DIPREC_RL_GRADIENT_ACCUMULATION_STEPS",
            "diprec_rl_num_iterations": "DIPREC_RL_NUM_ITERATIONS",
            "diprec_rl_beta": "DIPREC_RL_BETA",
            "sid_beams": "SID_BEAMS",
            "eval_beams": "EVAL_BEAMS",
            "eval_candidate_budget": "EVAL_CANDIDATE_BUDGET",
            "split_strategy": "SPLIT_STRATEGY",
            "max_history_len": "MAX_HISTORY_LEN",
            "max_seq_len": "MAX_SEQ_LEN",
            "seed": "SEED",
        }
        for config_key, shell_key in mapping.items():
            with self.subTest(key=config_key, script="run_experiment"):
                self.assertEqual(self.runner[shell_key], self.config[config_key])
            batch_key = "SEEDS" if shell_key == "SEED" else shell_key
            with self.subTest(key=config_key, script="run_all_comparisons"):
                self.assertEqual(self.batch_runner[batch_key], self.config[config_key])
        self.assertIsNone(self.runner["BASELINE_RL_GENERATION_BATCH_SIZE"])
        self.assertIsNone(self.runner["DIPREC_RL_GENERATION_BATCH_SIZE"])
        self.assertEqual(
            self.config["baseline_rl_generation_batch_size"],
            "auto_effective_update_batch",
        )
        self.assertEqual(
            self.config["diprec_rl_generation_batch_size"],
            "auto_effective_update_batch",
        )

    def test_json_matches_python_entrypoint_defaults(self):
        sft = parser_defaults(build_sft_parser())
        baseline = parser_defaults(build_baseline_parser())
        diprec_rl = parser_defaults(build_diprec_rl_parser())
        evaluation = parser_defaults(build_evaluation_parser())
        expectations = {
            "interest_topk": (sft, diprec_rl, evaluation),
            "interest_strategy": (sft, diprec_rl, evaluation),
            "time_decay": (sft, diprec_rl, evaluation),
            "interest_parameterization": (sft, diprec_rl, evaluation),
            "conditioning": (sft, diprec_rl, evaluation),
            "max_history_len": (sft, baseline, diprec_rl, evaluation),
            "max_seq_len": (sft, baseline, diprec_rl, evaluation),
            "seed": (sft, baseline, diprec_rl, evaluation),
        }
        for key, defaults in expectations.items():
            for values in defaults:
                with self.subTest(key=key):
                    self.assertEqual(values[key], self.config[key])
        self.assertEqual(sft["micro_batch_size"], self.config["sft_micro_batch_size"])
        self.assertEqual(
            sft["gradient_accumulation_steps"],
            self.config["sft_gradient_accumulation_steps"],
        )
        baseline_mapping = {
            "num_generations": "baseline_rl_num_generations",
            "per_device_batch_size": "baseline_rl_per_device_batch_size",
            "gradient_accumulation_steps": "baseline_rl_gradient_accumulation_steps",
            "beta": "baseline_rl_beta",
            "clip_ratio": "baseline_rl_clip_ratio",
            "num_iterations": "baseline_rl_num_iterations",
            "learning_rate": "baseline_rl_learning_rate",
            "num_epochs": "baseline_rl_num_epochs",
        }
        for parser_key, config_key in baseline_mapping.items():
            self.assertEqual(baseline[parser_key], self.config[config_key])
        diprec_mapping = {
            "num_plans": "num_plans",
            "sid_beams": "sid_beams",
            "per_device_batch_size": "diprec_rl_per_device_batch_size",
            "gradient_accumulation_steps": "diprec_rl_gradient_accumulation_steps",
            "beta": "diprec_rl_beta",
            "clip_ratio": "diprec_rl_clip_ratio",
            "num_iterations": "diprec_rl_num_iterations",
            "learning_rate": "diprec_rl_learning_rate",
            "num_epochs": "diprec_rl_num_epochs",
            "interest_loss_weight": "interest_loss_weight",
            "sid_loss_weight": "sid_loss_weight",
        }
        for parser_key, config_key in diprec_mapping.items():
            self.assertEqual(diprec_rl[parser_key], self.config[config_key])
        self.assertEqual(evaluation["eval_beams"], self.config["eval_beams"])
        self.assertEqual(
            evaluation["eval_candidate_budget"], self.config["eval_candidate_budget"]
        )

    def test_diprec_rl_exposes_checkpoint_resume(self):
        parser = build_diprec_rl_parser()
        self.assertIsNone(parser_defaults(parser)["resume_from_checkpoint"])
        parsed = parser.parse_args(
            [
                "--mode",
                "plan_grpo",
                "--model",
                "parent",
                "--train_file",
                "train.jsonl",
                "--sid_index",
                "index.json",
                "--item_meta",
                "items.json",
                "--output_dir",
                "output",
                "--resume_from_checkpoint",
                "checkpoint-12",
            ]
        )
        self.assertEqual(parsed.resume_from_checkpoint, "checkpoint-12")

    def test_core_requirements_exclude_optional_training_stacks(self):
        requirements = (ROOT / "requirements-diprec.txt").read_text(encoding="utf-8")
        active = {
            line.strip().split("=", 1)[0].split(">", 1)[0].lower()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertNotIn("peft", active)
        self.assertNotIn("wandb", active)
        self.assertIn("trl", active)
        self.assertIn("transformers[hf_xet]", active)


if __name__ == "__main__":
    unittest.main()
