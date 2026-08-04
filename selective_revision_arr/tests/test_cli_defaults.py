import sys
import unittest
from unittest.mock import patch

import evaluate
from model_registry import DEFAULT_MODELS, PREQUANTIZED_4BIT_MODELS


class CliDefaultsTest(unittest.TestCase):
    def test_expensive_conversation_tests_are_opt_in(self):
        with patch.object(sys, "argv", ["evaluate.py"]):
            args = evaluate.parse_args()
        self.assertFalse(args.run_multi_turn_persistence)
        self.assertFalse(args.run_memory_tests)
        self.assertTrue(args.run_same_turn_conflict_tests)

    def test_optional_conversation_flags_enable_their_tests(self):
        with patch.object(
            sys,
            "argv",
            ["evaluate.py", "--run_multi_turn_persistence", "--run_memory_tests"],
        ):
            args = evaluate.parse_args()
        self.assertTrue(args.run_multi_turn_persistence)
        self.assertTrue(args.run_memory_tests)

    def test_default_model_roster_is_unique(self):
        self.assertEqual(len(DEFAULT_MODELS), 10)
        self.assertEqual(len(DEFAULT_MODELS), len(set(DEFAULT_MODELS)))


    def test_single_record_jsonl_is_accepted(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.jsonl"
            path.write_text(json.dumps({"id": "one"}) + "\n", encoding="utf-8")
            self.assertEqual(evaluate.read_json_records(str(path)), [{"id": "one"}])

    def test_prequantized_models_are_registered(self):
        self.assertIn("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", PREQUANTIZED_4BIT_MODELS)
        self.assertIn("unsloth/Qwen2.5-72B-bnb-4bit", PREQUANTIZED_4BIT_MODELS)


if __name__ == "__main__":
    unittest.main()
