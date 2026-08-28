"""Checks for the public training command-line settings."""

import unittest

from configs.train_distillation import build_parser as build_distillation_parser
from configs.train_sft import build_parser as build_sft_parser


class TrainingConfigTests(unittest.TestCase):
    common_args = ["--train", "/tmp/train.xml", "--dataset", "fictbio", "--run-name", "test"]

    def test_distillation_defaults(self) -> None:
        args = build_distillation_parser().parse_args(self.common_args)
        self.assertEqual(args.world_size, 8)
        self.assertEqual(args.max_teacher_seq_len, 2048)
        self.assertEqual(args.gradient_accumulation, 1)
        self.assertEqual(args.weight_decay, 30.0)
        self.assertEqual(args.temperature, 2.0)
        self.assertTrue(args.lora_a_weight_decay_only)

    def test_distillation_settings_are_overridable(self) -> None:
        args = build_distillation_parser().parse_args(
            [
                *self.common_args,
                "--base-model",
                "qwen3-32b",
                "--val-interval",
                "7",
                "--checkpoint-interval",
                "11",
                "--log-interval",
                "3",
            ]
        )
        self.assertEqual(args.base_model, "qwen3-32b")
        self.assertEqual(args.val_interval, 7)
        self.assertEqual(args.checkpoint_interval, 11)
        self.assertEqual(args.log_interval, 3)

    def test_sft_defaults(self) -> None:
        args = build_sft_parser().parse_args(self.common_args)
        self.assertEqual(args.base_model, "qwen3-32b")
        self.assertEqual(args.val_interval, 160)
        self.assertEqual(args.checkpoint_interval, 120)


if __name__ == "__main__":
    unittest.main()
