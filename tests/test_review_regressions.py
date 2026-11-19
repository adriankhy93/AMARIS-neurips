import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from amaris_merge.compute_basis_by_percentage import compute_basis_from_covariances
from amaris_merge.merge_one_step import amaris_merge, amaris_merge_many_betas_streamed
from data.tokenize_dataset import find_assistant_start_seq
from utils.artifacts import beta_output_dirs, stage_state
from utils.checkpoints import SafeTensorCheckpoint
from utils.generalized_eigensolver import LOBPCGRetryExhausted

ROOT = Path(__file__).resolve().parents[1]


class DummyTokenizer:
    def save_pretrained(self, path):
        pass


def tiny_model(value):
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model


class ReviewRegressionTests(unittest.TestCase):
    def test_projected_tied_head_survives_both_merge_paths_and_reload(self):
        for streamed in (False, True):
            with self.subTest(streamed=streamed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                basis = root / "basis"
                basis.mkdir()
                torch.save(torch.eye(4)[:, :1], basis / "lm_head_input_basis.pt")
                base, specialist = tiny_model(1), tiny_model(3)
                if streamed:
                    base.save_pretrained(root / "base")
                    specialist.save_pretrained(root / "specialist")
                    amaris_merge_many_betas_streamed(
                        base,
                        SafeTensorCheckpoint(root / "base"),
                        SafeTensorCheckpoint(root / "specialist"),
                        DummyTokenizer(),
                        {0.0: root / "beta0", 1.0: root / "beta1"},
                        basis,
                    )
                    outputs = [(root / "beta0", 3), (root / "beta1", 1)]
                else:
                    amaris_merge(
                        base,
                        specialist,
                        DummyTokenizer(),
                        root / "merged",
                        basis_dir=basis,
                        beta=0.25,
                    )
                    outputs = [(root / "merged", 2.5)]
                for output, embedding_value in outputs:
                    loaded = LlamaForCausalLM.from_pretrained(output / "model")
                    expected = torch.ones(8, 4)
                    expected[:, 0] = 3
                    torch.testing.assert_close(loaded.lm_head.weight, expected)
                    torch.testing.assert_close(
                        loaded.model.embed_tokens.weight,
                        torch.full((8, 4), float(embedding_value)),
                    )
                    self.assertIsNot(
                        loaded.lm_head.weight, loaded.model.embed_tokens.weight
                    )
                    self.assertFalse(loaded.config.tie_word_embeddings)

    def test_assistant_start_supports_lists_and_encoding_objects(self):
        for encoded in (False, True):

            class Tokenizer:
                def apply_chat_template(self, messages, **kwargs):
                    ids = [1, 2, 3, 4] if kwargs["add_generation_prompt"] else [1, 2]
                    return SimpleNamespace(input_ids=ids) if encoded else ids

            self.assertEqual(find_assistant_start_seq(Tokenizer()), [3, 4])

    def test_beta_paths_preserve_precision_and_reject_duplicates(self):
        betas = [0.501, 0.504, 0.5, 1.0]
        paths = beta_output_dirs(betas, output_base_dir="output")
        self.assertEqual(list(paths), betas)
        self.assertEqual(len(set(paths.values())), len(betas))
        self.assertEqual(paths[0.504].name, "output-beta-0.504")
        self.assertEqual(paths[0.5].name, "output-beta-0.50")
        for invalid in ([0.5, 0.5], [], [float("nan")], [1.1]):
            with self.assertRaises(ValueError):
                beta_output_dirs(invalid, output_base_dir="output")

    def test_sweep_rejects_shared_tags_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "model1_dir: base",
                        "model2_dir: specialist",
                        "datasets1: common:first.jsonl",
                        "dataset2: common:second.jsonl",
                        "train_proportion: 0.8",
                        "max_seq_len: 4",
                        "max_num_tokens: 100",
                        "alpha: 0.1",
                        f'output_dir: {root / "output"}',
                    ]
                )
            )
            result = subprocess.run(
                ["bash", str(ROOT / "merge.sh"), str(config)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Dataset tags must be distinct", result.stderr)
            self.assertFalse((root / "output-shared").exists())

    def test_completion_requires_success_and_intact_outputs_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yaml"
            config.write_text("max_k: 256")
            source = root / "source.jsonl"
            source.write_text("original")
            delta = root / "delta"
            delta.mkdir()
            first, second = delta / "layer_0_delta.pt", delta / "layer_1_delta.pt"
            first.write_bytes(b"partial")
            args = (root, "delta", config, ["delta/*_delta.pt"], [source])
            self.assertFalse(stage_state("check", *args))
            stage_state("begin", *args)
            self.assertFalse(first.exists())
            first.write_bytes(b"complete")
            second.write_bytes(b"complete")
            stage_state("finish", *args)
            self.assertTrue(stage_state("check", *args))
            second.unlink()
            self.assertFalse(stage_state("check", *args))
            second.write_bytes(b"complete")
            stage_state("finish", *args)
            source.write_text("changed source")
            self.assertFalse(stage_state("check", *args))
            stage_state("finish", *args)
            config.write_text("max_k: 512")
            self.assertFalse(stage_state("check", *args))

    def test_cache_invalidates_when_matrix_changes_but_sum_and_norm_match(self):
        from utils import generalized_eigensolver

        common = dict(
            C1={"layer": torch.eye(3)},
            Delta={"layer": torch.eye(3)},
            alpha=0.34,
            max_k=1,
            basis_solver="dense",
        )
        with tempfile.TemporaryDirectory() as temp, patch(
            "amaris_merge.compute_basis_by_percentage.parse_devices",
            return_value=["cpu"],
        ), patch(
            "utils.generalized_eigensolver.generalized_eigh_torch_safe",
            wraps=generalized_eigensolver.generalized_eigh_torch_safe,
        ) as solve:
            compute_basis_from_covariances(
                C2={"layer": torch.diag(torch.tensor([3.0, 2.0, 1.0]))},
                candidate_cache_dir=Path(temp),
                **common,
            )
            changed = {"layer": torch.diag(torch.tensor([1.0, 2.0, 3.0]))}
            cached = compute_basis_from_covariances(
                C2=changed, candidate_cache_dir=Path(temp), **common
            )
            self.assertEqual(solve.call_count, 2)
            fresh = compute_basis_from_covariances(C2=changed, **common)
            torch.testing.assert_close(
                cached[0]["layer"] @ cached[0]["layer"].T,
                fresh[0]["layer"] @ fresh[0]["layer"].T,
            )
            self.assertAlmostEqual(cached[3]["layer"], 0.5)

    def test_all_fallback_build_can_merge_but_incomplete_metadata_cannot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            basis = root / "basis"
            basis.mkdir()
            error = LOBPCGRetryExhausted(
                "unstable", retries=2, eta=0.001, candidate_rank=1
            )
            with patch(
                "amaris_merge.compute_basis_by_percentage.parse_devices",
                return_value=["cpu"],
            ), patch(
                "amaris_merge.compute_basis_by_percentage.faster_generalized_eigen_ranking_projector_torch",
                side_effect=error,
            ):
                result = compute_basis_from_covariances(
                    C1={"lm_head": torch.eye(4)},
                    C2={"lm_head": torch.eye(4)},
                    Delta={"lm_head": torch.ones(8, 4)},
                    basis_save_dir=basis,
                )
            metadata = dict(
                completed=True,
                expected_layers=["lm_head"],
                rank_dict={},
                fallback_layers=result[-1],
            )
            (basis / "basis_metadata.json").write_text(json.dumps(metadata))
            base, specialist = tiny_model(1), tiny_model(3)
            base.save_pretrained(root / "base")
            specialist.save_pretrained(root / "specialist")
            amaris_merge_many_betas_streamed(
                base,
                SafeTensorCheckpoint(root / "base"),
                SafeTensorCheckpoint(root / "specialist"),
                DummyTokenizer(),
                {0.25: root / "merged"},
                basis,
            )
            loaded = LlamaForCausalLM.from_pretrained(root / "merged/model")
            for parameter in loaded.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, 2.5))
            amaris_merge(
                tiny_model(1),
                specialist,
                DummyTokenizer(),
                root / "nonstreamed",
                basis_dir=basis,
                beta=0.25,
            )
            for parameter in LlamaForCausalLM.from_pretrained(
                root / "nonstreamed/model"
            ).parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, 2.5))
            metadata["expected_layers"].append("layer_0_q_proj")
            (basis / "basis_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                amaris_merge(
                    tiny_model(1),
                    specialist,
                    DummyTokenizer(),
                    root / "invalid",
                    basis_dir=basis,
                    beta=0.25,
                )


if __name__ == "__main__":
    unittest.main()
