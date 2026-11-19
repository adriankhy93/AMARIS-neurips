import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from amaris_merge.merge_one_step import (
    amaris_merge,
    amaris_merge_one_step,
    amaris_merge_many_betas_streamed,
    mix_non_linear_parameters,
)
from utils.checkpoints import SafeTensorCheckpoint


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=True)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([TinyBlock()])
        self.norm = nn.LayerNorm(2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(2, 4, bias=True)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")


class DummyTokenizer:
    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer.json").write_text("{}")


def set_parameters(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


class MergeOneStepTests(unittest.TestCase):
    layer_name = "model.layers.0.self_attn.q_proj"

    def make_models(self):
        base = TinyModel()
        specialist = TinyModel()
        set_parameters(base, 1.0)
        set_parameters(specialist, 3.0)
        return base, specialist

    def test_projected_delta_changes_only_selected_linear_input_subspace(self):
        base, specialist = self.make_models()
        base_weight = base.model.layers[0].self_attn.q_proj.weight.detach().clone()
        basis = torch.tensor([[1.0], [0.0]])

        amaris_merge_one_step(base, specialist, {self.layer_name: basis})

        expected = torch.tensor([[3.0, 1.0], [3.0, 1.0]])
        self.assertTrue(
            torch.equal(base.model.layers[0].self_attn.q_proj.weight, expected)
        )
        self.assertTrue(
            torch.equal(base.model.layers[0].self_attn.q_proj.bias, torch.ones(2))
        )
        self.assertFalse(
            torch.equal(base.model.layers[0].self_attn.q_proj.weight, base_weight)
        )

    def test_ignored_linear_layer_remains_base(self):
        base, specialist = self.make_models()
        original = base.model.layers[0].self_attn.q_proj.weight.detach().clone()

        amaris_merge_one_step(
            base,
            specialist,
            {self.layer_name: torch.eye(2)},
            layers_to_ignore=self.layer_name,
        )

        self.assertTrue(
            torch.equal(base.model.layers[0].self_attn.q_proj.weight, original)
        )

    def test_beta_mixes_all_non_linear_weights_and_linear_biases(self):
        base, specialist = self.make_models()
        original_linear_weight = (
            base.model.layers[0].self_attn.q_proj.weight.detach().clone()
        )

        mixed_count = mix_non_linear_parameters(base, specialist, beta=0.25)

        self.assertEqual(mixed_count, len(list(base.named_parameters())) - 2)
        self.assertTrue(
            torch.equal(base.model.embed_tokens.weight, torch.full((4, 2), 2.5))
        )
        self.assertTrue(torch.equal(base.model.norm.weight, torch.full((2,), 2.5)))
        self.assertTrue(torch.equal(base.model.norm.bias, torch.full((2,), 2.5)))
        self.assertTrue(
            torch.equal(
                base.model.layers[0].self_attn.q_proj.bias, torch.full((2,), 2.5)
            )
        )
        self.assertTrue(torch.equal(base.lm_head.bias, torch.full((4,), 2.5)))
        self.assertTrue(
            torch.equal(
                base.model.layers[0].self_attn.q_proj.weight, original_linear_weight
            )
        )
        self.assertTrue(torch.equal(base.lm_head.weight, torch.ones((4, 2))))

    def test_beta_boundaries_and_validation(self):
        for beta, expected in ((0.0, 3.0), (1.0, 1.0)):
            base, specialist = self.make_models()
            mix_non_linear_parameters(base, specialist, beta=beta)
            self.assertTrue(
                torch.equal(
                    base.model.embed_tokens.weight, torch.full((4, 2), expected)
                )
            )
            self.assertTrue(
                torch.equal(
                    base.model.layers[0].self_attn.q_proj.weight, torch.ones((2, 2))
                )
            )

        base, specialist = self.make_models()
        with self.assertRaisesRegex(ValueError, "must be in"):
            mix_non_linear_parameters(base, specialist, beta=1.01)

    def test_merge_saves_a_reloadable_artifact_with_beta_mixing(self):
        base, specialist = self.make_models()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            basis_dir = root / "basis"
            basis_dir.mkdir()
            torch.save(torch.eye(2), basis_dir / "layer_0_q_proj_input_basis.pt")

            amaris_merge(
                base,
                specialist,
                DummyTokenizer(),
                output_path=root / "output",
                basis_dir=basis_dir,
                beta=0.5,
            )

            checkpoint = root / "output" / "model" / "pytorch_model.bin"
            self.assertTrue(checkpoint.exists())
            self.assertTrue((root / "output" / "model" / "tokenizer.json").exists())
            saved = torch.load(checkpoint, weights_only=True)
            self.assertTrue(
                torch.equal(
                    saved["model.layers.0.self_attn.q_proj.weight"],
                    torch.full((2, 2), 3.0),
                )
            )
            self.assertTrue(
                torch.equal(saved["model.embed_tokens.weight"], torch.full((4, 2), 2.0))
            )

    def test_ignored_linear_weight_is_beta_mixed(self):
        base, specialist = self.make_models()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            basis_dir = root / "basis"
            basis_dir.mkdir()
            torch.save(torch.eye(2), basis_dir / "layer_0_q_proj_input_basis.pt")
            amaris_merge(
                base,
                specialist,
                DummyTokenizer(),
                output_path=root / "output",
                basis_dir=basis_dir,
                beta=0.25,
                layers_to_ignore=self.layer_name,
            )
            # The ignored q_proj weight is not AMARIS-projected; it follows
            # beta * base + (1 - beta) * specialist like embeddings/norms.
            self.assertTrue(
                torch.equal(
                    base.model.layers[0].self_attn.q_proj.weight,
                    torch.full((2, 2), 2.5),
                )
            )
            manifest = json.loads((root / "output" / "amaris_run.json").read_text())
            self.assertEqual(manifest["projection"]["projected_layers"], [])
            self.assertIn(
                {"layer": self.layer_name, "reason": "layers_to_ignore"},
                manifest["projection"]["not_projected_layers"],
            )
            self.assertIn(
                "model.layers.0.self_attn.q_proj.weight",
                manifest["projection"]["weighted_average_layers"],
            )
            self.assertIn(
                "model.embed_tokens.weight",
                manifest["projection"]["weighted_average_layers"],
            )
            self.assertIn(
                "model.norm.weight", manifest["projection"]["weighted_average_layers"]
            )

    def test_streamed_merge_reuses_linear_update_for_multiple_betas(self):
        base, specialist = self.make_models()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir, specialist_dir, basis_dir = (
                root / "base",
                root / "specialist",
                root / "basis",
            )
            base_dir.mkdir()
            specialist_dir.mkdir()
            basis_dir.mkdir()
            save_file(base.state_dict(), base_dir / "model.safetensors")
            save_file(specialist.state_dict(), specialist_dir / "model.safetensors")
            torch.save(torch.eye(2), basis_dir / "layer_0_q_proj_input_basis.pt")
            shared_dir = root / "shared"
            (shared_dir / "cov_combined_1").mkdir(parents=True)
            (shared_dir / "cov_combined_2").mkdir(parents=True)
            (shared_dir / "metadata").mkdir()
            (shared_dir / "metadata" / "tiny_covariances_metadata.json").write_text(
                json.dumps(
                    {
                        "tag": "tiny",
                        "model": "base",
                        "timings_seconds": {
                            "model_load": 1.0,
                            "packed_data_load": 2.0,
                            "covariance_collection_and_save": 3.0,
                            "total": 6.0,
                        },
                    }
                )
            )
            covariance_links = root / "cov_combined"
            covariance_links.mkdir()
            (covariance_links / "cov_combined_1").symlink_to(
                shared_dir / "cov_combined_1"
            )
            (covariance_links / "cov_combined_2").symlink_to(
                shared_dir / "cov_combined_2"
            )
            (basis_dir / "basis_metadata.json").write_text(
                json.dumps(
                    {
                        "alpha": 0.4,
                        "enable_delta_variant": False,
                        "rank_dict": {self.layer_name: 2},
                        "full_rank_dict": {self.layer_name: 2},
                        "trace_dict": {
                            self.layer_name: {
                                "tr_C1": 10.0,
                                "tr_PC1": 8.0,
                                "tr_C2": 12.0,
                                "tr_PC2": 9.0,
                            }
                        },
                    }
                )
            )

            amaris_merge_many_betas_streamed(
                model1=base,
                base=SafeTensorCheckpoint(base_dir),
                specialist=SafeTensorCheckpoint(specialist_dir),
                tokenizer=DummyTokenizer(),
                output_dirs={0.0: root / "beta0", 1.0: root / "beta1"},
                basis_dir=basis_dir,
            )

            beta0 = torch.load(
                root / "beta0" / "model" / "pytorch_model.bin", weights_only=True
            )
            beta1 = torch.load(
                root / "beta1" / "model" / "pytorch_model.bin", weights_only=True
            )
            self.assertTrue(
                torch.equal(
                    beta0["model.layers.0.self_attn.q_proj.weight"],
                    torch.full((2, 2), 3.0),
                )
            )
            self.assertTrue(
                torch.equal(
                    beta1["model.layers.0.self_attn.q_proj.weight"],
                    torch.full((2, 2), 3.0),
                )
            )
            self.assertTrue(
                torch.equal(beta0["model.embed_tokens.weight"], torch.full((4, 2), 3.0))
            )
            self.assertTrue(
                torch.equal(beta1["model.embed_tokens.weight"], torch.full((4, 2), 1.0))
            )
            manifest = json.loads((root / "beta0" / "amaris_run.json").read_text())
            self.assertEqual(manifest["hyperparameters"]["alpha"], 0.4)
            self.assertEqual(manifest["hyperparameters"]["beta"], 0.0)
            self.assertFalse(manifest["hyperparameters"]["enable_delta_variant"])
            self.assertEqual(
                manifest["projection"]["projected_layers"], [self.layer_name]
            )
            self.assertEqual(
                manifest["projection"]["not_projected_layers"][0]["reason"],
                "no_basis_file",
            )
            self.assertIn(
                "model.embed_tokens.weight",
                manifest["projection"]["weighted_average_layers"],
            )
            self.assertIn(
                "model.layers.0.self_attn.q_proj.bias",
                manifest["projection"]["weighted_average_layers"],
            )
            self.assertNotIn(
                "model.layers.0.self_attn.q_proj.weight",
                manifest["projection"]["weighted_average_layers"],
            )
            self.assertEqual(
                manifest["projection"]["projected_layer_diagnostics"][self.layer_name][
                    "traces"
                ]["tr_PC2"],
                9.0,
            )
            covariance_timing = manifest["upstream_timings_seconds"][
                "covariance_computation"
            ]
            self.assertEqual(
                covariance_timing["total_covariance_collection_and_save"], 3.0
            )
            self.assertEqual(covariance_timing["per_tag"][0]["tag"], "tiny")
            self.assertTrue(
                all(value >= 0.0 for value in manifest["timings_seconds"].values())
            )


if __name__ == "__main__":
    unittest.main()
