import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from amaris_merge.get_delta import save_deltas, save_deltas_from_checkpoints
from utils.checkpoints import SafeTensorCheckpoint


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.k_proj = nn.Linear(2, 2, bias=False)
        self.v_proj = nn.Linear(2, 2, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)
        self.gate_proj = nn.Linear(2, 2, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMlp()


class TinyQwenLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(2, 2, bias=False)


def fill_parameters(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


class DeltaArtifactTests(unittest.TestCase):
    def test_saved_delta_and_gram_match_weight_difference(self):
        base = TinyQwenLikeModel()
        specialist = TinyQwenLikeModel()
        fill_parameters(base, 1.0)
        fill_parameters(specialist, 3.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_deltas(base, specialist, output_dir, save_gram=True)

            delta = torch.load(
                output_dir / "delta" / "layer_0_q_proj_delta.pt", weights_only=True
            )
            gram = torch.load(
                output_dir / "delta" / "layer_0_q_proj_gram.pt", weights_only=True
            )
            expected_delta = torch.full((2, 2), 2.0)
            self.assertTrue(torch.equal(delta, expected_delta))
            self.assertTrue(torch.equal(gram, expected_delta.T @ expected_delta))

            metadata = json.loads(
                (output_dir / "metadata" / "delta_metadata.json").read_text()
            )
            self.assertTrue(metadata["save_gram"])
            self.assertEqual(
                metadata["layers"]["model.layers.0.self_attn.q_proj"]["gram_shape"],
                [2, 2],
            )

    def test_streamed_safetensors_delta_matches_weight_difference(self):
        base = TinyQwenLikeModel()
        specialist = TinyQwenLikeModel()
        fill_parameters(base, 1.0)
        fill_parameters(specialist, 3.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir, specialist_dir = root / "base", root / "specialist"
            base_dir.mkdir()
            specialist_dir.mkdir()
            save_file(base.state_dict(), base_dir / "model.safetensors")
            save_file(specialist.state_dict(), specialist_dir / "model.safetensors")
            save_deltas_from_checkpoints(
                base_dir, specialist_dir, root / "output", save_gram=True
            )

            delta = torch.load(
                root / "output" / "delta" / "layer_0_q_proj_delta.pt", weights_only=True
            )
            gram = torch.load(
                root / "output" / "delta" / "layer_0_q_proj_gram.pt", weights_only=True
            )
            self.assertTrue(torch.equal(delta, torch.full((2, 2), 2.0)))
            self.assertTrue(torch.equal(gram, torch.full((2, 2), 8.0)))
            metadata = json.loads(
                (root / "output" / "metadata" / "delta_metadata.json").read_text()
            )
            self.assertTrue(metadata["streamed"])

    def test_ignored_layers_have_no_streamed_delta_artifact(self):
        base = TinyQwenLikeModel()
        specialist = TinyQwenLikeModel()
        fill_parameters(base, 1.0)
        fill_parameters(specialist, 3.0)
        ignored = "model.layers.0.self_attn.q_proj,lm_head"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir, specialist_dir = root / "base", root / "specialist"
            base_dir.mkdir()
            specialist_dir.mkdir()
            save_file(base.state_dict(), base_dir / "model.safetensors")
            save_file(specialist.state_dict(), specialist_dir / "model.safetensors")
            save_deltas_from_checkpoints(
                base_dir,
                specialist_dir,
                root / "output",
                layers_to_ignore=ignored,
            )
            self.assertFalse(
                (root / "output" / "delta" / "layer_0_q_proj_delta.pt").exists()
            )
            self.assertFalse((root / "output" / "delta" / "lm_head_delta.pt").exists())
            metadata = json.loads(
                (root / "output" / "metadata" / "delta_metadata.json").read_text()
            )
            self.assertEqual(metadata["layers_to_ignore"], sorted(ignored.split(",")))
            self.assertNotIn("model.layers.0.self_attn.q_proj", metadata["layers"])

    def test_streamed_reader_resolves_tied_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            embedding = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            save_file(
                {"model.embed_tokens.weight": embedding},
                model_dir / "model.safetensors",
            )
            checkpoint = SafeTensorCheckpoint(model_dir)
            self.assertIn("lm_head.weight", checkpoint.keys)
            self.assertTrue(
                torch.equal(checkpoint.get_tensor("lm_head.weight"), embedding)
            )

    def test_streamed_reader_prefers_explicit_untied_lm_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            embedding = torch.full((4, 2), 1.0)
            lm_head = torch.full((4, 2), 3.0)
            save_file(
                {"model.embed_tokens.weight": embedding, "lm_head.weight": lm_head},
                model_dir / "model.safetensors",
            )
            checkpoint = SafeTensorCheckpoint(model_dir)
            self.assertTrue(
                torch.equal(checkpoint.get_tensor("lm_head.weight"), lm_head)
            )


if __name__ == "__main__":
    unittest.main()
