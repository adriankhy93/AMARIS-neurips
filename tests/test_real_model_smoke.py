"""Optional real-checkpoint AMARIS smoke test.

Enable explicitly because this test loads two compatible checkpoints onto a GPU:
AMARIS_REAL_MODEL_SMOKE=1 AMARIS_MODEL1_DIR=/path/to/base \
AMARIS_MODEL2_DIR=/path/to/specialist AMARIS_TEST_DEVICE=cuda \
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_real_model_smoke -v
"""

import os
import unittest

import torch

from amaris_merge.merge_one_step import amaris_merge_one_step, mix_non_linear_parameters


@unittest.skipUnless(
    os.getenv("AMARIS_REAL_MODEL_SMOKE") == "1",
    "set AMARIS_REAL_MODEL_SMOKE=1 to load real checkpoints on a GPU",
)
class RealModelSmokeTests(unittest.TestCase):
    def test_projected_update_beta_mixing_and_forward_pass(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model1_dir = os.environ["AMARIS_MODEL1_DIR"]
        model2_dir = os.environ["AMARIS_MODEL2_DIR"]
        device = os.getenv("AMARIS_TEST_DEVICE", "cuda")
        beta = 0.25

        model1 = AutoModelForCausalLM.from_pretrained(
            model1_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model1_dir)

        layer_name = "model.layers.0.self_attn.q_proj"
        layer1 = model1.get_submodule(layer_name)
        layer2 = model2.get_submodule(layer_name)
        weight_before = layer1.weight.detach().clone()
        norm_before = model1.model.norm.weight.detach().clone()
        norm_specialist = model2.model.norm.weight.detach().clone()
        embedding_before = model1.model.embed_tokens.weight.detach().clone()
        embedding_specialist = model2.model.embed_tokens.weight.detach().clone()

        basis = torch.zeros((layer1.in_features, 1), device=device, dtype=torch.float32)
        basis[0, 0] = 1.0
        amaris_merge_one_step(model1, model2, {layer_name: basis})
        expected_weight = weight_before.float() + (
            layer2.weight.float() - weight_before.float()
        ) @ (basis @ basis.T)
        torch.testing.assert_close(
            layer1.weight.float(), expected_weight, rtol=0, atol=1e-2
        )

        mix_non_linear_parameters(model1, model2, beta)
        torch.testing.assert_close(
            model1.model.norm.weight.float(),
            beta * norm_before.float() + (1 - beta) * norm_specialist.float(),
            rtol=0,
            atol=1e-2,
        )
        torch.testing.assert_close(
            model1.model.embed_tokens.weight.float(),
            beta * embedding_before.float() + (1 - beta) * embedding_specialist.float(),
            rtol=0,
            atol=1e-2,
        )

        inputs = tokenizer("AMARIS merge smoke test.", return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model1(**inputs).logits
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
