import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook, add_hook_to_module

from amaris_merge.compute_covariances import collect_activations


class TinyDispatchedModel(nn.Module):
    def __init__(self, input_device, head_device):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, device=input_device))
        self.lm_head = nn.Linear(2, 3, bias=False, device=head_device)
        add_hook_to_module(self.lm_head, AlignDevicesHook(execution_device=head_device))

    def forward(self, input_ids, position_ids):
        return self.lm_head(input_ids.float())


class ComputeCovariancesTests(unittest.TestCase):
    def check_collection(self, input_device, head_device):
        model = TinyDispatchedModel(input_device, head_device)
        tokens = torch.tensor([[1, 2], [3, 4]])
        sample = {key: tokens for key in (
            'input_ids', 'position_ids', 'assistant_mask', 'labels'
        )}
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                model='tiny', cov_dtype='float32', layers='all',
                layers_to_ignore='', output_dir=tmp, num_chunks=1,
                tag='test', num_samples=1,
            )
            with patch('amaris_merge.compute_covariances.AutoModelForCausalLM.from_pretrained', return_value=model), patch(
                'amaris_merge.compute_covariances.load_packed', return_value=[sample]
            ):
                collect_activations(args)
            covariance = torch.load(Path(tmp) / 'cov_test' / 'lm_head_input_cov.pt')
        expected = tokens.float().T @ tokens.float() / tokens.shape[0]
        torch.testing.assert_close(covariance, expected)

    def test_cpu_covariance(self):
        self.check_collection('cpu', 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_dispatched_head_receives_input_from_another_device(self):
        self.check_collection('cpu', 'cuda:0')

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two GPUs required')
    def test_dispatched_head_across_gpus(self):
        self.check_collection('cuda:0', 'cuda:1')
