import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils import validate_models


ROOT = Path(__file__).resolve().parents[1]


class ValidationCommandTests(unittest.TestCase):
    def test_validation_cli_reports_dataset_means_and_best_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'val_packed.pt'
            torch.save([{'input_ids': torch.tensor([[1, 2]])}], data)
            base, merged = root / 'base', root / 'merged'
            base.mkdir()
            merged.mkdir()
            output = root / 'val-losses.json'
            argv = ['validate_models', '--model', 'base', str(base),
                    '--model', 'amaris-alpha-0.1-beta-0.50', str(merged),
                    '--dataset', 'core', str(data), '--dataset', 'specialist', str(data),
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(
                validate_models, 'load_model', side_effect=['base', 'merged']
            ), patch.object(validate_models, 'validate', side_effect=[2.0, 4.0, 1.0, 2.0]):
                validate_models.main()
            report = json.loads(output.read_text())
            self.assertEqual(report['models']['base']['average_val_loss'], 3.0)
            self.assertEqual(report['best_amaris_model']['average_val_loss'], 1.5)
            self.assertEqual(report['best_amaris_model']['label'], 'amaris-alpha-0.1-beta-0.50')

    def test_validation_cli_rejects_duplicate_dataset_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Duplicate dataset label'):
                validate_models.parse_pairs([['core', temp], ['core', temp]], 'dataset')


if __name__ == "__main__":
    unittest.main()
