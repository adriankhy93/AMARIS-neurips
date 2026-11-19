import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import DataCollatorWithFlattening

from data import pack, split
from data.pack import pack_sequences
from data.tokenize_dataset import Tokenize


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / 'tests' / 'sample_data'


class DeterministicChatTokenizer:
    """Small tokenizer stand-in that preserves chat boundaries for pipeline tests."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, truncation=False, max_length=None):
        ids = []
        for message in messages:
            role_id = {'system': 1, 'user': 2, 'assistant': 3, 'tool': 4}.get(message['role'], 5)
            ids.extend([role_id, len(message.get('content', '')) % 251 + 10])
        return ids[:max_length] if truncation and max_length is not None else ids


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


class DataPipelineTests(unittest.TestCase):
    def test_sample_data_tokenizes_splits_and_packs_without_losing_tokens(self):
        tokenizer = DeterministicChatTokenizer()
        source_path = SAMPLE_DATA / 'base.jsonl'
        tokenized, total_tokens = Tokenize(
            dataset_path=str(source_path),
            tokenizer=tokenizer,
            max_seq_len=32,
            max_num_tokens=10_000,
            num_proc=1,
        ).tokenize()

        self.assertEqual(len(tokenized), 10)
        self.assertEqual(total_tokens, sum(len(row['input_ids']) for row in tokenized))
        self.assertTrue(all(len(row['input_ids']) == len(row['assistant_mask']) for row in tokenized))
        self.assertTrue(any(any(row['assistant_mask']) for row in tokenized))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            tokenized_dir = output_dir / 'tokenized'
            tokenized_dir.mkdir()
            with (tokenized_dir / 'sample.jsonl').open('w', encoding='utf-8') as handle:
                for row in tokenized:
                    handle.write(json.dumps(row) + '\n')

            with patch.object(sys, 'argv', ['split.py', '--output_dir', str(output_dir), '--train_proportion', '0.5']):
                split.main()
            with patch.object(sys, 'argv', ['pack.py', '--output_dir', str(output_dir), '--max_seq_length', '4']):
                pack.main()

            split_rows = read_jsonl(output_dir / 'split_tokenized' / 'train_sample.jsonl')
            split_rows += read_jsonl(output_dir / 'split_tokenized' / 'val_sample.jsonl')
            packed_paths = sorted((output_dir / 'packed').glob('*_packed.pt'))
            packed = [chunk for path in packed_paths for chunk in torch.load(path, weights_only=True)]

            self.assertEqual(sum(len(row['input_ids']) for row in split_rows), sum(chunk['input_ids'].shape[1] for chunk in packed))
            for chunk in packed:
                self.assertLessEqual(chunk['input_ids'].shape[1], 4)
                self.assertEqual(chunk['input_ids'].shape, chunk['assistant_mask'].shape)

    def test_packing_can_truncate_document_overflow(self):
        seq_store = {'all_packed_sequences': [], 'current_seq': [], 'current_assistant_mask': []}
        pack_sequences(
            examples={'input_ids': [[1, 2, 3, 4, 5], [6, 7]], 'assistant_mask': [[0, 1, 1, 1, 1], [1, 0]]},
            collator=DataCollatorWithFlattening(),
            seq_store=seq_store,
            max_seq_length=4,
            truncate_documents=True,
        )
        if seq_store['current_seq']:
            collated = DataCollatorWithFlattening()(seq_store['current_seq'])
            collated['assistant_mask'] = torch.tensor(seq_store['current_assistant_mask']).unsqueeze(0)
            seq_store['all_packed_sequences'].append(collated)

        packed_ids = torch.cat([chunk['input_ids'][0] for chunk in seq_store['all_packed_sequences']]).tolist()
        self.assertEqual(packed_ids, [1, 2, 3, 4, 6, 7])

    def test_packing_preserves_overflow_after_a_full_chunk(self):
        seq_store = {'all_packed_sequences': [], 'current_seq': [], 'current_assistant_mask': []}
        pack_sequences(
            examples={'input_ids': [[1, 2, 3], [4, 5, 6]], 'assistant_mask': [[0, 1, 1], [1, 0, 0]]},
            collator=DataCollatorWithFlattening(),
            seq_store=seq_store,
            max_seq_length=4,
        )
        if seq_store['current_seq']:
            collated = DataCollatorWithFlattening()(seq_store['current_seq'])
            collated['assistant_mask'] = torch.tensor(seq_store['current_assistant_mask']).unsqueeze(0)
            seq_store['all_packed_sequences'].append(collated)

        packed_ids = torch.cat([chunk['input_ids'][0] for chunk in seq_store['all_packed_sequences']]).tolist()
        packed_mask = torch.cat([chunk['assistant_mask'][0] for chunk in seq_store['all_packed_sequences']]).tolist()
        self.assertEqual(packed_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(packed_mask, [0, 1, 1, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()
