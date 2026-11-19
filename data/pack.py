import argparse
import json
from pathlib import Path
from typing import Any, cast

import datasets
import torch
from datasets import disable_caching
from transformers import DataCollatorWithFlattening
from utils.load_data import load_raw_dataset

disable_caching()


def pack_sequences(
    examples,
    collator,
    seq_store,
    max_seq_length=4096,
    truncate_documents=False,
):
    """
    Pack sequences into fixed-length batches.

    Args:
        examples: Batch of examples with 'input_ids'
        collator: DataCollatorWithFlattening instance
        seq_store: Dictionary to store packed sequences
        max_seq_length: Maximum sequence length (default: 4096)
        truncate_documents: When true, discard tokens after ``max_seq_length``
            within each source document instead of carrying them into a later
            packed sequence.

    Returns:
        None (updates seq_store in place)
    """
    batch_seq = []
    seq = seq_store["current_seq"]
    assistant_mask_seq = seq_store["current_assistant_mask"]

    for ids, assistant_mask in zip(examples["input_ids"], examples["assistant_mask"]):
        if len(ids) != len(assistant_mask):
            raise ValueError("input_ids and assistant_mask must have the same length.")
        if truncate_documents:
            ids = ids[:max_seq_length]
            assistant_mask = assistant_mask[:max_seq_length]

        offset = 0
        while offset < len(ids):
            remaining_capacity = max_seq_length - len(assistant_mask_seq)
            take = min(remaining_capacity, len(ids) - offset)
            seq.append({"input_ids": ids[offset : offset + take]})
            assistant_mask_seq.extend(assistant_mask[offset : offset + take])
            offset += take

            if len(assistant_mask_seq) == max_seq_length:
                collated_seq = collator(seq)
                collated_seq["assistant_mask"] = torch.tensor(assistant_mask_seq).unsqueeze(0)
                batch_seq.append(collated_seq)
                seq = []
                assistant_mask_seq = []

    seq_store["current_seq"] = seq
    seq_store["current_assistant_mask"] = assistant_mask_seq
    seq_store["all_packed_sequences"].extend(batch_seq)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Pack tokenized sequences into fixed-length batches"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for packed sequences",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length for packing",
    )
    parser.add_argument(
        "--truncate_documents",
        action="store_true",
        help=(
            "Discard tokens after --max_seq_length in each source document "
            "instead of preserving overflow in later packed sequences."
        ),
    )

    args = parser.parse_args()

    # Initialize collator
    collator = DataCollatorWithFlattening()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_tokenized_dataset_dir = output_path / "split_tokenized"
    split_tokenized_files = list(split_tokenized_dataset_dir.glob("*.jsonl"))
    for file in split_tokenized_files:
        print(f"Found split tokenized dataset file: {file}")
        tag = file.stem

        split_tokenized_dataset_file = split_tokenized_dataset_dir / f"{tag}.jsonl"
        print(f"Loading split tokenized dataset from {split_tokenized_dataset_file}...")
        ds = load_raw_dataset(str(split_tokenized_dataset_file), streaming=False)
        print(f"Dataset loaded: {len(ds)} examples")

        # Initialize sequence store
        seq_store = {
            "all_packed_sequences": [],
            "current_seq": [],
            "current_assistant_mask": [],
        }

        # Pack sequences
        print(f"Packing sequences for tag: {tag}...")
        ds.map(
            lambda x: pack_sequences(
                x,
                collator,
                seq_store,
                args.max_seq_length,
                args.truncate_documents,
            ),
            batched=True,
            batch_size=1000,
            remove_columns=list(ds.column_names),
        )

        # Flush only once after all map batches are processed to avoid
        # duplicating partial sequences across callback invocations.
        if seq_store["current_seq"]:
            collated_seq = cast(dict[str, Any], collator(seq_store["current_seq"]))
            collated_seq["assistant_mask"] = torch.tensor(
                seq_store["current_assistant_mask"]
            ).unsqueeze(0)
            seq_store["all_packed_sequences"].append(collated_seq)
            seq_store["current_seq"] = []
            seq_store["current_assistant_mask"] = []

        # save sample packed sequence for inspection
        if seq_store["all_packed_sequences"]:
            sample_output_path = output_path / "samples" / f"sample_paked_{tag}.json"
            sample_output_path.parent.mkdir(parents=True, exist_ok=True)

            sample = seq_store["all_packed_sequences"][0]
            torch.save(sample, sample_output_path)
            print(f"Sample packed sequence saved to {sample_output_path}")

        # Save packed sequences
        output_path_tag = output_path / "packed" / f"{tag}_packed.pt"
        output_path_tag.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"Saving {len(seq_store['all_packed_sequences'])} packed sequences to {output_path_tag}"
        )
        torch.save(seq_store["all_packed_sequences"], output_path_tag)
        print("Done!")

        # save metadata
        metadata = {
            "dataset": str(split_tokenized_dataset_file),
            "num_packed_sequences": len(seq_store["all_packed_sequences"]),
            "num_total_tokens": sum(
                [seq["input_ids"].shape[1] for seq in seq_store["all_packed_sequences"]]
            ),
            "truncate_documents": args.truncate_documents,
        }

        metadata_output_path = output_path / "metadata" / f"{tag}_pack_metadata.json"
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_output_path}")


if __name__ == "__main__":
    main()
