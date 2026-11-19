import argparse
import json
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.iterable_dataset import IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.load_data import load_raw_dataset


def count_tokens(sample: dict) -> dict[str, int]:
    """Count the number of tokens in a sample based on the 'input_ids' field."""
    if "input_ids" not in sample:
        raise ValueError(
            "Sample does not contain 'input_ids' field for token counting."
        )

    return {"token_count": len(sample["input_ids"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a text dataset into fixed-length token chunks and save as JSONL."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--train_proportion",
        type=float,
        default=0.8,
        required=True,
        help="Proportion of the dataset to use for training (between 0 and 1)",
    )

    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset_dir = output_path / "tokenized"
    # get files in tokenized_dataset_dir
    tokenized_files = list(tokenized_dataset_dir.glob("*.jsonl"))
    for file in tokenized_files:
        print(f"Found tokenized dataset file: {file}")
        tag = file.stem

        tokenized_dataset_file = tokenized_dataset_dir / f"{tag}.jsonl"
        # load the tokenized dataset (expects each line to be a JSON object with an 'input_ids' field containing token ids)
        print(f"Loading tokenized dataset from {tokenized_dataset_file}...")

        ds = load_raw_dataset(str(tokenized_dataset_file), streaming=False)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        if isinstance(ds, IterableDataset):
            raise ValueError(
                "Expected non-streaming dataset for splitting, but got IterableDataset."
            )
        if not isinstance(ds, Dataset):
            raise TypeError(f"Unsupported dataset type: {type(ds)}")
        ds = ds.shuffle(seed=42)  # shuffle the dataset to ensure random train/val split
        ds = ds.map(count_tokens)

        # train-val split
        N = len(ds)
        train_size = int(N * args.train_proportion)
        print(
            f"Splitting dataset into {train_size} training samples and {N - train_size} validation samples..."
        )
        ds_train = ds.select(range(train_size))
        ds_val = ds.select(range(train_size, N))

        # save the training and validation sets as JSONL files
        train_output_path = output_path / "split_tokenized" / f"train_{tag}.jsonl"
        val_output_path = output_path / "split_tokenized" / f"val_{tag}.jsonl"

        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving training set to {train_output_path}...")
        with train_output_path.open("w", encoding="utf-8") as f:
            for item in tqdm(ds_train, desc="Saving training set"):
                json_line = json.dumps(item)
                f.write(json_line + "\n")

        print(f"Saving validation set to {val_output_path}...")
        with val_output_path.open("w", encoding="utf-8") as f:
            for item in tqdm(ds_val, desc="Saving validation set"):
                json_line = json.dumps(item)
                f.write(json_line + "\n")

        num_train_tokens = sum(ds_train["token_count"])
        num_val_tokens = sum(ds_val["token_count"])

        # Save metadata
        metadata_output_path = output_path / "metadata" / f"{tag}_split_metadata.json"
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset": str(tokenized_dataset_file),
            "num_train_tokens": num_train_tokens,
            "num_val_tokens": num_val_tokens,
            "total_tokens": num_train_tokens + num_val_tokens,
        }
        with open(metadata_output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
