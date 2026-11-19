import os
import json
from pathlib import Path


from datasets import load_dataset, load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.iterable_dataset import IterableDataset


def _normalize_mixed_primitive_lists(value):
    """Recursively normalize lists that mix primitive types.

    Arrow requires stable types per column. Some tool schemas contain arrays
    like [1, "1"] which fail conversion. For such mixed primitive lists, cast
    primitive elements to strings while preserving nested structures.
    """
    if isinstance(value, dict):
        return {k: _normalize_mixed_primitive_lists(v) for k, v in value.items()}

    if isinstance(value, list):
        normalized = [_normalize_mixed_primitive_lists(v) for v in value]

        primitive_types = {
            type(v)
            for v in normalized
            if v is not None and isinstance(v, (str, int, float, bool))
        }
        if len(primitive_types) > 1:
            return [str(v) if v is not None else None for v in normalized]
        return normalized

    return value


def _load_jsonl_list_rows(path: str) -> Dataset:
    """Load JSONL files where each line may be a list of chat messages.

    The default HF json loader expects object-like rows with stable columns.
    Some calibration datasets store each line as a JSON array of messages, so
    we normalize those to {"messages": [...]} rows.
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e

            parsed = _normalize_mixed_primitive_lists(parsed)

            if isinstance(parsed, dict):
                rows.append(parsed)
            elif isinstance(parsed, list):
                rows.append({"messages": parsed})
            else:
                raise ValueError(
                    f"Unsupported JSON row type on line {i} in {path}: {type(parsed).__name__}"
                )

    if not rows:
        raise ValueError(f"No rows found in JSONL file: {path}")

    try:
        return Dataset.from_list(rows)
    except Exception:
        # Final fallback: keep each row as a raw JSON string to avoid Arrow
        # schema inference issues on highly heterogeneous nested structures.
        return Dataset.from_dict(
            {"__raw_json_row__": [json.dumps(row, ensure_ascii=False) for row in rows]}
        )


def load_raw_dataset(
    path: str, streaming: bool = False, num_proc: int = 32
) -> IterableDataset | Dataset | DatasetDict:
    # Single file cases
    if path.endswith(".jsonl") or path.endswith(".json"):
        try:
            return load_dataset(
                "json", data_files=path, split="train", streaming=streaming
            )
        except Exception:
            # Fallback for JSONL files whose rows are arrays (e.g. chat messages).
            if path.endswith(".jsonl") and not streaming:
                return _load_jsonl_list_rows(path)
            raise
    elif path.endswith(".parquet"):
        return load_dataset(
            "parquet", data_files=path, split="train", streaming=streaming
        )
    elif path.endswith(".arrow"):
        return load_dataset(
            "arrow", data_files=path, split="train", streaming=streaming
        )

    # Directory cases
    elif os.path.isdir(path):
        path_obj = Path(path)

        # Check if it's a HuggingFace dataset directory (has dataset_info.json)
        if (path_obj / "dataset_info.json").exists():
            return load_from_disk(path)

        # Check for parquet files in directory
        elif any(path_obj.glob("*.parquet")):
            return load_dataset(
                "parquet",
                data_files=str(path_obj / "*.parquet"),
                split="train",
                streaming=streaming,
            )

        # Check for arrow files in directory
        elif any(path_obj.glob("*.arrow")):
            return load_dataset(
                "arrow",
                data_files=str(path_obj / "*.arrow"),
                split="train",
                streaming=streaming,
            )

        # Check for jsonl/json files in directory
        elif any(path_obj.glob("*.jsonl")) or any(path_obj.glob("*.json")):
            json_files = list(path_obj.glob("*.jsonl")) + list(path_obj.glob("*.json"))
            return load_dataset(
                "json",
                data_files=[str(f) for f in json_files],
                split="train",
                streaming=streaming,
            )

        else:
            raise ValueError(
                f"Directory {path} does not contain supported file formats (.json, .jsonl, .parquet, .arrow) or dataset_info.json"
            )

    # Assume it's a HuggingFace Hub dataset name
    else:
        return load_dataset(path, split="train", streaming=streaming)
