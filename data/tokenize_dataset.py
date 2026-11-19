import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm
from utils.load_data import load_raw_dataset


def find_assistant_start_seq(tokenizer):
    messages_without_reply = [{"role": "user", "content": "Hello"}]

    ids_without_generation_prompt = tokenizer.apply_chat_template(
        messages_without_reply, tokenize=True, add_generation_prompt=False
    )
    ids_with_generation_prompt = tokenizer.apply_chat_template(
        messages_without_reply, tokenize=True, add_generation_prompt=True
    )
    ids_without_generation_prompt = _extract_input_ids(ids_without_generation_prompt)
    ids_with_generation_prompt = _extract_input_ids(ids_with_generation_prompt)
    assistant_start_idx = len(ids_without_generation_prompt)
    assistant_start_seq = ids_with_generation_prompt[assistant_start_idx:]
    return assistant_start_seq


def find_last_sublist(input_ids, sublist):
    """Find the last occurrence of sublist in input_ids and return its starting index."""
    for i in range(len(input_ids) - len(sublist), -1, -1):
        if input_ids[i : i + len(sublist)] == sublist:
            return i
    return -1  # Not found


def add_assistant_mask(tokenizer, data):
    print("Adding assistant mask to pre-tokenized dataset...")
    assistant_start_seq = find_assistant_start_seq(tokenizer)

    def _add_mask(sample):
        input_ids = sample["_ids"]
        assistant_mask = [0] * len(input_ids)
        start_idx = find_last_sublist(input_ids, assistant_start_seq)
        if start_idx == -1:
            sample["assistant_mask"] = assistant_mask
            return sample

        for i in range(start_idx, len(input_ids)):
            assistant_mask[i] = 1
        sample["assistant_mask"] = assistant_mask
        return sample

    return data.map(_add_mask)


def load_tokenizer(tokenizer_name_or_path: str):
    """Load tokenizer, falling back to slow tokenizer if fast backend is unavailable."""
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    except ValueError as exc:
        # Common when fast tokenizer conversion requires sentencepiece/tiktoken.
        if "Couldn't instantiate the backend tokenizer" in str(exc):
            print(
                "Fast tokenizer backend could not be instantiated; retrying with use_fast=False."
            )
            return AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                use_fast=False,
            )
        raise


def extract_tools(sample: dict) -> list[dict]:
    """Extract tool definitions from a sample, if present."""
    tools = []
    if "tools" in sample:
        tools = sample["tools"]
    elif "tool_calls" in sample:
        # Some datasets might have a different structure for tools
        tools = sample["tool_calls"]
    return tools


def extract_messages(sample: dict) -> list[dict[str, str]]:
    """Extract conversation messages up to the last user message.

    Supports common dataset formats: 'messages' (OpenAI/HF), 'conversations'
    (ShareGPT), and 'prompt' (single-turn).
    """
    messages = None

    if "messages" in sample:
        # print("Found 'messages' key, using it for conversation.")
        messages = sample["messages"]
    elif "conversations" in sample:
        # print("Found 'conversations' key, using it for conversation with role mapping.")
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": role_map.get(c["from"], c["from"]), "content": c["value"]}
            for c in sample["conversations"]
        ]
    elif "prompt" in sample:
        # print("Found 'prompt' key, treating it as a single user message.")
        messages = [{"role": "user", "content": sample["prompt"]}]
    else:
        raise ValueError(
            f"Cannot extract messages from sample with keys: {list(sample.keys())}"
        )

    # # Truncate after the last user message (drop trailing assistant turns)
    # last_user_idx = -1
    # for i, msg in enumerate(messages):
    #     if msg["role"] == "user":
    #         last_user_idx = i
    # if last_user_idx == -1:
    #     raise ValueError("No user message found in sample")
    # return messages[: last_user_idx + 1]
    return messages


def _normalize_assistant_thinking(messages: list[dict]) -> list[dict]:
    """Convert inline <think>...</think> blocks into reasoning_content when present.

    This avoids duplicated empty think wrappers in chat templates that already render
    reasoning sections for assistant messages.
    """
    normalized: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            normalized.append(msg)
            continue

        content = msg.get("content")
        if not isinstance(content, str):
            normalized.append(msg)
            continue

        stripped = content.lstrip()
        if not stripped.startswith("<think>"):
            normalized.append(msg)
            continue

        think_open = len("<think>")
        think_close = stripped.find("</think>", think_open)

        new_msg = dict(msg)
        if think_close == -1:
            # Handle truncated assistant thoughts with no explicit closing tag.
            new_msg["reasoning_content"] = stripped[think_open:].strip()
            new_msg["content"] = ""
        else:
            new_msg["reasoning_content"] = stripped[think_open:think_close].strip()
            new_msg["content"] = stripped[think_close + len("</think>") :].lstrip()

        normalized.append(new_msg)

    return normalized


def _extract_input_ids(tokenized_output):
    """Support both list[int] and BatchEncoding return types from chat templating."""
    if hasattr(tokenized_output, "input_ids"):
        return tokenized_output.input_ids
    return tokenized_output


class Tokenize:
    """Loads/Streams a dataset and tokenizes it to the max_num_tokens"""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        max_num_tokens: int,
        batch_size: int = 1000,
        num_proc: int = 32,
        streaming: bool = False,
    ) -> None:
        ds = load_raw_dataset(dataset_path, streaming=streaming, num_proc=num_proc)
        ds = ds.shuffle(seed=42)  # Shuffle the dataset to ensure a random split

        # Extracts text from the dataset, assuming the text is under "text" or "content" keys, or uses the entire sample as text if those keys are not present
        def _extract_text(sample):
            # if "input_ids" already exists, assume dataset is pre-tokenized and skip text extraction
            if "input_ids" in sample:
                return sample
            elif "text" in sample:
                return {"text": sample["text"]}
            else:
                return {
                    "messages": extract_messages(sample),
                    "tools": extract_tools(sample),
                }

        print("Extracting text...")
        ds = ds.map(_extract_text)

        # Tokenizes the text using the provided tokenizer, truncating to max_seq_len tokens
        def _tokenize(row):
            if "text" in row:
                return {
                    "_ids": tokenizer(
                        row["text"], truncation=True, max_length=max_seq_len
                    ).input_ids
                }
            elif "messages" in row:
                messages = _normalize_assistant_thinking(row["messages"])
                tools = row.get("tools", [])
                tokenized_chat = {
                    "_ids": _extract_input_ids(
                        tokenizer.apply_chat_template(
                            messages,
                            tools=tools,
                            tokenize=True,
                            truncation=True,
                            max_length=max_seq_len,
                        )
                    )
                }

                # to get the assistant mask, we tokenize full messages and messages without last assistant message, then take the difference to find which tokens belong to the last assistant message
                # print("messages: ", messages)
                # print("messages[-1]:", messages[-1])
                # STOP for debugging
                # raise ValueError("Stop for debugging")

                # if last message is from assistant, remove it for the second round of tokenization to get the assistant mask
                if messages[-1]["role"] == "assistant":
                    batch_messages_without_last_assistant = messages[:-1]
                else:
                    batch_messages_without_last_assistant = messages

                tokenized_chat_no_last_assistant = {
                    "_ids": _extract_input_ids(
                        tokenizer.apply_chat_template(
                            batch_messages_without_last_assistant,
                            tools=tools,
                            tokenize=True,
                            truncation=True,
                            max_length=max_seq_len,
                        )
                    )
                }

                ids = tokenized_chat["_ids"]
                ids_no_assistant = tokenized_chat_no_last_assistant["_ids"]

                assistant_mask = [0] * len(ids)
                for i in range(len(ids)):
                    if i >= len(ids_no_assistant):
                        assistant_mask[i] = 1

                tokenized_chat["assistant_mask"] = assistant_mask
                return tokenized_chat

        sample = next(iter(ds))

        # if "input_ids" already exists, assume dataset is pre-tokenized and skip text extraction
        if "input_ids" in sample.keys():
            print("Dataset already tokenized, skipping tokenization step.")
            ds = ds.rename_column("input_ids", "_ids")

            # find assistant_sequence in the dataset to create assistant_mask
            ds = add_assistant_mask(tokenizer, ds)

        else:
            print("Tokenizing text...")
            ds = ds.map(_tokenize, batched=False)

        self.ds = ds
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len

    def tokenize(self):
        # tokenizes
        tokenized_ds = []
        total_tokens = 0

        print("Tokenizing dataset...")

        pbar = tqdm(self.ds, desc="Tokenizing")
        for sample in pbar:
            ids = sample["_ids"]
            assistant_mask = sample.get("assistant_mask")
            if assistant_mask is None:
                assistant_mask = [0] * len(ids)

            num_tokens = len(ids)
            if (
                self.max_num_tokens is not None
                and total_tokens + num_tokens <= self.max_num_tokens
            ):
                tokenized_ds.append(
                    {"input_ids": ids, "assistant_mask": assistant_mask}
                )
                total_tokens += num_tokens
            else:
                remaining = self.max_num_tokens - total_tokens
                if remaining > 0:
                    tokenized_ds.append(
                        {
                            "input_ids": ids[:remaining],
                            "assistant_mask": assistant_mask[:remaining],
                        }
                    )
                break
            pbar.set_postfix(total_tokens=total_tokens)

        print("split done. total tokens: ", total_tokens)

        return tokenized_ds, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a text dataset into fixed-length token chunks and save as JSONL."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated list of dataset paths, optionally with tags (e.g. 'mydataset:/path/to/dataset.jsonl,mydataset2:/path/to/dataset2.jsonl'). If no tags are provided, they will be inferred from the filename.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer name or path",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Token chunk length (default: 4096)",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        required=True,
        help="Cap total tokens for both training and validation sets before splitting (default: use all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to load the dataset in streaming mode (default: False)",
    )

    args = parser.parse_args()

    if "," in args.datasets:
        # split into tag and path by comma, e.g. "mydataset:/path/to/dataset.jsonl,mydataset2:/path/to/dataset2.jsonl" -> [("mydataset", "/path/to/dataset.jsonl"), ("mydataset2", "/path/to/dataset2.jsonl")]
        datasets = []
        for ds in args.datasets.split(","):
            if ":" in ds:
                tag, dataset_path = ds.split(":")
            else:
                tag = ds.split("/")[-1].split(".")[0]
                dataset_path = ds
            datasets.append((tag, dataset_path))
        print(f"Found multiple datasets: {datasets}")
    else:
        datasets = []
        if ":" in args.datasets:
            tag, dataset_path = args.datasets.split(":")
            print("tag: ", tag)
            print("dataset_path: ", dataset_path)
        else:
            tag = args.datasets.split("/")[-1].split(".")[0]
            dataset_path = args.datasets
        datasets.append((tag, dataset_path))
        print(f"Found single dataset: {datasets[0]}")

    print("datasets: ", datasets)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_output_dir = output_path / "samples"
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)

    for tag, dataset_path in datasets:
        tokenize_ds = Tokenize(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            max_num_tokens=args.max_num_tokens,
            streaming=args.streaming,
        )

        # save out text sample to verify the tokenization looks correct (save in same folder as train_output_path)
        sample = next(iter(tokenize_ds.ds))
        sample_text = tokenizer.decode(sample["_ids"])
        # save out a sample to verify the tokenization looks correct (save in same folder as train_output_path)
        with open(sample_output_dir / f"sample_{tag}_tokenize_text.json", "w") as f:
            json.dump(sample_text, f, indent=4)

        tokenized_ds, total_tokens = tokenize_ds.tokenize()

        # save out tokenized sample to verify the tokenization looks correct (save in same folder as train_output_path)
        sample = tokenized_ds[0]
        # save out a sample to verify the tokenization looks correct (save in same folder as train_output_path)
        with open(sample_output_dir / f"sample_{tag}_tokenize_tokens.json", "w") as f:
            json.dump(sample, f, indent=4)

        # save out the tokenized dataset as JSONL
        print(f"Saving {len(tokenized_ds)} tokenized dataset to {output_path}...")
        tokenized_output_path = output_path / "tokenized" / f"{tag}.jsonl"
        tokenized_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tokenized_output_path, "w") as f:
            for i in range(len(tokenized_ds)):
                f.write(json.dumps(tokenized_ds[i]) + "\n")

    # Save metadata
    metadata_output_path = output_path / "metadata" / f"{tag}_tokenize_metadata.json"
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "datasets": args.datasets,
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "max_num_tokens": args.max_num_tokens,
    }

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
