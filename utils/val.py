import torch
from tqdm import tqdm

import argparse
from typing import cast
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def validate(model, data):
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    num_samples = len(data)
    total_num_tokens = 0

    # Some packed validation sets contain no assistant turns. In that case,
    # score their existing labels rather than discarding the entire set.
    use_assistant_mask = any(
        (
            (cast(torch.Tensor, batch["labels"]) != -100)
            & (cast(torch.Tensor, batch["assistant_mask"]) == 1)
        ).any().item()
        for batch in data
    )
    if not use_assistant_mask:
        print("No supervised assistant tokens found; validating without assistant mask.")

    with torch.no_grad():
        for i in tqdm(
            range(num_samples),
            desc="Validating model",
            leave=False,
        ):
            batch = data[i]
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(
                device
            )  # shape (1, seq_len)
            position_ids = cast(torch.Tensor, batch["position_ids"]).to(
                device
            )  # shape (1, seq_len)
            assistant_mask = cast(torch.Tensor, batch["assistant_mask"]).to(
                device
            )  # shape (1, seq_len)
            labels = cast(torch.Tensor, batch["labels"]).to(
                device
            )  # shape (1, seq_len)

            if use_assistant_mask:
                # Mask out the loss for non-assistant tokens by setting them to -100.
                labels[:, assistant_mask[0] != 1] = -100

            num_tokens = (labels != -100).sum().item()
            if num_tokens == 0:
                continue  # skip this sample if it has no supervised tokens
            outputs = model(
                input_ids=input_ids, position_ids=position_ids, labels=labels
            )
            loss = outputs.loss

            total_loss += loss.item() * num_tokens
            total_num_tokens += num_tokens

            # print("loss: ", loss.item())

    if total_num_tokens == 0:
        raise ValueError("Validation set produced zero supervised tokens")

    avg_loss = total_loss / total_num_tokens
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Validate a model on a dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint to validate",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        required=True,
        help="comma-separated paths to the .pt files containing the validation data (each file should contain a list of dictionaries with keys 'input_ids', 'position_ids', 'assistant_mask', and 'labels')",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="validation_results.json",
        help="Path to save the validation results JSON file",
    )
    args = parser.parse_args()

    # Load model and data
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, attn_implementation="flash_attention_2", device_map="auto"
    )

    results = {}

    # load packed .pt data
    data_paths = args.data_paths.split(",")
    for path in data_paths:
        print(f"Loading data from {path}...")
        data = torch.load(path)
        print(f"Loaded {len(data)} samples from {path}")

        val_loss = validate(model, data)
        print(f"Validation loss for {path}: {val_loss:.4f}")

        results[path] = val_loss

    model_name = args.model_path.split("/")[-1]
    output_path = Path(args.output_path)
    output_path = output_path / f"{model_name}_validation_results-assistant-mask.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save results to a json file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
