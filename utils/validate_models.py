"""Validate source and AMARIS checkpoints against packed validation data."""

import argparse
import json
from pathlib import Path
from statistics import fmean

import torch
from transformers import AutoModelForCausalLM

from utils.val import validate


def parse_pairs(values: list[list[str]], kind: str) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for label, path in values:
        if label in labels:
            raise ValueError(f"Duplicate {kind} label: {label}")
        labels.add(label)
        resolved_path = Path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"{kind.capitalize()} path does not exist for {label}: {resolved_path}")
        pairs.append((label, resolved_path))
    return pairs


def load_model(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="flash_attention_2", device_map="auto"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute validation loss for multiple checkpoints.")
    parser.add_argument("--model", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--dataset", action="append", nargs=2, metavar=("TAG", "PACKED_PT"), required=True)
    parser.add_argument("--output", required=True, help="Path for the JSON report.")
    args = parser.parse_args()

    models = parse_pairs(args.model, "model")
    datasets = parse_pairs(args.dataset, "dataset")
    loaded_datasets: dict[str, list[dict[str, torch.Tensor]]] = {}
    for tag, path in datasets:
        print(f"Loading packed validation data for {tag}: {path}")
        loaded_datasets[tag] = torch.load(path, weights_only=True)
        print(f"Loaded {len(loaded_datasets[tag])} packed sequences for {tag}")

    report = {"datasets": {tag: str(path) for tag, path in datasets}, "models": {}}
    model_results = report["models"]
    for label, model_path in models:
        print(f"Loading model {label}: {model_path}")
        model = load_model(model_path)
        losses: dict[str, float] = {}
        for tag, _ in datasets:
            loss = validate(model, loaded_datasets[tag])
            losses[tag] = loss
            print(f"Validation loss for {label} on {tag}: {loss:.6f}")
        average_loss = fmean(losses.values())
        print(f"Average validation loss for {label}: {average_loss:.6f}")
        model_results[label] = {
            "path": str(model_path),
            "val_losses": losses,
            "average_val_loss": average_loss,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    amaris_results = {
        label: result
        for label, result in model_results.items()
        if label.startswith("amaris-alpha-")
    }
    if amaris_results:
        best_label = min(amaris_results, key=lambda label: amaris_results[label]["average_val_loss"])
        report["best_amaris_model"] = {"label": best_label, **amaris_results[best_label]}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Wrote validation losses to {output}")


if __name__ == "__main__":
    main()
