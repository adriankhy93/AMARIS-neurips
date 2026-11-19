"""Collect per-capability activation covariances with forward hooks.

For every selected linear layer, this accumulates the normalized input
covariance ``X^T X / N`` over calibration activations.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

from utils.layers import get_linear_layers, layer_name_to_filename
from utils.gpu_memory import peak_gpu_memory_bytes, reset_peak_gpu_memory


def load_packed(path: str) -> list[dict[str, Any]]:
    """Load packed data from a torch file produced by generate.py."""
    return torch.load(path)


def collect_activations(args: argparse.Namespace) -> None:
    """Main logic: collect and persist per-layer activation covariances."""
    overall_started = time.perf_counter()
    # Include model loading and all covariance chunks in this process-stage peak.
    reset_peak_gpu_memory()

    cov_dtype_map = {"float32": torch.float32, "float64": torch.float64}
    cov_dtype = cov_dtype_map[args.cov_dtype]

    model_load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model_load_seconds = time.perf_counter() - model_load_started

    model.eval()

    linear_layers = get_linear_layers(model, args.layers)
    ignored_layers = {name.strip() for name in args.layers_to_ignore.split(",") if name.strip()}
    linear_layers = {name: layer for name, layer in linear_layers.items() if name not in ignored_layers}
    layer_names = list(linear_layers.keys())
    if not layer_names:
        raise ValueError("No linear layers remain after --layers_to_ignore filtering.")

    output_path = Path(args.output_dir)
    packed_dir = output_path / "packed"

    rank_info: dict[str, dict[str, dict]] = {}
    # Layer chunking: split layers into num_chunks groups so that
    # activation memory is balanced across GPUs.
    # Group layers by device, preserving order

    device_layers: dict[torch.device, list[str]] = defaultdict(list)
    for lname in layer_names:
        # Assume all parameters of a layer are on the same device, so just check the weight
        dev = linear_layers[lname].weight.device
        device_layers[dev].append(lname)

    # Split each device's layers into num_chunks roughly-equal parts
    def split_even(lst: list, n: int) -> list[list]:
        k, r = divmod(len(lst), n)
        parts: list[list] = []
        idx = 0
        for i in range(n):
            size = k + (1 if i < r else 0)
            parts.append(lst[idx : idx + size])
            idx += size
        return parts

    per_device_chunks = {
        dev: split_even(lnames, args.num_chunks)
        for dev, lnames in device_layers.items()
    }

    # Chunk i = union of chunk-i from every device
    layer_groups = []
    for ci in range(args.num_chunks):
        group: list[str] = []
        for dev in device_layers:
            group.extend(per_device_chunks[dev][ci])
        if group:
            layer_groups.append(group)

    # Process each capability

    print(f"Processing capability: {args.tag}")
    packed_file = packed_dir / f"train_{args.tag}_packed.pt"

    # Accumulate covariances in a forward-only pass.
    packed_load_started = time.perf_counter()
    samples = load_packed(str(packed_file))
    packed_load_seconds = time.perf_counter() - packed_load_started
    covariance_started = time.perf_counter()

    # Forward hooks consume each layer's inputs immediately.

    def make_fwd_hook(name: str) -> Any:
        def hook_fn(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            X = inp[0].detach()  # [batch, seq, d_in]

            # Accelerate can move inputs inside forward; PyTorch hooks still
            # receive the original inputs, possibly on another device.
            X = X.to(device=input_covariances[name].device, dtype=cov_dtype)
            X = X.reshape(-1, X.shape[-1])

            N = X.shape[0]
            # X = X / (
            # N**0.5
            # )  # normalize by number of tokens so that each sample gives equal weight to the covariance, regardless of sequence length
            num_tokens[name] += N
            input_covariances[name].addmm_(X.t(), X)

        return hook_fn

    embedding_layer_device = next(model.parameters()).device
    max_num_samples = min(args.num_samples, len(samples))

    for group in tqdm(layer_groups, desc="chunks", leave=False):
        input_covariances: dict[str, torch.Tensor] = {}
        num_tokens: dict[str, int] = {}
        handles = []

        for lname in group:
            # Initialize covariance accumulators on each layer's device (GPU)
            d_in = linear_layers[lname].in_features
            layer_device = linear_layers[lname].weight.device
            input_covariances[lname] = torch.zeros(
                d_in, d_in, dtype=cov_dtype, device=layer_device
            )
            num_tokens[lname] = 0

            m = linear_layers[lname]
            handles.append(m.register_forward_hook(make_fwd_hook(lname)))

        for i in tqdm(
            range(max_num_samples),
            desc=f"Activation covariances for: {args.tag} (chunk {layer_groups.index(group)+1}/{len(layer_groups)})",
            leave=False,
        ):
            batch = samples[i]
            input_ids = cast(torch.Tensor, batch["input_ids"]).to(
                embedding_layer_device
            )  # shape (1, seq_len)
            position_ids = cast(torch.Tensor, batch["position_ids"]).to(
                embedding_layer_device
            )  # shape (1, seq_len)
            assistant_mask = cast(torch.Tensor, batch["assistant_mask"]).to(
                embedding_layer_device
            )  # shape (1, seq_len)
            labels = cast(torch.Tensor, batch["labels"]).to(
                embedding_layer_device
            )  # shape (1, seq_len)

            # labels[:, assistant_mask[0] != 1] = -100

            with torch.inference_mode():
                outputs = model(input_ids=input_ids, position_ids=position_ids)

        for h in handles:
            h.remove()

        for lname in group:

            C_input = input_covariances[lname]
            C_input /= num_tokens[lname]

            input_cov_save_path = (
                output_path
                / f"cov_{args.tag}"
                / f"{layer_name_to_filename(lname)}_input_cov.pt"
            )
            input_cov_save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the normalized covariance on CPU; the solver handles
            # numerical re-symmetrization before decomposition.
            torch.save(C_input.cpu().float(), input_cov_save_path)

        input_covariances.clear()
        torch.cuda.empty_cache()

        # Release this chunk before processing the next layer group.

    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
    covariance_seconds = time.perf_counter() - covariance_started
    metadata = {
        "model": args.model,
        "cov_dtype": args.cov_dtype,
        "layers": args.layers,
        "layers_to_ignore": sorted(ignored_layers),
        "layer_names": layer_names,
        "tag": args.tag,
        "rank_info": rank_info,
        "timings_seconds": {
            "model_load": model_load_seconds,
            "packed_data_load": packed_load_seconds,
            "covariance_collection_and_save": covariance_seconds,
            "total": time.perf_counter() - overall_started,
        },
        "gpu_memory_bytes": {
            "scope": "Peak PyTorch allocator memory for covariance collection, including model loading.",
            "per_device": peak_gpu_memory_bytes(),
        },
    }
    metadata_save_path = output_path / "metadata" / f"{args.tag}_covariances_metadata.json"
    metadata_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved to {metadata_save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-capability activation covariances and eigendecompose."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save covariance matrices and load all the processed data",
    )
    parser.add_argument(
        "--cov_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Dtype for covariance accumulation (default: float32)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        choices=["all", "attn", "mlp"],
        help="Which linear layers to target: all, attn, or mlp (default: all)",
    )
    parser.add_argument(
        "--layers_to_ignore",
        type=str,
        default="",
        help="Comma-separated linear module names to exclude from covariance collection.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples for forward passes (default: 10000)",
    )

    parser.add_argument(
        "--num_chunks",
        type=int,
        default=2,
        help="Split layers into N chunks for forward passes, balanced across GPUs (1=all layers at once)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="tag identifying the capability being processed, e.g. 'math' or 'code'",
    )
    args = parser.parse_args()

    collect_activations(args)


if __name__ == "__main__":
    main()
