import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from utils.checkpoints import SafeTensorCheckpoint
from utils.layers import get_linear_layers, layer_name_to_filename


TARGET_SUFFIXES = {
    "all": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "up_proj.weight", "down_proj.weight", "gate_proj.weight", "lm_head.weight"),
    "attn": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"),
    "mlp": ("up_proj.weight", "down_proj.weight", "gate_proj.weight", "lm_head.weight"),
}


def save_deltas_from_checkpoints(
    model1_dir: str | Path,
    model2_dir: str | Path,
    output_dir: Path,
    layers: str = "all",
    save_gram: bool = False,
    layers_to_ignore: str | None = None,
) -> None:
    """Stream target linear weights from safetensors checkpoints and save W2-W1."""
    checkpoint1 = SafeTensorCheckpoint(model1_dir)
    checkpoint2 = SafeTensorCheckpoint(model2_dir)
    suffixes = TARGET_SUFFIXES[layers]
    ignored_layers = {name.strip() for name in (layers_to_ignore or "").split(",") if name.strip()}
    names1 = {name for name in checkpoint1.keys if name.endswith(suffixes) and name.removesuffix(".weight") not in ignored_layers}
    names2 = {name for name in checkpoint2.keys if name.endswith(suffixes) and name.removesuffix(".weight") not in ignored_layers}
    if names1 != names2:
        raise ValueError("Checkpoint target linear-weight sets do not match.")

    delta_dir = output_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, float | int | list[int]]] = {}
    for parameter_name in tqdm(sorted(names1), desc="Streaming/saving deltas"):
        delta = (checkpoint2.get_tensor(parameter_name).float() - checkpoint1.get_tensor(parameter_name).float()).contiguous()
        layer_name = parameter_name.removesuffix(".weight")
        fname = layer_name_to_filename(layer_name)
        torch.save(delta, delta_dir / f"{fname}_delta.pt")
        info: dict[str, float | int | list[int]] = {
            "shape": list(delta.shape),
            "fro_norm": float(torch.linalg.norm(delta).item()),
        }
        if save_gram:
            gram = (delta.T @ delta).contiguous()
            torch.save(gram, delta_dir / f"{fname}_gram.pt")
            info["gram_shape"] = list(gram.shape)
        metadata[layer_name] = info

    meta_path = output_dir / "metadata" / "delta_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"num_layers": len(metadata), "layers": metadata, "save_gram": save_gram, "layers_to_ignore": sorted(ignored_layers), "streamed": True}, indent=2), encoding="utf-8")



def save_deltas(
    model1: nn.Module,
    model2: nn.Module,
    output_dir: Path,
    layers: str = "all",
    save_gram: bool = False,
    layers_to_ignore: str | None = None,
) -> None:
    linear1 = get_linear_layers(model1, layers)
    linear2 = get_linear_layers(model2, layers)
    ignored_layers = {name.strip() for name in (layers_to_ignore or "").split(",") if name.strip()}
    linear1 = {name: layer for name, layer in linear1.items() if name not in ignored_layers}
    linear2 = {name: layer for name, layer in linear2.items() if name not in ignored_layers}

    names1 = set(linear1.keys())
    names2 = set(linear2.keys())
    if names1 != names2:
        only1 = sorted(list(names1 - names2))[:10]
        only2 = sorted(list(names2 - names1))[:10]
        raise ValueError(
            "Model layer sets do not match. "
            f"Only in model1 (sample): {only1} | Only in model2 (sample): {only2}"
        )

    delta_dir = output_dir / "delta"
    delta_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, float | int | list[int]]] = {}

    # We keep saves on CPU to avoid GPU memory pressure.
    def _compute_and_save(lname: str) -> tuple[str, dict[str, float | int | list[int]]]:
        w1 = linear1[lname].weight.detach().float().cpu()
        w2 = linear2[lname].weight.detach().float().cpu()
        delta = (w2 - w1).contiguous()

        fname = layer_name_to_filename(lname)
        delta_path = delta_dir / f"{fname}_delta.pt"
        torch.save(delta, delta_path)

        info: dict[str, float | int | list[int]] = {
            "shape": list(delta.shape),
            "fro_norm": float(torch.linalg.norm(delta).item()),
        }

        if save_gram:
            g = (delta.T @ delta).contiguous()
            gram_path = delta_dir / f"{fname}_gram.pt"
            torch.save(g, gram_path)
            info["gram_shape"] = list(g.shape)

        return lname, info

    with ThreadPoolExecutor() as executor:
        for lname, info in tqdm(
            executor.map(_compute_and_save, sorted(names1)),
            total=len(names1),
            desc="Computing/saving deltas",
        ):
            metadata[lname] = info

    meta_path = output_dir / "metadata" / "delta_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_layers": len(metadata),
                "layers": metadata,
                "save_gram": save_gram,
                "layers_to_ignore": sorted(ignored_layers),
            },
            f,
            indent=2,
        )
    print(f"Saved deltas to {delta_dir}")
    print(f"Saved delta metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute per-layer deltas W2-W1.")
    parser.add_argument("--model1_dir", type=str, required=True)
    parser.add_argument("--model2_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        choices=["all", "attn", "mlp"],
        help="Layer subset consistent with utils.layers.get_linear_layers",
    )
    parser.add_argument(
        "--layers_to_ignore",
        type=str,
        default="",
        help="Comma-separated linear module names to exclude from delta computation.",
    )
    parser.add_argument(
        "--save_gram",
        action="store_true",
        help="Also save per-layer G=Delta^T Delta tensors as *_gram.pt",
    )
    args = parser.parse_args()

    if Path(args.model1_dir).resolve() == Path(args.model2_dir).resolve():
        raise ValueError("--model1_dir and --model2_dir are the same path.")

    save_deltas_from_checkpoints(
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir,
        output_dir=Path(args.output_dir),
        layers=args.layers,
        save_gram=args.save_gram,
        layers_to_ignore=args.layers_to_ignore,
    )


if __name__ == "__main__":
    main()
