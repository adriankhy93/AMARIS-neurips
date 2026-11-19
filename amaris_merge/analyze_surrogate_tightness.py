"""Measure AMARIS output-surrogate tightness on held-out packed examples.

The analysis evaluates the exact linearized output deviation of a saved AMARIS
projector, ``||X P Delta^T||_F^2 / N``, and its spectral-norm upper bound,
``||X P||_F^2 ||Delta||_2^2 / N``.  Activations are collected with forward
hooks from the *base* model on a held-out packed dataset; no merge or benchmark
evaluation is rerun.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# Support both ``python -m amaris_merge.analyze_surrogate_tightness`` and direct
# invocation as ``python amaris_merge/analyze_surrogate_tightness.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from utils.layers import get_linear_layers, layer_name_to_filename


def _json_number(value: float) -> float | None:
    """Convert non-finite values to JSON null rather than invalid JSON."""
    return value if math.isfinite(value) else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def spectral_norm_squared(
    delta: torch.Tensor,
    method: str,
    power_iterations: int,
) -> tuple[float, bool]:
    """Return ||delta||_2^2 and whether it is a numerical upper bound.

    ``svd`` is the faithful spectral-norm bound. ``power`` is much cheaper for
    large matrices, but returns an estimate (normally a lower estimate), so its
    resulting ratio must not be described as a rigorous bound-tightness ratio.
    """
    if method == "svd":
        sigma = torch.linalg.matrix_norm(delta, ord=2)
        return float((sigma * sigma).item()), True

    generator = torch.Generator(device=delta.device)
    generator.manual_seed(0)
    vector = torch.randn(
        delta.shape[1], device=delta.device, dtype=delta.dtype, generator=generator
    )
    vector /= torch.linalg.vector_norm(vector).clamp_min(torch.finfo(delta.dtype).eps)
    for _ in range(power_iterations):
        left = delta @ vector
        vector = delta.T @ left
        vector /= torch.linalg.vector_norm(vector).clamp_min(
            torch.finfo(delta.dtype).eps
        )
    sigma_squared = torch.linalg.vector_norm(delta @ vector).square()
    return float(sigma_squared.item()), False


def batch_metrics(
    activations: torch.Tensor,
    basis: torch.Tensor,
    delta_basis: torch.Tensor,
    spectral_norm_squared_value: float,
) -> tuple[float, float, float, int]:
    """Compute exact deviation and spectral bound contribution for one batch.

    If ``Z = XU`` and ``D_U = Delta U``, then
    ``||X P Delta^T||_F^2 = tr(Z^T Z D_U^T D_U)``.  This avoids constructing
    either the full projector P or the potentially enormous output matrix.
    """
    x = activations.detach().reshape(-1, activations.shape[-1]).float()
    z = x @ basis
    projected_input_energy = z.square().sum()
    delta_gram_in_basis = delta_basis.T @ delta_basis
    exact_deviation = ((z @ delta_gram_in_basis) * z).sum()
    spectral_bound = projected_input_energy * spectral_norm_squared_value
    return (
        float(exact_deviation.item()),
        float(spectral_bound.item()),
        float(projected_input_energy.item()),
        int(z.shape[0]),
    )


def _load_layer_state(
    layer_name: str,
    layer: nn.Linear,
    delta_dir: Path,
    basis_dir: Path,
    spectral_norm_method: str,
    power_iterations: int,
) -> dict[str, Any]:
    filename = layer_name_to_filename(layer_name)
    basis_path = basis_dir / f"{filename}_input_basis.pt"
    delta_path = delta_dir / f"{filename}_delta.pt"
    if not basis_path.exists() or not delta_path.exists():
        raise FileNotFoundError(
            f"Missing matched basis or delta for {layer_name}: {basis_path}, {delta_path}"
        )
    device = layer.weight.device
    basis = (
        torch.load(basis_path, map_location=device, weights_only=True)
        .float()
        .contiguous()
    )
    delta = (
        torch.load(delta_path, map_location=device, weights_only=True)
        .float()
        .contiguous()
    )
    if basis.ndim != 2 or delta.ndim != 2 or basis.shape[0] != delta.shape[1]:
        raise ValueError(
            f"Incompatible shapes for {layer_name}: basis={tuple(basis.shape)}, delta={tuple(delta.shape)}"
        )
    if basis.shape[1] == 0:
        # The exact deviation and the upper bound are both identically zero.
        return {
            "basis": basis,
            "delta_basis": delta[:, :0],
            "sigma_sq": 0.0,
            "bound_is_rigorous": True,
        }
    sigma_sq, rigorous = spectral_norm_squared(
        delta, spectral_norm_method, power_iterations
    )
    delta_basis = (delta @ basis).contiguous()
    del delta
    return {
        "basis": basis,
        "delta_basis": delta_basis,
        "sigma_sq": sigma_sq,
        "bound_is_rigorous": rigorous,
    }


def gpu_memory_snapshot(reset_peak: bool = False) -> list[dict[str, int]]:
    """Return per-device allocator memory statistics in bytes.

    With ``reset_peak=True``, reset PyTorch's peak counters after recording the
    model-resident baseline. The later snapshot then measures the incremental
    peak of this analysis (basis/delta loading, spectral norm, and forwards).
    """
    snapshots = []
    for index in range(torch.cuda.device_count()):
        snapshots.append(
            {
                "device_index": index,
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(index)
    return snapshots


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    val_pack_path = Path(args.val_pack_path)
    delta_dir = Path(args.delta_dir)
    basis_dir = Path(args.basis_dir)
    for path, description in (
        (val_pack_path, "validation pack"),
        (delta_dir, "delta directory"),
        (basis_dir, "basis directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} does not exist: {path}")

    samples = torch.load(val_pack_path, map_location="cpu", weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty packed list in {val_pack_path}")
    max_samples = (
        min(args.max_samples, len(samples)) if args.max_samples else len(samples)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    linear_layers = get_linear_layers(model, args.layers)
    available = {
        name: layer
        for name, layer in linear_layers.items()
        if (basis_dir / f"{layer_name_to_filename(name)}_input_basis.pt").exists()
        and (delta_dir / f"{layer_name_to_filename(name)}_delta.pt").exists()
    }
    if not available:
        raise ValueError(
            "No model linear layers have matched basis and delta artifacts."
        )

    gpu_memory_before_analysis = (
        gpu_memory_snapshot(reset_peak=True) if torch.cuda.is_available() else []
    )

    embedding_device = next(model.parameters()).device
    layer_names = sorted(available)
    all_results: dict[str, dict[str, Any]] = {}

    for start in tqdm(
        range(0, len(layer_names), args.layer_chunk_size), desc="layer chunks"
    ):
        chunk_names = layer_names[start : start + args.layer_chunk_size]
        states = {
            name: _load_layer_state(
                name,
                available[name],
                delta_dir,
                basis_dir,
                args.spectral_norm_method,
                args.power_iterations,
            )
            for name in chunk_names
        }
        totals = {
            name: {"exact": 0.0, "bound": 0.0, "projected": 0.0, "tokens": 0}
            for name in chunk_names
        }
        handles = []

        def make_hook(name: str):
            def hook(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ) -> None:
                exact, bound, projected, tokens = batch_metrics(
                    inputs[0],
                    states[name]["basis"],
                    states[name]["delta_basis"],
                    states[name]["sigma_sq"],
                )
                totals[name]["exact"] += exact
                totals[name]["bound"] += bound
                totals[name]["projected"] += projected
                totals[name]["tokens"] += tokens

            return hook

        for name in chunk_names:
            handles.append(available[name].register_forward_hook(make_hook(name)))
        try:
            for sample in tqdm(
                samples[:max_samples], desc="held-out forward pass", leave=False
            ):
                input_ids = cast(torch.Tensor, sample["input_ids"]).to(embedding_device)
                position_ids = cast(torch.Tensor, sample["position_ids"]).to(
                    embedding_device
                )
                with torch.inference_mode():
                    model(input_ids=input_ids, position_ids=position_ids)
        finally:
            for handle in handles:
                handle.remove()

        for name in chunk_names:
            total = totals[name]
            tokens = total["tokens"]
            if tokens == 0:
                raise RuntimeError(f"No activations were observed for {name}")
            exact = total["exact"] / tokens
            bound = total["bound"] / tokens
            projected = total["projected"] / tokens
            ratio = (
                exact / bound if bound > 0 else (0.0 if exact == 0 else float("nan"))
            )
            all_results[name] = {
                "rank": int(states[name]["basis"].shape[1]),
                "num_tokens": tokens,
                "realized_deviation_per_token": _json_number(exact),
                "spectral_bound_per_token": _json_number(bound),
                "realized_to_bound_ratio": _json_number(ratio),
                "projected_input_energy_per_token": _json_number(projected),
                "spectral_norm_squared": _json_number(states[name]["sigma_sq"]),
                "spectral_bound_is_rigorous": states[name]["bound_is_rigorous"],
            }
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finite_ratios = [
        record["realized_to_bound_ratio"]
        for record in all_results.values()
        if record["realized_to_bound_ratio"] is not None
    ]
    total_tokens = sum(record["num_tokens"] for record in all_results.values())
    total_exact = sum(
        record["realized_deviation_per_token"] * record["num_tokens"]
        for record in all_results.values()
        if record["realized_deviation_per_token"] is not None
    )
    total_bound = sum(
        record["spectral_bound_per_token"] * record["num_tokens"]
        for record in all_results.values()
        if record["spectral_bound_per_token"] is not None
    )
    result = {
        "analysis": "held_out_linearized_output_surrogate_tightness",
        "formulae": {
            "realized": "||X P Delta^T||_F^2 / N",
            "spectral_bound": "||X P||_F^2 ||Delta||_2^2 / N",
            "ratio": "realized / spectral_bound",
        },
        "inputs": {
            "base_model": args.base_model,
            "val_pack_path": str(val_pack_path),
            "delta_dir": str(delta_dir),
            "basis_dir": str(basis_dir),
            "num_packed_examples": max_samples,
            "spectral_norm_method": args.spectral_norm_method,
            "power_iterations": (
                args.power_iterations if args.spectral_norm_method == "power" else None
            ),
        },
        "aggregate": {
            "num_layers": len(all_results),
            "layer_token_count_sum": total_tokens,
            "realized_deviation_per_layer_token": _json_number(
                total_exact / total_tokens
            ),
            "spectral_bound_per_layer_token": _json_number(total_bound / total_tokens),
            "realized_to_bound_ratio": (
                _json_number(total_exact / total_bound) if total_bound > 0 else None
            ),
            "median_layer_realized_to_bound_ratio": _median(
                cast(list[float], finite_ratios)
            ),
            "spectral_bounds_are_rigorous": args.spectral_norm_method == "svd",
        },
        "gpu_memory_bytes": {
            "scope": (
                "PyTorch allocator memory after loading the base model and peak memory during this "
                "held-out analysis; it does not include base-model loading or the AMARIS merge pipeline."
            ),
            "before_analysis": gpu_memory_before_analysis,
            "peak_during_analysis": (
                gpu_memory_snapshot() if torch.cuda.is_available() else []
            ),
        },
        "layers": all_results,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val_pack_path", required=True, help="Held-out packed .pt file."
    )
    parser.add_argument(
        "--delta_dir", required=True, help="Directory containing *_delta.pt artifacts."
    )
    parser.add_argument(
        "--basis_dir",
        required=True,
        help="Directory containing *_input_basis.pt artifacts.",
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Base checkpoint used to obtain held-out activations.",
    )
    parser.add_argument(
        "--output_path",
        "--output_dir",
        dest="output_path",
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument("--layers", choices=("all", "attn", "mlp"), default="all")
    parser.add_argument(
        "--layer_chunk_size",
        type=int,
        default=1,
        help="Number of hooked layers processed per forward pass.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--spectral_norm_method", choices=("svd", "power"), default="svd"
    )
    parser.add_argument("--power_iterations", type=int, default=50)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.layer_chunk_size < 1:
        parser.error("--layer_chunk_size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max_samples must be positive")
    result = analyze(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote held-out surrogate-tightness analysis to {output_path}")


if __name__ == "__main__":
    main()
