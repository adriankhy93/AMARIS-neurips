from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import time

import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.checkpoints import SafeTensorCheckpoint
from utils.layers import get_linear_layers, layer_name_to_filename
from utils.gpu_memory import peak_gpu_memory_bytes, reset_peak_gpu_memory
from utils.artifacts import beta_output_dirs
from tqdm import tqdm


def setup_amaris_merge(
    model: nn.Module,
    basis_dir: str | Path,
) -> dict[str, torch.Tensor]:
    """Load amaris basis

    Returns:
    dict mapping parameter name (e.g. 'model.layers.0.self_attn.q_proj.weight') to the corresponding input basis
    """
    basis_dir = Path(basis_dir)

    linear_layers = get_linear_layers(model)
    input_basis_dict: dict[str, torch.Tensor] = {}

    # Collect valid layers and their file paths before loading
    load_tasks = []
    for lname in linear_layers:
        fname = layer_name_to_filename(lname)
        input_basis_pt_path = basis_dir / f"{fname}_input_basis.pt"
        if input_basis_pt_path.exists():
            load_tasks.append((lname, input_basis_pt_path))

    def _load_pair(
        task: tuple[str, Path | None],
    ) -> tuple[str, torch.Tensor | None]:
        fname, input_path = task
        # Leave basis on CPU to minimise GPU memory usage
        input_basis = (
            torch.load(input_path, map_location="cpu", weights_only=True)
            if input_path is not None
            else None
        )

        return fname, input_basis

    with ThreadPoolExecutor() as executor:
        for fname, input_basis in tqdm(
            executor.map(_load_pair, load_tasks),
            desc="Loading amaris basis",
            total=len(load_tasks),
        ):
            if input_basis is not None:
                input_basis_dict[fname] = input_basis

    return input_basis_dict


def _read_basis_metadata(basis_dir: Path) -> dict:
    """Read the metadata belonging to this exact alpha basis, when available."""
    for candidate in (
        basis_dir / "basis_metadata.json",
        basis_dir.parent / "metadata" / "basis_metadata.json",
    ):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _read_covariance_timings(basis_dir: Path) -> dict[str, object]:
    """Collect per-tag covariance timing records from the shared artifact tree."""
    metadata_paths: set[Path] = set()
    alpha_dir = basis_dir.parent
    for side in ("cov_combined_1", "cov_combined_2"):
        covariance_dir = alpha_dir / "cov_combined" / side
        if covariance_dir.exists():
            metadata_paths.update(
                covariance_dir.resolve().parent.glob(
                    "metadata/*_covariances_metadata.json"
                )
            )

    records = []
    for metadata_path in sorted(metadata_paths):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        timings = metadata.get("timings_seconds")
        if timings is not None:
            records.append(
                {
                    "tag": metadata.get("tag"),
                    "model": metadata.get("model"),
                    "timings_seconds": timings,
                }
            )
    return {
        "per_tag": records,
        "total_covariance_collection_and_save": sum(
            record["timings_seconds"].get("covariance_collection_and_save", 0.0)
            for record in records
        ),
    }


def _validate_basis_artifacts(model, basis_dir, bases):
    metadata = _read_basis_metadata(basis_dir)
    if metadata.get("completed"):
        expected = set(metadata.get("expected_layers", []))
        projected = set(metadata.get("rank_dict", {}))
        fallbacks = metadata.get("fallback_layers", {})
        available = {layer_name_to_filename(name) for name in get_linear_layers(model)}
        loaded = {layer_name_to_filename(name) for name in bases}
        if (
            not expected
            or not expected <= available
            or projected & set(fallbacks)
            or projected | set(fallbacks) != expected
            or loaded != projected
            or any(
                value.get("strategy") != "beta_weighted_average"
                for value in fallbacks.values()
            )
        ):
            raise ValueError(
                f"Incomplete or incompatible basis metadata in {basis_dir}"
            )
    elif not bases:
        raise ValueError(f"No basis tensors were loaded from {basis_dir}")
    return metadata


def _untie_projected_weights(model, projected_layers):
    """Give projected linear weights their own storage before updating them.

    Embeddings retain their original value for beta mixing. Persist the changed
    tying policy so Transformers does not retie the head when loading the result.
    """
    alias_counts = {}
    for _, parameter in model.named_parameters(remove_duplicate=False):
        alias_counts[id(parameter)] = alias_counts.get(id(parameter), 0) + 1
    for name in projected_layers:
        layer = model.get_submodule(name)
        if alias_counts.get(id(layer.weight), 0) > 1:
            layer.weight = nn.Parameter(
                layer.weight.detach().clone(), requires_grad=layer.weight.requires_grad
            )
            if name == "lm_head" and hasattr(model, "config"):
                model.config.tie_word_embeddings = False


def _projection_coverage(
    model: nn.Module,
    input_basis_dict: dict[str, torch.Tensor],
    layers_to_ignore: str | None,
    fallback_layers: dict[str, object] | None = None,
) -> dict[str, object]:
    """Describe AMARIS-projected weights and every beta-mixed parameter."""
    all_linear_layers = sorted(
        name.rsplit(".weight", 1)[0] for name in _linear_weight_parameter_names(model)
    )
    basis_layers = set(input_basis_dict)
    ignored = {
        name.strip() for name in (layers_to_ignore or "").split(",") if name.strip()
    }
    projected = sorted(basis_layers - ignored)
    fallback_layers = fallback_layers or {}
    not_projected = []
    for name in all_linear_layers:
        if name in ignored:
            reason = "layers_to_ignore"
        elif name not in basis_layers:
            reason = (
                "lobpcg_retry_exhausted_beta_mix"
                if layer_name_to_filename(name) in fallback_layers
                else "no_basis_file"
            )
        else:
            continue
        not_projected.append({"layer": name, "reason": reason})
    # A basis could refer to a layer absent from a differently configured base.
    # Preserve it in the manifest rather than silently hiding it.
    unavailable_basis_layers = sorted(basis_layers - set(all_linear_layers))
    projected_weight_parameters = _linear_weight_parameter_names(model, set(projected))
    weighted_average_parameters = sorted(
        name
        for name, _ in model.named_parameters()
        if name not in projected_weight_parameters
    )
    return {
        "linear_layers_total": len(all_linear_layers),
        "projected_layers": projected,
        "amaris_projected_layers": projected,
        "not_projected_layers": not_projected,
        # Kept under the legacy key, but now includes all beta-mixed parameters:
        # embeddings, norms, linear biases, and non-projected linear weights.
        "weighted_average_layers": weighted_average_parameters,
        "weighted_average_parameter_count": len(weighted_average_parameters),
        "basis_layers_not_in_model": unavailable_basis_layers,
    }


def _write_run_manifest(
    output_dir: Path,
    *,
    beta: float,
    basis_dir: Path,
    basis_metadata: dict,
    coverage: dict[str, object],
    timings: dict[str, float],
    mixed_non_linear_parameters: int,
    model1_dir: str | None,
    model2_dir: str | None,
    gpu_memory: list[dict[str, int]] | None = None,
) -> None:
    """Write a self-contained, JSON-serializable AMARIS run record."""
    trace_dict = basis_metadata.get("trace_dict", {})
    rank_dict = basis_metadata.get("rank_dict", {})
    full_rank_dict = basis_metadata.get("full_rank_dict", {})
    D1_dict = basis_metadata.get("D1_dict", {})
    V2_dict = basis_metadata.get("V2_dict", {})
    VG_dict = basis_metadata.get("VG_dict", {})

    def lookup(diagnostics: dict, layer: str):
        # New basis metadata uses artifact names; canonical keys are accepted
        # for manifests created from earlier metadata schemas.
        return diagnostics.get(layer_name_to_filename(layer), diagnostics.get(layer))

    projected_diagnostics = {
        layer: {
            "rank": lookup(rank_dict, layer),
            "full_rank": lookup(full_rank_dict, layer),
            # C1/C2 refer to base/specialist activation covariances. P is the
            # selected orthogonal input projector for this layer.
            "traces": lookup(trace_dict, layer),
            "selection_quality": {
                "constraint_fraction": lookup(D1_dict, layer),
                "specialist_objective_fraction": lookup(V2_dict, layer),
                "delta_capture_fraction": lookup(VG_dict, layer),
            },
        }
        for layer in coverage["projected_layers"]
    }
    basis_computation_seconds = basis_metadata.get("timings_seconds", {}).get(
        "basis_computation"
    )
    if basis_computation_seconds is not None:
        timings = {"basis_computation": basis_computation_seconds, **timings}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "AMARIS",
        "inputs": {
            "base_model": model1_dir,
            "specialist_model": model2_dir,
            "basis_dir": str(basis_dir),
            "basis_metadata_found": bool(basis_metadata),
        },
        "hyperparameters": {
            "alpha": basis_metadata.get("alpha"),
            "beta": beta,
            "enable_delta_variant": basis_metadata.get("enable_delta_variant"),
            "basis_solver": basis_metadata.get("basis_solver"),
            "lobpcg_tolerance": basis_metadata.get("lobpcg_tolerance"),
            "lobpcg_max_iterations": basis_metadata.get("lobpcg_max_iterations"),
            "lobpcg_max_retries": basis_metadata.get("lobpcg_max_retries"),
            "lobpcg_max_ridge_scale": basis_metadata.get("lobpcg_max_ridge_scale"),
        },
        "timings_seconds": timings,
        "gpu_memory_bytes": {
            "scope": "Peak PyTorch allocator memory for basis loading, projected linear merge, beta mixing, and checkpoint save; model loading is excluded.",
            "per_device": (
                gpu_memory if gpu_memory is not None else peak_gpu_memory_bytes()
            ),
        },
        "upstream_timings_seconds": {
            "covariance_computation": _read_covariance_timings(basis_dir),
        },
        "projection": {
            **coverage,
            "projected_layer_diagnostics": projected_diagnostics,
        },
        "parameter_mixing": {
            "linear_weights": "AMARIS projected update",
            "other_parameters": "beta * base + (1 - beta) * specialist (including non-projected linear weights)",
            "non_linear_parameters_mixed": mixed_non_linear_parameters,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "amaris_run.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def amaris_merge_one_step(
    model1,
    model2,
    input_basis_dict: dict[str, torch.Tensor],
    layers_to_ignore: str | None = None,
):
    layers_to_ignore_list = {
        name.strip() for name in (layers_to_ignore or "").split(",") if name.strip()
    }
    _untie_projected_weights(model1, set(input_basis_dict) - layers_to_ignore_list)
    with torch.no_grad():
        for name, input_basis in input_basis_dict.items():
            if name in layers_to_ignore_list:
                print(f"Skipping layer {name} as it is in layers_to_ignore")
                continue
            layer1 = model1.get_submodule(name)
            layer2 = model2.get_submodule(name)

            W1 = layer1.weight.data.detach()
            device = W1.device
            dtype = W1.dtype

            W2 = layer2.weight.to(device).data.detach()
            B = input_basis.to(device=device, dtype=torch.float32)

            delta = W2.float() - W1.float()
            projected_delta = delta @ B @ B.T

            # Apply only the basis-projected incremental update for this step.
            new_w1 = W1.float() + projected_delta
            layer1.weight.data.copy_(new_w1.to(dtype=dtype))


def amaris_merge_one_step_streamed(
    model1: nn.Module,
    specialist: SafeTensorCheckpoint,
    input_basis_dict: dict[str, torch.Tensor],
    layers_to_ignore: str | None = None,
) -> None:
    """Apply the linear AMARIS update while streaming specialist weights from CPU."""
    ignored = {
        name.strip() for name in (layers_to_ignore or "").split(",") if name.strip()
    }
    _untie_projected_weights(model1, set(input_basis_dict) - ignored)
    with torch.no_grad():
        for name, input_basis in input_basis_dict.items():
            if name in ignored:
                continue
            layer1 = model1.get_submodule(name)
            W1 = layer1.weight.detach()
            W2 = specialist.get_tensor(f"{name}.weight").to(
                device=W1.device, dtype=torch.float32
            )
            basis = input_basis.to(device=W1.device, dtype=torch.float32)
            merged = W1.float() + (W2 - W1.float()) @ basis @ basis.T
            layer1.weight.copy_(merged.to(dtype=W1.dtype))


def _linear_weight_parameter_names(
    model: nn.Module,
    projected_layers: set[str] | None = None,
) -> set[str]:
    """Return linear weight parameter names, optionally only AMARIS targets."""
    return {
        f"{module_name}.{parameter_name}" if module_name else parameter_name
        for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        if projected_layers is None or module_name in projected_layers
        for parameter_name, _ in module.named_parameters(recurse=False)
        if parameter_name == "weight"
    }


def mix_non_linear_parameters_streamed(
    model1: nn.Module,
    base: SafeTensorCheckpoint,
    specialist: SafeTensorCheckpoint,
    beta: float,
    projected_layers: set[str] | None = None,
) -> int:
    """Restore base parameters and beta-mix every non-projected parameter."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"--beta must be in [0, 1], got {beta}")
    _untie_projected_weights(
        model1,
        (
            projected_layers
            if projected_layers is not None
            else {
                name
                for name, module in model1.named_modules()
                if isinstance(module, nn.Linear)
            }
        ),
    )
    linear_weights = _linear_weight_parameter_names(model1, projected_layers)
    mixed_count = 0
    with torch.no_grad():
        for name, parameter in model1.named_parameters():
            if name in linear_weights:
                continue
            base_value = base.get_tensor(name)
            specialist_value = specialist.get_tensor(name)
            if (
                base_value.shape != parameter.shape
                or specialist_value.shape != parameter.shape
            ):
                raise ValueError(
                    f"Checkpoint shape mismatch for non-linear parameter: {name}"
                )
            mixed = beta * base_value.to(device=parameter.device, dtype=torch.float32)
            mixed.add_(
                (1.0 - beta)
                * specialist_value.to(device=parameter.device, dtype=torch.float32)
            )
            parameter.copy_(mixed.to(dtype=parameter.dtype))
            mixed_count += 1
    return mixed_count


def amaris_merge_many_betas_streamed(
    model1: nn.Module,
    base: SafeTensorCheckpoint,
    specialist: SafeTensorCheckpoint,
    tokenizer,
    output_dirs: dict[float, Path],
    basis_dir: str | Path,
    layers_to_ignore: str | None = None,
    model1_dir: str | None = None,
    model2_dir: str | None = None,
) -> None:
    """Load/apply the alpha-linear merge once, then save one model per beta.

    Each output directory receives ``amaris_run.json``. The expensive stages
    shared by a beta sweep are recorded in every manifest for comparability.
    """
    overall_started = time.perf_counter()
    reset_peak_gpu_memory()
    basis_dir = Path(basis_dir)
    basis_load_started = time.perf_counter()
    input_basis_dict = setup_amaris_merge(model1, basis_dir)
    basis_load_seconds = time.perf_counter() - basis_load_started
    basis_metadata = _validate_basis_artifacts(model1, basis_dir, input_basis_dict)
    coverage = _projection_coverage(
        model1,
        input_basis_dict,
        layers_to_ignore,
        basis_metadata.get("fallback_layers"),
    )

    linear_projection_started = time.perf_counter()
    amaris_merge_one_step_streamed(
        model1, specialist, input_basis_dict, layers_to_ignore
    )
    linear_projection_seconds = time.perf_counter() - linear_projection_started

    for beta, output_dir in output_dirs.items():
        beta_started = time.perf_counter()
        mixing_started = time.perf_counter()
        mixed_count = mix_non_linear_parameters_streamed(
            model1,
            base,
            specialist,
            beta,
            projected_layers=set(coverage["projected_layers"]),
        )
        mixing_seconds = time.perf_counter() - mixing_started
        model_output_path = output_dir / "model"
        save_model_started = time.perf_counter()
        model1.save_pretrained(model_output_path)
        model_save_seconds = time.perf_counter() - save_model_started
        save_tokenizer_started = time.perf_counter()
        tokenizer.save_pretrained(model_output_path)
        tokenizer_save_seconds = time.perf_counter() - save_tokenizer_started
        timings = {
            "basis_load": basis_load_seconds,
            "linear_projection_shared": linear_projection_seconds,
            "non_linear_beta_mixing": mixing_seconds,
            "model_save": model_save_seconds,
            "tokenizer_save": tokenizer_save_seconds,
            "this_output_total": time.perf_counter() - beta_started,
            "pipeline_elapsed_at_output": time.perf_counter() - overall_started,
        }
        _write_run_manifest(
            output_dir,
            beta=beta,
            basis_dir=basis_dir,
            basis_metadata=basis_metadata,
            coverage=coverage,
            timings=timings,
            mixed_non_linear_parameters=mixed_count,
            model1_dir=model1_dir,
            model2_dir=model2_dir,
            gpu_memory=peak_gpu_memory_bytes(),
        )
        print(
            f"Saved {model_output_path} (linear layers={len(coverage['projected_layers'])}, beta={beta:.2f}, non-linear parameters={mixed_count})"
        )


def mix_non_linear_parameters(
    model1: nn.Module,
    model2: nn.Module,
    beta: float,
    projected_layers: set[str] | None = None,
) -> int:
    """Beta-mix every parameter except AMARIS-projected linear weights."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"--beta must be in [0, 1], got {beta}")

    _untie_projected_weights(
        model1,
        (
            projected_layers
            if projected_layers is not None
            else {
                name
                for name, module in model1.named_modules()
                if isinstance(module, nn.Linear)
            }
        ),
    )
    linear_parameter_names = _linear_weight_parameter_names(model1, projected_layers)
    model2_parameters = dict(model2.named_parameters())
    mixed_count = 0
    with torch.no_grad():
        for name, parameter1 in model1.named_parameters():
            if name in linear_parameter_names:
                continue
            parameter2 = model2_parameters.get(name)
            if parameter2 is None:
                raise ValueError(f"Model 2 is missing non-linear parameter: {name}")
            if parameter1.shape != parameter2.shape:
                raise ValueError(
                    f"Non-linear parameter shape mismatch for {name}: "
                    f"{tuple(parameter1.shape)} != {tuple(parameter2.shape)}"
                )
            mixed = beta * parameter1.float() + (1.0 - beta) * parameter2.to(
                device=parameter1.device, dtype=torch.float32
            )
            parameter1.copy_(mixed.to(dtype=parameter1.dtype))
            mixed_count += 1
    return mixed_count


def amaris_merge(
    model1,
    model2,
    tokenizer,
    output_path,
    alpha=None,
    basis_dir=None,
    layers_to_ignore=None,
    beta: float = 1.0,
):
    overall_started = time.perf_counter()
    reset_peak_gpu_memory()
    output_path = Path(output_path)

    if basis_dir is not None:
        basis_dir = Path(basis_dir)
    elif alpha is not None:
        basis_dir = output_path / f"basis_alpha_{alpha:.12g}"
    else:
        raise ValueError("Either alpha or basis_dir must be provided")

    basis_load_started = time.perf_counter()
    input_basis_dict = setup_amaris_merge(model1, basis_dir)
    basis_load_seconds = time.perf_counter() - basis_load_started

    if not basis_dir.exists():
        raise ValueError(f"Basis directory does not exist: {basis_dir}")
    print(f"Loaded {len(input_basis_dict)} basis layers from {basis_dir}")

    basis_metadata = _validate_basis_artifacts(model1, basis_dir, input_basis_dict)
    coverage = _projection_coverage(
        model1,
        input_basis_dict,
        layers_to_ignore,
        basis_metadata.get("fallback_layers"),
    )
    linear_projection_started = time.perf_counter()
    amaris_merge_one_step(
        model1, model2, input_basis_dict, layers_to_ignore=layers_to_ignore
    )
    linear_projection_seconds = time.perf_counter() - linear_projection_started
    mixing_started = time.perf_counter()
    mixed_count = mix_non_linear_parameters(
        model1,
        model2,
        beta,
        projected_layers=set(coverage["projected_layers"]),
    )
    mixing_seconds = time.perf_counter() - mixing_started

    # Persist best checkpoint as the final saved model artifact.
    model_output_path = output_path / "model/"
    model_save_started = time.perf_counter()
    model1.save_pretrained(model_output_path)
    model_save_seconds = time.perf_counter() - model_save_started
    tokenizer_save_started = time.perf_counter()
    tokenizer.save_pretrained(model_output_path)
    tokenizer_save_seconds = time.perf_counter() - tokenizer_save_started
    _write_run_manifest(
        output_path,
        beta=beta,
        basis_dir=basis_dir,
        basis_metadata=basis_metadata,
        coverage=coverage,
        timings={
            "basis_load": basis_load_seconds,
            "linear_projection_shared": linear_projection_seconds,
            "non_linear_beta_mixing": mixing_seconds,
            "model_save": model_save_seconds,
            "tokenizer_save": tokenizer_save_seconds,
            "this_output_total": time.perf_counter() - overall_started,
            "pipeline_elapsed_at_output": time.perf_counter() - overall_started,
        },
        mixed_non_linear_parameters=mixed_count,
        model1_dir=None,
        model2_dir=None,
        gpu_memory=peak_gpu_memory_bytes(),
    )
    print(
        f"Saved best model to {model_output_path} "
        f"(merged {len(input_basis_dict)} linear layers; "
        f"mixed {mixed_count} non-linear parameters with beta={beta:.2f}; "
        f"basis_dir={basis_dir})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_dir", type=str, required=True)
    parser.add_argument("--model2_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for one beta value.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Prefix used to save one output per --betas value.",
    )
    parser.add_argument(
        "--betas",
        type=str,
        default=None,
        help="Comma-separated base coefficients; reuses one linear merge.",
    )
    parser.add_argument("--basis_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--layers_to_ignore", type=str, default=None)
    args = parser.parse_args()

    if Path(args.model1_dir).resolve() == Path(args.model2_dir).resolve():
        raise ValueError("--model1_dir and --model2_dir point to the same model path")
    if (args.output_dir is None) == (args.output_base_dir is None):
        raise ValueError("Provide exactly one of --output_dir or --output_base_dir")
    if args.output_dir is not None and args.betas is not None:
        raise ValueError("--betas requires --output_base_dir")
    if args.output_base_dir is not None and args.betas is None:
        raise ValueError("--output_base_dir requires --betas")
    if args.output_dir is not None:
        outputs = beta_output_dirs([args.beta], output_dir=args.output_dir)
    else:
        betas = [
            float(value.strip()) for value in args.betas.split(",") if value.strip()
        ]
        outputs = beta_output_dirs(betas, output_base_dir=args.output_base_dir)

    model1 = AutoModelForCausalLM.from_pretrained(
        args.model1_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model1_dir)
    base = SafeTensorCheckpoint(args.model1_dir)
    specialist = SafeTensorCheckpoint(args.model2_dir)
    amaris_merge_many_betas_streamed(
        model1=model1,
        base=base,
        specialist=specialist,
        tokenizer=tokenizer,
        output_dirs=outputs,
        basis_dir=args.basis_dir,
        layers_to_ignore=args.layers_to_ignore,
        model1_dir=args.model1_dir,
        model2_dir=args.model2_dir,
    )


if __name__ == "__main__":
    main()
