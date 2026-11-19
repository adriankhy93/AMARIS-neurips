import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import torch
import argparse
from utils.layers import (
    layer_name_to_filename,
)
from utils.gpu_memory import peak_gpu_memory_bytes, reset_peak_gpu_memory
from utils.generalized_eigensolver import (
    LOBPCGRetryExhausted,
    faster_generalized_eigen_ranking_projector_torch,
)
from tqdm import tqdm


def setup_amaris_covariances(
    cov_dir: str | Path,
) -> dict[str, torch.Tensor]:
    """Load amaris covariances

    Returns: dict mapping parameter name (e.g. 'model.layers.0.self_attn.q_proj.weight') to the corresponding input cov matrix

    """
    cov_dir = Path(cov_dir)

    # get linear layers by finding all files matching input_cov*.pt and extracting layer names, to ensure we only attempt to load covariances for layers that were actually computed
    linear_layers = [
        f.stem.replace("_input_cov", "") for f in cov_dir.glob("*input_cov.pt")
    ]
    # linear_layers = [filename_to_layer_name(fname) for fname in linear_layers]
    input_cov_dict: dict[str, torch.Tensor] = {}

    # Collect valid layers and their file paths before loading
    load_tasks = []
    for fname in linear_layers:
        input_cov_pt_path = cov_dir / f"{fname}_input_cov.pt"

        if input_cov_pt_path.exists():
            load_tasks.append((fname, input_cov_pt_path))
        else:
            raise ValueError(
                f"Expected covariance file {input_cov_pt_path} does not exist"
            )

    def _load_pair(
        task: tuple[str, Path],
    ) -> tuple[str, torch.Tensor]:
        fname, input_path = task
        # Leave covariances on CPU to minimise GPU memory usage
        try:
            # Newer PyTorch supports weights_only=True; fall back for older versions.
            input_cov = torch.load(
                input_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            input_cov = torch.load(
                input_path,
                map_location="cpu",
            )
        return fname, input_cov

    with ThreadPoolExecutor() as executor:
        for fname, input_cov in tqdm(
            executor.map(_load_pair, load_tasks),
            desc="Loading amaris covariances",
            total=len(load_tasks),
        ):
            input_cov_dict[fname] = input_cov

    return input_cov_dict


def setup_amaris_deltas(
    delta_dir: str | Path,
) -> dict[str, torch.Tensor]:
    """Load precomputed per-layer deltas (W2 - W1)."""
    delta_dir = Path(delta_dir)

    linear_layers = [f.stem.replace("_delta", "") for f in delta_dir.glob("*delta.pt")]
    delta_dict: dict[str, torch.Tensor] = {}

    load_tasks = []
    for fname in linear_layers:
        delta_pt_path = delta_dir / f"{fname}_delta.pt"

        if delta_pt_path.exists():
            load_tasks.append((fname, delta_pt_path))
        else:
            raise ValueError(f"Expected delta file {delta_pt_path} does not exist")

    def _load_pair(
        task: tuple[str, Path],
    ) -> tuple[str, torch.Tensor]:
        fname, input_path = task
        try:
            delta = torch.load(
                input_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            delta = torch.load(
                input_path,
                map_location="cpu",
            )
        return fname, delta

    with ThreadPoolExecutor() as executor:
        for fname, delta in tqdm(
            executor.map(_load_pair, load_tasks),
            desc="Loading deltas",
            total=len(load_tasks),
        ):
            delta_dict[fname] = delta

    return delta_dict


def trace_normalize(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a trace-normalized PSD matrix, or zeros when it has no energy."""
    matrix = 0.5 * (matrix + matrix.T)
    trace = torch.trace(matrix)
    if trace <= eps:
        return torch.zeros_like(matrix)
    return matrix / trace


def delta_aware_ranking_matrices(
    C1: torch.Tensor,
    C2: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-12,
    G: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized ranking matrices without changing the raw alpha budget."""
    G = delta.T @ delta if G is None else G
    return trace_normalize(C1, eps), trace_normalize(C2, eps) + trace_normalize(G, eps)


def parse_devices(gpus: str | None = None) -> list[str]:
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return ["cpu"]

    device_count = torch.cuda.device_count()
    if gpus is None:
        return [f"cuda:{i}" for i in range(device_count)]

    gpu_ids = [int(x.strip()) for x in gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("--gpus was provided but no GPU ids were found")

    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= device_count:
            raise ValueError(
                f"Invalid GPU id {gpu_id}. Available ids: 0 to {device_count - 1}"
            )

    return [f"cuda:{i}" for i in gpu_ids]


def _tensor_signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    """Fingerprint every byte, preserving coordinate order and precision."""
    data = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()
    return tuple(data.shape), str(data.dtype), digest


def _candidate_cache_path(cache_dir: Path, layer_name: str) -> Path:
    return cache_dir / f"{layer_name_to_filename(layer_name)}_generalized_eigenpairs.pt"


def _load_candidate_cache(path: Path, expected: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError):
        return None
    if payload.get("cache_key") != expected:
        return None
    return payload.get("eigensystem")


def _save_candidate_cache(path: Path, cache_key: dict, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "eigensystem": {
            "scores": result["scores"].detach().cpu(),
            "vecs": result["U_candidates"].detach().cpu(),
            "eta": float(result["eta"]),
            "solver": result["solver"],
            "solver_diagnostics": result["solver_diagnostics"],
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


@torch.no_grad()
def compute_basis_from_covariances(
    C1: dict[str, torch.Tensor],
    C2: dict[str, torch.Tensor],
    Delta: dict[str, torch.Tensor],
    alpha: float = 0.1,
    layer_chunk_size: int = 16,
    basis_save_dir: Path | None = None,
    gpus: str | None = None,
    max_k: int = 1024,
    basis_solver: str = "lobpcg",
    lobpcg_tolerance: float = 1e-3,
    lobpcg_max_iterations: int = 50,
    lobpcg_max_retries: int = 10,
    lobpcg_max_ridge_scale: float = 1e-2,
    enable_delta_variant: bool = False,
    candidate_cache_dir: Path | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, int],
    dict[str, dict[str, float]],
    dict[str, dict[str, object]],
]:
    if layer_chunk_size <= 0:
        raise ValueError("layer_chunk_size must be > 0")

    c1_layers = set(C1.keys())
    if not c1_layers:
        raise ValueError("No covariance layers were provided")
    c2_layers = set(C2.keys())
    delta_layers = set(Delta.keys())
    if c1_layers != c2_layers:
        missing_in_c2 = sorted(c1_layers - c2_layers)
        missing_in_c1 = sorted(c2_layers - c1_layers)
        mismatch_details = []
        if missing_in_c2:
            mismatch_details.append(
                f"Missing in C2 (present in C1): {missing_in_c2[:10]}"
            )
        if missing_in_c1:
            mismatch_details.append(
                f"Missing in C1 (present in C2): {missing_in_c1[:10]}"
            )
        raise ValueError(
            "C1 and C2 must have exactly the same layer set. "
            + " | ".join(mismatch_details)
        )

    if c1_layers != delta_layers:
        missing_in_delta = sorted(c1_layers - delta_layers)
        missing_in_cov = sorted(delta_layers - c1_layers)
        mismatch_details = []
        if missing_in_delta:
            mismatch_details.append(
                f"Missing in Delta (present in C1): {missing_in_delta[:10]}"
            )
        if missing_in_cov:
            mismatch_details.append(
                f"Missing in C1 (present in Delta): {missing_in_cov[:10]}"
            )
        raise ValueError(
            "C1 and Delta must have exactly the same layer set. "
            + " | ".join(mismatch_details)
        )

    basis_dict: dict[str, torch.Tensor] = {}
    rank_dict: dict[str, int] = {}
    D1_dict: dict[str, float] = {}
    V2_dict: dict[str, float] = {}
    VG_dict: dict[str, float] = {}
    full_rank_dict: dict[str, int] = {}
    trace_dict: dict[str, dict[str, object]] = {}
    fallback_dict: dict[str, dict[str, object]] = {}

    layer_names = list(C1.keys())

    # get available devices for parallel processing
    devices = parse_devices(gpus)

    # Balance work by covariance size so heavy layers are spread across devices.
    layer_costs: list[tuple[str, int]] = []
    for lname in layer_names:
        # Covariances are square; numel is a good proxy for memory/compute cost.
        layer_costs.append((lname, int(C1[lname].numel())))

    # Largest-first assignment reduces tail skew and late OOM risk.
    layer_costs.sort(key=lambda x: x[1], reverse=True)

    total_cost = sum(cost for _, cost in layer_costs)
    print(
        f"Using dynamic scheduling across {len(devices)} devices for {len(layer_names)} layers "
        f"(~{total_cost:,} covariance elements total)"
    )

    progress_lock = threading.Lock()
    task_lock = threading.Lock()
    next_task_idx = {"idx": 0}

    def _next_layer_name() -> str | None:
        with task_lock:
            idx = next_task_idx["idx"]
            if idx >= len(layer_costs):
                return None
            next_task_idx["idx"] = idx + 1
            return layer_costs[idx][0]

    def _record_completed_layer(
        progress_bar: tqdm,
        *,
        device: str,
        outcome: str,
    ) -> None:
        """Advance the shared layer-level progress bar exactly once."""
        with progress_lock:
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"{device}: {outcome}", refresh=False)

    def _process_chunks_on_device(
        device: str,
        progress_bar: tqdm,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, int],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, int],
        dict[str, dict[str, object]],
        dict[str, dict[str, object]],
    ]:

        local_basis: dict[str, torch.Tensor] = {}
        local_rank: dict[str, int] = {}
        local_D1: dict[str, float] = {}
        local_V2: dict[str, float] = {}
        local_VG: dict[str, float] = {}
        local_full_rank: dict[str, int] = {}
        local_traces: dict[str, dict[str, object]] = {}
        local_fallbacks: dict[str, dict[str, object]] = {}

        if device != "cpu":
            torch.cuda.set_device(device)

        while True:
            lname = _next_layer_name()
            if lname is None:
                break

            cov1 = C1[lname].float().contiguous()
            cov2 = C2[lname].float().contiguous()
            delta = Delta[lname].float().contiguous()

            def solve_layer(cov1_tensor, cov2_tensor, delta_tensor):
                gram = delta_tensor.T @ delta_tensor
                gram = 0.5 * (gram + gram.T)
                if enable_delta_variant:
                    ranking_C1, ranking_C2 = delta_aware_ranking_matrices(
                        cov1_tensor, cov2_tensor, delta_tensor, G=gram
                    )
                    solver_C1, solver_C2 = ranking_C1, ranking_C2
                    budget_C1, diagnostic_C2 = cov1_tensor, cov2_tensor
                else:
                    solver_C1, solver_C2 = cov1_tensor, cov2_tensor
                    budget_C1, diagnostic_C2 = None, None

                cache_key = {
                    "schema_version": 3,
                    "ranking_C1": _tensor_signature(solver_C1),
                    "ranking_C2": _tensor_signature(solver_C2),
                    "max_k": max_k,
                    "basis_solver": basis_solver,
                    "lobpcg_tolerance": lobpcg_tolerance,
                    "lobpcg_max_iterations": lobpcg_max_iterations,
                    "lobpcg_max_retries": lobpcg_max_retries,
                    "lobpcg_max_ridge_scale": lobpcg_max_ridge_scale,
                    "enable_delta_variant": enable_delta_variant,
                }
                cache_path = (
                    _candidate_cache_path(candidate_cache_dir, lname)
                    if candidate_cache_dir is not None
                    else None
                )
                eigensystem = (
                    _load_candidate_cache(cache_path, cache_key)
                    if cache_path is not None
                    else None
                )
                result = faster_generalized_eigen_ranking_projector_torch(
                    C1=solver_C1,
                    C2=solver_C2,
                    delta=delta_tensor,
                    alpha=alpha,
                    add_ridge_to_budget_metric=True,
                    budget_C1=budget_C1,
                    diagnostic_C2=diagnostic_C2,
                    use_incremental_prefix_scan=False,
                    max_rank=max_k,
                    solver=basis_solver,
                    lobpcg_tolerance=lobpcg_tolerance,
                    lobpcg_max_iterations=lobpcg_max_iterations,
                    lobpcg_max_retries=lobpcg_max_retries,
                    lobpcg_max_ridge_scale=lobpcg_max_ridge_scale,
                    precomputed_eigensystem=eigensystem,
                    precomputed_G=gram,
                )
                if cache_path is not None and eigensystem is None:
                    _save_candidate_cache(cache_path, cache_key, result)
                return result

            try:
                # Keep only one layer's tensors resident on each GPU worker.
                if device != "cpu":
                    cov1 = cov1.to(device=device, non_blocking=True)
                    cov2 = cov2.to(device=device, non_blocking=True)
                    delta = delta.to(device=device, non_blocking=True)
                result = solve_layer(cov1, cov2, delta)
            except LOBPCGRetryExhausted as exc:
                # No basis file means merge will beta-mix this linear weight.
                if basis_save_dir is not None:
                    (
                        basis_save_dir
                        / f"{layer_name_to_filename(lname)}_input_basis.pt"
                    ).unlink(missing_ok=True)
                local_fallbacks[lname] = {
                    "strategy": "beta_weighted_average",
                    "reason": "lobpcg_retry_exhausted",
                    "retries": exc.retries,
                    "final_eta": float(exc.eta),
                    "final_candidate_rank": exc.candidate_rank,
                    "error": str(exc),
                }
                del cov1, cov2, delta
                _record_completed_layer(
                    progress_bar,
                    device=device,
                    outcome=f"beta fallback: {lname}",
                )
                continue
            except RuntimeError as exc:
                err_msg = str(exc).lower()
                # Recover from CUDA OOM by retrying the current layer on CPU.
                if device != "cpu" and "out of memory" in err_msg:
                    print(f"CUDA OOM on {device} for layer {lname}; retrying on CPU")
                    torch.cuda.empty_cache()
                    cov1_cpu = C1[lname].float().contiguous()
                    cov2_cpu = C2[lname].float().contiguous()
                    delta_cpu = Delta[lname].float().contiguous()
                    result = solve_layer(cov1_cpu, cov2_cpu, delta_cpu)
                    del cov1_cpu, cov2_cpu, delta_cpu
                else:
                    raise

            local_rank[lname] = int(result["rank"])
            basis_cpu = result["U"].cpu()
            if basis_save_dir is None:
                local_basis[lname] = basis_cpu
            else:
                cap_name = layer_name_to_filename(lname)
                torch.save(
                    basis_cpu,
                    basis_save_dir / f"{cap_name}_input_basis.pt",
                )
            local_D1[lname] = float(result["D1"])
            local_V2[lname] = float(result["V2"])
            local_VG[lname] = float(result["VG"])
            local_full_rank[lname] = int(result["full_rank"])
            local_traces[lname] = {
                "tr_C1": float(result["tr_C1"]),
                "tr_PC1": float(result["tr_PC1"]),
                "tr_C2": float(result["tr_C2"]),
                "tr_PC2": float(result["tr_PC2"]),
                "tr_G": float(result["tr_G"]),
                "tr_PG": float(result["tr_PG"]),
                "constraint_value": float(result["constraint_value"]),
                "constraint_budget": float(result["constraint_budget"]),
                "constraint_fraction": float(result["constraint_fraction"]),
                "objective_value": float(result["objective_value"]),
                "objective_fraction": float(result["objective_fraction"]),
                "ranking_objective_value": float(result["ranking_objective_value"]),
                "delta_capture_fraction": float(result["delta_capture_fraction"]),
                "eta": float(result["eta"]),
                "solver": result["solver"],
                "solver_convergence": result["solver_diagnostics"],
            }
            convergence_status = (
                "converged"
                if result["solver_diagnostics"].get("converged")
                else "NOT converged"
            )
            del basis_cpu
            del result
            del cov1, cov2, delta

            _record_completed_layer(
                progress_bar,
                device=device,
                outcome=f"{convergence_status}, rank={local_rank[lname]}: {lname}",
            )

        if device != "cpu":
            torch.cuda.synchronize(device)

        return (
            local_basis,
            local_rank,
            local_D1,
            local_V2,
            local_VG,
            local_full_rank,
            local_traces,
            local_fallbacks,
        )

    futures = []
    with tqdm(
        total=len(layer_names),
        desc="Computing bases from covariances",
        unit="layer",
        dynamic_ncols=True,
    ) as progress_bar, ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for device in devices:
            futures.append(
                executor.submit(_process_chunks_on_device, device, progress_bar)
            )

        for future in as_completed(futures):
            (
                local_basis,
                local_rank,
                local_D1,
                local_V2,
                local_VG,
                local_full_rank,
                local_traces,
                local_fallbacks,
            ) = future.result()
            basis_dict.update(local_basis)
            rank_dict.update(local_rank)
            D1_dict.update(local_D1)
            V2_dict.update(local_V2)
            VG_dict.update(local_VG)
            full_rank_dict.update(local_full_rank)
            trace_dict.update(local_traces)
            fallback_dict.update(local_fallbacks)

    return (
        basis_dict,
        rank_dict,
        D1_dict,
        V2_dict,
        VG_dict,
        full_rank_dict,
        trace_dict,
        fallback_dict,
    )


def subtract_covariances(
    cov_dict1: dict[str, torch.Tensor],
    cov_dict2: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """C1-C2"""
    subtracted_cov_dict: dict[str, torch.Tensor] = {}
    for lname in cov_dict1.keys():
        if lname in cov_dict2:
            subtracted_cov_dict[lname] = cov_dict1[lname] - cov_dict2[lname]
        else:
            subtracted_cov_dict[lname] = cov_dict1[lname]
    return subtracted_cov_dict


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-capability activation covariances and eigendecompose."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Fraction of model-1 activation energy to preserve in the basis (default: 0.1).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save covariance matrices",
    )
    parser.add_argument(
        "--layer_chunk_size",
        type=int,
        default=2,
        help="Number of layers to process in each chunk (default: 16)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Optional comma-separated GPU ids to use (e.g. '0,1'). Default: all visible GPUs",
    )
    parser.add_argument(
        "--delta_dir",
        type=str,
        default=None,
        help="Directory containing *_delta.pt files. Default: <output_dir>/delta",
    )
    parser.add_argument(
        "--candidate_cache_dir",
        type=str,
        default=None,
        help="Directory for reusable per-layer generalized-eigenpair candidates.",
    )
    parser.add_argument(
        "--enable_delta_variant",
        action="store_true",
        help="Use trace-normalized C2 + Delta^T Delta for generalized-eigen ranking only.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        required=True,
        help="Maximum candidate and selected basis rank.",
    )
    parser.add_argument(
        "--basis_solver",
        choices=["lobpcg", "dense"],
        default="lobpcg",
        help="Partial iterative solver (recommended for large models) or exact dense reference solver.",
    )
    parser.add_argument(
        "--lobpcg_tolerance",
        type=float,
        default=1e-3,
        help="LOBPCG convergence tolerance (larger values stop earlier).",
    )
    parser.add_argument(
        "--lobpcg_max_iterations",
        type=int,
        default=50,
        help="Maximum LOBPCG iterations per layer.",
    )
    parser.add_argument("--lobpcg_max_retries", type=int, default=10)
    parser.add_argument("--lobpcg_max_ridge_scale", type=float, default=1e-2)

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    # Include covariance/delta loading, eigensolving, and greedy selection.
    reset_peak_gpu_memory()

    # load C1 and C2
    cov1_dir = output_path / "cov_combined" / "cov_combined_1"
    cov2_dir = output_path / "cov_combined" / "cov_combined_2"
    delta_dir = (
        Path(args.delta_dir) if args.delta_dir is not None else output_path / "delta"
    )
    C1_dict = setup_amaris_covariances(cov1_dir)
    C2_dict = setup_amaris_covariances(cov2_dir)
    Delta_dict = setup_amaris_deltas(delta_dir)
    # Preserve small alpha values in artifact names (e.g. 0.001, not 0.00).
    basis_save_dir = output_path / f"basis_alpha_{args.alpha:.12g}"
    basis_save_dir.mkdir(parents=True, exist_ok=True)

    basis_computation_started = time.perf_counter()
    (
        basis_dict,
        rank_dict,
        D1_dict,
        V2_dict,
        VG_dict,
        full_rank_dict,
        trace_dict,
        fallback_dict,
    ) = compute_basis_from_covariances(
        C1=C1_dict,
        C2=C2_dict,
        Delta=Delta_dict,
        alpha=args.alpha,
        layer_chunk_size=args.layer_chunk_size,
        basis_save_dir=basis_save_dir,
        gpus=args.gpus,
        max_k=args.max_k,
        basis_solver=args.basis_solver,
        lobpcg_tolerance=args.lobpcg_tolerance,
        lobpcg_max_iterations=args.lobpcg_max_iterations,
        lobpcg_max_retries=args.lobpcg_max_retries,
        lobpcg_max_ridge_scale=args.lobpcg_max_ridge_scale,
        enable_delta_variant=args.enable_delta_variant,
        candidate_cache_dir=(
            Path(args.candidate_cache_dir) if args.candidate_cache_dir else None
        ),
    )

    basis_computation_seconds = time.perf_counter() - basis_computation_started

    if basis_dict:
        for lname in tqdm(basis_dict.keys(), desc="Saving basis - input"):
            cap_name = layer_name_to_filename(lname)
            torch.save(
                basis_dict[lname],
                basis_save_dir / f"{cap_name}_input_basis.pt",
            )
            print(
                f"Saved basis for {args.tag2} - input - layer {lname}: {basis_dict[lname].shape[1]} basis vectors"
            )

    metadata = {
        "completed": True,
        "expected_layers": sorted(C1_dict),
        "alpha": args.alpha,
        "delta_dir": str(delta_dir),
        "candidate_cache_dir": args.candidate_cache_dir,
        "layer_chunk_size": args.layer_chunk_size,
        "enable_delta_variant": args.enable_delta_variant,
        "basis_solver": args.basis_solver,
        "lobpcg_tolerance": args.lobpcg_tolerance,
        "lobpcg_max_iterations": args.lobpcg_max_iterations,
        "lobpcg_max_retries": args.lobpcg_max_retries,
        "lobpcg_max_ridge_scale": args.lobpcg_max_ridge_scale,
        "fallback_layers": fallback_dict,
        "rank_dict": rank_dict,
        "D1_dict": D1_dict,
        "V2_dict": V2_dict,
        "VG_dict": VG_dict,
        "full_rank_dict": full_rank_dict,
        "trace_dict": trace_dict,
        "timings_seconds": {"basis_computation": basis_computation_seconds},
        "gpu_memory_bytes": {
            "scope": "Peak PyTorch allocator memory for basis loading, generalized eigensolving, and greedy selection.",
            "per_device": peak_gpu_memory_bytes(),
        },
    }
    # Keep metadata beside the exact alpha basis so merge manifests cannot pick
    # up another alpha's diagnostics. Retain the historical root location for
    # downstream tools that read it.
    metadata_paths = [
        basis_save_dir / "basis_metadata.json",
        output_path / "metadata" / "basis_metadata.json",
    ]
    for meta_path in metadata_paths:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
