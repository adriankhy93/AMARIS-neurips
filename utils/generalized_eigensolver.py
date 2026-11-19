import torch


def safe_cholesky_spd(
    A,
    initial_eta=None,
    ridge_scale=1e-5,
    eps=1e-12,
    max_tries=8,
    verbose=False,
):
    """
    Computes Cholesky of A + eta I with adaptive eta.

    Returns:
        L: lower Cholesky factor
        eta: final ridge used
        B: A + eta I
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]

    device = A.device
    dtype = A.dtype
    d = A.shape[0]

    A = 0.5 * (A + A.T)

    tr_A = torch.trace(A).clamp_min(torch.as_tensor(eps, device=device, dtype=dtype))

    if initial_eta is None:
        eta = ridge_scale * tr_A / max(d, 1)
    else:
        eta = torch.as_tensor(initial_eta, device=device, dtype=dtype)

    eta = eta.clamp_min(torch.as_tensor(eps, device=device, dtype=dtype))

    I = torch.eye(d, device=device, dtype=dtype)

    last_err = None

    for attempt in range(max_tries):
        B = A + eta * I

        try:
            L = torch.linalg.cholesky(B)
            if verbose and attempt > 0:
                print(f"Cholesky succeeded with eta={eta.item():.3e}")
            return L, eta, B
        except RuntimeError as err:
            err_msg = str(err).lower()
            # Some PyTorch builds can fail with:
            # "lazy wrapper should be called at most once"
            # Materialize a dense contiguous tensor and retry once, then CPU fallback.
            if "lazy wrapper should be called at most once" in err_msg:
                try:
                    B_mat = B.clone().contiguous()
                    L = torch.linalg.cholesky(B_mat)
                    if verbose:
                        print(
                            "Cholesky succeeded after materializing lazy tensor wrapper"
                        )
                    return L, eta, B_mat
                except RuntimeError:
                    B_cpu = B.detach().to(device="cpu").contiguous()
                    L_cpu = torch.linalg.cholesky(B_cpu)
                    if verbose:
                        print("Cholesky succeeded on CPU after lazy-wrapper failure")
                    return (
                        L_cpu.to(device=device, dtype=dtype),
                        eta,
                        B_cpu.to(device=device, dtype=dtype),
                    )
            # Non-lazy runtime errors are typically numerical failures
            # (e.g. matrix not positive definite). Keep adaptive ridge retries.
            last_err = err
            eta = eta * 10.0
            if verbose:
                print(f"Cholesky failed; increasing eta to {eta.item():.3e}")

    if last_err is not None:
        raise last_err

    raise RuntimeError("Cholesky failed without capturing an error")



class LOBPCGRetryExhausted(RuntimeError):
    """Raised when bounded numerical retries cannot stabilize LOBPCG."""

    def __init__(self, message, *, retries, eta, candidate_rank):
        super().__init__(message)
        self.retries = retries
        self.eta = eta
        self.candidate_rank = candidate_rank


def ridge_stabilize_spd(A, ridge_scale=1e-5, eps=1e-12):
    """Construct the PSD covariance metric plus its deterministic ridge.

    Activation covariances are PSD; adding this positive diagonal is sufficient
    for LOBPCG's generalized metric and avoids an unnecessary dense Cholesky.
    """
    A = 0.5 * (A + A.T)
    d = A.shape[0]
    trace = torch.trace(A).clamp_min(torch.as_tensor(eps, device=A.device, dtype=A.dtype))
    eta = (ridge_scale * trace / max(d, 1)).clamp_min(
        torch.as_tensor(eps, device=A.device, dtype=A.dtype)
    )
    return eta, A + eta * torch.eye(d, device=A.device, dtype=A.dtype)


@torch.no_grad()
def generalized_residual_diagnostics(
    C2: torch.Tensor,
    B: torch.Tensor,
    scores: torch.Tensor,
    uvecs: torch.Tensor,
    tolerance: float,
    eps: float = 1e-12,
) -> dict[str, float | int | bool]:
    """Evaluate relative residuals for ``C2 u = score * B u``."""
    C2u = C2 @ uvecs
    Bu = B @ uvecs
    residual = C2u - Bu * scores.unsqueeze(0)
    residual_norms = torch.linalg.vector_norm(residual, dim=0)
    denominators = (
        torch.linalg.vector_norm(C2u, dim=0)
        + scores.abs() * torch.linalg.vector_norm(Bu, dim=0)
        + eps
    )
    relative_residuals = residual_norms / denominators
    finite = bool(torch.isfinite(relative_residuals).all())
    max_relative_residual = float(relative_residuals.max()) if finite else float("inf")
    return {
        "converged": finite and max_relative_residual <= tolerance,
        "tolerance": float(tolerance),
        "candidate_count": int(relative_residuals.numel()),
        "max_relative_residual": max_relative_residual,
        "mean_relative_residual": float(relative_residuals.mean()) if finite else float("inf"),
        "median_relative_residual": float(relative_residuals.median()) if finite else float("inf"),
    }


def generalized_eigh_torch_safe(
    C2,
    C1,
    ridge_scale=1e-5,
    eps=1e-12,
    descending=True,
    solver="lobpcg",
    max_eigenpairs=None,
    lobpcg_tolerance=1e-3,
    lobpcg_max_iterations=50,
    lobpcg_max_retries=10,
    lobpcg_max_ridge_scale=1e-2,
):
    """Solve ``C2 u = rho (C1 + eta I) u`` with bounded LOBPCG retries."""
    C1 = 0.5 * (C1 + C1.T)
    C2 = 0.5 * (C2 + C2.T)
    d = C1.shape[0]

    if solver == "lobpcg":
        requested = d if max_eigenpairs is None else min(max_eigenpairs, d)
        k = min(requested, d // 3)
        if k > 0:
            ridge = ridge_scale
            last_error = None
            for attempt in range(lobpcg_max_retries):
                eta, B = ridge_stabilize_spd(C1, ridge_scale=ridge, eps=eps)
                try:
                    scores, uvecs = torch.lobpcg(
                        C2, k=k, B=B, largest=descending, tol=lobpcg_tolerance,
                        niter=lobpcg_max_iterations, method="ortho",
                    )
                    order = torch.argsort(scores, descending=descending)
                    scores, uvecs = scores[order], uvecs[:, order]
                    diagnostics = {
                        "retries": attempt,
                        "candidate_rank": k,
                        "ridge_scale": ridge,
                        "max_iterations": lobpcg_max_iterations,
                    }
                    diagnostics.update(
                        generalized_residual_diagnostics(
                            C2, B, scores, uvecs, lobpcg_tolerance, eps
                        )
                    )
                    return scores, uvecs, eta, B, "lobpcg", diagnostics
                except RuntimeError as exc:
                    error_text = str(exc).lower()
                    is_numerical_lobpcg_error = (
                        "cholesky" in error_text
                        or "positive-definite" in error_text
                        or "eigh" in error_text
                        or "failed to converge" in error_text
                        or "ill-conditioned" in error_text
                        or "repeated eigenvalues" in error_text
                    )
                    if not is_numerical_lobpcg_error:
                        raise
                    last_error = exc
                    final_eta, final_k = eta, k
                    if attempt + 1 == lobpcg_max_retries:
                        break
                    # Keep eta small relative to the average covariance
                    # variance, while reducing the nearly dependent block.
                    ridge = min(ridge * 2.0, lobpcg_max_ridge_scale)
                    k = max(1, int(k * 0.9))
            raise LOBPCGRetryExhausted(
                f"LOBPCG failed after {lobpcg_max_retries} numerical retries: {last_error}",
                retries=lobpcg_max_retries,
                eta=final_eta,
                candidate_rank=final_k,
            )
        solver = "dense"

    if solver != "dense":
        raise ValueError(f"Unknown basis solver: {solver}")

    L, eta, B = safe_cholesky_spd(C1, ridge_scale=ridge_scale, eps=eps, max_tries=10, verbose=False)
    X = torch.linalg.solve_triangular(L, C2, upper=False, left=True)
    Y = torch.linalg.solve_triangular(L, X.T, upper=False, left=True)
    A = 0.5 * (Y.T + Y)
    scores, zvecs = torch.linalg.eigh(A)
    order = torch.argsort(scores, descending=descending)
    scores = scores[order]
    zvecs = zvecs[:, order]
    uvecs = torch.linalg.solve_triangular(L.T, zvecs, upper=True, left=True)
    diagnostics = {"retries": 0, "candidate_rank": d, "ridge_scale": ridge_scale}
    diagnostics.update(generalized_residual_diagnostics(C2, B, scores, uvecs, eps, eps))
    return scores, uvecs, eta, B, "dense", diagnostics


@torch.no_grad()
def select_largest_feasible_prefix(
    vecs,
    C1,
    budget,
    max_rank=None,
    use_incremental_prefix_scan=False,
    orth_eps=1e-12,
):
    """
    Select largest prefix k such that

        tr(orth(vecs[:, :k]).T @ C1 @ orth(vecs[:, :k])) <= budget.

    When use_incremental_prefix_scan=False, uses binary search because
    prefix subspaces are nested.

    When use_incremental_prefix_scan=True, uses an incremental
    orthonormalization pass over prefixes and cumulative cost updates.
    This can reduce repeated QR work for large dimensions.
    """
    d, r_total = vecs.shape

    if max_rank is None:
        max_rank = r_total
    else:
        max_rank = min(max_rank, r_total)

    lo = 0
    hi = max_rank
    best_k = 0
    best_U = torch.empty((d, 0), device=vecs.device, dtype=vecs.dtype)
    best_cost = torch.zeros((), device=vecs.device, dtype=vecs.dtype)

    if use_incremental_prefix_scan:
        U_running = torch.empty((d, 0), device=vecs.device, dtype=vecs.dtype)
        running_cost = torch.zeros((), device=vecs.device, dtype=vecs.dtype)

        for k in range(1, max_rank + 1):
            v = vecs[:, k - 1]

            if U_running.shape[1] > 0:
                v = v - U_running @ (U_running.T @ v)

            v_norm = torch.linalg.norm(v)

            # If the new vector is numerically dependent, the prefix subspace
            # (and therefore the cost) is unchanged.
            if (v_norm > orth_eps).item():
                q = v / v_norm
                U_running = torch.cat([U_running, q[:, None]], dim=1)
                running_cost = running_cost + (q @ (C1 @ q))

            if (running_cost <= budget).item():
                best_k = k
                best_U = U_running
                best_cost = running_cost

        return best_U, best_k, best_cost

    while lo <= hi:
        mid = (lo + hi) // 2

        if mid == 0:
            cost = torch.zeros((), device=vecs.device, dtype=vecs.dtype)
            U_mid = torch.empty((d, 0), device=vecs.device, dtype=vecs.dtype)
        else:
            U_mid, _ = torch.linalg.qr(vecs[:, :mid], mode="reduced")
            cost = torch.trace(U_mid.T @ C1 @ U_mid)

        if (cost <= budget).item():
            best_k = mid
            best_U = U_mid
            best_cost = cost
            lo = mid + 1
        else:
            hi = mid - 1

    return best_U, best_k, best_cost


@torch.no_grad()
def select_budgeted_greedy_subspace_exact(
    vecs,
    C1,
    G,
    budget,
    max_rank=None,
    orth_eps=1e-12,
):
    """
    Greedy accept/skip selection under the exact AMARIS feasibility check:

        tr(P C1 P G) <= budget, with P = U U^T.

    Candidates are scanned in the provided order (typically descending generalized
    eigen-score). Each accepted direction is Euclidean-orthonormalized against
    previously accepted directions.
    """
    d, r_total = vecs.shape

    if max_rank is None:
        max_rank = r_total
    else:
        max_rank = min(max_rank, r_total)

    U_running = torch.empty((d, 0), device=vecs.device, dtype=vecs.dtype)
    running_cost = torch.zeros((), device=vecs.device, dtype=vecs.dtype)

    for k in range(max_rank):
        v = vecs[:, k]

        if U_running.shape[1] > 0:
            v = v - U_running @ (U_running.T @ v)

        v_norm = torch.linalg.norm(v)
        if (v_norm <= orth_eps).item():
            continue

        q = v / v_norm
        C1q = C1 @ q
        Gq = G @ q
        c1_diag = q @ C1q
        g_diag = q @ Gq

        if U_running.shape[1] == 0:
            # tr([c] [g]) = c g
            cost_try = c1_diag * g_diag
        else:
            c1_cross = U_running.T @ C1q
            g_cross = U_running.T @ Gq
            # Appending q extends A=U^T C1 U and H=U^T G U as
            # [A a; a^T c] and [H h; h^T g]. Therefore
            # tr(A' H') = tr(AH) + 2 a^T h + c g. This retains all
            # cross terms without recomputing either projected Gram matrix.
            cost_try = running_cost + 2 * (c1_cross @ g_cross) + c1_diag * g_diag

        if (cost_try <= budget).item():
            U_running = torch.cat([U_running, q[:, None]], dim=1)
            running_cost = cost_try

    return U_running, U_running.shape[1], running_cost


@torch.no_grad()
def faster_generalized_eigen_ranking_projector_torch(
    C1,
    C2,
    delta,
    alpha,
    ridge_scale=1e-5,
    eps=1e-12,
    max_rank=None,
    return_P=False,
    add_ridge_to_budget_metric=False,
    budget_C1=None,
    diagnostic_C2=None,
    use_incremental_prefix_scan=False,
    orth_eps=1e-12,
    verbose=True,
    solver="lobpcg",
    lobpcg_tolerance=1e-3,
    lobpcg_max_iterations=50,
    lobpcg_max_retries=10,
    lobpcg_max_ridge_scale=1e-2,
    precomputed_eigensystem=None,
    precomputed_G=None,
):
    assert C1.ndim == 2 and C2.ndim == 2
    assert C1.shape == C2.shape
    assert delta.ndim == 2
    assert delta.shape[1] == C1.shape[0]
    assert 0.0 <= alpha <= 1.0

    device = C1.device
    dtype = C1.dtype
    # symmetrize to prevent numerical issues from tiny asymmetry
    C1 = 0.5 * (C1 + C1.T)
    C2 = 0.5 * (C2 + C2.T)
    if budget_C1 is not None:
        assert budget_C1.shape == C1.shape
        budget_C1 = 0.5 * (budget_C1 + budget_C1.T)
    if diagnostic_C2 is not None:
        assert diagnostic_C2.shape == C2.shape
        diagnostic_C2 = 0.5 * (diagnostic_C2 + diagnostic_C2.T)
    G = delta.T @ delta if precomputed_G is None else precomputed_G
    G = 0.5 * (G + G.T)

    # Preserve raw covariance traces for audit logs. These are deliberately
    # distinct from the (possibly normalized/ridge-adjusted) matrices used by
    # the solver and alpha budget below.
    diagnostic_C1 = C1 if budget_C1 is None else budget_C1
    tr_C1 = torch.trace(diagnostic_C1)
    diagnostic_C2 = C2 if diagnostic_C2 is None else diagnostic_C2
    tr_C2 = torch.trace(diagnostic_C2)

    full_rank = C1.shape[0]

    if precomputed_eigensystem is None:
        scores, vecs, eta, B, solver_used, solver_diagnostics = generalized_eigh_torch_safe(
            C2,
            C1,
            ridge_scale=ridge_scale,
            eps=eps,
            descending=True,
            solver=solver,
            max_eigenpairs=max_rank,
            lobpcg_tolerance=lobpcg_tolerance,
            lobpcg_max_iterations=lobpcg_max_iterations,
            lobpcg_max_retries=lobpcg_max_retries,
            lobpcg_max_ridge_scale=lobpcg_max_ridge_scale,
        )
    else:
        scores = precomputed_eigensystem["scores"].to(device=device, dtype=dtype)
        vecs = precomputed_eigensystem["vecs"].to(device=device, dtype=dtype)
        eta = torch.as_tensor(precomputed_eigensystem["eta"], device=device, dtype=dtype)
        B = C1 + eta * torch.eye(C1.shape[0], device=device, dtype=dtype)
        solver_used = precomputed_eigensystem["solver"]
        solver_diagnostics = precomputed_eigensystem["solver_diagnostics"]

    # The optional budget matrix decouples ranking geometry from exact
    # alpha-budget accounting. The default path preserves prior behavior.
    if budget_C1 is None:
        C1_budget = B if add_ridge_to_budget_metric else C1
    elif add_ridge_to_budget_metric:
        _, _, C1_budget = safe_cholesky_spd(
            budget_C1,
            ridge_scale=ridge_scale,
            eps=eps,
            max_tries=10,
            verbose=False,
        )
    else:
        C1_budget = budget_C1

    tr_C1G = torch.trace(C1_budget @ G).clamp_min(
        torch.as_tensor(eps, device=device, dtype=dtype)
    )
    budget = alpha * tr_C1G

    if max_rank is not None:
        scores = scores[:max_rank]
        vecs = vecs[:, :max_rank]

    # Kept for backward API compatibility; exact AMARIS feasibility requires
    # greedy accept/skip with recomputed Gram-form cost per candidate.
    _ = use_incremental_prefix_scan
    U, rank, D1_raw = select_budgeted_greedy_subspace_exact(
        vecs=vecs,
        C1=C1_budget,
        G=G,
        budget=budget,
        max_rank=max_rank,
        orth_eps=orth_eps,
    )

    if rank == 0:
        PC1_raw = torch.zeros((), device=device, dtype=dtype)
        V2_raw = torch.zeros((), device=device, dtype=dtype)
        VG_raw = torch.zeros((), device=device, dtype=dtype)
        ranking_objective_raw = torch.zeros((), device=device, dtype=dtype)
    else:
        PC1_raw = torch.trace(U.T @ diagnostic_C1 @ U)
        V2_raw = torch.trace(U.T @ diagnostic_C2 @ U)
        VG_raw = torch.trace(U.T @ G @ U)
        # This is the exact ranking objective, including the normalized
        # delta-Gram term when enable_delta_variant is active upstream.
        ranking_objective_raw = torch.trace(U.T @ C2 @ U)

    tr_G_raw = torch.trace(G)
    tr_G = tr_G_raw.clamp_min(torch.as_tensor(eps, device=device, dtype=dtype))
    D1_normalized = D1_raw / (tr_C1G + eps)
    V2_normalized = V2_raw / (tr_C2 + eps)
    VG_normalized = VG_raw / (tr_G + eps)

    result = {
        "U": U,
        "D1": D1_normalized,
        "V2": V2_normalized,
        "VG": VG_normalized,
        "rank": rank,
        "eta": eta,
        "scores": scores,
        "U_candidates": vecs,
        "full_rank": full_rank,
        "candidate_rank": int(vecs.shape[1]),
        "solver": solver_used,
        "solver_diagnostics": solver_diagnostics,
        "tr_C1": tr_C1,
        "tr_PC1": PC1_raw,
        "tr_C2": tr_C2,
        "tr_PC2": V2_raw,
        # Exact final feasibility values for the selected Euclidean projector.
        "constraint_value": D1_raw,
        "constraint_budget": budget,
        "constraint_fraction": D1_normalized,
        # Objective/capture values are evaluated after selection, independent
        # of LOBPCG's convergence tolerance.
        "objective_value": V2_raw,
        "objective_fraction": V2_normalized,
        "ranking_objective_value": ranking_objective_raw,
        "tr_G": tr_G_raw,
        "tr_PG": VG_raw,
        "delta_capture_fraction": VG_normalized,
    }

    if return_P:
        result["P"] = U @ U.T

    if verbose:
        print(
            f"Selected {rank} directions "
            f"with D1={D1_normalized.item():.6f}, "
            f"V2={V2_normalized.item():.6f}, "
            f"alpha={alpha}, "
            f"eta={eta.item():.3e}"
        )

    return result
