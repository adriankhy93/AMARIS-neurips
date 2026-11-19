import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from amaris_merge.compute_basis_by_percentage import (
    compute_basis_from_covariances,
    delta_aware_ranking_matrices,
    trace_normalize,
)
from utils.generalized_eigensolver import (
    LOBPCGRetryExhausted,
    faster_generalized_eigen_ranking_projector_torch,
    generalized_eigh_torch_safe,
    select_budgeted_greedy_subspace_exact,
)


class GeneralizedEigensolverTests(unittest.TestCase):
    def test_lobpcg_returns_leading_dense_generalized_eigenvalues(self):
        C1 = torch.diag(torch.arange(1, 13, dtype=torch.float32))
        C2 = torch.diag(torch.arange(12, 0, -1, dtype=torch.float32))
        dense_scores, _, _, _, dense_used, _ = generalized_eigh_torch_safe(
            C2,
            C1,
            solver="dense",
            max_eigenpairs=3,
        )
        partial_scores, partial_vectors, _, _, partial_used, _ = (
            generalized_eigh_torch_safe(
                C2,
                C1,
                solver="lobpcg",
                max_eigenpairs=3,
                lobpcg_tolerance=1e-6,
                lobpcg_max_iterations=100,
            )
        )
        self.assertEqual(dense_used, "dense")
        self.assertEqual(partial_used, "lobpcg")
        self.assertEqual(partial_vectors.shape, (12, 3))
        self.assertTrue(torch.allclose(partial_scores, dense_scores[:3], atol=1e-4))

    def test_dense_solver_reports_residual_convergence(self):
        C1 = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        C2 = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
        _, _, _, _, _, diagnostics = generalized_eigh_torch_safe(
            C2,
            C1,
            solver="dense",
            max_eigenpairs=2,
        )
        self.assertTrue(diagnostics["converged"])
        self.assertLessEqual(
            diagnostics["max_relative_residual"], diagnostics["tolerance"]
        )
        self.assertEqual(diagnostics["candidate_count"], 3)

    def test_lobpcg_retries_internal_eigh_convergence_failure(self):
        C1 = torch.eye(12)
        C2 = torch.eye(12)
        with patch(
            "torch.lobpcg",
            side_effect=RuntimeError(
                "linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned"
            ),
        ) as lobpcg:
            with self.assertRaises(LOBPCGRetryExhausted) as caught:
                generalized_eigh_torch_safe(
                    C2,
                    C1,
                    solver="lobpcg",
                    max_eigenpairs=3,
                    lobpcg_max_retries=2,
                )
        self.assertEqual(lobpcg.call_count, 2)
        self.assertEqual(caught.exception.retries, 2)
        self.assertEqual(caught.exception.candidate_rank, 2)

    def test_alpha_zero_selects_no_positive_cost_direction(self):
        result = faster_generalized_eigen_ranking_projector_torch(
            C1=torch.eye(2, dtype=torch.float64),
            C2=torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64)),
            delta=torch.eye(2, dtype=torch.float64),
            alpha=0.0,
            add_ridge_to_budget_metric=True,
            verbose=False,
        )
        self.assertEqual(result["rank"], 0)
        self.assertEqual(result["U"].shape, (2, 0))

    def test_selected_basis_is_orthonormal_and_within_ridge_budget(self):
        C1 = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
        C2 = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
        delta = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        alpha = 0.35
        result = faster_generalized_eigen_ranking_projector_torch(
            C1=C1,
            C2=C2,
            delta=delta,
            alpha=alpha,
            add_ridge_to_budget_metric=True,
            return_P=True,
            verbose=False,
        )

        U = result["U"]
        self.assertTrue(
            torch.allclose(
                U.T @ U, torch.eye(result["rank"], dtype=U.dtype), atol=1e-10
            )
        )
        B = C1 + result["eta"] * torch.eye(2, dtype=C1.dtype)
        G = delta.T @ delta
        cost = torch.trace(result["P"] @ B @ result["P"] @ G)
        budget = alpha * torch.trace(B @ G)
        self.assertLessEqual(cost.item(), budget.item() + 1e-10)
        self.assertLessEqual(result["D1"].item(), alpha + 1e-10)

    def test_larger_alpha_cannot_reduce_selected_rank(self):
        kwargs = dict(
            C1=torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64)),
            C2=torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)),
            delta=torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)),
            add_ridge_to_budget_metric=True,
            verbose=False,
        )
        conservative = faster_generalized_eigen_ranking_projector_torch(
            alpha=0.2, **kwargs
        )
        permissive = faster_generalized_eigen_ranking_projector_torch(
            alpha=0.9, **kwargs
        )
        self.assertLessEqual(conservative["rank"], permissive["rank"])

    def test_greedy_selection_recomputes_full_cross_term_cost(self):
        vecs = torch.eye(2, dtype=torch.float64)
        C1 = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
        G = C1.clone()
        U, rank, cost = select_budgeted_greedy_subspace_exact(
            vecs=vecs,
            C1=C1,
            G=G,
            budget=torch.tensor(2.0, dtype=torch.float64),
        )

        self.assertEqual(rank, 1)
        self.assertAlmostEqual(cost.item(), 1.0)
        direct_cost = torch.trace((U.T @ C1 @ U) @ (U.T @ G @ U))
        self.assertAlmostEqual(cost.item(), direct_cost.item())
        full_cost = torch.trace(C1 @ G)
        self.assertGreater(full_cost.item(), 2.0)
        self.assertTrue(torch.allclose(U.T @ U, torch.eye(1, dtype=torch.float64)))

    def test_singular_base_covariance_is_stabilized(self):
        result = faster_generalized_eigen_ranking_projector_torch(
            C1=torch.zeros((2, 2), dtype=torch.float64),
            C2=torch.eye(2, dtype=torch.float64),
            delta=torch.eye(2, dtype=torch.float64),
            alpha=1.0,
            add_ridge_to_budget_metric=True,
            verbose=False,
        )
        self.assertGreater(result["eta"].item(), 0.0)
        self.assertEqual(result["rank"], 2)

    def test_delta_variant_trace_normalizes_and_adds_delta_gram(self):
        C1 = torch.diag(torch.tensor([2.0, 6.0], dtype=torch.float64))
        C2 = torch.diag(torch.tensor([4.0, 0.0], dtype=torch.float64))
        delta = torch.diag(torch.tensor([0.0, 3.0], dtype=torch.float64))

        ranking_C1, ranking_C2 = delta_aware_ranking_matrices(C1, C2, delta)

        self.assertTrue(
            torch.allclose(
                ranking_C1, torch.diag(torch.tensor([0.25, 0.75], dtype=torch.float64))
            )
        )
        self.assertTrue(torch.allclose(ranking_C2, torch.eye(2, dtype=torch.float64)))
        self.assertEqual(
            torch.trace(
                trace_normalize(torch.zeros((2, 2), dtype=torch.float64))
            ).item(),
            0.0,
        )

    def test_delta_variant_keeps_raw_alpha_budget(self):
        raw_C1 = torch.diag(torch.tensor([2.0, 6.0], dtype=torch.float64))
        raw_C2 = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
        delta = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
        ranking_C1, ranking_C2 = delta_aware_ranking_matrices(raw_C1, raw_C2, delta)
        alpha = 0.4

        result = faster_generalized_eigen_ranking_projector_torch(
            C1=ranking_C1,
            C2=ranking_C2,
            delta=delta,
            alpha=alpha,
            budget_C1=raw_C1,
            diagnostic_C2=raw_C2,
            add_ridge_to_budget_metric=True,
            return_P=True,
            verbose=False,
        )

        eta_budget = 1e-5 * torch.trace(raw_C1) / raw_C1.shape[0]
        budget_metric = raw_C1 + eta_budget * torch.eye(2, dtype=raw_C1.dtype)
        G = delta.T @ delta
        cost = torch.trace(result["P"] @ budget_metric @ result["P"] @ G)
        self.assertLessEqual(
            cost.item(), (alpha * torch.trace(budget_metric @ G)).item() + 1e-10
        )
        self.assertGreaterEqual(result["VG"].item(), 0.0)
        self.assertLessEqual(result["VG"].item(), 1.0 + 1e-10)

    def test_basis_builder_marks_exhausted_lobpcg_layer_for_beta_fallback(self):
        C1 = {"layer": torch.eye(2)}
        C2 = {"layer": torch.eye(2)}
        delta = {"layer": torch.eye(2)}
        error = LOBPCGRetryExhausted(
            "unstable", retries=10, eta=torch.tensor(1e-3), candidate_rank=80
        )
        with patch(
            "amaris_merge.compute_basis_by_percentage.parse_devices",
            return_value=["cpu"],
        ), patch(
            "amaris_merge.compute_basis_by_percentage.faster_generalized_eigen_ranking_projector_torch",
            side_effect=error,
        ):
            basis, ranks, _, _, _, _, _, fallbacks = compute_basis_from_covariances(
                C1=C1,
                C2=C2,
                Delta=delta,
                max_k=2,
            )
        self.assertEqual(basis, {})
        self.assertEqual(ranks, {})
        self.assertEqual(fallbacks["layer"]["strategy"], "beta_weighted_average")
        self.assertEqual(fallbacks["layer"]["reason"], "lobpcg_retry_exhausted")

    def test_candidate_cache_reuses_generalized_eigensolve_across_alphas(self):
        C1 = {"layer": torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))}
        C2 = {"layer": torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))}
        delta = {"layer": torch.eye(3, dtype=torch.float64)}
        from utils import generalized_eigensolver

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.generalized_eigensolver.generalized_eigh_torch_safe",
            wraps=generalized_eigensolver.generalized_eigh_torch_safe,
        ) as solve, patch(
            "amaris_merge.compute_basis_by_percentage.parse_devices",
            return_value=["cpu"],
        ):
            first = compute_basis_from_covariances(
                C1=C1,
                C2=C2,
                Delta=delta,
                alpha=0.2,
                max_k=2,
                candidate_cache_dir=Path(temp_dir),
            )
            second = compute_basis_from_covariances(
                C1=C1,
                C2=C2,
                Delta=delta,
                alpha=0.8,
                max_k=2,
                candidate_cache_dir=Path(temp_dir),
            )

        self.assertEqual(solve.call_count, 1)
        self.assertIn("layer", first[1])
        self.assertIn("layer", second[1])

    def test_basis_builder_executes_delta_variant_and_records_delta_capture(self):
        C1 = {"layer": torch.diag(torch.tensor([2.0, 6.0], dtype=torch.float64))}
        C2 = {"layer": torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))}
        delta = {"layer": torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))}

        with patch(
            "amaris_merge.compute_basis_by_percentage.parse_devices",
            return_value=["cpu"],
        ):
            _, ranks, d1, v2, vg, full_ranks, traces, fallbacks = (
                compute_basis_from_covariances(
                    C1=C1,
                    C2=C2,
                    Delta=delta,
                    alpha=0.4,
                    max_k=2,
                    enable_delta_variant=True,
                )
            )

        self.assertIn("layer", ranks)
        self.assertLessEqual(d1["layer"], 0.4 + 1e-10)
        self.assertGreaterEqual(v2["layer"], 0.0)
        self.assertGreaterEqual(vg["layer"], 0.0)
        self.assertEqual(full_ranks["layer"], 2)
        self.assertTrue(
            {
                "tr_C1",
                "tr_PC1",
                "tr_C2",
                "tr_PC2",
                "tr_G",
                "tr_PG",
                "constraint_value",
                "constraint_budget",
                "constraint_fraction",
                "objective_value",
                "objective_fraction",
                "ranking_objective_value",
                "delta_capture_fraction",
                "eta",
                "solver",
                "solver_convergence",
            }.issubset(traces["layer"])
        )
        self.assertAlmostEqual(traces["layer"]["tr_C1"], 8.0)
        self.assertAlmostEqual(traces["layer"]["tr_C2"], 5.0)
        self.assertLessEqual(
            traces["layer"]["constraint_value"],
            traces["layer"]["constraint_budget"] + 1e-10,
        )
        self.assertEqual(fallbacks, {})


if __name__ == "__main__":
    unittest.main()
