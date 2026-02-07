"""Block-structured linear algebra helpers for Newton deterministic solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from dsgekit.exceptions import SolverError

LinearSolverMode = Literal["auto", "dense", "sparse"]
ResolvedLinearSolverMode = Literal["dense", "sparse"]
JitBackendMode = Literal["none", "numba"]
ResolvedJitBackendMode = Literal["none", "numba"]

_NUMBA_ASSEMBLE_DENSE_BLOCKS: Any = None


def _require_numba() -> Any:
    try:
        import numba as nb
    except ImportError as exc:  # pragma: no cover - runtime branch
        raise SolverError(
            "jit_backend='numba' requires the optional speed dependency. "
            "Install with: pip install dsgekit[speed]"
        ) from exc
    return nb


def _assemble_dense_blocks_numba(
    a_blocks: NDArray[np.float64],
    b_blocks: NDArray[np.float64],
    c_blocks: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int]:
    global _NUMBA_ASSEMBLE_DENSE_BLOCKS
    if _NUMBA_ASSEMBLE_DENSE_BLOCKS is None:
        nb = _require_numba()

        @nb.njit(cache=True)  # type: ignore[misc]
        def _kernel(
            a: NDArray[np.float64],
            b: NDArray[np.float64],
            c: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], int]:
            n_periods, n_eq, n_vars = b.shape
            n_rows = n_periods * n_eq
            n_cols = n_periods * n_vars
            jac = np.zeros((n_rows, n_cols), dtype=np.float64)
            nnz = 0

            for t in range(n_periods):
                row_offset = t * n_eq
                curr_col_offset = t * n_vars
                prev_col_offset = (t - 1) * n_vars
                next_col_offset = (t + 1) * n_vars

                for i in range(n_eq):
                    row = row_offset + i
                    for j in range(n_vars):
                        val_b = b[t, i, j]
                        jac[row, curr_col_offset + j] = val_b
                        if val_b != 0.0:
                            nnz += 1

                        if t > 0:
                            val_a = a[t, i, j]
                            jac[row, prev_col_offset + j] = val_a
                            if val_a != 0.0:
                                nnz += 1

                        if t < (n_periods - 1):
                            val_c = c[t, i, j]
                            jac[row, next_col_offset + j] = val_c
                            if val_c != 0.0:
                                nnz += 1

            return jac, nnz

        _NUMBA_ASSEMBLE_DENSE_BLOCKS = _kernel

    return _NUMBA_ASSEMBLE_DENSE_BLOCKS(a_blocks, b_blocks, c_blocks)


@dataclass(slots=True)
class JacobianAssembly:
    """Assembled Jacobian and sparsity metadata."""

    matrix: NDArray[np.float64] | sparse.csr_matrix
    nnz: int


class BlockTridiagonalJacobianBuilder:
    """Assemble Jacobians with time-block tridiagonal structure."""

    def __init__(
        self,
        *,
        n_periods: int,
        n_eq: int,
        n_vars: int,
        use_sparse: bool,
        jit_backend: ResolvedJitBackendMode = "none",
    ) -> None:
        if n_periods < 1:
            raise SolverError(f"n_periods must be >= 1, got {n_periods}")
        if n_eq < 1:
            raise SolverError(f"n_eq must be >= 1, got {n_eq}")
        if n_vars < 1:
            raise SolverError(f"n_vars must be >= 1, got {n_vars}")

        self.n_periods = int(n_periods)
        self.n_eq = int(n_eq)
        self.n_vars = int(n_vars)
        self.use_sparse = bool(use_sparse)
        self.jit_backend = jit_backend
        self.n_rows = self.n_periods * self.n_eq
        self.n_cols = self.n_periods * self.n_vars
        self._nnz = 0

        if self.use_sparse:
            self._rows: list[int] = []
            self._cols: list[int] = []
            self._data: list[float] = []
            self._dense = None
            self._a_blocks = None
            self._b_blocks = None
            self._c_blocks = None
        elif self.jit_backend == "numba":
            self._rows = []
            self._cols = []
            self._data = []
            self._dense = None
            self._a_blocks = np.zeros(
                (self.n_periods, self.n_eq, self.n_vars),
                dtype=np.float64,
            )
            self._b_blocks = np.zeros(
                (self.n_periods, self.n_eq, self.n_vars),
                dtype=np.float64,
            )
            self._c_blocks = np.zeros(
                (self.n_periods, self.n_eq, self.n_vars),
                dtype=np.float64,
            )
        else:
            self._rows = []
            self._cols = []
            self._data = []
            self._dense = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            self._a_blocks = None
            self._b_blocks = None
            self._c_blocks = None

    def _check_block_shape(self, block: NDArray[np.float64], label: str) -> None:
        if block.shape != (self.n_eq, self.n_vars):
            raise SolverError(
                f"Jacobian block '{label}' must have shape ({self.n_eq}, {self.n_vars}), "
                f"got {block.shape}"
            )

    def _add_block(
        self,
        block: NDArray[np.float64],
        *,
        row_offset: int,
        col_offset: int,
        label: str,
    ) -> None:
        self._check_block_shape(block, label)
        if self.use_sparse:
            row_idx, col_idx = np.nonzero(block)
            if row_idx.size == 0:
                return
            values = block[row_idx, col_idx]
            self._rows.extend((row_idx + row_offset).tolist())
            self._cols.extend((col_idx + col_offset).tolist())
            self._data.extend(values.tolist())
            self._nnz += int(values.size)
            return

        assert self._dense is not None
        row_slice = slice(row_offset, row_offset + self.n_eq)
        col_slice = slice(col_offset, col_offset + self.n_vars)
        self._dense[row_slice, col_slice] = block
        self._nnz += int(np.count_nonzero(block))

    def add_period_blocks(
        self,
        *,
        period: int,
        a_t: NDArray[np.float64],
        b_t: NDArray[np.float64],
        c_t: NDArray[np.float64],
    ) -> None:
        """Add period Jacobian blocks A_t/B_t/C_t to global matrix."""
        if period < 0 or period >= self.n_periods:
            raise SolverError(
                f"period out of range for Jacobian assembly: {period} "
                f"(n_periods={self.n_periods})"
            )
        self._check_block_shape(a_t, "A")
        self._check_block_shape(b_t, "B")
        self._check_block_shape(c_t, "C")

        if self.jit_backend == "numba" and not self.use_sparse:
            assert self._a_blocks is not None
            assert self._b_blocks is not None
            assert self._c_blocks is not None
            self._a_blocks[period, :, :] = a_t
            self._b_blocks[period, :, :] = b_t
            self._c_blocks[period, :, :] = c_t
            return

        row_offset = period * self.n_eq
        self._add_block(
            b_t,
            row_offset=row_offset,
            col_offset=period * self.n_vars,
            label="B",
        )
        if period > 0:
            self._add_block(
                a_t,
                row_offset=row_offset,
                col_offset=(period - 1) * self.n_vars,
                label="A",
            )
        if period < (self.n_periods - 1):
            self._add_block(
                c_t,
                row_offset=row_offset,
                col_offset=(period + 1) * self.n_vars,
                label="C",
            )

    def build(self) -> JacobianAssembly:
        """Finalize global Jacobian assembly."""
        if self.use_sparse:
            matrix = sparse.csr_matrix(
                (
                    np.array(self._data, dtype=np.float64),
                    (
                        np.array(self._rows, dtype=np.int32),
                        np.array(self._cols, dtype=np.int32),
                    ),
                ),
                shape=(self.n_rows, self.n_cols),
                dtype=np.float64,
            )
            matrix.eliminate_zeros()
            return JacobianAssembly(matrix=matrix, nnz=int(matrix.nnz))

        if self.jit_backend == "numba":
            assert self._a_blocks is not None
            assert self._b_blocks is not None
            assert self._c_blocks is not None
            matrix, nnz = _assemble_dense_blocks_numba(
                self._a_blocks,
                self._b_blocks,
                self._c_blocks,
            )
            return JacobianAssembly(matrix=matrix, nnz=int(nnz))

        assert self._dense is not None
        return JacobianAssembly(matrix=self._dense, nnz=int(self._nnz))


def resolve_linear_solver_mode(
    *,
    linear_solver: LinearSolverMode | str,
    n_unknowns: int,
    sparse_threshold: int,
) -> ResolvedLinearSolverMode:
    """Resolve auto mode to either dense or sparse linear solve."""
    if n_unknowns < 1:
        raise SolverError(f"n_unknowns must be >= 1, got {n_unknowns}")
    if sparse_threshold < 1:
        raise SolverError(f"sparse_threshold must be >= 1, got {sparse_threshold}")

    mode = str(linear_solver).lower()
    if mode == "auto":
        return "sparse" if n_unknowns >= sparse_threshold else "dense"
    if mode in {"dense", "sparse"}:
        return mode
    raise SolverError(
        "linear_solver must be one of {'auto', 'dense', 'sparse'}, "
        f"got '{linear_solver}'"
    )


def resolve_jit_backend_mode(
    *,
    jit_backend: JitBackendMode | str,
    linear_solver: ResolvedLinearSolverMode,
) -> ResolvedJitBackendMode:
    """Resolve and validate optional JIT backend mode."""
    mode = str(jit_backend).lower()
    if mode == "none":
        return "none"
    if mode != "numba":
        raise SolverError(
            "jit_backend must be one of {'none', 'numba'}, "
            f"got '{jit_backend}'"
        )
    if linear_solver != "dense":
        # Sparse branch uses SciPy sparse operators and does not use dense assembly.
        return "none"
    _require_numba()
    return "numba"


def solve_newton_step(
    *,
    jacobian: NDArray[np.float64] | sparse.csr_matrix,
    rhs: NDArray[np.float64],
    solver_mode: ResolvedLinearSolverMode,
) -> NDArray[np.float64]:
    """Solve linear Newton step, with robust least-squares fallback."""
    rhs_vec = np.asarray(rhs, dtype=np.float64).reshape(-1)

    if solver_mode == "sparse":
        jac_sparse = (
            jacobian.tocsr() if sparse.issparse(jacobian) else sparse.csr_matrix(jacobian)
        )
        try:
            delta = sparse_linalg.spsolve(jac_sparse.tocsc(), rhs_vec)
        except Exception:
            delta = sparse_linalg.lsmr(jac_sparse, rhs_vec)[0]
        return np.asarray(delta, dtype=np.float64).reshape(-1)

    jac_dense = np.asarray(jacobian, dtype=np.float64)
    try:
        delta = np.linalg.solve(jac_dense, rhs_vec)
    except np.linalg.LinAlgError:
        delta, *_ = np.linalg.lstsq(jac_dense, rhs_vec, rcond=None)
    return np.asarray(delta, dtype=np.float64).reshape(-1)
