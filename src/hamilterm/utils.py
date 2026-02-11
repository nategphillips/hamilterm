# module utils.py
"""Contains various utility functions for computing the Hamiltonian."""

# Copyright (C) 2025-2026 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING

import numpy as np

from hamilterm import elements as mel
from hamilterm import options

if TYPE_CHECKING:
    from numpy.typing import NDArray


def construct_n_operator_matrices(
    s_qn: float,
    j_qn: float,
    sigma_basis_vec: NDArray[np.float64],
    omega_basis_vec: NDArray[np.float64],
    max_n_index: int,
    dim: int,
) -> list[NDArray[np.float64]]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        omega_basis_vec: Basis vector of Ω, one value per row of the Hamiltonian.
        max_n_index: Index of the maximum N^(2k) matrix to compute.
        dim: Dimension of the Hamiltonian.

    Returns:
        N operator matrices.
    """
    # Each N^{2k} operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc. Always initialize the
    # operator matrices up to N^12 - if MAX_N_POWER is less than 12, the unused matrices will have
    # all their elements equal to zero.
    n_op_mats = [np.zeros((dim, dim)) for _ in range(6)]

    # Form the N^2 matrix using the matrix elements above.
    n_op_mats[0] = mel.n_squared(sigma_basis_vec, omega_basis_vec, s_qn, j_qn, dim)

    # The following N^(2k) matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, max_n_index):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def parse_term_symbol(term_symbol: str) -> tuple[float, int]:
    """Parse the molecular term symbol into the quantum numbers S and Λ.

    Args:
        term_symbol: Molecular term symbol, e.g., "2P" or "3S".

    Returns:
        Quantum numbers S and Λ.
    """
    spin_multiplicity = int(term_symbol[0])
    s_qn = 0.5 * (spin_multiplicity - 1)
    term = term_symbol[1:]
    lambda_qn = options.LAMBDA_INT_MAP[term]

    return s_qn, lambda_qn


def generate_basis_fns(s_qn: float, lambda_qn: int) -> list[tuple[int, float, float]]:
    """Construct the Hund's case (a) basis set |Λ, Σ; Ω⟩.

    Args:
        s_qn: Quantum number S.
        lambda_qn: Quantum number Λ.

    Returns:
        List of basis vectors |Λ, Σ; Ω⟩.
    """
    # Possible values for Σ = S, S - 1, ..., -S. There are 2S + 1 total values of Σ.
    sigmas = [-s_qn + i for i in range(int(2 * s_qn) + 1)]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, float, float]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: float = lam + sigma
            basis_fns.append((lam, sigma, omega))

    return basis_fns


def basis_vectors(
    basis_fns: list[tuple[int, float, float]], dim: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Construct basis vectors of Λ, Σ, and Ω for use with vectorized functions.

    Args:
        basis_fns: List of basis functions |Λ, Σ; Ω⟩.
        dim: Dimension of the Hamiltonian.

    Returns:
        Basis vectors for Λ, Σ, and Ω.
    """
    lambda_basis = np.empty(dim, dtype=np.int64)
    sigma_basis = np.empty(dim, dtype=np.float64)
    omega_basis = np.empty(dim, dtype=np.float64)

    for i, (lam, sig, omg) in enumerate(basis_fns):
        lambda_basis[i] = lam
        sigma_basis[i] = sig
        omega_basis[i] = omg

    return lambda_basis, sigma_basis, omega_basis


def form_basis_matrices(basis_vector: NDArray) -> tuple[NDArray, NDArray]:
    """Constructs basis matrices for i and j given a basis vector.

    A basis vector will look something like `[-1, 0, 1]`. To create masks and perform element-wise
    logic, we need to form basis matrices. For the i matrices, each row is filled with the same
    number. For the j matrices, each column is filled with the same number. As an example, a basis
    of `[-1, 0, 1]` would form:

    ```
    M_j = [[-1, 0, 1], and M_i = [[-1, -1, -1],
           [-1, 0, 1],            [0 ,  0,  0],
           [-1, 0, 1]]            [1 ,  1,  1]]
    ```

    The i matrices are formed by stacking column vectors rightward, while the j matrices are
    formed by stacking row vectors downward. This is more obvious when a sample Hamiltonian is
    labeled with its basis states:

    ```
                j
           -1   0   1
      -1 |             |
    i  0 | Hamiltonian |
       1 |             |
    ```

    Args:
        basis_vector: A basis vector for Λ, Σ, or Ω

    Returns:
        Basis matrices for i and j.
    """
    basis_matrix_j = np.tile(basis_vector, (basis_vector.size, 1))
    basis_matrix_i = basis_matrix_j.T

    return basis_matrix_i, basis_matrix_j
