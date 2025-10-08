# module utils.py
"""Contains various utility functions for computing the Hamiltonian."""

# Copyright (C) 2025 Nathan G. Phillips

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

from typing import cast

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy import Integer, MutableDenseMatrix, Rational, Symbol

from hamilterm import elements as mel
from hamilterm import options


def safe_rational(num: int, denom: int) -> Rational:
    """Ensure a rational return value.

    Args:
        num (int): Numerator
        denom (int): Denominator

    Raises:
        ValueError: If NaN or Infinity is encountered

    Returns:
        sp.Rational: A guaranteed sp.Rational type
    """
    r = Rational(num, denom)

    if not isinstance(r, Rational):
        raise ValueError(f"Expected sp.Rational, got {r}.")

    return r


def construct_n_operator_matrices_num(
    s_qn: float,
    j_qn: float,
    sigma_basis: NDArray[np.float64],
    omega_basis: NDArray[np.float64],
    max_n_index: int,
    dim: int,
) -> list[NDArray[np.float64]]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S
        j_qn (float): Quantum number J
        max_n_index (int): Index of the maximum N^{2k} matrix to compute

    Returns:
        list[NDArray[np.float64]]: N operator matrices
    """
    # Each N^{2k} operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc. Always initialize the
    # operator matrices up to N^12 - if MAX_N_POWER is less than 12, the unused matrices will have
    # all their elements equal to zero.
    n_op_mats: list[NDArray[np.float64]] = [np.zeros((dim, dim)) for _ in range(6)]

    # Form the N^2 matrix using the matrix elements above.
    n_op_mats[0] = mel.n_squared_num(sigma_basis, omega_basis, s_qn, j_qn, dim)

    # The following N^{2k} matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, max_n_index):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def construct_n_operator_matrices_sym(
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
    max_n_index: int,
    dim: int,
) -> list[MutableDenseMatrix]:
    n_op_mats: list[MutableDenseMatrix] = [sp.zeros(dim) for _ in range(6)]

    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel.n_squared_sym(i, j, basis_fns, s_qn, j_qn)

    for i in range(1, max_n_index):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def parse_term_symbol_num(term_symbol: str) -> tuple[float, int]:
    """Parse the molecular term symbol into the quantum numbers S and Λ.

    Args:
        term_symbol (str): Molecular term symbol, e.g., "2P" or "3S"

    Returns:
        tuple[float, int]: Quantum numbers S and Λ
    """
    spin_multiplicity: int = int(term_symbol[0])
    s_qn: float = 0.5 * (spin_multiplicity - 1)
    term: str = term_symbol[1:]
    lambda_qn: int = options.LAMBDA_INT_MAP[term]

    return s_qn, lambda_qn


def parse_term_symbol_sym(term_symbol: str) -> tuple[Rational, Integer]:
    spin_multiplicity: int = int(term_symbol[0])
    s_qn: Rational = safe_rational(spin_multiplicity - 1, 2)
    term: str = term_symbol[1:]
    lambda_qn: Integer = Integer(options.LAMBDA_INT_MAP[term])

    return s_qn, lambda_qn


def generate_basis_fns_num(s_qn: float, lambda_qn: int) -> list[tuple[int, float, float]]:
    """Construct the Hund's case (a) basis set |Λ, Σ; Ω>.

    Args:
        s_qn (float): Quantum number S
        lambda_qn (int): Quantum number Λ

    Returns:
        list[tuple[int, float, float]]: List of basis vectors |Λ, Σ; Ω>
    """
    # Possible values for Σ = S, S - 1, ..., -S. There are 2S + 1 total values of Σ.
    sigmas: list[float] = [-s_qn + i for i in range(int(2 * s_qn) + 1)]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas: list[int] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, float, float]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: float = lam + sigma
            basis_fns.append((lam, sigma, omega))

    return basis_fns


def generate_basis_fns_sym(
    s_qn: Rational, lambda_qn: Integer
) -> list[tuple[Integer, Rational, Rational]]:
    sigmas: list[Rational] = [cast("Rational", -s_qn + i) for i in range(int(2 * s_qn) + 1)]

    lambdas: list[Integer] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[Integer, Rational, Rational]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: Rational = cast("Rational", lam + sigma)
            basis_fns.append((lam, sigma, omega))

    return basis_fns


def basis_vectors_num(
    basis_fns: list[tuple[int, float, float]], dim: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Construct basis arrays of Λ, Σ, and Ω for use with vectorized functions.

    Args:
        basis_fns (list[tuple[int, float, float]]): List of basis functions |Λ, Σ; Ω>

    Returns:
        tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]: Basis vectors for Λ, Σ,
            and Ω
    """
    lambda_basis: NDArray[np.int64] = np.empty(dim, dtype=np.int64)
    sigma_basis: NDArray[np.float64] = np.empty(dim, dtype=np.float64)
    omega_basis: NDArray[np.float64] = np.empty(dim, dtype=np.float64)

    for i, (lam, sig, omg) in enumerate(basis_fns):
        lambda_basis[i] = lam
        sigma_basis[i] = sig
        omega_basis[i] = omg

    return lambda_basis, sigma_basis, omega_basis


def form_basis_matrices_num(basis_vector: NDArray) -> tuple[NDArray, NDArray]:
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
        basis_vector (NDArray): A basis vector for Λ, Σ, or Ω

    Returns:
        tuple[NDArray, NDArray]: Basis matrices for i and j
    """
    basis_matrix_j: NDArray = np.tile(basis_vector, (basis_vector.size, 1))
    basis_matrix_i: NDArray = basis_matrix_j.T

    return basis_matrix_i, basis_matrix_j
