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

from typing import cast

import sympy as sp
from sympy import Integer, MutableDenseMatrix, Rational, Symbol

from hamilterm import elements as mel
from hamilterm import options


def safe_rational(num: int, denom: int) -> Rational:
    """Ensure a rational return value.

    Args:
        num: Numerator.
        denom: Denominator.

    Raises:
        ValueError: If NaN or Infinity is encountered.

    Returns:
        A guaranteed `sympy.Rational` type.
    """
    r = Rational(num, denom)

    if not isinstance(r, Rational):
        raise ValueError(f"Expected sympy.Rational, got {r}.")

    return r


def construct_n_operator_matrices(
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
    max_n_index: int,
    dim: int,
) -> list[MutableDenseMatrix]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        max_n_index: Index of the maximum N^(2k) matrix to compute.
        dim: Dimension of the Hamiltonian.

    Returns:
        N operator matrices.
    """
    # Each N^{2k} operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc. Always initialize the
    # operator matrices up to N^12 - if MAX_N_POWER is less than 12, the unused matrices will have
    # all their elements equal to zero.
    n_op_mats: list[MutableDenseMatrix] = [sp.zeros(dim) for _ in range(6)]

    # Form the N^2 matrix using the matrix elements above.
    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel.n_squared(i, j, basis_fns, s_qn, j_qn)

    # The following N^(2k) matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, max_n_index):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def parse_term_symbol(term_symbol: str) -> tuple[Rational, Integer]:
    """Parse the molecular term symbol into the quantum numbers S and Λ.

    Args:
        term_symbol: Molecular term symbol, e.g., "2P" or "3S".

    Returns:
        Quantum numbers S and Λ.
    """
    spin_multiplicity = int(term_symbol[0])
    s_qn = safe_rational(spin_multiplicity - 1, 2)
    term = term_symbol[1:]
    lambda_qn = Integer(options.LAMBDA_INT_MAP[term])

    return s_qn, lambda_qn


def generate_basis_fns(
    s_qn: Rational, lambda_qn: Integer
) -> list[tuple[Integer, Rational, Rational]]:
    """Construct the Hund's case (a) basis set |Λ, Σ; Ω⟩.

    Args:
        s_qn: Quantum number S.
        lambda_qn: Quantum number Λ.

    Returns:
        List of basis vectors |Λ, Σ; Ω⟩.
    """
    sigmas = [cast("Rational", -s_qn + i) for i in range(int(2 * s_qn) + 1)]

    lambdas = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[Integer, Rational, Rational]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega = cast("Rational", lam + sigma)
            basis_fns.append((lam, sigma, omega))

    return basis_fns
