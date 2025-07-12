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

from fractions import Fraction
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from hamilterm import constants, options, terms
from hamilterm import elements as mel

if TYPE_CHECKING:
    from hamilterm.symmat import SymbolicMatrix


@overload
def construct_n_operator_matrices(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int,
) -> list[NDArray[np.float64]]: ...


@overload
def construct_n_operator_matrices(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: sp.Symbol,
) -> list[sp.MutableDenseMatrix]: ...


def construct_n_operator_matrices(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int | sp.Symbol,
) -> list[NDArray[np.float64]] | list[sp.MutableDenseMatrix]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J

    Returns:
        list[NDArray[np.float64]]: N operator matrices
    """
    # The number of basis functions determine the size of the N operator matrix.
    dim: int = len(basis_fns)

    # Each N^{2k} operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc. Always initialize the
    # operator matrices up to N^12 - if MAX_N_POWER is less than 12, the unused matrices will have
    # all their elements equal to zero.
    n_op_mats: list[NDArray[np.float64]] | list[sp.MutableDenseMatrix]
    if isinstance(j_qn, int):
        n_op_mats = [np.zeros((dim, dim)) for _ in range(6)]
    else:
        n_op_mats = [sp.zeros(dim) for _ in range(6)]

    # Form the N^2 matrix using the matrix elements above.
    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel.n_squared(i, j, basis_fns, s_qn, j_qn)

    # The following N^{2k} matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, options.MAX_N_POWER // 2):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


@overload
def build_hamiltonian(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int,
    consts: constants.NumericConstants,
) -> NDArray[np.float64]: ...


@overload
def build_hamiltonian(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: sp.Symbol,
    consts: constants.SymbolicConstants,
) -> sp.MutableDenseMatrix: ...


def build_hamiltonian(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int | sp.Symbol,
    consts: constants.NumericConstants | constants.SymbolicConstants,
) -> NDArray[np.float64] | sp.MutableDenseMatrix:
    """Build Hamiltonian matrix for either numeric or symbolic computation.

    This function is overloaded to provide proper type hints while maintaining
    a single implementation.
    """
    dim: int = len(basis_fns)
    is_symbolic = isinstance(j_qn, sp.Symbol)
    n_op_mats = construct_n_operator_matrices(basis_fns, s_qn, j_qn)

    h_mat: sp.MutableDenseMatrix | NDArray[np.float64] = (
        sp.zeros(dim) if is_symbolic else np.zeros((dim, dim))
    )

    switch_r, switch_so, switch_ss, switch_sr, switch_ld = map(
        int,
        [
            options.INCLUDE_R,
            options.INCLUDE_SO,
            options.INCLUDE_SS,
            options.INCLUDE_SR,
            options.INCLUDE_LD,
        ],
    )

    # TODO: 25/07/12 - I'm aware that this is really janky, but I'm not sure how to get the type
    #       checker to narrow down while not repeating code. Oh, well; it works fine.

    if is_symbolic:
        assert isinstance(consts, constants.SymbolicConstants)
        n_op_mats = cast("list[SymbolicMatrix[sp.Expr]]", n_op_mats)

        for i in range(dim):
            for j in range(dim):
                h_mat[i, j] = (
                    switch_r * terms.rotational(i, j, n_op_mats, consts.rotational)
                    + switch_so
                    * terms.spin_orbit(i, j, basis_fns, s_qn, n_op_mats, consts.spin_orbit)
                    + switch_ss
                    * terms.spin_spin(i, j, basis_fns, s_qn, n_op_mats, consts.spin_spin)
                    + switch_sr
                    * terms.spin_rotation(
                        i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.spin_rotation
                    )
                    + switch_ld
                    * terms.lambda_doubling(
                        i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.lambda_doubling
                    )
                )
    else:
        assert isinstance(consts, constants.NumericConstants)
        n_op_mats = cast("list[NDArray[np.float64]]", n_op_mats)

        for i in range(dim):
            for j in range(dim):
                h_mat[i, j] = (
                    switch_r * terms.rotational(i, j, n_op_mats, consts.rotational)
                    + switch_so
                    * terms.spin_orbit(i, j, basis_fns, s_qn, n_op_mats, consts.spin_orbit)
                    + switch_ss
                    * terms.spin_spin(i, j, basis_fns, s_qn, n_op_mats, consts.spin_spin)
                    + switch_sr
                    * terms.spin_rotation(
                        i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.spin_rotation
                    )
                    + switch_ld
                    * terms.lambda_doubling(
                        i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.lambda_doubling
                    )
                )

    return h_mat


def parse_term_symbol(term_symbol: str) -> tuple[Fraction, int]:
    """Parse the molecular term symbol into the quantum numbers S and Λ.

    Args:
        term_symbol (str): Molecular term symbol, e.g., "2Pi" or "3Sigma"

    Returns:
        tuple[Fraction, int]: Quantum numbers S and Λ
    """
    spin_multiplicity: int = int(term_symbol[0])
    s_qn: Fraction = Fraction(spin_multiplicity - 1, 2)
    term: str = term_symbol[1:]
    lambda_qn: int = options.LAMBDA_INT_MAP[term]

    return s_qn, lambda_qn


def generate_basis_fns(s_qn: Fraction, lambda_qn: int) -> list[tuple[int, Fraction, Fraction]]:
    """Construct the Hund's case (a) basis set |Λ, Σ; Ω>.

    Args:
        s_qn (Fraction): Quantum number S
        lambda_qn (int): Quantum number Λ

    Returns:
        list[tuple[int, Fraction, Fraction]]: List of basis vectors |Λ, Σ; Ω>
    """
    # Possible values for Σ = S, S - 1, ..., -S. There are 2S + 1 total values of Σ.
    sigmas: list[Fraction] = [-s_qn + i for i in range(int(2 * s_qn) + 1)]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas: list[int] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, Fraction, Fraction]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: Fraction = lam + sigma
            basis_fns.append((lam, sigma, omega))

    return basis_fns
