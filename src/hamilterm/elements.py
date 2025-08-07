# module elements.py
"""Contains functions for computing matrix elements."""

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

import math

import numpy as np
from numpy.typing import NDArray

from hamilterm import utils


def j_squared(j_qn: float) -> float:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn (float): Quantum number J

    Returns:
        int: Matrix element J(J + 1)
    """
    return j_qn * (j_qn + 1)


def j_plus(j_qn: float, omega_qn_j: float) -> float:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Args:
        j_qn (float): Quantum number J
        omega_qn_j (float): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2)
    """
    return math.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1))


def j_plus_vec(j_qn: float, omega_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1))


def j_minus(j_qn: float, omega_qn_j: float) -> float:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Args:
        j_qn (float): Quantum number J
        omega_qn_j (float): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2)
    """
    return math.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1))


def j_minus_vec(j_qn: float, omega_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1))


def s_squared(s_qn: float) -> float:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn (float): Quantum number S

    Returns:
        float: Matrix element S(S + 1)
    """
    return s_qn * (s_qn + 1)


def s_squared_vec(s_qn: float) -> float:
    return s_qn * (s_qn + 1)


def s_plus(s_qn: float, sigma_qn_j: float) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Args:
        s_qn (float): Quantum number S
        sigma_qn_j (float): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def s_plus_vec(s_qn: float, sigma_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def s_minus(s_qn: float, sigma_qn_j: float) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Args:
        s_qn (float): Quantum number S
        sigma_qn_j (float): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


def s_minus_vec(s_qn: float, sigma_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


def n_squared(
    i: int, j: int, basis_fns: list[tuple[int, float, float]], s_qn: float, j_qn: float
) -> float:
    """Return matrix elements for the N^2 operator.

    N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S
        j_qn (float): Quantum number J

    Returns:
        float | float: Matrix elements for J^2 + S^2 - 2JzSz - (J+S- + J-S+)
    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    # ⟨J, S, Ω, Σ|J^2 + S^2 - 2JzSz|J, S, Ω, Σ⟩ = J(J + 1) + S(S + 1) - 2ΩΣ
    if i == j:
        return j_squared(j_qn) + s_squared(s_qn) - 2 * omega_qn_j * sigma_qn_j

    # ⟨J, S, Ω - 1, Σ - 1|-(J+S-)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if omega_qn_i == omega_qn_j - 1 and sigma_qn_i == sigma_qn_j - 1:
        return -j_plus(j_qn, omega_qn_j) * s_minus(s_qn, sigma_qn_j)

    # ⟨J, S, Ω + 1, Σ + 1|-(J-S+)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if omega_qn_i == omega_qn_j + 1 and sigma_qn_i == sigma_qn_j + 1:
        return -j_minus(j_qn, omega_qn_j) * s_plus(s_qn, sigma_qn_j)

    return 0.0


def lz_sz(m: int, n: int, basis_fns: list[tuple[int, float, float]]) -> int | float:
    """Return matrix elements for the LzSz operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>

    Returns:
        int | float: Matrix elements for LzSz
    """
    lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

    # Operator is completely diagonal, so only m = n terms exist.
    if m == n:
        # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
        return lambda_qn_n * sigma_qn_n

    return 0


def lz_sz_vec(
    lambda_basis: NDArray[np.int64], sigma_basis: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Operator is completely diagonal, so only m = n terms exist.

    # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
    return np.diag(lambda_basis * sigma_basis)


def three_sz2_minus_s2(
    m: int, n: int, basis_fns: list[tuple[int, float, float]], s_qn: float
) -> int | float:
    """Return matrix elements for the 3Sz^2 - S^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S

    Returns:
        int | float: Matrix elements for 3Sz^2 - S^2
    """
    sigma_qn_n = basis_fns[n][1]

    # Operator is completely diagonal, so only m = n terms exist.
    if m == n:
        # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
        return 3 * sigma_qn_n**2 - s_squared(s_qn)

    return 0


def three_sz2_minus_s2_vec(sigma_basis: NDArray[np.float64], s_qn: float) -> NDArray[np.float64]:
    # Operator is completely diagonal, so only m = n terms exist.

    # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
    return np.diag(3 * sigma_basis**2 - s_squared_vec(s_qn))


def n_dot_s(
    m: int, n: int, basis_fns: list[tuple[int, float, float]], s_qn: float, j_qn: float
) -> float:
    """Return matrix elements for the N·S operator.

    N·S = JzSz + 0.5(J+S- + J-S+) - S^2

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S
        j_qn (float): Quantum number J

    Returns:
        float: Matrix elements for JzSz + 0.5(J+S- + J-S+) - S^2
    """
    _, sigma_qn_m, omega_qn_m = basis_fns[m]
    _, sigma_qn_n, omega_qn_n = basis_fns[n]

    # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
    if m == n:
        return omega_qn_n * sigma_qn_n - s_squared(s_qn)

    # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if omega_qn_m == omega_qn_n - 1 and sigma_qn_m == sigma_qn_n - 1:
        return 0.5 * j_plus(j_qn, omega_qn_n) * s_minus(s_qn, sigma_qn_n)

    # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if omega_qn_m == omega_qn_n + 1 and sigma_qn_m == sigma_qn_n + 1:
        return 0.5 * j_minus(j_qn, omega_qn_n) * s_plus(s_qn, sigma_qn_n)

    return 0.0


def n_dot_s_vec(
    sigma_basis: NDArray[np.float64],
    omega_basis: NDArray[np.float64],
    s_qn: float,
    j_qn: float,
) -> NDArray[np.float64]:
    dim: int = sigma_basis.size

    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis)

    result: NDArray[np.float64] = np.zeros((dim, dim))

    # Fully diagonal matrix element ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
    np.fill_diagonal(result, omega_basis * sigma_basis - s_squared_vec(s_qn))

    # Create masks to denote where the off-diagonal array elements are
    # Denote the areas in the array where Ω_i = Ω_j - 1 and Σ_i = Σ_j - 1
    mask_minus: NDArray[np.bool] = (omega_i == omega_j - 1) & (sigma_i == sigma_j - 1)
    # Denote the areas in the array where Ω_i = Ω_j + 1 and Σ_i = Σ_j + 1
    mask_plus: NDArray[np.bool] = (omega_i == omega_j + 1) & (sigma_i == sigma_j + 1)

    # TODO: 25/07/16 - This method computes the following elements for all locations in the matrix,
    #       after which the masks are applied. It might be a bit more efficient to only compute
    #       these where the mask is true, maybe by applying the mask to the Ω and Σ matrices first.

    # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    term_minus: NDArray[np.float64] = 0.5 * j_plus_vec(j_qn, omega_j) * s_minus_vec(s_qn, sigma_j)
    # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    term_plus: NDArray[np.float64] = 0.5 * j_minus_vec(j_qn, omega_j) * s_plus_vec(s_qn, sigma_j)

    result[mask_minus] = term_minus[mask_minus]
    result[mask_plus] = term_plus[mask_plus]

    return result


def sp2_plus_sm2(m: int, n: int, basis_fns: list[tuple[int, float, float]], s_qn: float) -> float:
    """Return matrix elements for the S+^2 + S-^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S

    Returns:
        float: Matrix elements for S+^2 + S-^2
    """
    lambda_qn_m, sigma_qn_m, _ = basis_fns[m]
    lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

    # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
    if lambda_qn_m == lambda_qn_n - 2 and sigma_qn_m == sigma_qn_n + 2:
        return s_plus(s_qn, sigma_qn_n) * s_plus(s_qn, sigma_qn_n + 1)

    # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
    if lambda_qn_m == lambda_qn_n + 2 and sigma_qn_m == sigma_qn_n - 2:
        return s_minus(s_qn, sigma_qn_n) * s_minus(s_qn, sigma_qn_n - 1)

    return 0.0


def sp2_plus_sm2_vec(
    lambda_basis: NDArray[np.int64], sigma_basis: NDArray[np.float64], s_qn: float
) -> NDArray[np.float64]:
    dim: int = lambda_basis.size

    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis)
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis)

    result: NDArray[np.float64] = np.zeros((dim, dim))

    # Create masks to denote where the off-diagonal array elements are
    # Denote the areas in the array where Λ_i = Λ_j - 2 and Σ_i = Σ_j + 2
    mask_plus: NDArray[np.bool] = (lambda_i == lambda_j - 2) & (sigma_i == sigma_j + 2)
    # Denote the areas in the array where Λ_i = Λ_j + 2 and Σ_i = Σ_j - 2
    mask_minus: NDArray[np.bool] = (lambda_i == lambda_j + 2) & (sigma_i == sigma_j - 2)

    # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
    term_plus: NDArray[np.float64] = s_plus_vec(s_qn, sigma_j) * s_plus_vec(s_qn, sigma_j + 1)
    # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
    term_minus: NDArray[np.float64] = s_minus_vec(s_qn, sigma_j) * s_minus_vec(s_qn, sigma_j - 1)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result


def jpsp_plus_jmsm(
    m: int, n: int, basis_fns: list[tuple[int, float, float]], s_qn: float, j_qn: float
) -> float:
    """Return matrix elements for the J+S+ + J-S- operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (float): Quantum number S
        j_qn (float): Quantum number J

    Returns:
        float: Matrix elements for J+S+ + J-S-
    """
    lambda_qn_m, sigma_qn_m, omega_qn_m = basis_fns[m]
    lambda_qn_n, sigma_qn_n, omega_qn_n = basis_fns[n]

    # ⟨Λ - 2, Ω - 1, Σ + 1|J+S+|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if (
        lambda_qn_m == lambda_qn_n - 2
        and sigma_qn_m == sigma_qn_n + 1
        and omega_qn_m == omega_qn_n - 1
    ):
        return j_plus(j_qn, omega_qn_n) * s_plus(s_qn, sigma_qn_n)

    # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if (
        lambda_qn_m == lambda_qn_n + 2
        and sigma_qn_m == sigma_qn_n - 1
        and omega_qn_m == omega_qn_n + 1
    ):
        return j_minus(j_qn, omega_qn_n) * s_minus(s_qn, sigma_qn_n)

    return 0.0


def jpsp_plus_jmsm_vec(
    lambda_basis: NDArray[np.int64],
    sigma_basis: NDArray[np.float64],
    omega_basis: NDArray[np.float64],
    s_qn: float,
    j_qn: float,
) -> NDArray[np.float64]:
    dim: int = lambda_basis.size

    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis)
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis)

    result: NDArray[np.float64] = np.zeros((dim, dim))

    # Create masks to denote where the off-diagonal array elements are
    # Denote the areas in the array where Λ_i = Λ_j - 2, Ω_i = Ω_j - 1, and Σ_i = Σ_j + 1
    mask_plus: NDArray[np.bool] = (
        (lambda_i == lambda_j - 2) & (omega_i == omega_j - 1) & (sigma_i == sigma_j + 1)
    )
    # Denote the areas in the array where Λ_i = Λ_j + 2, Ω_i = Ω_j + 1, and Σ_i = Σ_j - 1
    mask_minus: NDArray[np.bool] = (
        (lambda_i == lambda_j + 2) & (omega_i == omega_j + 1) & (sigma_i == sigma_j - 1)
    )

    # ⟨Λ - 2, Ω - 1, Σ + 1|J+S+|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    term_plus: NDArray[np.float64] = j_plus_vec(j_qn, omega_j) * s_plus_vec(s_qn, sigma_j)
    # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    term_minus: NDArray[np.float64] = j_minus_vec(j_qn, omega_j) * s_minus_vec(s_qn, sigma_j)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result


def jp2_plus_jm2(
    m: int,
    n: int,
    basis_fns: list[tuple[int, float, float]],
    j_qn: float,
) -> float:
    """Return matrix elements for the J+^2 + J-^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, float, float]]): List of basis vectors |Λ, Σ; Ω>
        j_qn (float): Quantum number J

    Returns:
        float: Matrix elements for J+^2 + J-^2
    """
    lambda_qn_m, _, omega_qn_m = basis_fns[m]
    lambda_qn_n, _, omega_qn_n = basis_fns[n]

    # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J
    #       is an anomalously commutative operator.
    # ⟨Λ - 2, Ω - 2|J+^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
    if lambda_qn_m == lambda_qn_n - 2 and omega_qn_m == omega_qn_n - 2:
        return j_plus(j_qn, omega_qn_n) * j_plus(j_qn, omega_qn_n - 1)

    # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
    # ⟨Λ + 2, Ω + 2|J-^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
    if lambda_qn_m == lambda_qn_n + 2 and omega_qn_m == omega_qn_n + 2:
        return j_minus(j_qn, omega_qn_n) * j_minus(j_qn, omega_qn_n + 1)

    return 0.0


def jp2_plus_jm2_vec(
    lambda_basis: NDArray[np.int64], omega_basis: NDArray[np.float64], j_qn: float
) -> NDArray[np.float64]:
    dim: int = lambda_basis.size

    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis)

    result: NDArray[np.float64] = np.zeros((dim, dim))

    # Create masks to denote where the off-diagonal array elements are
    # Denote the areas in the array where Λ_i = Λ_j - 2 and Ω_i = Ω_j - 2
    mask_plus: NDArray[np.bool] = (lambda_i == lambda_j - 2) & (omega_i == omega_j - 2)
    # Denote the areas in the array where Λ_i = Λ_j + 2 and Ω_i = Ω_j + 2
    mask_minus: NDArray[np.bool] = (lambda_i == lambda_j + 2) & (omega_i == omega_j + 2)

    # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J
    #       is an anomalously commutative operator.
    # ⟨Λ - 2, Ω - 2|J+^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
    term_plus: NDArray[np.float64] = j_plus_vec(j_qn, omega_j) * j_plus_vec(j_qn, omega_j - 1)
    # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
    # ⟨Λ + 2, Ω + 2|J-^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
    term_minus: NDArray[np.float64] = j_minus_vec(j_qn, omega_j) * j_minus_vec(j_qn, omega_j + 1)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result
