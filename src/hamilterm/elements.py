# module elements.py
"""Contains functions for computing matrix elements."""

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

from hamilterm import utils

if TYPE_CHECKING:
    from numpy.typing import NDArray


def j_squared(j_qn: float) -> float:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn: Quantum number J.

    Returns:
        The diagonal matrix element J(J + 1).
    """
    return j_qn * (j_qn + 1)


def j_plus(j_qn: float, omega_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Note that J is an anomalously commutative angular momentum operator.

    Args:
        j_qn: Quantum number J.
        omega_qn_j: Quantum number Ω.

    Returns:
        The off-diagonal matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2).
    """
    term = j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1)

    # NOTE: 25/08/12 - This check must be performed to avoid runtime errors in numpy. Unfortunately,
    #       it slows down the computations by a noticeable margin. In the future, it might be worth
    #       returning an error elsewhere if the values of J or S will result in a term that's less
    #       than zero. The term.min() < 0.0 approach is noticeably faster than np.any(term < 0.0)
    #       for some reason.
    if term.min() < 0.0:
        return np.zeros_like(omega_qn_j)

    return np.sqrt(term)


def j_minus(j_qn: float, omega_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Note that J is an anomalously commutative angular momentum operator.

    Args:
        j_qn: Quantum number J.
        omega_qn_j: Quantum number Ω.

    Returns:
        The off-diagonal matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2).
    """
    term = j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1)

    # See note in `j_plus`.
    if term.min() < 0.0:
        return np.zeros_like(omega_qn_j)

    return np.sqrt(term)


def s_squared(s_qn: float) -> float:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn: Quantum number S.

    Returns:
        The diagonal matrix element S(S + 1).
    """
    return s_qn * (s_qn + 1)


def s_plus(s_qn: float, sigma_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Note that S is a normally commutative angular momentum operator.

    Args:
        s_qn: Quantum number S.
        sigma_qn_j: Quantum number Σ.

    Returns:
        The off-diagonal matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2).
    """
    term = s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1)

    # See note in `j_plus`.
    if term.min() < 0.0:
        return np.zeros_like(sigma_qn_j)

    return np.sqrt(term)


def s_minus(s_qn: float, sigma_qn_j: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Note that S is a normally commutative angular momentum operator.

    Args:
        s_qn: Quantum number S.
        sigma_qn_j: Quantum number Σ.

    Returns:
        The off-diagonal matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2).
    """
    term = s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1)

    # See note in `j_plus`.
    if term.min() < 0.0:
        return np.zeros_like(sigma_qn_j)

    return np.sqrt(term)


def n_squared(
    sigma_basis_vec: NDArray[np.float64],
    omega_basis_vec: NDArray[np.float64],
    s_qn: float,
    j_qn: float,
    dim: int,
) -> NDArray[np.float64]:
    """Return matrix elements for the N^2 operator.

    N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

    Args:
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        omega_basis_vec: Basis vector of Ω, one value per row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        dim: Dimension of the Hamiltonian.

    Returns:
        The matrix elements for N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).
    """
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis_vec)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis_vec)

    result = np.zeros((dim, dim))

    # ⟨J, S, Ω, Σ|J^2 + S^2 - 2JzSz|J, S, Ω, Σ⟩ = J(J + 1) + S(S + 1) - 2ΩΣ
    np.fill_diagonal(result, j_squared(j_qn) + s_squared(s_qn) - 2.0 * omega_j * sigma_j)

    # Denote the areas in the array where Ω_i = Ω_j - 1 and Σ_i = Σ_j - 1.
    mask_minus: NDArray[np.bool] = (omega_i == omega_j - 1) & (sigma_i == sigma_j - 1)
    # Denote the areas in the array where Ω_i = Ω_j + 1 and Σ_i = Σ_j + 1.
    mask_plus: NDArray[np.bool] = (omega_i == omega_j + 1) & (sigma_i == sigma_j + 1)

    # ⟨J, S, Ω - 1, Σ - 1|-(J+S-)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    term_minus = -j_plus(j_qn, omega_j) * s_minus(s_qn, sigma_j)
    # ⟨J, S, Ω + 1, Σ + 1|-(J-S+)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    term_plus = -j_minus(j_qn, omega_j) * s_plus(s_qn, sigma_j)

    result[mask_minus] = term_minus[mask_minus]
    result[mask_plus] = term_plus[mask_plus]

    return result


def lz_sz(
    lambda_basis_vec: NDArray[np.int64], sigma_basis_vec: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return matrix elements for the LzSz operator.

    Args:
        lambda_basis_vec: Basis vector of Λ, one value per row of the Hamiltonian.
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.

    Returns:
        The matrix elements for LzSz.
    """
    # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
    return np.diag(lambda_basis_vec * sigma_basis_vec)


def three_sz2_minus_s2(sigma_basis_vec: NDArray[np.float64], s_qn: float) -> NDArray[np.float64]:
    """Return matrix elements for the 3Sz^2 - S^2 operator.

    Args:
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        s_qn: Quantum number S

    Returns:
        The matrix elements for 3Sz^2 - S^2.
    """
    # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
    return np.diag(3 * sigma_basis_vec**2 - s_squared(s_qn))


def n_dot_s(
    sigma_basis_vec: NDArray[np.float64],
    omega_basis_vec: NDArray[np.float64],
    s_qn: float,
    j_qn: float,
    dim: int,
) -> NDArray[np.float64]:
    """Return matrix elements for the N·S operator.

    N·S = JzSz + 0.5(J+S- + J-S+) - S^2.

    Args:
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        omega_basis_vec: Basis vector of Ω, one value per row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        dim: Dimension of the Hamiltonian.

    Returns:
        The matrix elements for N·S = JzSz + 0.5(J+S- + J-S+) - S^2.
    """
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis_vec)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis_vec)

    result = np.zeros((dim, dim))

    # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
    np.fill_diagonal(result, omega_basis_vec * sigma_basis_vec - s_squared(s_qn))

    # Denote the areas in the array where Ω_i = Ω_j - 1 and Σ_i = Σ_j - 1
    mask_minus: NDArray[np.bool] = (omega_i == omega_j - 1) & (sigma_i == sigma_j - 1)
    # Denote the areas in the array where Ω_i = Ω_j + 1 and Σ_i = Σ_j + 1
    mask_plus: NDArray[np.bool] = (omega_i == omega_j + 1) & (sigma_i == sigma_j + 1)

    # TODO: 25/07/16 - This method computes the following elements for all locations in the matrix,
    #       after which the masks are applied. It might be a bit more efficient to only compute
    #       these where the mask is true, maybe by applying the mask to the Ω and Σ matrices first.

    # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    term_minus = 0.5 * j_plus(j_qn, omega_j) * s_minus(s_qn, sigma_j)
    # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    term_plus = 0.5 * j_minus(j_qn, omega_j) * s_plus(s_qn, sigma_j)

    result[mask_minus] = term_minus[mask_minus]
    result[mask_plus] = term_plus[mask_plus]

    return result


def sp2_plus_sm2(
    lambda_basis_vec: NDArray[np.int64], sigma_basis_vec: NDArray[np.float64], s_qn: float, dim: int
) -> NDArray[np.float64]:
    """Return matrix elements for the S+^2 + S-^2 operator.

    Args:
        lambda_basis_vec: Basis vector of Λ, one value per row of the Hamiltonian.
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        s_qn: Quantum number S.
        dim: Dimension of the Hamiltonian.

    Returns:
        The matrix elements for S+^2 + S-^2.
    """
    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis_vec)
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis_vec)

    result = np.zeros((dim, dim))

    # Denote the areas in the array where Λ_i = Λ_j - 2 and Σ_i = Σ_j + 2
    mask_plus: NDArray[np.bool] = (lambda_i == lambda_j - 2) & (sigma_i == sigma_j + 2)
    # Denote the areas in the array where Λ_i = Λ_j + 2 and Σ_i = Σ_j - 2
    mask_minus: NDArray[np.bool] = (lambda_i == lambda_j + 2) & (sigma_i == sigma_j - 2)

    # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
    term_plus = s_plus(s_qn, sigma_j) * s_plus(s_qn, sigma_j + 1)
    # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
    term_minus = s_minus(s_qn, sigma_j) * s_minus(s_qn, sigma_j - 1)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result


def jpsp_plus_jmsm(
    lambda_basis_vec: NDArray[np.int64],
    sigma_basis_vec: NDArray[np.float64],
    omega_basis_vec: NDArray[np.float64],
    s_qn: float,
    j_qn: float,
    dim: int,
) -> NDArray[np.float64]:
    """Return matrix elements for the J+S+ + J-S- operator.

    Args:
        lambda_basis_vec: Basis vector of Λ, one value per row of the Hamiltonian.
        sigma_basis_vec: Basis vector of Σ, one value per row of the Hamiltonian.
        omega_basis_vec: Basis vector of Ω, one value per row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        dim: Dimension of the Hamiltonian.

    Returns:
        The matrix elements for J+S+ + J-S-.
    """
    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis_vec)
    sigma_i, sigma_j = utils.form_basis_matrices(sigma_basis_vec)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis_vec)

    result = np.zeros((dim, dim))

    # Denote the areas in the array where Λ_i = Λ_j - 2, Ω_i = Ω_j - 1, and Σ_i = Σ_j + 1
    mask_plus: NDArray[np.bool] = (
        (lambda_i == lambda_j - 2) & (omega_i == omega_j - 1) & (sigma_i == sigma_j + 1)
    )
    # Denote the areas in the array where Λ_i = Λ_j + 2, Ω_i = Ω_j + 1, and Σ_i = Σ_j - 1
    mask_minus: NDArray[np.bool] = (
        (lambda_i == lambda_j + 2) & (omega_i == omega_j + 1) & (sigma_i == sigma_j - 1)
    )

    # ⟨Λ - 2, Ω - 1, Σ + 1|J+S+|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    term_plus = j_plus(j_qn, omega_j) * s_plus(s_qn, sigma_j)
    # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    term_minus = j_minus(j_qn, omega_j) * s_minus(s_qn, sigma_j)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result


def jp2_plus_jm2(
    lambda_basis: NDArray[np.int64], omega_basis: NDArray[np.float64], j_qn: float, dim: int
) -> NDArray[np.float64]:
    """Return matrix elements for the J+^2 + J-^2 operator.

    Args:
        lambda_basis_vec: Basis vector of Λ, one value per row of the Hamiltonian.
        omega_basis_vec: Basis vector of Ω, one value per row of the Hamiltonian.
        j_qn: Quantum number J.
        dim: Dimension of the Hamiltonian.

    Returns:
        The matrix elements for J+^2 + J-^2.
    """
    lambda_i, lambda_j = utils.form_basis_matrices(lambda_basis)
    omega_i, omega_j = utils.form_basis_matrices(omega_basis)

    result = np.zeros((dim, dim))

    # Denote the areas in the array where Λ_i = Λ_j - 2 and Ω_i = Ω_j - 2
    mask_plus: NDArray[np.bool] = (lambda_i == lambda_j - 2) & (omega_i == omega_j - 2)
    # Denote the areas in the array where Λ_i = Λ_j + 2 and Ω_i = Ω_j + 2
    mask_minus: NDArray[np.bool] = (lambda_i == lambda_j + 2) & (omega_i == omega_j + 2)

    # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J
    #       is an anomalously commutative operator.
    # ⟨Λ - 2, Ω - 2|J+^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
    term_plus = j_plus(j_qn, omega_j) * j_plus(j_qn, omega_j - 1)
    # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
    # ⟨Λ + 2, Ω + 2|J-^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
    term_minus = j_minus(j_qn, omega_j) * j_minus(j_qn, omega_j + 1)

    result[mask_plus] = term_plus[mask_plus]
    result[mask_minus] = term_minus[mask_minus]

    return result
