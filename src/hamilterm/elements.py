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

import sympy as sp
from sympy import Expr, Integer, Rational, Symbol


def j_squared(j_qn: Symbol) -> Expr:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn: Quantum number J.

    Returns:
        The diagonal matrix element J(J + 1).
    """
    return j_qn * (j_qn + 1)


def j_plus(j_qn: Symbol, omega_qn_j: Rational) -> Expr:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Note that J is an anomalously commutative angular momentum operator.

    Args:
        j_qn: Quantum number J.
        omega_qn_j: Quantum number Ω.

    Returns:
        The off-diagonal matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2).
    """
    return sp.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1))


def j_minus(j_qn: Symbol, omega_qn_j: Rational) -> Expr:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Note that J is an anomalously commutative angular momentum operator.

    Args:
        j_qn: Quantum number J.
        omega_qn_j: Quantum number Ω.

    Returns:
        The off-diagonal matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2).
    """
    return sp.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1))


def s_squared(s_qn: Rational) -> Expr:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn: Quantum number S.

    Returns:
        The diagonal matrix element S(S + 1).
    """
    return s_qn * (s_qn + 1)


def s_plus(s_qn: Rational, sigma_qn_j: Rational) -> Expr:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Note that S is a normally commutative angular momentum operator.

    Args:
        s_qn: Quantum number S.
        sigma_qn_j: Quantum number Σ.

    Returns:
        The off-diagonal matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2).
    """
    return sp.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def s_minus(s_qn: Rational, sigma_qn_j: Rational) -> Expr:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Note that S is a normally commutative angular momentum operator.

    Args:
        s_qn: Quantum number S.
        sigma_qn_j: Quantum number Σ.

    Returns:
        The off-diagonal matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2).
    """
    return sp.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


def n_squared(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
) -> Expr:
    """Return matrix elements for the N^2 operator.

    N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.

    Returns:
        The matrix elements for N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).
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

    return Integer(0)


def lz_sz(i: int, j: int, basis_fns: list[tuple[Integer, Rational, Rational]]) -> Expr:
    """Return matrix elements for the LzSz operator.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.

    Returns:
        The matrix elements for LzSz.
    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
    if i == j:
        return lambda_qn_j * sigma_qn_j

    return Integer(0)


def three_sz2_minus_s2(
    i: int, j: int, basis_fns: list[tuple[Integer, Rational, Rational]], s_qn: Rational
) -> Expr:
    """Return matrix elements for the 3Sz^2 - S^2 operator.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.

    Returns:
        The matrix elements for 3Sz^2 - S^2.
    """
    sigma_qn_j = basis_fns[j][1]

    # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
    if i == j:
        return 3 * sigma_qn_j**2 - s_squared(s_qn)

    return Integer(0)


def n_dot_s(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
) -> Expr:
    """Return matrix elements for the N·S operator.

    N·S = JzSz + 0.5(J+S- + J-S+) - S^2.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.

    Returns:
        The matrix elements for N·S = JzSz + 0.5(J+S- + J-S+) - S^2.
    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
    if i == j:
        return omega_qn_j * sigma_qn_j - s_squared(s_qn)

    # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if omega_qn_i == omega_qn_j - 1 and sigma_qn_i == sigma_qn_j - 1:
        return Rational(1, 2) * j_plus(j_qn, omega_qn_j) * s_minus(s_qn, sigma_qn_j)

    # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if omega_qn_i == omega_qn_j + 1 and sigma_qn_i == sigma_qn_j + 1:
        return Rational(1, 2) * j_minus(j_qn, omega_qn_j) * s_plus(s_qn, sigma_qn_j)

    return Integer(0)


def sp2_plus_sm2(
    i: int, j: int, basis_fns: list[tuple[Integer, Rational, Rational]], s_qn: Rational
) -> Expr:
    """Return matrix elements for the S+^2 + S-^2 operator.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.

    Returns:
        The matrix elements for S+^2 + S-^2.
    """
    lambda_qn_i, sigma_qn_i, _ = basis_fns[i]
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j - 2 and sigma_qn_i == sigma_qn_j + 2:
        return s_plus(s_qn, sigma_qn_j) * s_plus(s_qn, sigma_qn_j + 1)

    # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j + 2 and sigma_qn_i == sigma_qn_j - 2:
        return s_minus(s_qn, sigma_qn_j) * s_minus(s_qn, sigma_qn_j - 1)

    return Integer(0)


def jpsp_plus_jmsm(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
) -> Expr:
    """Return matrix elements for the J+S+ + J-S- operator.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.

    Returns:
        The matrix elements for J+S+ + J-S-.
    """
    lambda_qn_i, sigma_qn_i, omega_qn_i = basis_fns[i]
    lambda_qn_j, sigma_qn_j, omega_qn_j = basis_fns[j]

    # ⟨Λ - 2, Ω - 1, Σ + 1|J+S+|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if (
        lambda_qn_i == lambda_qn_j - 2
        and sigma_qn_i == sigma_qn_j + 1
        and omega_qn_i == omega_qn_j - 1
    ):
        return j_plus(j_qn, omega_qn_j) * s_plus(s_qn, sigma_qn_j)

    # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if (
        lambda_qn_i == lambda_qn_j + 2
        and sigma_qn_i == sigma_qn_j - 1
        and omega_qn_i == omega_qn_j + 1
    ):
        return j_minus(j_qn, omega_qn_j) * s_minus(s_qn, sigma_qn_j)

    return Integer(0)


def jp2_plus_jm2_sym(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    j_qn: Symbol,
) -> Expr:
    """Return matrix elements for the J+^2 + J-^2 operator.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        j_qn: Quantum number J.

    Returns:
        The matrix elements for J+^2 + J-^2.
    """
    lambda_qn_i, _, omega_qn_i = basis_fns[i]
    lambda_qn_j, _, omega_qn_j = basis_fns[j]

    # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J
    #       is an anomalously commutative operator.
    # ⟨Λ - 2, Ω - 2|J+^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j - 2 and omega_qn_i == omega_qn_j - 2:
        return j_plus(j_qn, omega_qn_j) * j_plus(j_qn, omega_qn_j - 1)

    # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
    # ⟨Λ + 2, Ω + 2|J-^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j + 2 and omega_qn_i == omega_qn_j + 2:
        return j_minus(j_qn, omega_qn_j) * j_minus(j_qn, omega_qn_j + 1)

    return Integer(0)
