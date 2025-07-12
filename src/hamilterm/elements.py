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
from fractions import Fraction
from typing import overload

import sympy as sp


@overload
def j_squared(j_qn: int) -> int: ...


@overload
def j_squared(j_qn: sp.Symbol) -> sp.Expr: ...


def j_squared(j_qn: int | sp.Symbol) -> int | sp.Expr:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn (int): Quantum number J

    Returns:
        int: Matrix element J(J + 1)
    """
    return j_qn * (j_qn + 1)


@overload
def j_plus(j_qn: int, omega_qn_j: Fraction) -> float: ...


@overload
def j_plus(j_qn: sp.Symbol, omega_qn_j: Fraction) -> sp.Expr: ...


def j_plus(j_qn: int | sp.Symbol, omega_qn_j: Fraction) -> float | sp.Expr:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Args:
        j_qn (int): Quantum number J
        omega_qn_j (Fraction): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2)
    """
    result: Fraction | sp.Expr = j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1)

    if isinstance(j_qn, int):
        return math.sqrt(result)

    return sp.sqrt(result)


@overload
def j_minus(j_qn: int, omega_qn_j: Fraction) -> float: ...


@overload
def j_minus(j_qn: sp.Symbol, omega_qn_j: Fraction) -> sp.Expr: ...


def j_minus(j_qn: int | sp.Symbol, omega_qn_j: Fraction) -> float | sp.Expr:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Args:
        j_qn (int): Quantum number J
        omega_qn_j (Fraction): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2)
    """
    result: Fraction | sp.Expr = j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1)

    if isinstance(j_qn, int):
        return math.sqrt(result)

    return sp.sqrt(result)


def s_squared(s_qn: Fraction) -> Fraction:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn (Fraction): Quantum number S

    Returns:
        Fraction: Matrix element S(S + 1)
    """
    return s_qn * (s_qn + 1)


def s_plus(s_qn: Fraction, sigma_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Args:
        s_qn (Fraction): Quantum number S
        sigma_qn_j (Fraction): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def s_minus(s_qn: Fraction, sigma_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Args:
        s_qn (Fraction): Quantum number S
        sigma_qn_j (Fraction): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


@overload
def n_squared(
    i: int, j: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: int
) -> float | Fraction: ...


@overload
def n_squared(
    i: int, j: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: sp.Symbol
) -> sp.Expr: ...


def n_squared(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int | sp.Symbol,
) -> float | Fraction | sp.Expr:
    """Return matrix elements for the N^2 operator.

    N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J

    Returns:
        float | Fraction: Matrix elements for J^2 + S^2 - 2JzSz - (J+S- + J-S+)
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


def lz_sz(m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]]) -> int | Fraction:
    """Return matrix elements for the LzSz operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>

    Returns:
        int | Fraction: Matrix elements for LzSz
    """
    lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

    # Operator is completely diagonal, so only m = n terms exist.
    if m == n:
        # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
        return lambda_qn_n * sigma_qn_n

    return 0


def sz2_minus_s2(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction
) -> int | Fraction:
    """Return matrix elements for the 3Sz^2 - S^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S

    Returns:
        int | Fraction: Matrix elements for 3Sz^2 - S^2
    """
    sigma_qn_n = basis_fns[n][1]

    # Operator is completely diagonal, so only m = n terms exist.
    if m == n:
        # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
        return 3 * sigma_qn_n**2 - s_squared(s_qn)

    return 0


@overload
def n_dot_s(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: int
) -> float | Fraction: ...


@overload
def n_dot_s(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: sp.Symbol
) -> sp.Expr: ...


def n_dot_s(
    m: int,
    n: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int | sp.Symbol,
) -> float | Fraction | sp.Expr:
    """Return matrix elements for the N·S operator.

    N·S = JzSz + 0.5(J+S- + J-S+) - S^2

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J

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
        return Fraction(1, 2) * j_plus(j_qn, omega_qn_n) * s_minus(s_qn, sigma_qn_n)

    # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if omega_qn_m == omega_qn_n + 1 and sigma_qn_m == sigma_qn_n + 1:
        return Fraction(1, 2) * j_minus(j_qn, omega_qn_n) * s_plus(s_qn, sigma_qn_n)

    return 0.0


def sp2_plus_sm2(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction
) -> float:
    """Return matrix elements for the S+^2 + S-^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S

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


@overload
def jpsp_plus_jmsm(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: int
) -> float: ...


@overload
def jpsp_plus_jmsm(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: sp.Symbol
) -> sp.Expr: ...


def jpsp_plus_jmsm(
    m: int,
    n: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int | sp.Symbol,
) -> float | sp.Expr:
    """Return matrix elements for the J+S+ + J-S- operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J

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


@overload
def jp2_plus_jm2(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], j_qn: int
) -> float: ...


@overload
def jp2_plus_jm2(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], j_qn: sp.Symbol
) -> sp.Expr: ...


def jp2_plus_jm2(
    m: int, n: int, basis_fns: list[tuple[int, Fraction, Fraction]], j_qn: int | sp.Symbol
) -> float | sp.Expr:
    """Return matrix elements for the J+^2 + J-^2 operator.

    Args:
        m (int): Dummy index m for the bra vector (row)
        n (int): Dummy index n for the ket vector (col)
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        j_qn (int): Quantum number J

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
