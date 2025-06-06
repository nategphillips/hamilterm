# module main
"""Symbolically computes the diatomic Hamiltonian for Σ and Π states."""

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

import sympy as sp

# Rotational quantum number J and the shorthand x = J(J + 1).
j_qn, x = sp.symbols("J, x")
# Rotational constants.
B, D, H, L, M, P = sp.symbols("B, D, H, L, M, P")
# Spin-orbit constants.
A, A_D, A_H, A_L, A_M, eta = sp.symbols("A, A_D, A_H, A_L, A_M, eta")
# Spin-spin constants.
lamda, lambda_D, lambda_H, theta = sp.symbols("lambda, lambda_D, lambda_H, theta")
# Spin-rotation constants.
gamma, gamma_D, gamma_H, gamma_L, gamma_S = sp.symbols("gamma, gamma_D, gamma_H, gamma_L, gamma_S")
# Lambda doubling constants for Π states.
o, p, q, o_D, p_D, q_D, o_H, p_H, q_H, o_L, p_L, q_L = sp.symbols(
    "o, p, q, o_D, p_D, q_D, o_H, p_H, q_H, o_L, p_L, q_L"
)

include_r: bool = True
include_so: bool = True
include_ss: bool = True
include_sr: bool = True
include_ld: bool = True

f_r, f_so, f_ss, f_sr, f_ld = map(int, [include_r, include_so, include_ss, include_sr, include_ld])

# MAX_N_POWER can be 2, 4, 6, 8, 10, or 12. Powers above 12 have no associated constants and
# therefore will not contribute to the calculation.
MAX_N_POWER: int = 2
LAMBDA_MAP: dict[str, int] = {"Sigma": 0, "Pi": 1}


def mel_j2(j_qn: sp.Symbol) -> sp.Expr:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn (sp.Symbol): Quantum number J

    Returns:
        sp.Expr: Matrix element J(J + 1)

    """
    return j_qn * (j_qn + 1)


def mel_jp(j_qn: sp.Symbol, omega_qn_j: sp.Rational | sp.Expr) -> sp.Expr:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Args:
        j_qn (sp.Symbol): Quantum number J
        omega_qn_j (sp.Rational | sp.Expr): Quantum number Ω

    Returns:
        sp.Expr: Matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2)

    """
    return sp.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1))


def mel_jm(j_qn: sp.Symbol, omega_qn_j: sp.Rational | sp.Expr) -> sp.Expr:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Args:
        j_qn (sp.Symbol): Quantum number J
        omega_qn_j (sp.Rational | sp.Expr): Quantum number Ω

    Returns:
        sp.Expr: Matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2)

    """
    return sp.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1))


def mel_s2(s_qn: sp.Rational) -> sp.Rational:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Rational: Matrix element S(S + 1)

    """
    return s_qn * (s_qn + 1)


def mel_sp(s_qn: sp.Rational, sigma_qn_j: sp.Rational | sp.Expr) -> sp.Expr:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Args:
        s_qn (sp.Rational): Quantum number S
        sigma_qn_j (sp.Rational | sp.Expr): Quantum number Σ

    Returns:
        sp.Expr: Matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2)

    """
    return sp.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def mel_sm(s_qn: sp.Rational, sigma_qn_j: sp.Rational | sp.Expr) -> sp.Expr:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Args:
        s_qn (sp.Rational): Quantum number S
        sigma_qn_j (sp.Rational | sp.Expr): Quantum number Σ

    Returns:
        sp.Expr: Matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2)

    """
    return sp.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


def construct_n_operator_matrices(
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> list[sp.MutableDenseMatrix]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        list[sp.MutableDenseMatrix]: N operator matrices
    """
    # The number of basis functions determine the size of the N operator matrix.
    dim: int = len(basis_fns)

    # Each N^{2k} operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc. Always initialize the
    # operator matrices up to N^12 - if MAX_N_POWER is less than 12, the unused matrices will have
    # all their elements equal to zero.
    n_op_mats: list[sp.MutableDenseMatrix] = [sp.zeros(dim) for _ in range(6)]

    # Matrix elements for the N^2 operator.
    def mel_n2(i: int, j: int) -> sp.Expr:
        """Return matrix elements for the N^2 operator.

        N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

        Args:
            i (int): Index i (row) of the Hamiltonian matrix
            j (int): Index j (col) of the Hamiltonian matrix
            basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>

        Returns:
            sp.Expr: Matrix elements for J^2 + S^2 - 2JzSz - (J+S- + J-S+)
        """
        _, sigma_qn_i, omega_qn_i = basis_fns[i]
        _, sigma_qn_j, omega_qn_j = basis_fns[j]

        # ⟨J, S, Ω, Σ|J^2 + S^2 - 2JzSz|J, S, Ω, Σ⟩ = J(J + 1) + S(S + 1) - 2ΩΣ
        if i == j:
            return mel_j2(j_qn) + mel_s2(s_qn) - 2 * omega_qn_j * sigma_qn_j

        # ⟨J, S, Ω - 1, Σ - 1|-(J+S-)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if omega_qn_i == omega_qn_j - 1 and sigma_qn_i == sigma_qn_j - 1:
            return -mel_jp(j_qn, omega_qn_j) * mel_sm(s_qn, sigma_qn_j)

        # ⟨J, S, Ω + 1, Σ + 1|-(J-S+)|J, S, Ω, Σ⟩ = -([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
        if omega_qn_i == omega_qn_j + 1 and sigma_qn_i == sigma_qn_j + 1:
            return -mel_jm(j_qn, omega_qn_j) * mel_sp(s_qn, sigma_qn_j)

        return sp.S.Zero

    # Form the N^2 matrix using the matrix elements above.
    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel_n2(i, j)

    # The following N^{2k} matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, MAX_N_POWER // 2):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def h_rotational(
    i: int,
    j: int,
    n_op_mats: list[sp.MutableDenseMatrix],
) -> sp.Expr:
    """Return matrix elements for the rotational Hamiltonian.

    H_r = BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices

    Returns:
        sp.Expr: Matrix elements for BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12

    """
    # BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12
    return (
        B * n_op_mats[0][i, j]
        - D * n_op_mats[1][i, j]
        + H * n_op_mats[2][i, j]
        + L * n_op_mats[3][i, j]
        + M * n_op_mats[4][i, j]
        + P * n_op_mats[5][i, j]
    )


def h_spin_orbit(
    i: int,
    j: int,
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]],
    s_qn: sp.Rational,
    n_op_mats: list[sp.MutableDenseMatrix],
) -> sp.Expr:
    """Return matrix elements for the spin-orbit Hamiltonian.

    H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices

    Returns:
        sp.Expr: Matrix elements for A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+
            + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+ + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]

    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    result: sp.Expr = sp.S.Zero

    def mel_lzsz(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the LzSz operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for LzSz
        """
        lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

        # Operator is completely diagonal, so only m = n terms exist.
        if m == n:
            # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
            return lambda_qn_n * sigma_qn_n

        return sp.S.Zero

    # Spin-orbit coupling is only defined for states with Λ > 0 and S > 0.
    if abs(lambda_qn_j) > 0 and s_qn > 0:
        # A(LzSz)
        result += A * mel_lzsz(i, j)

        for k in range(len(basis_fns)):
            # ⟨i|A_D/2[N^2, LzSz]+|j⟩ = A_D/2[⟨i|N^2(LzSz)|j⟩ + ⟨i|(LzSz)N^2|j⟩]
            #                         = A_D/2(∑_k⟨i|N^2|k⟩⟨k|LzSz|j⟩ + ∑_k⟨i|LzSz|k⟩⟨k|N^2|j⟩)
            #                         = A_D/2[(N^2)_{ik}(LzSz)_{kj} + (LzSz)_{ik}(N^2)_{kj}]
            result += (
                safe_rational(1, 2)
                * A_D
                * (n_op_mats[0][i, k] * mel_lzsz(k, j) + mel_lzsz(i, k) * n_op_mats[0][k, j])
            )

            # ⟨i|A_H/2[N^4, LzSz]+|j⟩ = A_H/2[⟨i|N^4(LzSz)|j⟩ + ⟨i|(LzSz)N^4|j⟩]
            #                         = A_H/2(∑_k⟨i|N^4|k⟩⟨k|LzSz|j⟩ + ∑_k⟨i|LzSz|k⟩⟨k|N^4|j⟩)
            #                         = A_H/2[(N^4)_{ik}(LzSz)_{kj} + (LzSz)_{ik}(N^4)_{kj}]
            result += (
                safe_rational(1, 2)
                * A_H
                * (n_op_mats[1][i, k] * mel_lzsz(k, j) + mel_lzsz(i, k) * n_op_mats[1][k, j])
            )

            # ⟨i|A_L/2[N^6, LzSz]+|j⟩ = A_L/2[⟨i|N^6(LzSz)|j⟩ + ⟨i|(LzSz)N^6|j⟩]
            #                         = A_L/2(∑_k⟨i|N^6|k⟩⟨k|LzSz|j⟩ + ∑_k⟨i|LzSz|k⟩⟨k|N^6|j⟩)
            #                         = A_L/2[(N^6)_{ik}(LzSz)_{kj} + (LzSz)_{ik}(N^6)_{kj}]
            result += (
                safe_rational(1, 2)
                * A_L
                * (n_op_mats[2][i, k] * mel_lzsz(k, j) + mel_lzsz(i, k) * n_op_mats[2][k, j])
            )

            # ⟨i|A_M/2[N^8, LzSz]+|j⟩ = A_M/2[⟨i|N^8(LzSz)|j⟩ + ⟨i|(LzSz)N^8|j⟩]
            #                         = A_M/2(∑_k⟨i|N^8|k⟩⟨k|LzSz|j⟩ + ∑_k⟨i|LzSz|k⟩⟨k|N^8|j⟩)
            #                         = A_M/2[(N^8)_{ik}(LzSz)_{kj} + (LzSz)_{ik}(N^8)_{kj}]
            result += (
                safe_rational(1, 2)
                * A_M
                * (n_op_mats[3][i, k] * mel_lzsz(k, j) + mel_lzsz(i, k) * n_op_mats[3][k, j])
            )

        # Term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨Λ, Σ|ηLzSz[Sz^2 - 1/5(3S^2 - 1)]|Λ, Σ⟩ = ηΛΣ[Σ^2 - 1/5(3S(S + 1) - 1)]
            result += (
                eta
                * mel_lzsz(i, j)
                * (sigma_qn_j**2 - safe_rational(1, 5) * (3 * mel_s2(s_qn) - 1))
            )

    return result


def h_spin_spin(
    i: int,
    j: int,
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]],
    s_qn: sp.Rational,
    n_op_mats: list[sp.MutableDenseMatrix],
) -> sp.Expr:
    """Return matrix elements for the spin-spin Hamiltonian.

    H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices

    Returns:
        sp.Expr: Matrix elements for 2λ/3(3Sz^2 - S^2) + λ_D/2[2/3(3Sz^2 - S^2), N^2]+
            + λ_H/2[2/3(3Sz^2 - S^2), N^4]+ + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)

    """
    sigma_qn_j: sp.Rational = basis_fns[j][1]

    result: sp.Expr = sp.S.Zero

    def mel_sz2ms2(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the 3Sz^2 - S^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for 3Sz^2 - S^2
        """
        sigma_qn_n = basis_fns[n][1]

        # Operator is completely diagonal, so only m = n terms exist.
        if m == n:
            # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
            return 3 * sigma_qn_n**2 - mel_s2(s_qn)

        return sp.S.Zero

    # Spin-spin coupling is only defined for states with S > 1/2.
    if s_qn > safe_rational(1, 2):
        # 2λ/3(3Sz^2 - S^2)
        result += safe_rational(2, 3) * lamda * mel_sz2ms2(i, j)

        for k in range(len(basis_fns)):
            # ⟨i|λ_D/3[(3Sz^2 - S^2), N^2]+|j⟩ = λ_D/3[⟨i|(3Sz^2 - S^2)N^2|j⟩ + ⟨i|N^2(3Sz^2 - S^2)|j⟩]
            #                                  = λ_D/3(∑_k⟨i|(3Sz^2 - S^2)|k⟩⟨k|N^2|j⟩ + ∑_k⟨i|N^2|k⟩⟨k|(3Sz^2 - S^2)|j⟩)
            #                                  = λ_D/3[(3Sz^2 - S^2)_{ik}(N^2)_{kj} + (N^2)_{ik}(3Sz^2 - S^2)_{kj}]
            result += (
                safe_rational(1, 3)
                * lambda_D
                * (mel_sz2ms2(i, k) * n_op_mats[0][k, j] + n_op_mats[0][i, k] * mel_sz2ms2(k, j))
            )

            # ⟨i|λ_H/3[(3Sz^2 - S^2), N^4]+|j⟩ = λ_H/3[⟨i|(3Sz^2 - S^2)N^4|j⟩ + ⟨i|N^4(3Sz^2 - S^2)|j⟩]
            #                                  = λ_H/3(∑_k⟨i|(3Sz^2 - S^2)|k⟩⟨k|N^4|j⟩ + ∑_k⟨i|N^4|k⟩⟨k|(3Sz^2 - S^2)|j⟩)
            #                                  = λ_H/3[(3Sz^2 - S^2)_{ik}(N^4)_{kj} + (N^4)_{ik}(3Sz^2 - S^2)_{kj}]
            result += (
                safe_rational(1, 3)
                * lambda_H
                * (mel_sz2ms2(i, k) * n_op_mats[1][k, j] + n_op_mats[1][i, k] * mel_sz2ms2(k, j))
            )

        # Term only valid for states with S > 3/2.
        if i == j and s_qn > safe_rational(3, 2):
            # ⟨S, Σ|θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)|S, Σ⟩
            #   = θ/12(35Σ^4 - 30S(S + 1)Σ^2 + 25Σ^2 - 6S(S + 1) + 3[S(S + 1)]^2)
            result += (
                safe_rational(1, 12)
                * theta
                * (
                    35 * sigma_qn_j**4
                    - 30 * mel_s2(s_qn) * sigma_qn_j**2
                    + 25 * sigma_qn_j**2
                    - 6 * mel_s2(s_qn)
                    + 3 * mel_s2(s_qn) ** 2
                )
            )

    return result


def h_spin_rotation(
    i: int,
    j: int,
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]],
    s_qn: sp.Rational,
    n_op_mats: list[sp.MutableDenseMatrix],
) -> sp.Expr:
    """Return matrix elements for the spin-rotation Hamiltonian.

    H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices

    Returns:
        sp.Expr: Matrix elements for γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
            + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}

    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    result: sp.Expr = sp.S.Zero

    def mel_ndots(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the N·S operator.

        N·S = JzSz + 0.5(J+S- + J-S+) - S^2

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for JzSz + 0.5(J+S- + J-S+) - S^2
        """
        _, sigma_qn_m, omega_qn_m = basis_fns[m]
        _, sigma_qn_n, omega_qn_n = basis_fns[n]

        # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
        if m == n:
            return omega_qn_n * sigma_qn_n - mel_s2(s_qn)

        # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if omega_qn_m == omega_qn_n - 1 and sigma_qn_m == sigma_qn_n - 1:
            return safe_rational(1, 2) * mel_jp(j_qn, omega_qn_n) * mel_sm(s_qn, sigma_qn_n)

        # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
        if omega_qn_m == omega_qn_n + 1 and sigma_qn_m == sigma_qn_n + 1:
            return safe_rational(1, 2) * mel_jm(j_qn, omega_qn_n) * mel_sp(s_qn, sigma_qn_n)

        return sp.S.Zero

    # Spin-rotation coupling is only defined for states with S > 0.
    if s_qn > 0:
        # γ(N·S)
        result += gamma * mel_ndots(i, j)

        for k in range(len(basis_fns)):
            # ⟨i|γ_D/2[N·S, N^2]+|j⟩ = γ_D/2[⟨i|(N·S)N^2|j⟩ + ⟨i|N^2(N·S)|j⟩]
            #                        = γ_D/2(∑_k⟨i|N·S|k⟩⟨k|N^2|j⟩ + ∑_k⟨i|N^2|k⟩⟨k|N·S|j⟩)
            #                        = γ_D/2[(N·S)_{ik}(N^2)_{kj} + (N^2)_{ik}(N·S)_{kj}]
            result += (
                safe_rational(1, 2)
                * gamma_D
                * (mel_ndots(i, k) * n_op_mats[0][k, j] + n_op_mats[0][i, k] * mel_ndots(k, j))
            )

            # ⟨i|γ_H/2[N·S, N^4]+|j⟩ = γ_H/2[⟨i|(N·S)N^4|j⟩ + ⟨i|N^4(N·S)|j⟩]
            #                        = γ_H/2(∑_k⟨i|N·S|k⟩⟨k|N^4|j⟩ + ∑_k⟨i|N^4|k⟩⟨k|N·S|j⟩)
            #                        = γ_H/2[(N·S)_{ik}(N^4)_{kj} + (N^4)_{ik}(N·S)_{kj}]
            result += (
                safe_rational(1, 2)
                * gamma_H
                * (mel_ndots(i, k) * n_op_mats[1][k, j] + n_op_mats[1][i, k] * mel_ndots(k, j))
            )

            # ⟨i|γ_L/2[N·S, N^6]+|j⟩ = γ_L/2[⟨i|(N·S)N^6|j⟩ + ⟨i|N^6(N·S)|j⟩]
            #                        = γ_L/2(∑_k⟨i|N·S|k⟩⟨k|N^6|j⟩ + ∑_k⟨i|N^6|k⟩⟨k|N·S|j⟩)
            #                        = γ_L/2[(N·S)_{ik}(N^6)_{kj} + (N^6)_{ik}(N·S)_{kj}]
            result += (
                safe_rational(1, 2)
                * gamma_L
                * (mel_ndots(i, k) * n_op_mats[2][k, j] + n_op_mats[2][i, k] * mel_ndots(k, j))
            )

        # Term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨J, S, Ω + 1, Σ + 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_S/2[S(S + 1) - 5Σ(Σ + 1) + 2]([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j + 1 and omega_qn_i == omega_qn_j + 1:
                result += (
                    -safe_rational(1, 2)
                    * gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j + 1) - 2)
                    * mel_jm(j_qn, omega_qn_j)
                    * mel_sp(s_qn, sigma_qn_j)
                )

            # ⟨J, S, Ω - 1, Σ - 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_s/2[S(S + 1) - 5Σ(Σ - 1) + 2]([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
                result += (
                    -safe_rational(1, 2)
                    * gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j - 1) - 2)
                    * mel_jp(j_qn, omega_qn_j)
                    * mel_sm(s_qn, sigma_qn_j)
                )

    return result


def h_lambda_doubling(
    i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.Expr:
    """Return matrix elements for the lambda doubling Hamiltonian.

    H_ld = 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Expr: Matrix elements for 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-)
            + q/2(J+^2 + J-^2)

    """
    # TODO: 25/06/06 - In the future, evaluate the lambda doubling commutators explicitly as in
    #       all the other terms to account for centrifugal distortion terms.
    lambda_qn_i, sigma_qn_i, omega_qn_i = basis_fns[i]
    lambda_qn_j, sigma_qn_j, omega_qn_j = basis_fns[j]

    # Lambda doubling is only defined for Λ ± 2 transitions.
    if abs(lambda_qn_i - lambda_qn_j) != 2:
        return sp.S.Zero

    # ⟨Σ + 2|S+^2|Σ⟩ = 0.5(o + p + q)([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j - 2 and sigma_qn_i == sigma_qn_j + 2:
        return 0.5 * (o + p + q) * mel_sp(s_qn, sigma_qn_j) * mel_sp(s_qn, sigma_qn_j + 1)

    # ⟨Σ - 2|S-^2|Σ⟩ = 0.5(o + p + q)([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j + 2 and sigma_qn_i == sigma_qn_j - 2:
        return 0.5 * (o + p + q) * mel_sm(s_qn, sigma_qn_j) * mel_sm(s_qn, sigma_qn_j - 1)

    # ⟨Ω - 1, Σ + 1|J+S+|Ω, Σ⟩ = -0.5(p + 2q)([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if (
        lambda_qn_i == lambda_qn_j - 2
        and sigma_qn_i == sigma_qn_j + 1
        and omega_qn_i == omega_qn_j - 1
    ):
        return -0.5 * (p + 2 * q) * mel_jp(j_qn, omega_qn_j) * mel_sp(s_qn, sigma_qn_j)

    # ⟨Ω + 1, Σ - 1|J-S-|Ω, Σ⟩ = -0.5(p + 2q)([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if (
        lambda_qn_i == lambda_qn_j + 2
        and sigma_qn_i == sigma_qn_j - 1
        and omega_qn_i == omega_qn_j + 1
    ):
        return -0.5 * (p + 2 * q) * mel_jm(j_qn, omega_qn_j) * mel_sm(s_qn, sigma_qn_j)

    # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J is
    #       an anomalously commutative operator.
    # ⟨Ω - 2|J+^2|Ω⟩ = 0.5 * q([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j - 2 and omega_qn_i == omega_qn_j - 2:
        return 0.5 * q * mel_jp(j_qn, omega_qn_j) * mel_jp(j_qn, omega_qn_j - 1)

    # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
    # ⟨Ω + 2|J-^2|Ω⟩ = 0.5 * q([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
    if lambda_qn_i == lambda_qn_j + 2 and omega_qn_i == omega_qn_j + 2:
        return 0.5 * q * mel_jm(j_qn, omega_qn_j) * mel_jm(j_qn, omega_qn_j + 1)

    return sp.S.Zero


def build_hamiltonian(
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.MutableDenseMatrix:
    """Build the symbolic Hamiltonian matrix.

    Args:
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.MutableDenseMatrix: Hamiltonian matrix H

    """
    dim: int = len(basis_fns)
    n_op_mats: list[sp.MutableDenseMatrix] = construct_n_operator_matrices(basis_fns, s_qn)
    h_mat: sp.MutableDenseMatrix = sp.zeros(dim)

    for i in range(dim):
        for j in range(dim):
            h_mat[i, j] = (
                f_r * h_rotational(i, j, n_op_mats)
                + f_so * h_spin_orbit(i, j, basis_fns, s_qn, n_op_mats)
                + f_ss * h_spin_spin(i, j, basis_fns, s_qn, n_op_mats)
                + f_sr * h_spin_rotation(i, j, basis_fns, s_qn, n_op_mats)
                + f_ld * h_lambda_doubling(i, j, basis_fns, s_qn)
            )

    return h_mat


def parse_term_symbol(term_symbol: str) -> tuple[sp.Rational, int]:
    """Parse the molecular term symbol into the quantum numbers S and Λ.

    Args:
        term_symbol (str): Molecular term symbol, e.g., "2Pi" or "3Sigma"

    Returns:
        tuple[sp.Rational, int]: Quantum numbers S and Λ

    """
    spin_multiplicity: int = int(term_symbol[0])
    s_qn: sp.Rational = safe_rational(spin_multiplicity - 1, 2)
    term: str = term_symbol[1:]
    lambda_qn: int = LAMBDA_MAP[term]

    return s_qn, lambda_qn


def generate_basis_fns(
    s_qn: sp.Rational, lambda_qn: int
) -> list[tuple[int, sp.Rational, sp.Rational]]:
    """Construct the Hund's case (a) basis set |Λ, Σ; Ω>.

    Args:
        s_qn (sp.Rational): Quantum number S
        lambda_qn (int): Quantum number Λ

    Returns:
        list[tuple[int, sp.Rational, sp.Rational]]: List of basis vectors |Λ, Σ; Ω>

    """
    # Possible values for Σ = S, S - 1, ..., -S. There are 2S + 1 total values of Σ.
    sigmas: list[sp.Rational] = [
        cast("sp.Rational", -s_qn + safe_rational(i, 1)) for i in range(int(2 * s_qn) + 1)
    ]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas: list[int] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: sp.Rational = cast("sp.Rational", lam + sigma)
            basis_fns.append((lam, sigma, omega))

    return basis_fns


def safe_rational(num: int, denom: int) -> sp.Rational:
    """Ensure a rational return value.

    Args:
        num (int): Numerator
        denom (int): Denominator

    Raises:
        ValueError: If NaN or Infinity is encountered

    Returns:
        sp.Rational: A guaranteed sp.Rational type

    """
    r = sp.Rational(num, denom)

    if not isinstance(r, sp.Rational):
        raise ValueError(f"Expected sp.Rational, got {r}.")

    return cast("sp.Rational", r)


def fsn(num: int | sp.Rational | sp.Expr) -> str:
    """Format signed number for printing.

    Args:
        num (int | sp.Rational | sp.Expr): Number

    Returns:
        str: Print-friendly string with a +, ±, or -

    """
    s: str = str(num)

    if num > 0:
        return f"+{s}"
    if num == 0:
        return f"±{s}"

    return s


def main() -> None:
    """Entry point."""
    term_symbol: str = "2Sigma"

    s_qn, lambda_qn = parse_term_symbol(term_symbol)
    print("Term symbol:")
    print(f"{term_symbol}: S={s_qn}, Λ={lambda_qn}")

    print("\nBasis states |Λ, Σ, Ω>:")
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]] = generate_basis_fns(s_qn, lambda_qn)
    for state in basis_fns:
        print(f"|{fsn(state[0])}, {fsn(state[1])}, {fsn(state[2])}>")

    print("\nHamiltonian matrix:")
    h_mat: sp.MutableDenseMatrix = build_hamiltonian(basis_fns, s_qn).subs(j_qn * (j_qn + 1), x)
    sp.pprint(sp.nsimplify(h_mat))

    print("\nEigenvalues:")
    eigenvals: dict[sp.Expr, int] = cast("dict[sp.Expr, int]", h_mat.eigenvals())
    for eigenvalue in eigenvals:
        sp.pprint(sp.nsimplify(eigenvalue))


if __name__ == "__main__":
    main()
