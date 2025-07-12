# module symbolics.py
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

import subprocess
from fractions import Fraction
from pathlib import Path
from typing import cast

import sympy as sp

from hamilterm import constants
from hamilterm import elements as mel

# Manually select which terms contribute to the molecular Hamiltonian.
include_r: bool = True
include_so: bool = True
include_ss: bool = True
include_sr: bool = True
include_ld: bool = True

# MAX_N_POWER can be 2, 4, 6, 8, 10, or 12. Powers above 12 have no associated constants and
# therefore will not contribute to the calculation.
MAX_N_POWER: int = 4
# Specify the maximum power of N used when evaluating anticommutators. A value of 0 will skip the
# evaluation of all anticommutators.
MAX_N_ACOMM_POWER: int = 2

# Output to the terminal, LaTeX, or both.
PRINT_TERM: bool = False
PRINT_TEX: bool = True

MAX_POWER_INDEX: int = MAX_N_POWER // 2
MAX_ACOMM_INDEX: int = MAX_N_ACOMM_POWER // 2
LAMBDA_INT_MAP: dict[str, int] = {"Sigma": 0, "Pi": 1}
LAMBDA_STR_MAP: dict[str, str] = {"Sigma": "Σ", "Pi": "Π"}


def construct_n_operator_matrices(
    basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: sp.Symbol
) -> list[sp.MutableDenseMatrix]:
    """Construct the N operator matrices, where N is the total angular momentum w/o any spin.

    Args:
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (sp.Symbol): Quantum number J

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

    # Form the N^2 matrix using the matrix elements above.
    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel.n_squared(i, j, basis_fns, s_qn, j_qn)

    # The following N^{2k} matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, MAX_N_POWER // 2):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def h_rotational(
    i: int,
    j: int,
    n_op_mats: list[sp.MutableDenseMatrix],
    r_consts: constants.RotationalConsts[sp.Symbol],
) -> sp.Expr:
    """Return matrix elements for the rotational Hamiltonian.

    H_r = BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices
        r_consts (RotationalConsts): Rotational constants

    Returns:
        sp.Expr: Matrix elements for BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12
    """
    # BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12
    return (
        r_consts.B * n_op_mats[0][i, j]
        - r_consts.D * n_op_mats[1][i, j]
        + r_consts.H * n_op_mats[2][i, j]
        + r_consts.L * n_op_mats[3][i, j]
        + r_consts.M * n_op_mats[4][i, j]
        + r_consts.P * n_op_mats[5][i, j]
    )


def h_spin_orbit(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    n_op_mats: list[sp.MutableDenseMatrix],
    so_consts: constants.SpinOrbitConsts[sp.Symbol],
) -> sp.Expr:
    """Return matrix elements for the spin-orbit Hamiltonian.

    H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices
        so_consts (SpinOrbitConsts): Spin-orbit constants

    Returns:
        sp.Expr: Matrix elements for A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+
            + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+ + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]
    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    result: sp.Expr = sp.S.Zero

    # Spin-orbit coupling is only defined for states with Λ > 0 and S > 0.
    if abs(lambda_qn_j) > 0 and s_qn > 0:
        # A(LzSz)
        result += so_consts.A * mel.lz_sz(i, j, basis_fns)

        spin_orbit_cd_consts: list[sp.Symbol] = [
            so_consts.A_D,
            so_consts.A_H,
            so_consts.A_L,
            so_consts.A_M,
        ]

        # A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        for k in range(len(basis_fns)):
            # ⟨i|A_x/2[N^{2n}, LzSz]+|j⟩ = A_x/2[⟨i|N^{2n}(LzSz)|j⟩ + ⟨i|(LzSz)N^{2n}|j⟩]
            #                            = A_x/2(∑_k⟨i|N^{2n}|k⟩⟨k|LzSz|j⟩ + ∑_k⟨i|LzSz|k⟩⟨k|N^{2n}|j⟩)
            #                            = A_x/2[(N^{2n})_{ik}(LzSz)_{kj} + (LzSz)_{ik}(N^{2n})_{kj}]
            for idx, const in enumerate(spin_orbit_cd_consts[:MAX_ACOMM_INDEX]):
                result += (
                    Fraction(1, 2)
                    * const
                    * (
                        n_op_mats[idx][i, k] * mel.lz_sz(k, j, basis_fns)
                        + mel.lz_sz(i, k, basis_fns) * n_op_mats[idx][k, j]
                    )
                )

        # ηLzSz[Sz^2 - 1/5(3S^2 - 1)] term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨Λ, Σ|ηLzSz[Sz^2 - 1/5(3S^2 - 1)]|Λ, Σ⟩ = ηΛΣ[Σ^2 - 1/5(3S(S + 1) - 1)]
            result += (
                so_consts.eta
                * mel.lz_sz(i, j, basis_fns)
                * (sigma_qn_j**2 - Fraction(1, 5) * (3 * mel.s_squared(s_qn) - 1))
            )

    return result


def h_spin_spin(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    n_op_mats: list[sp.MutableDenseMatrix],
    ss_consts: constants.SpinSpinConsts[sp.Symbol],
) -> sp.Expr:
    """Return matrix elements for the spin-spin Hamiltonian.

    H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices
        ss_consts (SpinSpinConsts): Spin-spin constants

    Returns:
        sp.Expr: Matrix elements for 2λ/3(3Sz^2 - S^2) + λ_D/2[2/3(3Sz^2 - S^2), N^2]+
            + λ_H/2[2/3(3Sz^2 - S^2), N^4]+ + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)
    """
    sigma_qn_j: Fraction = basis_fns[j][1]

    result: sp.Expr = sp.S.Zero

    # Spin-spin coupling is only defined for states with S > 1/2.
    if s_qn > Fraction(1, 2):
        # 2λ/3(3Sz^2 - S^2)
        result += Fraction(2, 3) * ss_consts.lamda * mel.sz2_minus_s2(i, j, basis_fns, s_qn)

        spin_spin_cd_consts: list[sp.Symbol] = [ss_consts.lambda_D, ss_consts.lambda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for k in range(len(basis_fns)):
            # ⟨i|λ_x/3[(3Sz^2 - S^2), N^{2n}]+|j⟩
            #   = λ_x/3[⟨i|(3Sz^2 - S^2)N^{2n}|j⟩ + ⟨i|N^{2n}(3Sz^2 - S^2)|j⟩]
            #   = λ_x/3(∑_k⟨i|(3Sz^2 - S^2)|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|(3Sz^2 - S^2)|j⟩)
            #   = λ_x/3[(3Sz^2 - S^2)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(3Sz^2 - S^2)_{kj}]
            for idx, const in enumerate(spin_spin_cd_consts[:MAX_ACOMM_INDEX]):
                result += (
                    Fraction(1, 3)
                    * const
                    * (
                        mel.sz2_minus_s2(i, k, basis_fns, s_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.sz2_minus_s2(k, j, basis_fns, s_qn)
                    )
                )

        # θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4) term only valid for states with S > 3/2.
        if i == j and s_qn > Fraction(3, 2):
            # ⟨S, Σ|θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)|S, Σ⟩
            #   = θ/12(35Σ^4 - 30S(S + 1)Σ^2 + 25Σ^2 - 6S(S + 1) + 3[S(S + 1)]^2)
            result += (
                Fraction(1, 12)
                * ss_consts.theta
                * (
                    35 * sigma_qn_j**4
                    - 30 * mel.s_squared(s_qn) * sigma_qn_j**2
                    + 25 * sigma_qn_j**2
                    - 6 * mel.s_squared(s_qn)
                    + 3 * mel.s_squared(s_qn) ** 2
                )
            )

    return result


def h_spin_rotation(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: sp.Symbol,
    n_op_mats: list[sp.MutableDenseMatrix],
    sr_consts: constants.SpinRotationConsts[sp.Symbol],
) -> sp.Expr:
    """Return matrix elements for the spin-rotation Hamiltonian.

    H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (sp.Symbol): Quantum number J
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices
        sr_consts (SpinRotationConsts): Spin-rotation constants

    Returns:
        sp.Expr: Matrix elements for γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
            + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}
    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    result: sp.Expr = sp.S.Zero

    # Spin-rotation coupling is only defined for states with S > 0.
    if s_qn > 0:
        # γ(N·S)
        result += sr_consts.gamma * mel.n_dot_s(i, j, basis_fns, s_qn, j_qn)

        spin_rotation_cd_consts: list[sp.Symbol] = [
            sr_consts.gamma_D,
            sr_consts.gamma_H,
            sr_consts.gamma_L,
        ]

        # γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        for k in range(len(basis_fns)):
            # ⟨i|γ_x/2[N·S, N^{2n}]+|j⟩ = γ_x/2[⟨i|(N·S)N^{2n}|j⟩ + ⟨i|N^{2n}(N·S)|j⟩]
            #                           = γ_x/2(∑_k⟨i|N·S|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|N·S|j⟩)
            #                           = γ_x/2[(N·S)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(N·S)_{kj}]
            for idx, const in enumerate(spin_rotation_cd_consts[:MAX_ACOMM_INDEX]):
                result += (
                    Fraction(1, 2)
                    * const
                    * (
                        mel.n_dot_s(i, k, basis_fns, s_qn, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.n_dot_s(k, j, basis_fns, s_qn, j_qn)
                    )
                )

        # -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)} term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨J, S, Ω + 1, Σ + 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_S/2[S(S + 1) - 5Σ(Σ + 1) + 2]([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j + 1 and omega_qn_i == omega_qn_j + 1:
                result += (
                    -Fraction(1, 2)
                    * sr_consts.gamma_S
                    * (mel.s_squared(s_qn) - 5 * sigma_qn_j * (sigma_qn_j + 1) - 2)
                    * mel.j_minus(j_qn, omega_qn_j)
                    * mel.s_plus(s_qn, sigma_qn_j)
                )

            # ⟨J, S, Ω - 1, Σ - 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_s/2[S(S + 1) - 5Σ(Σ - 1) + 2]([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
                result += (
                    -Fraction(1, 2)
                    * sr_consts.gamma_S
                    * (mel.s_squared(s_qn) - 5 * sigma_qn_j * (sigma_qn_j - 1) - 2)
                    * mel.j_plus(j_qn, omega_qn_j)
                    * mel.s_minus(s_qn, sigma_qn_j)
                )

    return result


def h_lambda_doubling(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: sp.Symbol,
    n_op_mats: list[sp.MutableDenseMatrix],
    ld_consts: constants.LambdaDoublingConsts,
) -> sp.Expr:
    """Return matrix elements for the lambda doubling Hamiltonian.

    H_ld = 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
        + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
        + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
        + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+.

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (sp.Symbol): Quantum number J
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices
        ld_consts (LambdaDoublingConsts): Lambda-doubling constants

    Returns:
        sp.Expr: Matrix elements for 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
            + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
            + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
            + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+
    """
    lambda_qn_i = basis_fns[i][0]
    lambda_qn_j = basis_fns[j][0]

    result: sp.Expr = sp.S.Zero

    # Lambda doubling is only defined for Λ ± 2 transitions.
    if abs(lambda_qn_i - lambda_qn_j) == 2:
        # 0.5(o + p + q)(S+^2 + S-^2)
        result += (
            Fraction(1, 2)
            * (ld_consts.o + ld_consts.p + ld_consts.q)
            * mel.sp2_plus_sm2(i, j, basis_fns, s_qn)
        )

        # -0.5(p + 2q)(J+S+ + J-S-)
        result += (
            -Fraction(1, 2)
            * (ld_consts.p + 2 * ld_consts.q)
            * mel.jpsp_plus_jmsm(i, j, basis_fns, s_qn, j_qn)
        )

        # q/2(J+^2 + J-^2)
        result += Fraction(1, 2) * ld_consts.q * mel.jp2_plus_jm2(i, j, basis_fns, j_qn)

        lambda_doubling_cd_consts_opq: list[sp.Expr] = [
            ld_consts.o_D + ld_consts.p_D + ld_consts.q_D,
            ld_consts.o_H + ld_consts.p_H + ld_consts.q_H,
            ld_consts.o_L + ld_consts.p_L + ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_pq: list[sp.Expr] = [
            ld_consts.p_D + 2 * ld_consts.q_D,
            ld_consts.p_H + 2 * ld_consts.q_H,
            ld_consts.p_L + 2 * ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_q: list[sp.Symbol] = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

        # 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
        #   + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
        #   + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+
        for k in range(len(basis_fns)):
            # ⟨i|[0.25(o_x + p_x + q_x)(S+^2 + S-^2), N^{2n}]+|j⟩
            #   = 0.25(o_x + p_x + q_x)[⟨i|(S+^2 + S-^2)N^{2n}|j⟩ + ⟨i|N^{2n}(S+^2 + S-^2)|j⟩]
            #   = 0.25(o_x + p_x + q_x)(∑_k⟨i|S+^2 + S-^2|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|S+^2 + S-^2|j⟩)
            #   = 0.25(o_x + p_x + q_x)[(S+^2 + S-^2)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(S+^2 + S-^2)_{kj}]
            for idx, const in enumerate(lambda_doubling_cd_consts_opq[:MAX_ACOMM_INDEX]):
                result += (
                    Fraction(1, 4)
                    * const
                    * (
                        mel.sp2_plus_sm2(i, k, basis_fns, s_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.sp2_plus_sm2(k, j, basis_fns, s_qn)
                    )
                )

            # ⟨i|[-0.25(p_x + 2 * q_x)(J+S+ + J-S-), N^{2n}]+|j⟩
            #   = -0.25(p_x + 2 * q_x)[⟨i|(J+S+ + J-S-)N^{2n}|j⟩ + ⟨i|N^{2n}(J+S+ + J-S-)|j⟩]
            #   = -0.25(p_x + 2 * q_x)(∑_k⟨i|J+S+ + J-S-|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|J+S+ + J-S-|j⟩)
            #   = -0.25(p_x + 2 * q_x)[(J+S+ + J-S-)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(J+S+ + J-S-)_{kj}]
            for idx, const in enumerate(lambda_doubling_cd_consts_pq[:MAX_ACOMM_INDEX]):
                result += (
                    -Fraction(1, 4)
                    * const
                    * (
                        mel.jpsp_plus_jmsm(i, k, basis_fns, s_qn, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.jpsp_plus_jmsm(k, j, basis_fns, s_qn, j_qn)
                    )
                )

            # ⟨i|[0.25 * q_x(J+^2 + J-^2), N^{2n}]+|j⟩
            #   = 0.25 * q_x[⟨i|(J+^2 + J-^2)N^{2n}|j⟩ + ⟨i|N^{2n}(J+^2 + J-^2)|j⟩]
            #   = 0.25 * q_x(∑_k⟨i|J+^2 + J-^2|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|J+^2 + J-^2|j⟩)
            #   = 0.25 * q_x[(J+^2 + J-^2)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(J+^2 + J-^2)_{kj}]
            for idx, const in enumerate(lambda_doubling_cd_consts_q[:MAX_ACOMM_INDEX]):
                result += (
                    Fraction(1, 4)
                    * const
                    * (
                        mel.jp2_plus_jm2(i, k, basis_fns, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.jp2_plus_jm2(k, j, basis_fns, j_qn)
                    )
                )

    return result


def build_hamiltonian(
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: sp.Symbol,
    consts: constants.SymbolicConstants,
) -> sp.MutableDenseMatrix:
    """Build the symbolic Hamiltonian matrix.

    Args:
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (sp.Symbol): Quantum number J
        consts (Constants): Molecular constants

    Returns:
        sp.MutableDenseMatrix: Hamiltonian matrix H
    """
    dim: int = len(basis_fns)
    n_op_mats: list[sp.MutableDenseMatrix] = construct_n_operator_matrices(basis_fns, s_qn, j_qn)
    h_mat: sp.MutableDenseMatrix = sp.zeros(dim)

    switch_r, switch_so, switch_ss, switch_sr, switch_ld = map(
        int, [include_r, include_so, include_ss, include_sr, include_ld]
    )

    for i in range(dim):
        for j in range(dim):
            h_mat[i, j] = (
                switch_r * h_rotational(i, j, n_op_mats, consts.rotational)
                + switch_so * h_spin_orbit(i, j, basis_fns, s_qn, n_op_mats, consts.spin_orbit)
                + switch_ss * h_spin_spin(i, j, basis_fns, s_qn, n_op_mats, consts.spin_spin)
                + switch_sr
                * h_spin_rotation(i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.spin_rotation)
                + switch_ld
                * h_lambda_doubling(i, j, basis_fns, s_qn, j_qn, n_op_mats, consts.lambda_doubling)
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
    lambda_qn: int = LAMBDA_INT_MAP[term]

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
    sigmas: list[Fraction] = [
        cast("Fraction", -s_qn + Fraction(i, 1)) for i in range(int(2 * s_qn) + 1)
    ]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas: list[int] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, Fraction, Fraction]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: Fraction = cast("Fraction", lam + sigma)
            basis_fns.append((lam, sigma, omega))

    return basis_fns


class AntiCommutator(sp.Expr):
    """Represents the anticommutator [A, B]_+."""

    def _sympystr(self, printer):
        # TODO: 25/06/11 - This is the fallback for the Unicode pretty printer, which means the
        #       superscripts on the N^{2n} terms and the `*` symbols aren't converted to their
        #       Unicode counterparts. Doesn't matter, just looks slightly worse than it could.
        a, b = self.args
        return f"[{printer._print(a)}, {printer._print(b)}]_+"

    def _latex(self, printer):
        a, b = self.args
        return rf"\left[{printer._print(a)}, {printer._print(b)}\right]_+"


class DotSymbol(sp.Symbol):
    r"""Overrides the LaTeX output for \cdot."""

    def _latex(self, _):
        return r"\cdot"


def included_hamiltonian_terms(
    s_qn: Fraction, j_qn: sp.Symbol, lambda_qn: int, consts: constants.SymbolicConstants
) -> list[sp.Expr]:
    """Return symbolic expressions for the terms included in the diatomic Hamiltonian.

    Args:
        s_qn (Fraction): Quantum number S
        j_qn (sp.Symbol): Quantum number J
        lambda_qn (int): Quantum number Λ
        consts (Constants): Molecular constants

    Returns:
        list[sp.Expr]: Terms included in the diatomic Hamiltonian
    """
    N, S, S_z = sp.symbols("N, S, S_z")

    h_r: sp.Expr = sp.S.Zero
    h_so: sp.Expr = sp.S.Zero
    h_ss: sp.Expr = sp.S.Zero
    h_sr: sp.Expr = sp.S.Zero
    h_ld: sp.Expr = sp.S.Zero

    # H_r = BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12
    if include_r:
        r_consts: constants.RotationalConsts[sp.Symbol] = consts.rotational
        coeffs: list[tuple[sp.Expr, int]] = [
            (r_consts.B, 2),
            (-r_consts.D, 4),
            (r_consts.H, 6),
            (r_consts.L, 8),
            (r_consts.M, 10),
            (r_consts.P, 12),
        ]

        for symbol, exponent in coeffs[:MAX_POWER_INDEX]:
            h_r += symbol * N**exponent

    # H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
    #   + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]
    if include_so and lambda_qn > 0 and s_qn > 0:
        so_consts: constants.SpinOrbitConsts[sp.Symbol] = consts.spin_orbit
        L_z: sp.Symbol = sp.Symbol("L_z")

        # A(LzSz)
        h_so += so_consts.A * L_z * S_z

        spin_orbit_cd_consts: list[sp.Symbol] = [
            so_consts.A_D,
            so_consts.A_H,
            so_consts.A_L,
            so_consts.A_M,
        ]

        # A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        for idx, symbol in enumerate(spin_orbit_cd_consts[:MAX_ACOMM_INDEX]):
            h_so += Fraction(1, 2) * symbol * AntiCommutator(N ** (2 * idx + 2), L_z * S_z)

        # ηLzSz[Sz^2 - 1/5(3S^2 - 1)] term only valid for states with S > 1.
        if s_qn > 1:
            h_so += so_consts.eta * L_z * S_z * (S_z**2 - Fraction(1, 5) * (3 * S**2 - 1))

    # H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
    #   + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)
    if include_ss and s_qn > Fraction(1, 2):
        ss_consts: constants.SpinSpinConsts[sp.Symbol] = consts.spin_spin
        # 2λ/3(3Sz^2 - S^2)
        h_ss += Fraction(2, 3) * ss_consts.lamda * (3 * S_z**2 - S**2)

        spin_spin_cd_consts: list[sp.Symbol] = [ss_consts.lambda_D, ss_consts.lambda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for idx, symbol in enumerate(spin_spin_cd_consts[:MAX_ACOMM_INDEX]):
            h_ss += (
                Fraction(1, 2)
                * symbol
                * AntiCommutator(Fraction(2, 3) * (3 * S_z**2 - S**2), N ** (2 * idx + 2))
            )

        # θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4) term only valid for states with S > 3/2.
        if s_qn > Fraction(3, 2):
            h_ss += (
                Fraction(1, 12)
                * ss_consts.theta
                * (35 * S_z**4 - 30 * S**2 * S_z**2 + 25 * S_z**2 - 6 * S**2 + 3 * S**4)
            )

    # H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
    #   + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}
    if include_sr and s_qn > 0:
        sr_consts: constants.SpinRotationConsts[sp.Symbol] = consts.spin_rotation
        dot: DotSymbol = DotSymbol("dot", commutative=False)
        T_0: sp.Symbol = sp.Symbol("T_0")
        ndots: sp.Expr = sp.Mul(N, dot, S, evaluate=False)
        TJ: sp.Expr = sp.Function("T")(j_qn)
        TS: sp.Expr = sp.Function("T")(S)

        # γ(N·S)
        h_sr += sp.Mul(sr_consts.gamma, ndots, evaluate=False)

        spin_rotation_cd_consts: list[sp.Symbol] = [
            sr_consts.gamma_D,
            sr_consts.gamma_H,
            sr_consts.gamma_L,
        ]

        # γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        for idx, symbol in enumerate(spin_rotation_cd_consts[:MAX_ACOMM_INDEX]):
            h_sr += Fraction(1, 2) * symbol * AntiCommutator(ndots, N ** (2 * idx + 2))

        # -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)} term only valid for states with S > 1.
        if s_qn > 1:
            h_sr += (
                -sp.sqrt(Fraction(70, 3)) * sr_consts.gamma_S * T_0**2 * AntiCommutator(TJ, TS**3)
            )

    # H_ld = o/2(S+^2 + S-^2) - p/2(N+S+ + N-S-) + q/2(N+^2 + N-^2)
    #   + 0.25[S+^2 + S-^2, o_D * N^2 + o_H * N^4 + o_L * N^6]+
    #   - 0.25[N+S+ + N-S-, p_D * N^2 + p_H * N^4 + p_L * N^6]+
    #   + 0.25[N+^2 + N-^2, q_D * N^2 + q_H * N^4 + q_L * N^6]+
    if include_ld and lambda_qn == 1:
        ld_consts: constants.LambdaDoublingConsts[sp.Symbol] = consts.lambda_doubling
        Np, Nm, Sp, Sm = sp.symbols("N_+, N_-, S_+, S_-")

        # q/2(N+^2 + N-^2)
        h_ld += Fraction(1, 2) * ld_consts.q * (Np**2 + Nm**2)

        lambda_doubling_cd_consts_q: list[sp.Symbol] = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

        # 0.25[N+^2 + N-^2, q_D * N^2 + q_H * N^4 + q_L * N^6]+
        for idx, symbol in enumerate(lambda_doubling_cd_consts_q[:MAX_ACOMM_INDEX]):
            h_ld += Fraction(1, 4) * symbol * AntiCommutator(Sp**2 + Sm**2, N ** (2 * idx + 2))

        lambda_doubling_cd_consts_p: list[sp.Symbol] = [ld_consts.p_D, ld_consts.p_H, ld_consts.p_L]

        if s_qn > 0:
            # -p/2(N+S+ + N-S-)
            h_ld += -Fraction(1, 2) * ld_consts.p * (Np * Sp + Nm * Sm)

            # -0.25[N+S+ + N-S-, p_D * N^2 + p_H * N^4 + p_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_p[:MAX_ACOMM_INDEX]):
                h_ld += (
                    -Fraction(1, 4) * symbol * AntiCommutator(Np * Sp + Nm * Sm, N ** (2 * idx + 2))
                )

        lambda_doubling_cd_consts_o: list[sp.Symbol] = [ld_consts.o_D, ld_consts.o_H, ld_consts.o_L]

        if s_qn > Fraction(1, 2):
            # o/2(S+^2 + S-^2)
            h_ld += Fraction(1, 2) * ld_consts.o * (Sp**2 + Sm**2)

            # 0.25[S+^2 + S-^2, o_D * N^2 + o_H * N^4 + o_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_o[:MAX_ACOMM_INDEX]):
                h_ld += Fraction(1, 4) * symbol * AntiCommutator(Sp**2 + Sm**2, N ** (2 * idx + 2))

    return [h_r, h_so, h_ss, h_sr, h_ld]


def fsn(num: int | Fraction | sp.Expr, tex: bool = False) -> str:
    """Format signed number for printing.

    Args:
        num (int | Fraction | sp.Expr): Number

    Returns:
        str: Print-friendly string with a +, ±, or -
    """
    s: str = str(num)

    if num > 0:
        return f"+{s}"
    if num == 0:
        if tex:
            return rf"\pm{s}"
        return f"±{s}"

    return s


def main() -> None:
    """Entry point."""
    # Rotational quantum number J and the shorthand x = J(J + 1).
    j_qn, x = sp.symbols("J, x")
    term_symbol: str = "2Sigma"

    consts: constants.SymbolicConstants = constants.SymbolicConstants()

    s_qn, lambda_qn = parse_term_symbol(term_symbol)
    basis_fns: list[tuple[int, Fraction, Fraction]] = generate_basis_fns(s_qn, lambda_qn)
    h_mat: sp.MutableDenseMatrix = (
        build_hamiltonian(basis_fns, s_qn, j_qn, consts)
        .subs(j_qn * (j_qn + 1), x)
        .applyfunc(sp.simplify)
    )
    eigenval_dict: dict[sp.Expr, int] = cast("dict[sp.Expr, int]", h_mat.eigenvals())
    eigenval_list: list[sp.Expr] = [eigenval.simplify() for eigenval in eigenval_dict]

    h_r, h_so, h_ss, h_sr, h_ld = included_hamiltonian_terms(s_qn, j_qn, lambda_qn, consts)

    if PRINT_TERM:
        print("Info:")
        print(f" • Computed up to N^{MAX_N_POWER}")
        print(f" • max anticommutator value N^{MAX_N_ACOMM_POWER}")

        print("\nTerm symbol:")
        print(f" • {term_symbol[0]}{LAMBDA_STR_MAP[term_symbol[1:]]}: S={s_qn}, Λ={lambda_qn}")

        print("\nBasis states |Λ, Σ, Ω>:")
        for state in basis_fns:
            print(rf" • |{fsn(state[0])}, {fsn(state[1])}, {fsn(state[2])}>")

        print("\nHamiltonian H = H_r + H_so + H_ss + H_sr + H_ld:")
        print("H_r:")
        sp.pprint(h_r)
        print("H_so:")
        sp.pprint(h_so)
        print("H_ss:")
        sp.pprint(h_ss)
        print("H_sr:")
        sp.pprint(h_sr)
        print("H_ld:")
        sp.pprint(h_ld)

        print("\nHamiltonian matrix:")
        sp.pprint(h_mat)

        print("\nEigenvalues:")
        for eigenval in eigenval_list:
            sp.pprint(eigenval)

    if PRINT_TEX:
        tex_term: str = (
            rf"\item $^{term_symbol[0]}\{term_symbol[1:]}:\quad S={s_qn},\;\Lambda={lambda_qn}$"
        )
        tex_ham: str = sp.latex(h_mat)

        basis_items: list[str] = []
        for lam, sig, ome in basis_fns:
            basis_items.append(
                rf"\item $\lvert {fsn(lam, tex=True)},\,{fsn(sig, tex=True)},\,{fsn(ome, tex=True)}\rangle$"
            )

        lines: list[str] = [
            r"\documentclass[12pt,fleqn]{article}",
            r"\usepackage[margin=0.25in]{geometry}",
            r"\usepackage{amsmath,amssymb,bm,parskip}",
            r"\usepackage{graphicx}",
            r"\begin{document}",
            r"\pagestyle{empty}",
            "",
            r"\textbf{Info:}",
            r"\begin{itemize}",
            rf"\item Computed up to $\bm{{N}}^{MAX_N_POWER}$",
            rf"\item Max anticommutator value $\bm{{N}}^{MAX_N_ACOMM_POWER}",
            r"\end{itemize}",
            "",
            r"\textbf{Term symbol:}",
            r"\begin{itemize}",
            tex_term,
            r"\end{itemize}",
            "",
            r"\textbf{Basis states $\lvert \Lambda,\Sigma,\Omega\rangle$:}",
            r"\begin{itemize}",
        ]
        lines += basis_items
        lines += [
            r"\end{itemize}",
            "",
            r"\textbf{Hamiltonian $H = H_r + H_{so} + H_{ss} + H_{sr} + H_{ld}$:}",
            r"\begin{equation*}",
            r"\begin{aligned}",
            rf"H_r &= {sp.latex(h_r)} \\",
            rf"H_{{so}} &= {sp.latex(h_so)} \\",
            rf"H_{{ss}} &= {sp.latex(h_ss)} \\",
            rf"H_{{sr}} &= {sp.latex(h_sr)} \\",
            rf"H_{{ld}} &= {sp.latex(h_ld)}",
            r"\end{aligned}",
            r"\end{equation*}",
            "",
            r"\textbf{Hamiltonian matrix:}",
            r"\begin{equation*}",
            r"\resizebox{\linewidth}{!}{$",
            tex_ham,
            r"$}",
            r"\end{equation*}",
            "",
            r"\textbf{Eigenvalues:}",
            r"\begin{equation*}",
            r"\resizebox{\linewidth}{!}{$",
            r"\begin{aligned}",
        ]

        for idx, eigenval in enumerate(eigenval_list, start=1):
            lines.append(rf"F_{{{idx}}} &= {sp.latex(eigenval)} \\")

        lines += [r"\end{aligned}", r"$}", r"\end{equation*}", "", r"\end{document}"]

        tex_str: str = "\n".join(lines)

        filedir: Path = Path("../../docs/main.tex")
        workdir: Path = Path("../../docs/")

        with open(filedir, "w") as file:
            file.write(tex_str)

        subprocess.run(["pdflatex", "-interaction=batchmode", "main.tex"], cwd=workdir, check=False)


if __name__ == "__main__":
    main()
