# module terms.py
"""Contains functions for computing terms in the diatomic Hamiltonian."""

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

from sympy import Expr, Integer, MutableDenseMatrix, Rational, Symbol

from hamilterm import constants
from hamilterm import elements as mel


def rotational(
    i: int,
    j: int,
    n_op_mats: list[MutableDenseMatrix],
    r_consts: constants.RotationalConsts,
) -> Expr:
    """Return matrix elements for the rotational Hamiltonian.

    H_r = BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        n_op_mats: N operator matrices.
        r_consts: Rotational constants.

    Returns:
        The matrix elements for BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12.
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


def spin_orbit(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    n_op_mats: list[MutableDenseMatrix],
    so_consts: constants.SpinOrbitConsts,
    max_acomm_index: int,
) -> Expr:
    """Return matrix elements for the spin-orbit Hamiltonian.

    H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
         + ηLzSz[Sz^2 - 1/5(3S^2 - 1)].

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        n_op_mats: N operator matrices.
        so_consts: Spin-orbit constants.
        max_acomm_index: Index of the maximum anticommutator term to compute.

    Returns:
        The matrix elements for A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+
        + A_M/2[N^8, LzSz]+ + ηLzSz[Sz^2 - 1/5(3S^2 - 1)].
    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    result = Integer(0)

    # Spin-orbit coupling is only defined for states with Λ > 0 and S > 0. Since the Λ values in the
    # basis functions range from -Λ to +Λ, make sure to get the absolute value, |Λ|.
    if abs(lambda_qn_j) > 0 and s_qn > 0:
        # A(LzSz)
        result += so_consts.A * mel.lz_sz(i, j, basis_fns)

        spin_orbit_cd_consts = [so_consts.A_D, so_consts.A_H, so_consts.A_L, so_consts.A_M]

        # A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        for k in range(len(basis_fns)):
            for idx, const in enumerate(spin_orbit_cd_consts[:max_acomm_index]):
                result += (
                    Rational(1, 2)
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
                * (sigma_qn_j**2 - Rational(1, 5) * (3 * mel.s_squared(s_qn) - 1))
            )

    return result


def spin_spin(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    n_op_mats: list[MutableDenseMatrix],
    ss_consts: constants.SpinSpinConsts,
    max_acomm_index: int,
) -> Expr:
    """Return matrix elements for the spin-spin Hamiltonian.

    H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
         + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4).

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        n_op_mats: N operator matrices.
        ss_consts: Spin-spin constants.
        max_acomm_index: Index of the maximum anticommutator term to compute.

    Returns:
        The matrix elements for 2λ/3(3Sz^2 - S^2) + λ_D/2[2/3(3Sz^2 - S^2), N^2]+
        + λ_H/2[2/3(3Sz^2 - S^2), N^4]+ + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4).
    """
    sigma_qn_j = basis_fns[j][1]

    result = Integer(0)

    # Spin-spin coupling is only defined for states with S > 1/2.
    if s_qn > Rational(1, 2):
        # 2λ/3(3Sz^2 - S^2)
        result += Rational(2, 3) * ss_consts.lamda * mel.three_sz2_minus_s2(i, j, basis_fns, s_qn)

        spin_spin_cd_consts = [ss_consts.lamda_D, ss_consts.lamda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for k in range(len(basis_fns)):
            for idx, const in enumerate(spin_spin_cd_consts[:max_acomm_index]):
                result += (
                    Rational(1, 3)
                    * const
                    * (
                        mel.three_sz2_minus_s2(i, k, basis_fns, s_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.three_sz2_minus_s2(k, j, basis_fns, s_qn)
                    )
                )

        # θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4) term only valid for states with S > 3/2.
        if i == j and s_qn > Rational(3, 2):
            # ⟨S, Σ|θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)|S, Σ⟩
            #   = θ/12(35Σ^4 - 30S(S + 1)Σ^2 + 25Σ^2 - 6S(S + 1) + 3[S(S + 1)]^2)
            result += (
                Rational(1, 12)
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


def spin_rotation(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
    n_op_mats: list[MutableDenseMatrix],
    sr_consts: constants.SpinRotationConsts,
    max_acomm_index: int,
) -> Expr:
    """Return matrix elements for the spin-rotation Hamiltonian.

    H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
         - (70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        n_op_mats: N operator matrices.
        sr_consts: Spin-rotation constants.
        max_acomm_index: Index of the maximum anticommutator term to compute.

    Returns:
        The matrix elements for γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        - (70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}.
    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    result = Integer(0)

    # Spin-rotation coupling is only defined for states with S > 0.
    if s_qn > 0:
        # γ(N·S)
        result += sr_consts.gamma * mel.n_dot_s(i, j, basis_fns, s_qn, j_qn)

        spin_rotation_cd_consts = [sr_consts.gamma_D, sr_consts.gamma_H, sr_consts.gamma_L]

        # γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        for k in range(len(basis_fns)):
            for idx, const in enumerate(spin_rotation_cd_consts[:max_acomm_index]):
                result += (
                    Rational(1, 2)
                    * const
                    * (
                        mel.n_dot_s(i, k, basis_fns, s_qn, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.n_dot_s(k, j, basis_fns, s_qn, j_qn)
                    )
                )

        # -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)} term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨J, S, Ω - 1, Σ - 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_s/2[S(S + 1) - 5Σ(Σ - 1) + 2]([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
                result += (
                    -Rational(1, 2)
                    * sr_consts.gamma_S
                    * (mel.s_squared(s_qn) - 5 * sigma_qn_j * (sigma_qn_j - 1) - 2)
                    * mel.j_plus(j_qn, omega_qn_j)
                    * mel.s_minus(s_qn, sigma_qn_j)
                )

            # ⟨J, S, Ω + 1, Σ + 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_S/2[S(S + 1) - 5Σ(Σ + 1) + 2]([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j + 1 and omega_qn_i == omega_qn_j + 1:
                result += (
                    -Rational(1, 2)
                    * sr_consts.gamma_S
                    * (mel.s_squared(s_qn) - 5 * sigma_qn_j * (sigma_qn_j + 1) - 2)
                    * mel.j_minus(j_qn, omega_qn_j)
                    * mel.s_plus(s_qn, sigma_qn_j)
                )

    return result


def lambda_doubling(
    i: int,
    j: int,
    basis_fns: list[tuple[Integer, Rational, Rational]],
    s_qn: Rational,
    j_qn: Symbol,
    n_op_mats: list[MutableDenseMatrix],
    ld_consts: constants.LambdaDoublingConsts,
    max_acomm_index: int,
) -> Expr:
    """Return matrix elements for the lambda doubling Hamiltonian.

    H_ld = 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
         + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
         + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
         + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+.

    Args:
        i: Row index of the Hamiltonian.
        j: Column index of the Hamiltonian.
        basis_fns: Basis functions |Λ, Σ; Ω⟩ for each row of the Hamiltonian.
        s_qn: Quantum number S.
        j_qn: Quantum number J.
        n_op_mats: N operator matrices.
        ld_consts: Lambda-doubling constants.
        max_acomm_index: Index of the maximum anticommutator term to compute.

    Returns:
        The matrix elements for 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
        + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
        + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
        + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+.
    """
    lambda_qn_i = basis_fns[i][0]
    lambda_qn_j = basis_fns[j][0]

    result = Integer(0)

    # Lambda doubling is only defined for Λ ± 2 transitions, i.e., Π states.
    if abs(lambda_qn_i - lambda_qn_j) == 2:
        # 0.5(o + p + q)(S+^2 + S-^2)
        result += (
            Rational(1, 2)
            * (ld_consts.o + ld_consts.p + ld_consts.q)
            * mel.sp2_plus_sm2(i, j, basis_fns, s_qn)
        )

        # -0.5(p + 2q)(J+S+ + J-S-)
        result += (
            -Rational(1, 2)
            * (ld_consts.p + 2 * ld_consts.q)
            * mel.jpsp_plus_jmsm(i, j, basis_fns, s_qn, j_qn)
        )

        # q/2(J+^2 + J-^2)
        result += Rational(1, 2) * ld_consts.q * mel.jp2_plus_jm2_sym(i, j, basis_fns, j_qn)

        lambda_doubling_cd_consts_opq = [
            ld_consts.o_D + ld_consts.p_D + ld_consts.q_D,
            ld_consts.o_H + ld_consts.p_H + ld_consts.q_H,
            ld_consts.o_L + ld_consts.p_L + ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_pq = [
            ld_consts.p_D + 2 * ld_consts.q_D,
            ld_consts.p_H + 2 * ld_consts.q_H,
            ld_consts.p_L + 2 * ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_q = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

        # 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
        #   + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
        #   + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+
        for k in range(len(basis_fns)):
            for idx, const in enumerate(lambda_doubling_cd_consts_opq[:max_acomm_index]):
                result += (
                    Rational(1, 4)
                    * const
                    * (
                        mel.sp2_plus_sm2(i, k, basis_fns, s_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.sp2_plus_sm2(k, j, basis_fns, s_qn)
                    )
                )

            for idx, const in enumerate(lambda_doubling_cd_consts_pq[:max_acomm_index]):
                result += (
                    -Rational(1, 4)
                    * const
                    * (
                        mel.jpsp_plus_jmsm(i, k, basis_fns, s_qn, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.jpsp_plus_jmsm(k, j, basis_fns, s_qn, j_qn)
                    )
                )

            for idx, const in enumerate(lambda_doubling_cd_consts_q[:max_acomm_index]):
                result += (
                    Rational(1, 4)
                    * const
                    * (
                        mel.jp2_plus_jm2_sym(i, k, basis_fns, j_qn) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel.jp2_plus_jm2_sym(k, j, basis_fns, j_qn)
                    )
                )

    return result
