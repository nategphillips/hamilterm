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

import subprocess
from dataclasses import dataclass, field
from typing import cast

import sympy as sp

# Rotational quantum number J and the shorthand x = J(J + 1).
j_qn, x = sp.symbols("J, x")


@dataclass
class RotationalConsts:
    """Constants for the rotational operator."""

    B: sp.Symbol = sp.symbols("B")
    D: sp.Symbol = sp.symbols("D")
    H: sp.Symbol = sp.symbols("H")
    L: sp.Symbol = sp.symbols("L")
    M: sp.Symbol = sp.symbols("M")
    P: sp.Symbol = sp.symbols("P")


@dataclass
class SpinOrbitConsts:
    """Constants for the spin-orbit operator."""

    A: sp.Symbol = sp.symbols("A")
    A_D: sp.Symbol = sp.symbols("A_D")
    A_H: sp.Symbol = sp.symbols("A_H")
    A_L: sp.Symbol = sp.symbols("A_L")
    A_M: sp.Symbol = sp.symbols("A_M")
    eta: sp.Symbol = sp.symbols("eta")


@dataclass
class SpinSpinConsts:
    """Constants for the spin-spin operator."""

    lamda: sp.Symbol = sp.symbols("lambda")
    lambda_D: sp.Symbol = sp.symbols("lambda_D")
    lambda_H: sp.Symbol = sp.symbols("lambda_H")
    theta: sp.Symbol = sp.symbols("theta")


@dataclass
class SpinRotationConsts:
    """Constants for the spin-rotation operator."""

    gamma: sp.Symbol = sp.symbols("gamma")
    gamma_D: sp.Symbol = sp.symbols("gamma_D")
    gamma_H: sp.Symbol = sp.symbols("gamma_H")
    gamma_L: sp.Symbol = sp.symbols("gamma_L")
    gamma_S: sp.Symbol = sp.symbols("gamma_S")


@dataclass
class LambdaDoublingConsts:
    """Constants for the Λ-doubling operator."""

    o: sp.Symbol = sp.symbols("o")
    p: sp.Symbol = sp.symbols("p")
    q: sp.Symbol = sp.symbols("q")
    o_D: sp.Symbol = sp.symbols("o_D")
    p_D: sp.Symbol = sp.symbols("p_D")
    q_D: sp.Symbol = sp.symbols("q_D")
    o_H: sp.Symbol = sp.symbols("o_H")
    p_H: sp.Symbol = sp.symbols("p_H")
    q_H: sp.Symbol = sp.symbols("q_H")
    o_L: sp.Symbol = sp.symbols("o_L")
    p_L: sp.Symbol = sp.symbols("p_L")
    q_L: sp.Symbol = sp.symbols("q_L")


@dataclass
class Constants:
    """All molecular constants."""

    rotational: RotationalConsts = field(default_factory=RotationalConsts)
    spin_orbit: SpinOrbitConsts = field(default_factory=SpinOrbitConsts)
    spin_spin: SpinSpinConsts = field(default_factory=SpinSpinConsts)
    spin_rotation: SpinRotationConsts = field(default_factory=SpinRotationConsts)
    lambda_doubling: LambdaDoublingConsts = field(default_factory=LambdaDoublingConsts)


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
    i: int, j: int, n_op_mats: list[sp.MutableDenseMatrix], r_consts: RotationalConsts
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
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]],
    s_qn: sp.Rational,
    n_op_mats: list[sp.MutableDenseMatrix],
    so_consts: SpinOrbitConsts,
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
        so_consts (SpinOrbitConsts): Spin-orbit constants

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
        result += so_consts.A * mel_lzsz(i, j)

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
                    safe_rational(1, 2)
                    * const
                    * (
                        n_op_mats[idx][i, k] * mel_lzsz(k, j)
                        + mel_lzsz(i, k) * n_op_mats[idx][k, j]
                    )
                )

        # ηLzSz[Sz^2 - 1/5(3S^2 - 1)] term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨Λ, Σ|ηLzSz[Sz^2 - 1/5(3S^2 - 1)]|Λ, Σ⟩ = ηΛΣ[Σ^2 - 1/5(3S(S + 1) - 1)]
            result += (
                so_consts.eta
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
    ss_consts: SpinSpinConsts,
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
        ss_consts (SpinSpinConsts): Spin-spin constants

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
        result += safe_rational(2, 3) * ss_consts.lamda * mel_sz2ms2(i, j)

        spin_spin_cd_consts: list[sp.Symbol] = [ss_consts.lambda_D, ss_consts.lambda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for k in range(len(basis_fns)):
            # ⟨i|λ_x/3[(3Sz^2 - S^2), N^{2n}]+|j⟩
            #   = λ_x/3[⟨i|(3Sz^2 - S^2)N^{2n}|j⟩ + ⟨i|N^{2n}(3Sz^2 - S^2)|j⟩]
            #   = λ_x/3(∑_k⟨i|(3Sz^2 - S^2)|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|(3Sz^2 - S^2)|j⟩)
            #   = λ_x/3[(3Sz^2 - S^2)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(3Sz^2 - S^2)_{kj}]
            for idx, const in enumerate(spin_spin_cd_consts[:MAX_ACOMM_INDEX]):
                result += (
                    safe_rational(1, 3)
                    * const
                    * (
                        mel_sz2ms2(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_sz2ms2(k, j)
                    )
                )

        # θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4) term only valid for states with S > 3/2.
        if i == j and s_qn > safe_rational(3, 2):
            # ⟨S, Σ|θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)|S, Σ⟩
            #   = θ/12(35Σ^4 - 30S(S + 1)Σ^2 + 25Σ^2 - 6S(S + 1) + 3[S(S + 1)]^2)
            result += (
                safe_rational(1, 12)
                * ss_consts.theta
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
    sr_consts: SpinRotationConsts,
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
        sr_consts (SpinRotationConsts): Spin-rotation constants

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
        result += sr_consts.gamma * mel_ndots(i, j)

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
                    safe_rational(1, 2)
                    * const
                    * (
                        mel_ndots(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_ndots(k, j)
                    )
                )

        # -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)} term only valid for states with S > 1.
        if s_qn > 1:
            # ⟨J, S, Ω + 1, Σ + 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_S/2[S(S + 1) - 5Σ(Σ + 1) + 2]([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j + 1 and omega_qn_i == omega_qn_j + 1:
                result += (
                    -safe_rational(1, 2)
                    * sr_consts.gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j + 1) - 2)
                    * mel_jm(j_qn, omega_qn_j)
                    * mel_sp(s_qn, sigma_qn_j)
                )

            # ⟨J, S, Ω - 1, Σ - 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_s/2[S(S + 1) - 5Σ(Σ - 1) + 2]([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
                result += (
                    -safe_rational(1, 2)
                    * sr_consts.gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j - 1) - 2)
                    * mel_jp(j_qn, omega_qn_j)
                    * mel_sm(s_qn, sigma_qn_j)
                )

    return result


def h_lambda_doubling(
    i: int,
    j: int,
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]],
    s_qn: sp.Rational,
    n_op_mats: list[sp.MutableDenseMatrix],
    ld_consts: LambdaDoublingConsts,
) -> sp.Expr:
    """Return matrix elements for the lambda doubling Hamiltonian.

    H_ld = 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
        + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
        + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
        + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+.

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S
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

    def mel_sp2_sm2(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the S+^2 + S-^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for S+^2 + S-^2
        """
        lambda_qn_m, sigma_qn_m, _ = basis_fns[m]
        lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

        # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n - 2 and sigma_qn_m == sigma_qn_n + 2:
            return mel_sp(s_qn, sigma_qn_n) * mel_sp(s_qn, sigma_qn_n + 1)

        # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n + 2 and sigma_qn_m == sigma_qn_n - 2:
            return mel_sm(s_qn, sigma_qn_n) * mel_sm(s_qn, sigma_qn_n - 1)

        return sp.S.Zero

    def mel_jpsp_jmsm(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the J+S+ + J-S- operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for J+S+ + J-S-
        """
        lambda_qn_m, sigma_qn_m, omega_qn_m = basis_fns[m]
        lambda_qn_n, sigma_qn_n, omega_qn_n = basis_fns[n]

        # ⟨Λ - 2, Ω - 1, Σ + 1|J+S+|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
        if (
            lambda_qn_m == lambda_qn_n - 2
            and sigma_qn_m == sigma_qn_n + 1
            and omega_qn_m == omega_qn_n - 1
        ):
            return mel_jp(j_qn, omega_qn_n) * mel_sp(s_qn, sigma_qn_n)

        # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if (
            lambda_qn_m == lambda_qn_n + 2
            and sigma_qn_m == sigma_qn_n - 1
            and omega_qn_m == omega_qn_n + 1
        ):
            return mel_jm(j_qn, omega_qn_n) * mel_sm(s_qn, sigma_qn_n)

        return sp.S.Zero

    def mel_jp2_jm2(m: int, n: int) -> sp.Expr:
        """Return matrix elements for the J+^2 + J-^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            sp.Expr: Matrix elements for J+^2 + J-^2
        """
        lambda_qn_m, _, omega_qn_m = basis_fns[m]
        lambda_qn_n, _, omega_qn_n = basis_fns[n]

        # NOTE: 25/05/29 - The Ω - 1 being plugged into the second J+ matrix element occurs since J
        #       is an anomalously commutative operator.
        # ⟨Λ - 2, Ω - 2|J+^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω - 1)][J(J + 1) - (Ω - 1)(Ω - 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n - 2 and omega_qn_m == omega_qn_n - 2:
            return mel_jp(j_qn, omega_qn_n) * mel_jp(j_qn, omega_qn_n - 1)

        # NOTE: 25/05/29 - The same thing happens here with Ω + 1.
        # ⟨Λ + 2, Ω + 2|J-^2|Λ, Ω⟩ = ([J(J + 1) - Ω(Ω + 1)][J(J + 1) - (Ω + 1)(Ω + 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n + 2 and omega_qn_m == omega_qn_n + 2:
            return mel_jm(j_qn, omega_qn_n) * mel_jm(j_qn, omega_qn_n + 1)

        return sp.S.Zero

    # Lambda doubling is only defined for Λ ± 2 transitions.
    if abs(lambda_qn_i - lambda_qn_j) == 2:
        # 0.5(o + p + q)(S+^2 + S-^2)
        result += (
            safe_rational(1, 2) * (ld_consts.o + ld_consts.p + ld_consts.q) * mel_sp2_sm2(i, j)
        )

        # -0.5(p + 2q)(J+S+ + J-S-)
        result += -safe_rational(1, 2) * (ld_consts.p + 2 * ld_consts.q) * mel_jpsp_jmsm(i, j)

        # q/2(J+^2 + J-^2)
        result += safe_rational(1, 2) * ld_consts.q * mel_jp2_jm2(i, j)

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
                    safe_rational(1, 4)
                    * const
                    * (
                        mel_sp2_sm2(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_sp2_sm2(k, j)
                    )
                )

            # ⟨i|[-0.25(p_x + 2 * q_x)(J+S+ + J-S-), N^{2n}]+|j⟩
            #   = -0.25(p_x + 2 * q_x)[⟨i|(J+S+ + J-S-)N^{2n}|j⟩ + ⟨i|N^{2n}(J+S+ + J-S-)|j⟩]
            #   = -0.25(p_x + 2 * q_x)(∑_k⟨i|J+S+ + J-S-|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|J+S+ + J-S-|j⟩)
            #   = -0.25(p_x + 2 * q_x)[(J+S+ + J-S-)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(J+S+ + J-S-)_{kj}]
            for idx, const in enumerate(lambda_doubling_cd_consts_pq[:MAX_ACOMM_INDEX]):
                result += (
                    -safe_rational(1, 4)
                    * const
                    * (
                        mel_jpsp_jmsm(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_jpsp_jmsm(k, j)
                    )
                )

            # ⟨i|[0.25 * q_x(J+^2 + J-^2), N^{2n}]+|j⟩
            #   = 0.25 * q_x[⟨i|(J+^2 + J-^2)N^{2n}|j⟩ + ⟨i|N^{2n}(J+^2 + J-^2)|j⟩]
            #   = 0.25 * q_x(∑_k⟨i|J+^2 + J-^2|k⟩⟨k|N^{2n}|j⟩ + ∑_k⟨i|N^{2n}|k⟩⟨k|J+^2 + J-^2|j⟩)
            #   = 0.25 * q_x[(J+^2 + J-^2)_{ik}(N^{2n})_{kj} + (N^{2n})_{ik}(J+^2 + J-^2)_{kj}]
            for idx, const in enumerate(lambda_doubling_cd_consts_q[:MAX_ACOMM_INDEX]):
                result += (
                    safe_rational(1, 4)
                    * const
                    * (
                        mel_jp2_jm2(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_jp2_jm2(k, j)
                    )
                )

    return result


def build_hamiltonian(
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational, consts: Constants
) -> sp.MutableDenseMatrix:
    """Build the symbolic Hamiltonian matrix.

    Args:
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S
        consts (Constants): Molecular constants

    Returns:
        sp.MutableDenseMatrix: Hamiltonian matrix H
    """
    dim: int = len(basis_fns)
    n_op_mats: list[sp.MutableDenseMatrix] = construct_n_operator_matrices(basis_fns, s_qn)
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
                * h_spin_rotation(i, j, basis_fns, s_qn, n_op_mats, consts.spin_rotation)
                + switch_ld
                * h_lambda_doubling(i, j, basis_fns, s_qn, n_op_mats, consts.lambda_doubling)
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
    lambda_qn: int = LAMBDA_INT_MAP[term]

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
    s_qn: sp.Rational, lambda_qn: int, consts: Constants
) -> list[sp.Expr]:
    """Return symbolic expressions for the terms included in the diatomic Hamiltonian.

    Args:
        s_qn (sp.Rational): Quantum number S
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
        r_consts: RotationalConsts = consts.rotational
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
        so_consts: SpinOrbitConsts = consts.spin_orbit
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
            h_so += safe_rational(1, 2) * symbol * AntiCommutator(N ** (2 * idx + 2), L_z * S_z)

        # ηLzSz[Sz^2 - 1/5(3S^2 - 1)] term only valid for states with S > 1.
        if s_qn > 1:
            h_so += so_consts.eta * L_z * S_z * (S_z**2 - safe_rational(1, 5) * (3 * S**2 - 1))

    # H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
    #   + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)
    if include_ss and s_qn > safe_rational(1, 2):
        ss_consts: SpinSpinConsts = consts.spin_spin
        # 2λ/3(3Sz^2 - S^2)
        h_ss += safe_rational(2, 3) * ss_consts.lamda * (3 * S_z**2 - S**2)

        spin_spin_cd_consts: list[sp.Symbol] = [ss_consts.lambda_D, ss_consts.lambda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for idx, symbol in enumerate(spin_spin_cd_consts[:MAX_ACOMM_INDEX]):
            h_ss += (
                safe_rational(1, 2)
                * symbol
                * AntiCommutator(safe_rational(2, 3) * (3 * S_z**2 - S**2), N ** (2 * idx + 2))
            )

        # θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4) term only valid for states with S > 3/2.
        if s_qn > safe_rational(3, 2):
            h_ss += (
                safe_rational(1, 12)
                * ss_consts.theta
                * (35 * S_z**4 - 30 * S**2 * S_z**2 + 25 * S_z**2 - 6 * S**2 + 3 * S**4)
            )

    # H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
    #   + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}
    if include_sr and s_qn > 0:
        sr_consts: SpinRotationConsts = consts.spin_rotation
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
            h_sr += safe_rational(1, 2) * symbol * AntiCommutator(ndots, N ** (2 * idx + 2))

        # -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)} term only valid for states with S > 1.
        if s_qn > 1:
            h_sr += (
                -sp.sqrt(safe_rational(70, 3))
                * sr_consts.gamma_S
                * T_0**2
                * AntiCommutator(TJ, TS**3)
            )

    # H_ld = o/2(S+^2 + S-^2) - p/2(N+S+ + N-S-) + q/2(N+^2 + N-^2)
    #   + 0.25[S+^2 + S-^2, o_D * N^2 + o_H * N^4 + o_L * N^6]+
    #   - 0.25[N+S+ + N-S-, p_D * N^2 + p_H * N^4 + p_L * N^6]+
    #   + 0.25[N+^2 + N-^2, q_D * N^2 + q_H * N^4 + q_L * N^6]+
    if include_ld and lambda_qn == 1:
        ld_consts: LambdaDoublingConsts = consts.lambda_doubling
        Np, Nm, Sp, Sm = sp.symbols("N_+, N_-, S_+, S_-")

        # q/2(N+^2 + N-^2)
        h_ld += safe_rational(1, 2) * ld_consts.q * (Np**2 + Nm**2)

        lambda_doubling_cd_consts_q: list[sp.Symbol] = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

        # 0.25[N+^2 + N-^2, q_D * N^2 + q_H * N^4 + q_L * N^6]+
        for idx, symbol in enumerate(lambda_doubling_cd_consts_q[:MAX_ACOMM_INDEX]):
            h_ld += safe_rational(1, 4) * symbol * AntiCommutator(Sp**2 + Sm**2, N ** (2 * idx + 2))

        lambda_doubling_cd_consts_p: list[sp.Symbol] = [ld_consts.p_D, ld_consts.p_H, ld_consts.p_L]

        if s_qn > 0:
            # -p/2(N+S+ + N-S-)
            h_ld += -safe_rational(1, 2) * ld_consts.p * (Np * Sp + Nm * Sm)

            # -0.25[N+S+ + N-S-, p_D * N^2 + p_H * N^4 + p_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_p[:MAX_ACOMM_INDEX]):
                h_ld += (
                    -safe_rational(1, 4)
                    * symbol
                    * AntiCommutator(Np * Sp + Nm * Sm, N ** (2 * idx + 2))
                )

        lambda_doubling_cd_consts_o: list[sp.Symbol] = [ld_consts.o_D, ld_consts.o_H, ld_consts.o_L]

        if s_qn > safe_rational(1, 2):
            # o/2(S+^2 + S-^2)
            h_ld += safe_rational(1, 2) * ld_consts.o * (Sp**2 + Sm**2)

            # 0.25[S+^2 + S-^2, o_D * N^2 + o_H * N^4 + o_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_o[:MAX_ACOMM_INDEX]):
                h_ld += (
                    safe_rational(1, 4) * symbol * AntiCommutator(Sp**2 + Sm**2, N ** (2 * idx + 2))
                )

    return [h_r, h_so, h_ss, h_sr, h_ld]


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


def fsn(num: int | sp.Rational | sp.Expr, tex: bool = False) -> str:
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
        if tex:
            return rf"\pm{s}"
        return f"±{s}"

    return s


def main() -> None:
    """Entry point."""
    term_symbol: str = "2Sigma"

    consts: Constants = Constants()

    s_qn, lambda_qn = parse_term_symbol(term_symbol)
    basis_fns: list[tuple[int, sp.Rational, sp.Rational]] = generate_basis_fns(s_qn, lambda_qn)
    h_mat: sp.MutableDenseMatrix = (
        build_hamiltonian(basis_fns, s_qn, consts).subs(j_qn * (j_qn + 1), x).applyfunc(sp.simplify)
    )
    eigenval_dict: dict[sp.Expr, int] = cast("dict[sp.Expr, int]", h_mat.eigenvals())
    eigenval_list: list[sp.Expr] = [eigenval.simplify() for eigenval in eigenval_dict]

    h_r, h_so, h_ss, h_sr, h_ld = included_hamiltonian_terms(s_qn, lambda_qn, consts)

    if PRINT_TERM:
        print("Info:")
        print(f"Computed up to N^{MAX_N_POWER}, max anticommutator value N^{MAX_N_ACOMM_POWER}")

        print("\nTerm symbol:")
        print(f"{term_symbol[0]}{LAMBDA_STR_MAP[term_symbol[1:]]}: S={s_qn}, Λ={lambda_qn}")

        print("\nBasis states |Λ, Σ, Ω>:")
        for state in basis_fns:
            print(rf"|{fsn(state[0])}, {fsn(state[1])}, {fsn(state[2])}>")

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

        with open("../docs/main.tex", "w") as file:
            file.write(tex_str)

        subprocess.run(
            ["pdflatex", "-interaction=batchmode", "../docs/main.tex"], cwd="../docs/", check=False
        )


if __name__ == "__main__":
    main()
