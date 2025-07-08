# module numerics
"""Numerically computes the diatomic Hamiltonian for Σ and Π states."""

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
from dataclasses import dataclass, field
from fractions import Fraction
from typing import cast

import numpy as np
from numpy.typing import NDArray


@dataclass
class RotationalConsts:
    """Constants for the rotational operator."""

    B: float = 0.0
    D: float = 0.0
    H: float = 0.0
    L: float = 0.0
    M: float = 0.0
    P: float = 0.0


@dataclass
class SpinOrbitConsts:
    """Constants for the spin-orbit operator."""

    A: float = 0.0
    A_D: float = 0.0
    A_H: float = 0.0
    A_L: float = 0.0
    A_M: float = 0.0
    eta: float = 0.0


@dataclass
class SpinSpinConsts:
    """Constants for the spin-spin operator."""

    lamda: float = 0.0
    lambda_D: float = 0.0
    lambda_H: float = 0.0
    theta: float = 0.0


@dataclass
class SpinRotationConsts:
    """Constants for the spin-rotation operator."""

    gamma: float = 0.0
    gamma_D: float = 0.0
    gamma_H: float = 0.0
    gamma_L: float = 0.0
    gamma_S: float = 0.0


@dataclass
class LambdaDoublingConsts:
    """Constants for the Λ-doubling operator."""

    o: float = 0.0
    p: float = 0.0
    q: float = 0.0
    o_D: float = 0.0
    p_D: float = 0.0
    q_D: float = 0.0
    o_H: float = 0.0
    p_H: float = 0.0
    q_H: float = 0.0
    o_L: float = 0.0
    p_L: float = 0.0
    q_L: float = 0.0


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

MAX_POWER_INDEX: int = MAX_N_POWER // 2
MAX_ACOMM_INDEX: int = MAX_N_ACOMM_POWER // 2
LAMBDA_INT_MAP: dict[str, int] = {"Sigma": 0, "Pi": 1}
LAMBDA_STR_MAP: dict[str, str] = {"Sigma": "Σ", "Pi": "Π"}


def mel_j2(j_qn: int) -> int:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn (int): Quantum number J

    Returns:
        int: Matrix element J(J + 1)
    """
    return j_qn * (j_qn + 1)


def mel_jp(j_qn: int, omega_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨J, Ω - 1|J+|J, Ω⟩ = [J(J + 1) - Ω(Ω - 1)]^(1/2).

    Args:
        j_qn (int): Quantum number J
        omega_qn_j (Fraction): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω - 1)]^(1/2)
    """
    return math.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j - 1))


def mel_jm(j_qn: int, omega_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨J, Ω + 1|J-|J, Ω⟩ = [J(J + 1) - Ω(Ω + 1)]^(1/2).

    Args:
        j_qn (int): Quantum number J
        omega_qn_j (Fraction): Quantum number Ω

    Returns:
        float: Matrix element [J(J + 1) - Ω(Ω + 1)]^(1/2)
    """
    return math.sqrt(j_qn * (j_qn + 1) - omega_qn_j * (omega_qn_j + 1))


def mel_s2(s_qn: Fraction) -> Fraction:
    """Return the diagonal matrix element ⟨S|S^2|S⟩ = S(S + 1).

    Args:
        s_qn (Fraction): Quantum number S

    Returns:
        Fraction: Matrix element S(S + 1)
    """
    return s_qn * (s_qn + 1)


def mel_sp(s_qn: Fraction, sigma_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ + 1|S+|S, Σ⟩ = [S(S + 1) - Σ(Σ + 1)]^(1/2).

    Args:
        s_qn (Fraction): Quantum number S
        sigma_qn_j (Fraction): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ + 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j + 1))


def mel_sm(s_qn: Fraction, sigma_qn_j: Fraction) -> float:
    """Return the off-diagonal matrix element ⟨S, Σ - 1|S-|S, Σ⟩ = [S(S + 1) - Σ(Σ - 1)]^(1/2).

    Args:
        s_qn (Fraction): Quantum number S
        sigma_qn_j (Fraction): Quantum number Σ

    Returns:
        float: Matrix element [S(S + 1) - Σ(Σ - 1)]^(1/2)
    """
    return math.sqrt(s_qn * (s_qn + 1) - sigma_qn_j * (sigma_qn_j - 1))


def construct_n_operator_matrices(
    basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: int
) -> list[NDArray[np.float64]]:
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
    n_op_mats: list[NDArray[np.float64]] = [np.zeros((dim, dim)) for _ in range(6)]

    # Matrix elements for the N^2 operator.
    def mel_n2(i: int, j: int) -> float | Fraction:
        """Return matrix elements for the N^2 operator.

        N^2 = J^2 + S^2 - 2JzSz - (J+S- + J-S+).

        Args:
            i (int): Index i (row) of the Hamiltonian matrix
            j (int): Index j (col) of the Hamiltonian matrix

        Returns:
            float | Fraction: Matrix elements for J^2 + S^2 - 2JzSz - (J+S- + J-S+)
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

        return 0.0

    # Form the N^2 matrix using the matrix elements above.
    for i in range(dim):
        for j in range(dim):
            n_op_mats[0][i, j] = mel_n2(i, j)

    # The following N^{2k} matrices, where k > 1, are formed using matrix multiplication.
    for i in range(1, MAX_N_POWER // 2):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def h_rotational(
    i: int, j: int, n_op_mats: list[NDArray[np.float64]], r_consts: RotationalConsts
) -> float:
    """Return matrix elements for the rotational Hamiltonian.

    H_r = BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        n_op_mats (list[NDArray[np.float64]]): N operator matrices
        r_consts (RotationalConsts): Rotational constants

    Returns:
        float: Matrix elements for BN^2 - DN^4 + HN^6 + LN^8 + MN^10 + PN^12
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
    n_op_mats: list[NDArray[np.float64]],
    so_consts: SpinOrbitConsts,
) -> float:
    """Return matrix elements for the spin-orbit Hamiltonian.

    H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
        + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        n_op_mats (list[NDArray[np.float64]]): N operator matrices
        so_consts (SpinOrbitConsts): Spin-orbit constants

    Returns:
        float: Matrix elements for A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+
            + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+ + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]
    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    result: float = 0.0

    def mel_lzsz(m: int, n: int) -> int | Fraction:
        """Return matrix elements for the LzSz operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            int | Fraction: Matrix elements for LzSz
        """
        lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

        # Operator is completely diagonal, so only m = n terms exist.
        if m == n:
            # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
            return lambda_qn_n * sigma_qn_n

        return 0

    # Spin-orbit coupling is only defined for states with Λ > 0 and S > 0.
    if abs(lambda_qn_j) > 0 and s_qn > 0:
        # A(LzSz)
        result += so_consts.A * mel_lzsz(i, j)

        spin_orbit_cd_consts: list[float] = [
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
                * (sigma_qn_j**2 - Fraction(1, 5) * (3 * mel_s2(s_qn) - 1))
            )

    return result


def h_spin_spin(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    n_op_mats: list[NDArray[np.float64]],
    ss_consts: SpinSpinConsts,
) -> float:
    """Return matrix elements for the spin-spin Hamiltonian.

    H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        n_op_mats (list[NDArray[np.float64]]): N operator matrices
        ss_consts (SpinSpinConsts): Spin-spin constants

    Returns:
        float: Matrix elements for 2λ/3(3Sz^2 - S^2) + λ_D/2[2/3(3Sz^2 - S^2), N^2]+
            + λ_H/2[2/3(3Sz^2 - S^2), N^4]+ + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)
    """
    sigma_qn_j: Fraction = basis_fns[j][1]

    result: float = 0.0

    def mel_sz2ms2(m: int, n: int) -> int | Fraction:
        """Return matrix elements for the 3Sz^2 - S^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            int | Fraction: Matrix elements for 3Sz^2 - S^2
        """
        sigma_qn_n = basis_fns[n][1]

        # Operator is completely diagonal, so only m = n terms exist.
        if m == n:
            # ⟨Λ, Σ|3Sz^2 - S^2|Λ, Σ⟩ = 3Σ^2 - S(S + 1)
            return 3 * sigma_qn_n**2 - mel_s2(s_qn)

        return 0

    # Spin-spin coupling is only defined for states with S > 1/2.
    if s_qn > Fraction(1, 2):
        # 2λ/3(3Sz^2 - S^2)
        result += Fraction(2, 3) * ss_consts.lamda * mel_sz2ms2(i, j)

        spin_spin_cd_consts: list[float] = [ss_consts.lambda_D, ss_consts.lambda_H]

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
                        mel_sz2ms2(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_sz2ms2(k, j)
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
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int,
    n_op_mats: list[NDArray[np.float64]],
    sr_consts: SpinRotationConsts,
) -> float:
    """Return matrix elements for the spin-rotation Hamiltonian.

    H_sr = γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
        + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J
        n_op_mats (list[NDArray[np.float64]]): N operator matrices
        sr_consts (SpinRotationConsts): Spin-rotation constants

    Returns:
        float: Matrix elements for γ(N·S) + γ_D/2[N·S, N^2]+ + γ_H/2[N·S, N^4]+ + γ_L/2[N·S, N^6]+
            + -(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}
    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    result: float = 0.0

    def mel_ndots(m: int, n: int) -> float | Fraction:
        """Return matrix elements for the N·S operator.

        N·S = JzSz + 0.5(J+S- + J-S+) - S^2

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            float: Matrix elements for JzSz + 0.5(J+S- + J-S+) - S^2
        """
        _, sigma_qn_m, omega_qn_m = basis_fns[m]
        _, sigma_qn_n, omega_qn_n = basis_fns[n]

        # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
        if m == n:
            return omega_qn_n * sigma_qn_n - mel_s2(s_qn)

        # ⟨J, S, Ω - 1, Σ - 1|0.5(J+S-)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if omega_qn_m == omega_qn_n - 1 and sigma_qn_m == sigma_qn_n - 1:
            return Fraction(1, 2) * mel_jp(j_qn, omega_qn_n) * mel_sm(s_qn, sigma_qn_n)

        # ⟨J, S, Ω + 1, Σ + 1|0.5(J-S+)|J, S, Ω, Σ⟩ = 0.5([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
        if omega_qn_m == omega_qn_n + 1 and sigma_qn_m == sigma_qn_n + 1:
            return Fraction(1, 2) * mel_jm(j_qn, omega_qn_n) * mel_sp(s_qn, sigma_qn_n)

        return 0.0

    # Spin-rotation coupling is only defined for states with S > 0.
    if s_qn > 0:
        # γ(N·S)
        result += sr_consts.gamma * mel_ndots(i, j)

        spin_rotation_cd_consts: list[float] = [
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
                    -Fraction(1, 2)
                    * sr_consts.gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j + 1) - 2)
                    * mel_jm(j_qn, omega_qn_j)
                    * mel_sp(s_qn, sigma_qn_j)
                )

            # ⟨J, S, Ω - 1, Σ - 1|-(70/3)^(1/2)γ_S * T_0^2{T^1(J), T^3(S)}|J, S, Ω, Σ⟩
            #   = -γ_s/2[S(S + 1) - 5Σ(Σ - 1) + 2]([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
            if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
                result += (
                    -Fraction(1, 2)
                    * sr_consts.gamma_S
                    * (mel_s2(s_qn) - 5 * sigma_qn_j * (sigma_qn_j - 1) - 2)
                    * mel_jp(j_qn, omega_qn_j)
                    * mel_sm(s_qn, sigma_qn_j)
                )

    return result


def h_lambda_doubling(
    i: int,
    j: int,
    basis_fns: list[tuple[int, Fraction, Fraction]],
    s_qn: Fraction,
    j_qn: int,
    n_op_mats: list[NDArray[np.float64]],
    ld_consts: LambdaDoublingConsts,
) -> float:
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
        j_qn (int): Quantum number J
        n_op_mats (list[NDArray[np.float64]]): N operator matrices
        ld_consts (LambdaDoublingConsts): Lambda-doubling constants

    Returns:
        float: Matrix elements for 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + q/2(J+^2 + J-^2)
            + 0.25(o_D + p_D + q_D)[S+^2 + S-^2, N^2]+ - 0.25(p_D + 2 * q_D)[J+S+ + J-S-, N^2]+ + q_D/4[J+^2 + J-^2, N^2]+
            + 0.25(o_H + p_H + q_H)[S+^2 + S-^2, N^4]+ - 0.25(p_H + 2 * q_H)[J+S+ + J-S-, N^4]+ + q_H/4[J+^2 + J-^2, N^4]+
            + 0.25(o_L + p_L + q_L)[S+^2 + S-^2, N^6]+ - 0.25(p_L + 2 * q_L)[J+S+ + J-S-, N^6]+ + q_L/4[J+^2 + J-^2, N^6]+
    """
    lambda_qn_i = basis_fns[i][0]
    lambda_qn_j = basis_fns[j][0]

    result: float = 0.0

    def mel_sp2_sm2(m: int, n: int) -> float:
        """Return matrix elements for the S+^2 + S-^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            float: Matrix elements for S+^2 + S-^2
        """
        lambda_qn_m, sigma_qn_m, _ = basis_fns[m]
        lambda_qn_n, sigma_qn_n, _ = basis_fns[n]

        # ⟨Λ - 2, Σ + 2|S+^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ + 1)][S(S + 1) - (Σ + 1)(Σ + 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n - 2 and sigma_qn_m == sigma_qn_n + 2:
            return mel_sp(s_qn, sigma_qn_n) * mel_sp(s_qn, sigma_qn_n + 1)

        # ⟨Λ + 2, Σ - 2|S-^2|Λ, Σ⟩ = ([S(S + 1) - Σ(Σ - 1)][S(S + 1) - (Σ - 1)(Σ - 2)])^(1/2)
        if lambda_qn_m == lambda_qn_n + 2 and sigma_qn_m == sigma_qn_n - 2:
            return mel_sm(s_qn, sigma_qn_n) * mel_sm(s_qn, sigma_qn_n - 1)

        return 0.0

    def mel_jpsp_jmsm(m: int, n: int) -> float:
        """Return matrix elements for the J+S+ + J-S- operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

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
            return mel_jp(j_qn, omega_qn_n) * mel_sp(s_qn, sigma_qn_n)

        # ⟨Λ + 2, Ω + 1, Σ - 1|J-S-|Λ, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if (
            lambda_qn_m == lambda_qn_n + 2
            and sigma_qn_m == sigma_qn_n - 1
            and omega_qn_m == omega_qn_n + 1
        ):
            return mel_jm(j_qn, omega_qn_n) * mel_sm(s_qn, sigma_qn_n)

        return 0.0

    def mel_jp2_jm2(m: int, n: int) -> float:
        """Return matrix elements for the J+^2 + J-^2 operator.

        Args:
            m (int): Dummy index m for the bra vector (row)
            n (int): Dummy index n for the ket vector (col)

        Returns:
            float: Matrix elements for J+^2 + J-^2
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

        return 0.0

    # Lambda doubling is only defined for Λ ± 2 transitions.
    if abs(lambda_qn_i - lambda_qn_j) == 2:
        # 0.5(o + p + q)(S+^2 + S-^2)
        result += Fraction(1, 2) * (ld_consts.o + ld_consts.p + ld_consts.q) * mel_sp2_sm2(i, j)

        # -0.5(p + 2q)(J+S+ + J-S-)
        result += -Fraction(1, 2) * (ld_consts.p + 2 * ld_consts.q) * mel_jpsp_jmsm(i, j)

        # q/2(J+^2 + J-^2)
        result += Fraction(1, 2) * ld_consts.q * mel_jp2_jm2(i, j)

        lambda_doubling_cd_consts_opq: list[float] = [
            ld_consts.o_D + ld_consts.p_D + ld_consts.q_D,
            ld_consts.o_H + ld_consts.p_H + ld_consts.q_H,
            ld_consts.o_L + ld_consts.p_L + ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_pq: list[float] = [
            ld_consts.p_D + 2 * ld_consts.q_D,
            ld_consts.p_H + 2 * ld_consts.q_H,
            ld_consts.p_L + 2 * ld_consts.q_L,
        ]

        lambda_doubling_cd_consts_q: list[float] = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

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
                    -Fraction(1, 4)
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
                    Fraction(1, 4)
                    * const
                    * (
                        mel_jp2_jm2(i, k) * n_op_mats[idx][k, j]
                        + n_op_mats[idx][i, k] * mel_jp2_jm2(k, j)
                    )
                )

    return result


def build_hamiltonian(
    basis_fns: list[tuple[int, Fraction, Fraction]], s_qn: Fraction, j_qn: int, consts: Constants
) -> NDArray[np.float64]:
    """Build the symbolic Hamiltonian matrix.

    Args:
        basis_fns (list[tuple[int, Fraction, Fraction]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (Fraction): Quantum number S
        j_qn (int): Quantum number J
        consts (Constants): Molecular constants

    Returns:
        NDArray[np.float64]: Hamiltonian matrix H
    """
    dim: int = len(basis_fns)
    n_op_mats: list[NDArray[np.float64]] = construct_n_operator_matrices(basis_fns, s_qn, j_qn)
    h_mat: NDArray[np.float64] = np.zeros((dim, dim))

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
    sigmas: list[Fraction] = [-s_qn + i for i in range(int(2 * s_qn) + 1)]

    # For states with Λ > 1, include both +Λ and -Λ in the basis.
    lambdas: list[int] = [lambda_qn] if lambda_qn == 0 else [-lambda_qn, lambda_qn]
    basis_fns: list[tuple[int, Fraction, Fraction]] = []

    for lam in lambdas:
        for sigma in sigmas:
            omega: Fraction = cast("Fraction", lam + sigma)
            basis_fns.append((lam, sigma, omega))

    return basis_fns


def main() -> None:
    """Entry point."""
    j_qn: int = 1
    term_symbol: str = "2Sigma"

    # Constants for the v' = 0 B3Σu- state of O2.
    consts = Constants(
        rotational=RotationalConsts(B=0.8132, D=4.50e-06),
        spin_spin=SpinSpinConsts(lamda=1.69),
        spin_rotation=SpinRotationConsts(gamma=-0.028),
    )

    s_qn, lambda_qn = parse_term_symbol(term_symbol)
    basis_fns: list[tuple[int, Fraction, Fraction]] = generate_basis_fns(s_qn, lambda_qn)
    h_mat: NDArray[np.float64] = build_hamiltonian(basis_fns, s_qn, j_qn, consts)

    # The Hamiltonian matrix is always Hermitian, so eigvalsh can be used.
    eigenvals: NDArray[np.float64] = np.linalg.eigvalsh(h_mat).astype(np.float64)
    print(h_mat)
    print(eigenvals)


if __name__ == "__main__":
    main()
