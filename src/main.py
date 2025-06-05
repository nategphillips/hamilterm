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

A, B, D, j_qn, o, p, q, x, lamda, gamma = sp.symbols("A, B, D, J, o, p, q, x, lambda, gamma")

include_r: bool = True
include_so: bool = False
include_ss: bool = False
include_sr: bool = False
include_ld: bool = False

f_r, f_so, f_ss, f_sr, f_ld = map(int, [include_r, include_so, include_ss, include_sr, include_ld])

# MAX_N_POWER can be 2, 4, 6, 8, 10, or 12. Powers above 12 have no associated constants and
# therefore will not contribute to the calculation.
MAX_N_POWER: int = 12
LAMBDA_MAP: dict[str, int] = {"Sigma": 0, "Pi": 1}


def mel_j2(j_qn: sp.Symbol) -> sp.Expr:
    """Return the diagonal matrix element ⟨J|J^2|J⟩ = J(J + 1).

    Args:
        j_qn (sp.Symbol): Quantum number J

    Returns:
        sp.Expr: Matrix element J(J + 1)

    """
    return j_qn * (j_qn + 1)


def mel_jz(omega_qn: sp.Rational) -> sp.Rational:
    """Return the diagonal matrix element ⟨Ω|Jz|Ω⟩ = Ω.

    Args:
        omega_qn (sp.Rational): Quantum number Ω

    Returns:
        sp.Rational: Matrix element Ω

    """
    return omega_qn


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


def mel_sz(sigma_qn: sp.Rational) -> sp.Rational:
    """Return the diagonal matrix element ⟨Σ|Sz|Σ⟩ = Σ.

    Args:
        sigma_qn (sp.Rational): Quantum number Σ

    Returns:
        sp.Rational: Matrix element Σ

    """
    return sigma_qn


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


def mel_lz(lambda_qn: int) -> int:
    """Return the diagonal matrix element ⟨Λ|Lz|Λ⟩ = Λ.

    Args:
        lambda_qn (int): Quantum number Λ

    Returns:
        int: Matrix element Λ

    """
    return lambda_qn


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
    mat_size: int = len(basis_fns)

    # Each N^(2k) operator, where k is an integer, will have its own operator matrix. This notation
    # implies the N^2 operator occupies index 0, N^4 occupies index 1, etc.
    n_op_mats: list[sp.MutableDenseMatrix] = [sp.zeros(mat_size) for _ in range(MAX_N_POWER // 2)]

    # Matrix elements for the N^2 operator.
    def mel_n2(i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]]) -> sp.Expr:
        """Return matrix elements for the N^2 operator.

        N^2 = J^2 + S^2 - 2JzSz - J+S- - J-S+.

        Args:
            i (int): Index i (row) of the Hamiltonian matrix
            j (int): Index j (col) of the Hamiltonian matrix
            basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>

        Returns:
            sp.Expr: Matrix elements for J^2 + S^2 - 2JzSz - J+S- - J-S+
        """
        _, sigma_qn_i, omega_qn_i = basis_fns[i]
        _, sigma_qn_j, omega_qn_j = basis_fns[j]

        # ⟨J, S, Ω, Σ|J^2 + S^2 - 2JzSz|J, S, Ω, Σ⟩ = J(J + 1) + S(S + 1) - 2ΩΣ
        if i == j:
            return mel_j2(j_qn) + mel_s2(s_qn) - 2 * mel_jz(omega_qn_j) * mel_sz(sigma_qn_j)

        # ⟨J, S, Ω - 1, Σ - 1|J+S-|J, S, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
        if omega_qn_i == omega_qn_j - 1 and sigma_qn_i == sigma_qn_j - 1:
            return -mel_jp(j_qn, omega_qn_j) * mel_sm(s_qn, sigma_qn_j)

        # ⟨J, S, Ω + 1, Σ + 1|J-S+|J, S, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
        if omega_qn_i == omega_qn_j + 1 and sigma_qn_i == sigma_qn_j + 1:
            return -mel_jm(j_qn, omega_qn_j) * mel_sp(s_qn, sigma_qn_j)

        return sp.S.Zero

    # Form the N^2 matrix using the matrix elements above.
    for i in range(mat_size):
        for j in range(mat_size):
            n_op_mats[0][i, j] = mel_n2(i, j, basis_fns)

    # The following N^(2k), where k > 1, matrices are formed using matrix multiplication.
    for i in range(1, MAX_N_POWER // 2):
        n_op_mats[i] = n_op_mats[i - 1] @ n_op_mats[0]

    return n_op_mats


def h_rotational(
    i: int,
    j: int,
    n_op_mats: list[sp.MutableDenseMatrix],
) -> sp.Expr:
    """Return matrix elements for the rotational Hamiltonian.

    H_r = B * N^2 = B(J^2 + S^2 - 2JzSz - J+S- - J-S+).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        n_op_mats (list[sp.MutableDenseMatrix]): N operator matrices

    Returns:
        sp.Expr: Matrix elements for B(J^2 + S^2 - 2JzSz - J+S- - J-S+)

    """
    return B * n_op_mats[0][i, j]


def h_spin_orbit(
    i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.Expr:
    """Return matrix elements for the spin-orbit Hamiltonian.

    H_so = A(Lz * Sz).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Expr: Matrix elements for A(Lz * Sz)

    """
    lambda_qn_j, sigma_qn_j, _ = basis_fns[j]

    # Spin-orbit coupling is only defined for states with Λ > 0 and S > 0. (Neglecting the term for
    # states with Λ > 0 and S > 1 for now.)
    if abs(lambda_qn_j) == 0 and s_qn == 0:
        return sp.S.Zero

    # ⟨Λ, Σ|LzSz|Λ, Σ⟩ = ΛΣ
    if i == j:
        return A * mel_lz(lambda_qn_j) * mel_sz(sigma_qn_j)

    return sp.S.Zero


def h_spin_spin(
    i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.Expr:
    """Return matrix elements for the spin-spin Hamiltonian.

    H_ss = 2/3 * λ(3Sz^2 - S^2).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Expr: Matrix elements for 2/3 * λ(3Sz^2 - S^2)

    """
    sigma_qn_j: sp.Rational = basis_fns[j][1]

    # Spin-spin coupling is only defined for states with S > 1/2. (Neglecting the term for states
    # with S > 3/2 for now.)
    if s_qn < safe_rational(1, 2):
        return sp.S.Zero

    # ⟨S, Σ|3Sz^2 - S^2|S, Σ⟩ = 3Σ^2 − S(S + 1)
    if i == j:
        return sp.Rational(2, 3) * lamda * (3 * mel_sz(sigma_qn_j) ** 2 - mel_s2(s_qn))

    return sp.S.Zero


def h_spin_rotation(
    i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.Expr:
    """Return matrix elements for the spin-rotation Hamiltonian.

    H_sr = γ(N·S) = γ[JzSz + 0.5 * (J+S- + J-S+) - S^2].

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Expr: Matrix elements for γ[JzSz + 0.5 * (J+S- + J-S+) - S^2]

    """
    _, sigma_qn_i, omega_qn_i = basis_fns[i]
    _, sigma_qn_j, omega_qn_j = basis_fns[j]

    # Spin-rotation coupling is only defined for states with S > 0. (Neglecting the term for states
    # with S > 1 for now.)
    if s_qn == 0:
        return sp.S.Zero

    # ⟨S, Ω, Σ|JzSz - S^2|S, Ω, Σ⟩ = ΩΣ - S(S + 1)
    if i == j:
        return gamma * (mel_jz(omega_qn_j) * mel_sz(sigma_qn_j) - mel_s2(s_qn))

    # ⟨J, S, Ω - 1, Σ - 1|J+S-|J, S, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω - 1)][S(S + 1) - Σ(Σ - 1)])^(1/2)
    if sigma_qn_i == sigma_qn_j - 1 and omega_qn_i == omega_qn_j - 1:
        return gamma * (mel_jp(j_qn, omega_qn_j) * mel_sm(s_qn, sigma_qn_j)) / 2

    # ⟨J, S, Ω + 1, Σ + 1|J-S+|J, S, Ω, Σ⟩ = ([J(J + 1) - Ω(Ω + 1)][S(S + 1) - Σ(Σ + 1)])^(1/2)
    if sigma_qn_i == sigma_qn_j + 1 and omega_qn_i == omega_qn_j + 1:
        return gamma * (mel_jm(j_qn, omega_qn_j) * mel_sp(s_qn, sigma_qn_j)) / 2

    return sp.S.Zero


def h_lambda_doubling(
    i: int, j: int, basis_fns: list[tuple[int, sp.Rational, sp.Rational]], s_qn: sp.Rational
) -> sp.Expr:
    """Return matrix elements for the lambda doubling Hamiltonian.

    H_ld = 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) + 0.5 * q(J+^2 + J-^2).

    Args:
        i (int): Index i (row) of the Hamiltonian matrix
        j (int): Index j (col) of the Hamiltonian matrix
        basis_fns (list[tuple[int, sp.Rational, sp.Rational]]): List of basis vectors |Λ, Σ; Ω>
        s_qn (sp.Rational): Quantum number S

    Returns:
        sp.Expr: Matrix elements for 0.5(o + p + q)(S+^2 + S-^2) - 0.5(p + 2q)(J+S+ + J-S-) +
            0.5 * q(J+^2 + J-^2)

    """
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
    mat_size: int = len(basis_fns)
    n_op_mats: list[sp.MutableDenseMatrix] = construct_n_operator_matrices(basis_fns, s_qn)
    h_mat: sp.MutableDenseMatrix = sp.zeros(mat_size)

    for i in range(mat_size):
        for j in range(mat_size):
            h_mat[i, j] = (
                f_r * h_rotational(i, j, n_op_mats)
                + f_so * h_spin_orbit(i, j, basis_fns, s_qn)
                + f_ss * h_spin_spin(i, j, basis_fns, s_qn)
                + f_sr * h_spin_rotation(i, j, basis_fns, s_qn)
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
    h_mat: sp.MutableDenseMatrix = build_hamiltonian(basis_fns, s_qn)
    sp.pprint(h_mat.subs(j_qn * (j_qn + 1), x).applyfunc(sp.nsimplify))

    print("\nEigenvalues:")
    eigenvals: dict[sp.Expr, int] = cast("dict[sp.Expr, int]", h_mat.eigenvals())
    for eigenvalue in eigenvals:
        sp.pprint(sp.nsimplify(eigenvalue))


if __name__ == "__main__":
    main()
