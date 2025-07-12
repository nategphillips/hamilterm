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

from hamilterm import constants, options, utils


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
    if options.INCLUDE_R:
        r_consts: constants.RotationalConsts[sp.Symbol] = consts.rotational
        coeffs: list[tuple[sp.Expr, int]] = [
            (r_consts.B, 2),
            (-r_consts.D, 4),
            (r_consts.H, 6),
            (r_consts.L, 8),
            (r_consts.M, 10),
            (r_consts.P, 12),
        ]

        for symbol, exponent in coeffs[: options.MAX_POWER_INDEX]:
            h_r += symbol * N**exponent

    # H_so = A(LzSz) + A_D/2[N^2, LzSz]+ + A_H/2[N^4, LzSz]+ + A_L/2[N^6, LzSz]+ + A_M/2[N^8, LzSz]+
    #   + ηLzSz[Sz^2 - 1/5(3S^2 - 1)]
    if options.INCLUDE_SO and lambda_qn > 0 and s_qn > 0:
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
        for idx, symbol in enumerate(spin_orbit_cd_consts[: options.MAX_ACOMM_INDEX]):
            h_so += Fraction(1, 2) * symbol * AntiCommutator(N ** (2 * idx + 2), L_z * S_z)

        # ηLzSz[Sz^2 - 1/5(3S^2 - 1)] term only valid for states with S > 1.
        if s_qn > 1:
            h_so += so_consts.eta * L_z * S_z * (S_z**2 - Fraction(1, 5) * (3 * S**2 - 1))

    # H_ss = 2λ/3(3Sz^2 - S^2) + λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
    #   + θ/12(35Sz^4 - 30S^2Sz^2 + 25Sz^2 - 6S^2 + 3S^4)
    if options.INCLUDE_SS and s_qn > Fraction(1, 2):
        ss_consts: constants.SpinSpinConsts[sp.Symbol] = consts.spin_spin
        # 2λ/3(3Sz^2 - S^2)
        h_ss += Fraction(2, 3) * ss_consts.lamda * (3 * S_z**2 - S**2)

        spin_spin_cd_consts: list[sp.Symbol] = [ss_consts.lambda_D, ss_consts.lambda_H]

        # λ_D/3[(3Sz^2 - S^2), N^2]+ + λ_H/3[(3Sz^2 - S^2), N^4]+
        for idx, symbol in enumerate(spin_spin_cd_consts[: options.MAX_ACOMM_INDEX]):
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
    if options.INCLUDE_SR and s_qn > 0:
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
        for idx, symbol in enumerate(spin_rotation_cd_consts[: options.MAX_ACOMM_INDEX]):
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
    if options.INCLUDE_LD and lambda_qn == 1:
        ld_consts: constants.LambdaDoublingConsts[sp.Symbol] = consts.lambda_doubling
        Np, Nm, Sp, Sm = sp.symbols("N_+, N_-, S_+, S_-")

        # q/2(N+^2 + N-^2)
        h_ld += Fraction(1, 2) * ld_consts.q * (Np**2 + Nm**2)

        lambda_doubling_cd_consts_q: list[sp.Symbol] = [ld_consts.q_D, ld_consts.q_H, ld_consts.q_L]

        # 0.25[N+^2 + N-^2, q_D * N^2 + q_H * N^4 + q_L * N^6]+
        for idx, symbol in enumerate(lambda_doubling_cd_consts_q[: options.MAX_ACOMM_INDEX]):
            h_ld += Fraction(1, 4) * symbol * AntiCommutator(Sp**2 + Sm**2, N ** (2 * idx + 2))

        lambda_doubling_cd_consts_p: list[sp.Symbol] = [ld_consts.p_D, ld_consts.p_H, ld_consts.p_L]

        if s_qn > 0:
            # -p/2(N+S+ + N-S-)
            h_ld += -Fraction(1, 2) * ld_consts.p * (Np * Sp + Nm * Sm)

            # -0.25[N+S+ + N-S-, p_D * N^2 + p_H * N^4 + p_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_p[: options.MAX_ACOMM_INDEX]):
                h_ld += (
                    -Fraction(1, 4) * symbol * AntiCommutator(Np * Sp + Nm * Sm, N ** (2 * idx + 2))
                )

        lambda_doubling_cd_consts_o: list[sp.Symbol] = [ld_consts.o_D, ld_consts.o_H, ld_consts.o_L]

        if s_qn > Fraction(1, 2):
            # o/2(S+^2 + S-^2)
            h_ld += Fraction(1, 2) * ld_consts.o * (Sp**2 + Sm**2)

            # 0.25[S+^2 + S-^2, o_D * N^2 + o_H * N^4 + o_L * N^6]+
            for idx, symbol in enumerate(lambda_doubling_cd_consts_o[: options.MAX_ACOMM_INDEX]):
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

    s_qn, lambda_qn = utils.parse_term_symbol(term_symbol)
    basis_fns: list[tuple[int, Fraction, Fraction]] = utils.generate_basis_fns(s_qn, lambda_qn)
    h_mat: sp.MutableDenseMatrix = (
        utils.build_hamiltonian(basis_fns, s_qn, j_qn, consts)
        .subs(j_qn * (j_qn + 1), x)
        .applyfunc(sp.simplify)
    )
    eigenval_dict: dict[sp.Expr, int] = cast("dict[sp.Expr, int]", h_mat.eigenvals())
    eigenval_list: list[sp.Expr] = [eigenval.simplify() for eigenval in eigenval_dict]

    h_r, h_so, h_ss, h_sr, h_ld = included_hamiltonian_terms(s_qn, j_qn, lambda_qn, consts)

    if options.PRINT_TERM:
        print("Info:")
        print(f" • Computed up to N^{options.MAX_N_POWER}")
        print(f" • max anticommutator value N^{options.MAX_N_ACOMM_POWER}")

        print("\nTerm symbol:")
        print(
            f" • {term_symbol[0]}{options.LAMBDA_STR_MAP[term_symbol[1:]]}: S={s_qn}, Λ={lambda_qn}"
        )

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

    if options.PRINT_TEX:
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
            rf"\item Computed up to $\bm{{N}}^{options.MAX_N_POWER}$",
            rf"\item Max anticommutator value $\bm{{N}}^{options.MAX_N_ACOMM_POWER}",
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
