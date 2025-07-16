# module numerics.py
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

import timeit
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg._linalg import EighResult
from numpy.typing import NDArray

from hamilterm import constants, options, terms, utils

if TYPE_CHECKING:
    from fractions import Fraction


class NumericComputation:
    """Numerically compute the Hamiltonian for a given J."""

    def __init__(self, term_symbol: str, consts: constants.NumericConstants, j_qn: int) -> None:
        """Initialize class variables.

        Args:
            term_symbol (str): Molecular term symbol, e.g., "2Pi" or "3Sigma"
            consts (constants.NumericConstants): Molecular constants
            j_qn (int): Quantum number J
        """
        self.term_symbol: str = term_symbol
        self.consts: constants.NumericConstants = consts
        self.j_qn: int = j_qn

    @cached_property
    def hamiltonian(self) -> NDArray[np.float64]:
        """Build a Hamiltonian matrix for numeric computation.

        Returns:
            NDArray[np.float64]: Hamiltonian matrix
        """
        s_qn, lambda_qn = utils.parse_term_symbol(self.term_symbol)
        basis_fns: list[tuple[int, Fraction, Fraction]] = utils.generate_basis_fns(s_qn, lambda_qn)

        dim: int = len(basis_fns)
        n_op_mats = utils.construct_n_operator_matrices(basis_fns, s_qn, self.j_qn)

        h_mat: NDArray[np.float64] = np.zeros((dim, dim))

        switch_r, switch_so, switch_ss, switch_sr, switch_ld = map(
            int,
            [
                options.INCLUDE_R,
                options.INCLUDE_SO,
                options.INCLUDE_SS,
                options.INCLUDE_SR,
                options.INCLUDE_LD,
            ],
        )

        for i in range(dim):
            for j in range(dim):
                h_mat[i, j] = (
                    switch_r * terms.rotational(i, j, n_op_mats, self.consts.rotational)
                    + switch_so
                    * terms.spin_orbit(i, j, basis_fns, s_qn, n_op_mats, self.consts.spin_orbit)
                    + switch_ss
                    * terms.spin_spin(i, j, basis_fns, s_qn, n_op_mats, self.consts.spin_spin)
                    + switch_sr
                    * terms.spin_rotation(
                        i, j, basis_fns, s_qn, self.j_qn, n_op_mats, self.consts.spin_rotation
                    )
                    + switch_ld
                    * terms.lambda_doubling(
                        i, j, basis_fns, s_qn, self.j_qn, n_op_mats, self.consts.lambda_doubling
                    )
                )

        return h_mat

    @cached_property
    def hamiltonian_vec(self) -> NDArray[np.float64]:
        s_qn, lambda_qn = utils.parse_term_symbol(self.term_symbol)
        basis_fns: list[tuple[int, Fraction, Fraction]] = utils.generate_basis_fns(s_qn, lambda_qn)

        dim: int = len(basis_fns)
        n_op_mats = utils.construct_n_operator_matrices(basis_fns, s_qn, self.j_qn)

        h_mat: NDArray[np.float64] = np.zeros((dim, dim))

        switch_r, switch_so, switch_ss, switch_sr, switch_ld = map(
            int,
            [
                options.INCLUDE_R,
                options.INCLUDE_SO,
                options.INCLUDE_SS,
                options.INCLUDE_SR,
                options.INCLUDE_LD,
            ],
        )

        if switch_r:
            h_mat = terms.rotational_vec(n_op_mats, self.consts.rotational)

        for i in range(dim):
            for j in range(dim):
                h_mat[i, j] += (
                    switch_so
                    * terms.spin_orbit(i, j, basis_fns, s_qn, n_op_mats, self.consts.spin_orbit)
                    + switch_ss
                    * terms.spin_spin(i, j, basis_fns, s_qn, n_op_mats, self.consts.spin_spin)
                    + switch_sr
                    * terms.spin_rotation(
                        i, j, basis_fns, s_qn, self.j_qn, n_op_mats, self.consts.spin_rotation
                    )
                    + switch_ld
                    * terms.lambda_doubling(
                        i, j, basis_fns, s_qn, self.j_qn, n_op_mats, self.consts.lambda_doubling
                    )
                )

        return h_mat

    @cached_property
    def eigenvalues_eigenvectors(self) -> EighResult:
        """Cache the full eigendecomposition.

        Returns:
            EighResult: Eigenvalues and eigenvectors
        """
        # The Hamiltonian matrix is always Hermitian, so eigh can be used.
        return np.linalg.eigh(self.hamiltonian)

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the Hamiltonian matrix.

        Returns:
            NDArray[np.float64]: Eigenvalues
        """
        return self.eigenvalues_eigenvectors[0]

    @property
    def eigenvectors(self) -> NDArray[np.float64]:
        """Eigenvectors of the Hamiltonian matrix.

        Returns:
            NDArray[np.float64]: Eigenvectors
        """
        return self.eigenvalues_eigenvectors[1]


def main() -> None:
    """Entry point."""
    j_qn: int = 1
    term_symbol: str = "3Sigma"

    # Constants for the v' = 0 B3Σu- state of O2.
    consts: constants.NumericConstants = constants.NumericConstants(
        rotational=constants.RotationalConsts.numeric(B=0.8132, D=4.50e-06),
        spin_spin=constants.SpinSpinConsts.numeric(lamda=1.69),
        spin_rotation=constants.SpinRotationConsts.numeric(gamma=-0.028),
    )

    comp: NumericComputation = NumericComputation(term_symbol, consts, j_qn)

    def bench_orig():
        comp = NumericComputation(term_symbol, consts, j_qn)
        comp.hamiltonian

    def bench_vec():
        comp = NumericComputation(term_symbol, consts, j_qn)
        comp.hamiltonian_vec

    num = 1000
    print(f"{num} Hamiltonians - original:   {timeit.timeit(bench_orig, number=num)} s")
    print(f"{num} Hamiltonians - vectorized: {timeit.timeit(bench_vec, number=num)} s")

    evals_vec, evects_vec = np.linalg.eigh(comp.hamiltonian_vec)

    assert np.allclose(comp.hamiltonian, comp.hamiltonian_vec)
    assert np.allclose(comp.eigenvalues, evals_vec)
    assert np.allclose(comp.eigenvectors, evects_vec)


if __name__ == "__main__":
    main()
