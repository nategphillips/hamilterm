# module numerics.py
"""Numerically computes the diatomic Hamiltonian for Σ and Π states."""

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

import timeit
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from hamilterm import constants, options, terms, utils

if TYPE_CHECKING:
    from numpy.linalg._linalg import EighResult
    from numpy.typing import NDArray


class NumericComputation:
    """Numerically compute the Hamiltonian for a given J."""

    def __init__(
        self,
        term_symbol: str,
        consts: constants.Constants,
        j_qn: float,
        max_n_power: int = 4,
        max_acomm_power: int = 2,
    ) -> None:
        """Initialize class variables.

        Args:
            term_symbol: Molecular term symbol, e.g., "2Pi" or "3Sigma".
            consts: Molecular constants.
            j_qn: Quantum number J.
            max_n_power: Maximum power of N matrices to compute: can be 2, 4, 6, 8, 10, or 12.
            max_acomm_power: Maximum power of N used when evaluating anticommutators: can be 0, 2,
                4, 6, or 8.
        """
        self.term_symbol = term_symbol
        self.consts = consts
        self.j_qn = j_qn
        self.max_n_index = max_n_power // 2
        self.max_acomm_index = max_acomm_power // 2

    @cached_property
    def hamiltonian(self) -> NDArray[np.float64]:
        s_qn, lambda_qn = utils.parse_term_symbol(self.term_symbol)
        basis_fns = utils.generate_basis_fns(s_qn, lambda_qn)

        dim = len(basis_fns)

        lambda_basis, sigma_basis, omega_basis = utils.basis_vectors(basis_fns, dim)

        n_op_mats = utils.construct_n_operator_matrices(
            s_qn, self.j_qn, sigma_basis, omega_basis, self.max_n_index, dim
        )

        h_mat = np.zeros((dim, dim))

        if options.INCLUDE_R:
            terms.rotational(h_mat, n_op_mats, self.consts.rotational)
        if options.INCLUDE_SO:
            terms.spin_orbit(
                h_mat,
                lambda_basis,
                sigma_basis,
                s_qn,
                n_op_mats,
                self.consts.spin_orbit,
                self.max_acomm_index,
            )
        if options.INCLUDE_SS:
            terms.spin_spin(
                h_mat, sigma_basis, s_qn, n_op_mats, self.consts.spin_spin, self.max_acomm_index
            )
        if options.INCLUDE_SR:
            terms.spin_rotation(
                h_mat,
                sigma_basis,
                omega_basis,
                s_qn,
                self.j_qn,
                n_op_mats,
                self.consts.spin_rotation,
                self.max_acomm_index,
                dim,
            )
        if options.INCLUDE_LD:
            terms.lambda_doubling(
                h_mat,
                lambda_basis,
                sigma_basis,
                omega_basis,
                s_qn,
                self.j_qn,
                n_op_mats,
                self.consts.lambda_doubling,
                self.max_acomm_index,
                dim,
            )

        return h_mat

    @cached_property
    def eigenvalues_eigenvectors(self) -> EighResult:
        """Cache the full eigendecomposition.

        Returns:
            The eigenvalues and eigenvectors of the Hamiltonian.
        """
        # The Hamiltonian matrix is always Hermitian, so eigh can be used.
        return np.linalg.eigh(self.hamiltonian)

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the Hamiltonian matrix.

        Returns:
            Eigenvalues of the Hamiltonian matrix.
        """
        return self.eigenvalues_eigenvectors[0]

    @property
    def eigenvectors(self) -> NDArray[np.float64]:
        """Eigenvectors of the Hamiltonian matrix.

        Returns:
            Eigenvectors of the Hamiltonian matrix.
        """
        return self.eigenvalues_eigenvectors[1]


def three_sigma(num: int) -> None:
    """Test function for a 3Σ state.

    Args:
        num: The number of iterations to perform.
    """
    # Terms included via the inherent properties of a 3Σ state:
    #   - H_r (all) + H_ss (only S > 1/2 term) + H_sr (only S > 0 term)
    # These terms are further narrowed depending on the included constants below.
    j_qn = 1.0
    term_symbol = "3S"

    # Constants for the v' = 0 B3Σu- state of O2.
    consts = constants.Constants(
        rotational=constants.RotationalConsts(B=0.8132, D=4.50e-06),
        spin_spin=constants.SpinSpinConsts(lamda=1.69),
        spin_rotation=constants.SpinRotationConsts(gamma=-0.028),
    )

    def bench():
        comp = NumericComputation(term_symbol, consts, j_qn, max_n_power=12, max_acomm_power=8)
        comp.hamiltonian

    print(f"{num}\t3Σ Hamiltonians: {timeit.timeit(bench, number=num)} s")


def two_pi(num: int) -> None:
    """Test function for a 2Π state.

    Args:
        num: The number of iterations to perform.
    """
    # Terms included via the inherent properties of a 2Π state:
    #   - H_r (all) + H_so (only S > 0 term) + H_sr (only S > 0 term) + H_ld (all)
    # These terms are further narrowed depending on the included constants below.
    j_qn = 1.0
    term_symbol = "2P"

    # Constants for the X2Π ground state of OH.
    consts = constants.Constants(
        rotational=constants.RotationalConsts(B=18.55),
        spin_orbit=constants.SpinOrbitConsts(A=-139.21),
        lambda_doubling=constants.LambdaDoublingConsts(p=0.235, q=-0.0391),
    )

    def bench():
        comp = NumericComputation(term_symbol, consts, j_qn, max_n_power=12, max_acomm_power=8)
        comp.hamiltonian

    print(f"{num}\t2Π Hamiltonians: {timeit.timeit(bench, number=num)} s")


def five_pi(num: int) -> None:
    """Test function for a 5Π state.

    Args:
        num: The number of iterations to perform.
    """
    # Terms included via the inherent properties of a 5Π state:
    #   - H_r (all) + H_so (all) + H_ss (all) + H_sr (all) + H_ld (all)
    # These terms are further narrowed depending on the included constants below.
    j_qn = 5.0
    term_symbol = "5P"

    # Random 5Π state filled with all possible constants.
    consts = constants.Constants(
        rotational=constants.RotationalConsts(
            B=18.55, D=4.50e-06, H=4.50e-06, L=4.50e-06, M=4.50e-06, P=4.50e-06
        ),
        spin_orbit=constants.SpinOrbitConsts(A=-139.21, A_D=1, A_H=1, A_L=1, A_M=1, eta=1),
        spin_spin=constants.SpinSpinConsts(lamda=1.69, lamda_D=1, lamda_H=1, theta=1),
        spin_rotation=constants.SpinRotationConsts(
            gamma=-0.028, gamma_D=-0.028, gamma_H=-0.028, gamma_L=-0.028, gamma_S=-0.028
        ),
        lambda_doubling=constants.LambdaDoublingConsts(
            o=0.1,
            p=0.235,
            q=-0.0391,
            o_D=0.1,
            p_D=0.1,
            q_D=0.1,
            o_H=0.1,
            p_H=0.1,
            q_H=0.1,
            o_L=0.1,
            p_L=0.1,
            q_L=0.1,
        ),
    )

    def bench():
        comp = NumericComputation(term_symbol, consts, j_qn, max_n_power=12, max_acomm_power=8)
        comp.hamiltonian

    print(f"{num}\t5Π Hamiltonians: {timeit.timeit(bench, number=num)} s")


def main() -> None:
    """Entry point."""
    three_sigma(1000)
    two_pi(500)
    five_pi(200)


if __name__ == "__main__":
    main()
