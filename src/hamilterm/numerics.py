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

from typing import TYPE_CHECKING

import numpy as np

from hamilterm import constants, utils

if TYPE_CHECKING:
    from fractions import Fraction

    from numpy.typing import NDArray


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

    s_qn, lambda_qn = utils.parse_term_symbol(term_symbol)
    basis_fns: list[tuple[int, Fraction, Fraction]] = utils.generate_basis_fns(s_qn, lambda_qn)
    h_mat: NDArray[np.float64] = utils.build_hamiltonian(basis_fns, s_qn, j_qn, consts)

    # The Hamiltonian matrix is always Hermitian, so eigvalsh can be used.
    eigenvals: NDArray[np.float64] = np.linalg.eigvalsh(h_mat).astype(np.float64)

    # Example Hamiltonian and eigenvalues.
    print(h_mat)
    print(eigenvals)


if __name__ == "__main__":
    main()
