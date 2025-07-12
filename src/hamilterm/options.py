# module options.py
"""Input options for the user to select."""

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

# Manually select which terms contribute to the molecular Hamiltonian.
INCLUDE_R: bool = True
INCLUDE_SO: bool = True
INCLUDE_SS: bool = True
INCLUDE_SR: bool = True
INCLUDE_LD: bool = True

# MAX_N_POWER can be 2, 4, 6, 8, 10, or 12. Powers above 12 have no associated constants and
# therefore will not contribute to the calculation.
MAX_N_POWER: int = 4
# Specify the maximum power of N used when evaluating anticommutators. A value of 0 will skip the
# evaluation of all anticommutators.
MAX_N_ACOMM_POWER: int = 2

# Printing options for symbolic computations.
PRINT_TERM: bool = False
PRINT_TEX: bool = True

MAX_POWER_INDEX: int = MAX_N_POWER // 2
MAX_ACOMM_INDEX: int = MAX_N_ACOMM_POWER // 2
LAMBDA_INT_MAP: dict[str, int] = {"Sigma": 0, "Pi": 1}
LAMBDA_STR_MAP: dict[str, str] = {"Sigma": "Σ", "Pi": "Π"}
