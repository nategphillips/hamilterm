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

LAMBDA_INT_MAP: dict[str, int] = {"S": 0, "P": 1}
LAMBDA_STR_MAP: dict[str, str] = {"S": "Σ", "P": "Π"}
LAMBDA_LONG_MAP: dict[str, str] = {"S": "Sigma", "P": "Pi"}
