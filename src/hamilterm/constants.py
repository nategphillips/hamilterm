# module constants.py
"""Stores numeric molecular constants."""

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

from dataclasses import dataclass, field


@dataclass
class RotationalConsts:
    """Constants for the rotational operator."""

    B = 0.0
    D = 0.0
    H = 0.0
    L = 0.0
    M = 0.0
    P = 0.0


@dataclass
class SpinOrbitConsts:
    """Constants for the spin-orbit operator."""

    A = 0.0
    A_D = 0.0
    A_H = 0.0
    A_L = 0.0
    A_M = 0.0
    eta = 0.0


@dataclass
class SpinSpinConsts:
    """Constants for the spin-spin operator."""

    lamda = 0.0
    lamda_D = 0.0
    lamda_H = 0.0
    theta = 0.0


@dataclass
class SpinRotationConsts:
    """Constants for the spin-rotation operator."""

    gamma = 0.0
    gamma_D = 0.0
    gamma_H = 0.0
    gamma_L = 0.0
    gamma_S = 0.0


@dataclass
class LambdaDoublingConsts:
    """Constants for the Λ-doubling operator."""

    o = 0.0
    p = 0.0
    q = 0.0
    o_D = 0.0
    p_D = 0.0
    q_D = 0.0
    o_H = 0.0
    p_H = 0.0
    q_H = 0.0
    o_L = 0.0
    p_L = 0.0
    q_L = 0.0


@dataclass
class Constants:
    """Container for numeric molecular constants."""

    rotational: RotationalConsts = field(default_factory=RotationalConsts)
    spin_orbit: SpinOrbitConsts = field(default_factory=SpinOrbitConsts)
    spin_spin: SpinSpinConsts = field(default_factory=SpinSpinConsts)
    spin_rotation: SpinRotationConsts = field(default_factory=SpinRotationConsts)
    lambda_doubling: LambdaDoublingConsts = field(default_factory=LambdaDoublingConsts)
