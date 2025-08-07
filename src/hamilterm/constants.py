# module constants.py
"""Stores numeric and/or symbolic molecular constants."""

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

from dataclasses import dataclass, field


@dataclass
class RotationalConstsNum:
    """Constants for the rotational operator."""

    B: float = 0.0
    D: float = 0.0
    H: float = 0.0
    L: float = 0.0
    M: float = 0.0
    P: float = 0.0


@dataclass
class SpinOrbitConstsNum:
    """Constants for the spin-orbit operator."""

    A: float = 0.0
    A_D: float = 0.0
    A_H: float = 0.0
    A_L: float = 0.0
    A_M: float = 0.0
    eta: float = 0.0


@dataclass
class SpinSpinConstsNum:
    """Constants for the spin-spin operator."""

    lamda: float = 0.0
    lamda_D: float = 0.0
    lamda_H: float = 0.0
    theta: float = 0.0


@dataclass
class SpinRotationConstsNum:
    """Constants for the spin-rotation operator."""

    gamma: float = 0.0
    gamma_D: float = 0.0
    gamma_H: float = 0.0
    gamma_L: float = 0.0
    gamma_S: float = 0.0


@dataclass
class LambdaDoublingConstsNum:
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
class ConstantsNum:
    """Container for numeric molecular constants."""

    rotational: RotationalConstsNum = field(default_factory=RotationalConstsNum)
    spin_orbit: SpinOrbitConstsNum = field(default_factory=SpinOrbitConstsNum)
    spin_spin: SpinSpinConstsNum = field(default_factory=SpinSpinConstsNum)
    spin_rotation: SpinRotationConstsNum = field(default_factory=SpinRotationConstsNum)
    lambda_doubling: LambdaDoublingConstsNum = field(default_factory=LambdaDoublingConstsNum)
