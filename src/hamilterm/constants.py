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

from sympy import Symbol


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
class RotationalConstsSym:
    """Constants for the rotational operator."""

    B: Symbol = Symbol("B")
    D: Symbol = Symbol("D")
    H: Symbol = Symbol("H")
    L: Symbol = Symbol("L")
    M: Symbol = Symbol("M")
    P: Symbol = Symbol("P")


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
class SpinOrbitConstsSym:
    """Constants for the spin-orbit operator."""

    A: Symbol = Symbol("A")
    A_D: Symbol = Symbol("A_D")
    A_H: Symbol = Symbol("A_H")
    A_L: Symbol = Symbol("A_L")
    A_M: Symbol = Symbol("A_M")
    eta: Symbol = Symbol("eta")


@dataclass
class SpinSpinConstsNum:
    """Constants for the spin-spin operator."""

    lamda: float = 0.0
    lamda_D: float = 0.0
    lamda_H: float = 0.0
    theta: float = 0.0


@dataclass
class SpinSpinConstsSym:
    """Constants for the spin-spin operator."""

    lamda: Symbol = Symbol("lambda")
    lamda_D: Symbol = Symbol("lambda_D")
    lamda_H: Symbol = Symbol("lambda_H")
    theta: Symbol = Symbol("theta")


@dataclass
class SpinRotationConstsNum:
    """Constants for the spin-rotation operator."""

    gamma: float = 0.0
    gamma_D: float = 0.0
    gamma_H: float = 0.0
    gamma_L: float = 0.0
    gamma_S: float = 0.0


@dataclass
class SpinRotationConstsSym:
    """Constants for the spin-rotation operator."""

    gamma: Symbol = Symbol("gamma")
    gamma_D: Symbol = Symbol("gamma_D")
    gamma_H: Symbol = Symbol("gamma_H")
    gamma_L: Symbol = Symbol("gamma_L")
    gamma_S: Symbol = Symbol("gamma_S")


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
class LambdaDoublingConstsSym:
    """Constants for the Λ-doubling operator."""

    o: Symbol = Symbol("o")
    p: Symbol = Symbol("p")
    q: Symbol = Symbol("q")
    o_D: Symbol = Symbol("o_D")
    p_D: Symbol = Symbol("p_D")
    q_D: Symbol = Symbol("q_D")
    o_H: Symbol = Symbol("o_H")
    p_H: Symbol = Symbol("p_H")
    q_H: Symbol = Symbol("q_H")
    o_L: Symbol = Symbol("o_L")
    p_L: Symbol = Symbol("p_L")
    q_L: Symbol = Symbol("q_L")


@dataclass
class ConstantsNum:
    """Container for numeric molecular constants."""

    rotational: RotationalConstsNum = field(default_factory=RotationalConstsNum)
    spin_orbit: SpinOrbitConstsNum = field(default_factory=SpinOrbitConstsNum)
    spin_spin: SpinSpinConstsNum = field(default_factory=SpinSpinConstsNum)
    spin_rotation: SpinRotationConstsNum = field(default_factory=SpinRotationConstsNum)
    lambda_doubling: LambdaDoublingConstsNum = field(default_factory=LambdaDoublingConstsNum)


@dataclass
class ConstantsSym:
    """Container for numeric molecular constants."""

    rotational: RotationalConstsSym = field(default_factory=RotationalConstsSym)
    spin_orbit: SpinOrbitConstsSym = field(default_factory=SpinOrbitConstsSym)
    spin_spin: SpinSpinConstsSym = field(default_factory=SpinSpinConstsSym)
    spin_rotation: SpinRotationConstsSym = field(default_factory=SpinRotationConstsSym)
    lambda_doubling: LambdaDoublingConstsSym = field(default_factory=LambdaDoublingConstsSym)
