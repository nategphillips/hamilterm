# module constants.py
"""Stores numeric and/or symbolic molecular constants."""

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

from sympy import Symbol


@dataclass
class RotationalConsts:
    """Constants for the rotational operator."""

    B: Symbol = Symbol("B")
    D: Symbol = Symbol("D")
    H: Symbol = Symbol("H")
    L: Symbol = Symbol("L")
    M: Symbol = Symbol("M")
    P: Symbol = Symbol("P")


@dataclass
class SpinOrbitConsts:
    """Constants for the spin-orbit operator."""

    A: Symbol = Symbol("A")
    A_D: Symbol = Symbol("A_D")
    A_H: Symbol = Symbol("A_H")
    A_L: Symbol = Symbol("A_L")
    A_M: Symbol = Symbol("A_M")
    eta: Symbol = Symbol("eta")


@dataclass
class SpinSpinConsts:
    """Constants for the spin-spin operator."""

    lamda: Symbol = Symbol("lambda")
    lamda_D: Symbol = Symbol("lambda_D")
    lamda_H: Symbol = Symbol("lambda_H")
    theta: Symbol = Symbol("theta")


@dataclass
class SpinRotationConsts:
    """Constants for the spin-rotation operator."""

    gamma: Symbol = Symbol("gamma")
    gamma_D: Symbol = Symbol("gamma_D")
    gamma_H: Symbol = Symbol("gamma_H")
    gamma_L: Symbol = Symbol("gamma_L")
    gamma_S: Symbol = Symbol("gamma_S")


@dataclass
class LambdaDoublingConsts:
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
class Constants:
    """Container for numeric molecular constants."""

    rotational: RotationalConsts = field(default_factory=RotationalConsts)
    spin_orbit: SpinOrbitConsts = field(default_factory=SpinOrbitConsts)
    spin_spin: SpinSpinConsts = field(default_factory=SpinSpinConsts)
    spin_rotation: SpinRotationConsts = field(default_factory=SpinRotationConsts)
    lambda_doubling: LambdaDoublingConsts = field(default_factory=LambdaDoublingConsts)
