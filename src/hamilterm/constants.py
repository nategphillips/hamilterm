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

import sympy as sp


@dataclass
class RotationalConsts[T: (float, sp.Symbol)]:
    """Constants for the rotational operator."""

    B: T
    D: T
    H: T
    L: T
    M: T
    P: T

    @classmethod
    def numeric(cls: type["RotationalConsts[float]"], **kwargs: float) -> "RotationalConsts[float]":
        """Create numeric rotational constants."""
        defaults: dict[str, float] = {"B": 0.0, "D": 0.0, "H": 0.0, "L": 0.0, "M": 0.0, "P": 0.0}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def symbolic(cls: type["RotationalConsts[sp.Symbol]"]) -> "RotationalConsts[sp.Symbol]":
        """Create symbolic rotational constants."""
        return cls(*sp.symbols("B D H L M P"))


@dataclass
class SpinOrbitConsts[T: (float, sp.Symbol)]:
    """Constants for the spin-orbit operator."""

    A: T
    A_D: T
    A_H: T
    A_L: T
    A_M: T
    eta: T

    @classmethod
    def numeric(cls: type["SpinOrbitConsts[float]"], **kwargs: float) -> "SpinOrbitConsts[float]":
        """Create numeric spin-orbit constants."""
        defaults: dict[str, float] = {
            "A": 0.0,
            "A_D": 0.0,
            "A_H": 0.0,
            "A_L": 0.0,
            "A_M": 0.0,
            "eta": 0.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def symbolic(cls: type["SpinOrbitConsts[sp.Symbol]"]) -> "SpinOrbitConsts[sp.Symbol]":
        """Create symbolic spin-orbit constants."""
        return cls(*sp.symbols("A A_D A_H A_L A_M eta"))


@dataclass
class SpinSpinConsts[T: (float, sp.Symbol)]:
    """Constants for the spin-spin operator."""

    lamda: T
    lambda_D: T
    lambda_H: T
    theta: T

    @classmethod
    def numeric(cls: type["SpinSpinConsts[float]"], **kwargs: float) -> "SpinSpinConsts[float]":
        """Create numeric spin-spin constants."""
        defaults: dict[str, float] = {"lamda": 0.0, "lambda_D": 0.0, "lambda_H": 0.0, "theta": 0.0}
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def symbolic(cls: type["SpinSpinConsts[sp.Symbol]"]) -> "SpinSpinConsts[sp.Symbol]":
        """Create symbolic spin-spin constants."""
        return cls(*sp.symbols("lambda lambda_D lambda_H theta"))


@dataclass
class SpinRotationConsts[T: (float, sp.Symbol)]:
    """Constants for the spin-rotation operator."""

    gamma: T
    gamma_D: T
    gamma_H: T
    gamma_L: T
    gamma_S: T

    @classmethod
    def numeric(
        cls: type["SpinRotationConsts[float]"], **kwargs: float
    ) -> "SpinRotationConsts[float]":
        """Create numeric spin-rotation constants."""
        defaults: dict[str, float] = {
            "gamma": 0.0,
            "gamma_D": 0.0,
            "gamma_H": 0.0,
            "gamma_L": 0.0,
            "gamma_S": 0.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def symbolic(cls: type["SpinRotationConsts[sp.Symbol]"]) -> "SpinRotationConsts[sp.Symbol]":
        """Create symbolic spin-rotation constants."""
        return cls(*sp.symbols("gamma gamma_D gamma_H gamma_L gamma_S"))


@dataclass
class LambdaDoublingConsts[T: (float, sp.Symbol)]:
    """Constants for the Λ-doubling operator."""

    o: T
    p: T
    q: T
    o_D: T
    p_D: T
    q_D: T
    o_H: T
    p_H: T
    q_H: T
    o_L: T
    p_L: T
    q_L: T

    @classmethod
    def numeric(
        cls: type["LambdaDoublingConsts[float]"], **kwargs: float
    ) -> "LambdaDoublingConsts[float]":
        """Create numeric Λ-doubling constants."""
        defaults: dict[str, float] = {
            "o": 0.0,
            "p": 0.0,
            "q": 0.0,
            "o_D": 0.0,
            "p_D": 0.0,
            "q_D": 0.0,
            "o_H": 0.0,
            "p_H": 0.0,
            "q_H": 0.0,
            "o_L": 0.0,
            "p_L": 0.0,
            "q_L": 0.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def symbolic(cls: type["LambdaDoublingConsts[sp.Symbol]"]) -> "LambdaDoublingConsts[sp.Symbol]":
        """Create symbolic Λ-doubling constants."""
        return cls(*sp.symbols("o p q o_D p_D q_D o_H p_H q_H o_L p_L q_L"))


# TODO: 25/07/11 - Getting correct typing with generics is tricky, and I'm not sure if I like this
#       implementation all that much. It works well enough, but seems a bit cumbersome.


@dataclass
class NumericConstants:
    """Container for numeric molecular constants."""

    rotational: RotationalConsts[float] = field(default_factory=lambda: RotationalConsts.numeric())
    spin_orbit: SpinOrbitConsts[float] = field(default_factory=lambda: SpinOrbitConsts.numeric())
    spin_spin: SpinSpinConsts[float] = field(default_factory=lambda: SpinSpinConsts.numeric())
    spin_rotation: SpinRotationConsts[float] = field(
        default_factory=lambda: SpinRotationConsts.numeric()
    )
    lambda_doubling: LambdaDoublingConsts[float] = field(
        default_factory=lambda: LambdaDoublingConsts.numeric()
    )


@dataclass
class SymbolicConstants:
    """Container for symbolic molecular constants."""

    rotational: RotationalConsts[sp.Symbol] = field(
        default_factory=lambda: RotationalConsts.symbolic()
    )
    spin_orbit: SpinOrbitConsts[sp.Symbol] = field(
        default_factory=lambda: SpinOrbitConsts.symbolic()
    )
    spin_spin: SpinSpinConsts[sp.Symbol] = field(default_factory=lambda: SpinSpinConsts.symbolic())
    spin_rotation: SpinRotationConsts[sp.Symbol] = field(
        default_factory=lambda: SpinRotationConsts.symbolic()
    )
    lambda_doubling: LambdaDoublingConsts[sp.Symbol] = field(
        default_factory=lambda: LambdaDoublingConsts.symbolic()
    )
