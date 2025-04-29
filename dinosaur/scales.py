from collections import abc
from typing import Iterator, Protocol, Union
import jax.numpy as jnp
import numpy as np
import pint

units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
Quantity = units.Quantity
Unit = units.Unit
UnitsContainer = pint.util.UnitsContainer
Array = Union[np.ndarray, jnp.ndarray]
Numeric = Union[Array, float, int]
RADIUS = 6.37122e6 * units.m
ANGULAR_VELOCITY = OMEGA = 7.292e-5 / units.s
GRAVITY_ACCELERATION = 9.80616 * units.m / units.s**2
ISOBARIC_HEAT_CAPACITY = 1004 * units.J / units.kilogram / units.degK
WATER_VAPOR_CP = 1859 * units.J / units.kilogram / units.degK
KAPPA = 2 / 7 * units.dimensionless
IDEAL_GAS_CONSTANT = ISOBARIC_HEAT_CAPACITY * KAPPA
IDEAL_GAS_CONSTANT_H20 = 461.0 * units.J / units.kilogram / units.degK


def _get_dimension(quantity: Quantity) -> str:
    exponents = list(quantity.dimensionality.values())
    return str(quantity.dimensionality)


class ScaleProtocol(Protocol):

    def nondimensionalize(self, quantity: Quantity) -> Numeric:
        ...

    def dimensionalize(self, value: Numeric, unit: Unit) -> Quantity:
        ...


class Scale:

    def __init__(self, *scales: Quantity):
        self._scales = dict()
        for quantity in scales:
            dimension = _get_dimension(quantity)
            self._scales[_get_dimension(quantity)] = quantity.to_base_units()

    def _scaling_factor(self,
                        dimensionality: pint.util.UnitsContainer) -> Quantity:
        factor = Quantity(1)
        for dimension, exponent in dimensionality.items():
            quantity = self._scales.get(dimension)
            factor *= quantity**exponent
        assert factor.check(dimensionality)
        return factor

    def nondimensionalize(self, quantity: Quantity) -> Numeric:
        scaling_factor = self._scaling_factor(quantity.dimensionality)
        nondimensionalized = (quantity / scaling_factor).to(
            units.dimensionless)
        return nondimensionalized.magnitude

    def dimensionalize(self, value: Numeric, unit: Unit) -> Quantity:
        scaling_factor = self._scaling_factor(unit.dimensionality)
        dimensionalized = value * scaling_factor
        return dimensionalized.to(unit)


DEFAULT_SCALE = Scale(
    RADIUS,
    1 / 2 / OMEGA,
    1 * units.kilogram,
    1 * units.degK,
)
