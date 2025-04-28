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
MASS_OF_DRY_ATMOSPHERE = 5.18e18 * units.kg
KAPPA = 2 / 7 * units.dimensionless
LATENT_HEAT_OF_VAPORIZATION = 2.501e6 * units.J / units.kilogram
IDEAL_GAS_CONSTANT = ISOBARIC_HEAT_CAPACITY * KAPPA
IDEAL_GAS_CONSTANT_H20 = 461.0 * units.J / units.kilogram / units.degK
WATER_DENSITY = 997 * units.kg / units.m**3
def parse_units(units_str: str) -> Quantity:
    if units_str in {'(0 - 1)', '%', '~'}:
        units_str = 'dimensionless'
    return units.parse_expression(units_str)
def _get_dimension(quantity: Quantity) -> str:
    exponents = list(quantity.dimensionality.values())
    if len(quantity.dimensionality) != 1 or exponents[0] != 1:
        raise ValueError('All scales must describe a single dimension;'
                         f'got dimensionality {quantity.dimensionality}')
    return str(quantity.dimensionality)
class ScaleProtocol(Protocol):
    def nondimensionalize(self, quantity: Quantity) -> Numeric:
        ...
    def dimensionalize(self, value: Numeric, unit: Unit) -> Quantity:
        ...
class Scale(abc.Mapping):
    def __init__(self, *scales: Quantity):
        self._scales = dict()
        for quantity in scales:
            dimension = _get_dimension(quantity)
            if dimension in self._scales:
                raise ValueError(
                    f'Got duplicate scales for dimension {dimension}.')
            self._scales[_get_dimension(quantity)] = quantity.to_base_units()
    def __getitem__(self, key: str) -> Quantity:
        return self._scales[key]
    def __iter__(self) -> Iterator[str]:
        return iter(self._scales)
    def __len__(self) -> int:
        return len(self._scales)
    def __repr__(self) -> str:
        return '\n'.join(f'{dimension}: {quantity}'
                         for dimension, quantity in self._scales.items())
    def _scaling_factor(self,
                        dimensionality: pint.util.UnitsContainer) -> Quantity:
        factor = Quantity(1)
        for dimension, exponent in dimensionality.items():
            quantity = self._scales.get(dimension)
            if quantity is None:
                raise ValueError(f'No scale has been set for {dimension}.')
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
        return dimensionalized.to(unit)  # pytype: disable=attribute-error  # jax-ndarray
NEURALGCM_V1_SCALE = Scale(
    RADIUS,  # length
    1 / 2 / OMEGA,  # time
    1 * units.kilogram,  # mass
    1 * units.degK,  # temperature
)
ATMOSPHERIC_SCALE = Scale(
    RADIUS,  # length
    1 / 2 / OMEGA,  # time
    MASS_OF_DRY_ATMOSPHERE,  # mass
    1 * units.degK,  # temperature
)
SI_SCALE = Scale(
    1 * units.m,  # length
    1 * units.s,  # time
    1 * units.kilogram,  # mass
    1 * units.degK,  # temperature
)
DEFAULT_SCALE = NEURALGCM_V1_SCALE
