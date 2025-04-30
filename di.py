from __future__ import annotations
from jax import lax
from typing import Any
import dataclasses
import functools
import jax
import jax.numpy as jnp
import numpy as np
import pint
import scipy
import scipy.special as sps
import tree_math
import xarray

tree_map = jax.tree_util.tree_map
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)
units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
Unit = units.Unit
RADIUS = 6.37122e6 * units.m
ANGULAR_VELOCITY = 7.292e-5 / units.s
GRAVITY_ACCELERATION = 9.80616 * units.m / units.s**2
ISOBARIC_HEAT_CAPACITY = 1004 * units.J / units.kilogram / units.degK
KAPPA = 2 / 7 * units.dimensionless
IDEAL_GAS_CONSTANT = ISOBARIC_HEAT_CAPACITY * KAPPA
_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


class Scale:

    def __init__(self, *scales):
        self.scales = {}
        for quantity in scales:
            self.scales[str(
                quantity.dimensionality)] = quantity.to_base_units()

    def scaling_factor(self, dimensionality):
        factor = units.Quantity(1)
        for dimension, exponent in dimensionality.items():
            quantity = self.scales.get(dimension)
            factor *= quantity**exponent
        assert factor.check(dimensionality)
        return factor

    def nondimensionalize(self, quantity):
        scaling_factor = self.scaling_factor(quantity.dimensionality)
        nondimensionalized = (quantity / scaling_factor).to(
            units.dimensionless)
        return nondimensionalized.magnitude

    def dimensionalize(self, value, unit):
        scaling_factor = self.scaling_factor(unit.dimensionality)
        dimensionalized = value * scaling_factor
        return dimensionalized.to(unit)


DEFAULT_SCALE = Scale(
    RADIUS,
    1 / 2 / ANGULAR_VELOCITY,
    1 * units.kilogram,
    1 * units.degK,
)

radius = DEFAULT_SCALE.nondimensionalize(RADIUS)
angular_velocity = DEFAULT_SCALE.nondimensionalize(ANGULAR_VELOCITY)
gravity_acceleration = DEFAULT_SCALE.nondimensionalize(GRAVITY_ACCELERATION)
ideal_gas_constant = DEFAULT_SCALE.nondimensionalize(IDEAL_GAS_CONSTANT)
kappa = DEFAULT_SCALE.nondimensionalize(KAPPA)


def cumsum(x, axis):
    if axis < 0:
        axis = axis + x.ndim
    size = x.shape[axis]
    i = jnp.arange(size)[:, jnp.newaxis]
    j = jnp.arange(size)[jnp.newaxis, :]
    w = jnp.less_equal(i, j).astype(np.float32)
    out_axes = list(range(x.ndim))
    out_axes[axis] = x.ndim
    return jnp.einsum(
        w,
        [axis, x.ndim],
        x,
        list(range(x.ndim)),
        out_axes,
        precision=("bfloat16", "highest"),
    )


def pad_in_dim(x, pad_width, axis):
    padding_value = jnp.array(0, dtype=x.dtype)
    padding_config = [(0, 0, 0)] * x.ndim
    padding_config[axis] = pad_width + (0, )
    return lax.pad(x, padding_value, padding_config)


def shift(x, offset, axis):
    if offset > 0:
        sliced = lax.slice_in_dim(x, 0, x.shape[axis] - offset, axis=axis)
        return pad_in_dim(sliced, (offset, 0), axis=axis)
    else:
        sliced = lax.slice_in_dim(x, -offset, x.shape[axis], axis=axis)
        return pad_in_dim(sliced, (0, -offset), axis=axis)


def diff(x, axis=-1):
    upper = lax.slice_in_dim(x, 1, None, axis=axis)
    lower = lax.slice_in_dim(x, 0, -1, axis=axis)
    return upper - lower


def tree_map_over_nonscalars(f, x, *, scalar_fn=lambda x: x):

    def g(x):
        x = jnp.asarray(x)
        return f(x) if x.ndim else scalar_fn(x)

    return tree_map(g, x)


def as_dict(inputs):
    return_type = type(inputs)
    inputs = inputs.asdict()
    from_dict_fn = lambda dict_inputs: return_type(**dict_inputs)
    return inputs, from_dict_fn


def _slice_shape_along_axis(x, axis):
    x_shape = list(x.shape)
    x_shape[axis] = 1
    return tuple(x_shape)


def _with_f64_math(f):
    return lambda x: f(x.astype(np.float64)).astype(x.dtype)


@dataclasses.dataclass(frozen=True)
class SigmaCoordinates:
    boundaries: np.ndarray

    def __init__(self, boundaries):
        boundaries = np.asarray(boundaries)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def centers(self):
        return _with_f64_math(lambda x: (x[1:] + x[:-1]) / 2)(self.boundaries)

    @property
    def layer_thickness(self):
        return _with_f64_math(np.diff)(self.boundaries)

    @property
    def center_to_center(self):
        return _with_f64_math(np.diff)(self.centers)

    @property
    def layers(self):
        return len(self.boundaries) - 1

    @classmethod
    def equidistant(cls, layers: int, dtype=np.float32):
        boundaries = np.linspace(0, 1, layers + 1, dtype=dtype)
        return cls(boundaries)

    def asdict(self):
        return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

    def __hash__(self):
        return hash(tuple(self.centers.tolist()))


def centered_difference(x, coordinates, axis=-3):
    dx = diff(x, axis=axis)
    dx_axes = range(dx.ndim)
    inv_d𝜎 = 1 / coordinates.center_to_center
    inv_d𝜎_axes = [dx_axes[axis]]
    return einsum(dx,
                  dx_axes,
                  inv_d𝜎,
                  inv_d𝜎_axes,
                  dx_axes,
                  precision="float32")


def cumulative_sigma_integral(
    x,
    coordinates,
    axis=-3,
):
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return cumsum(xd𝜎, axis)


def sigma_integral(
    x,
    coordinates,
    axis=-3,
    keepdims=True,
):
    x_axes = range(x.ndim)
    d𝜎 = coordinates.layer_thickness
    d𝜎_axes = [x_axes[axis]]
    xd𝜎 = einsum(x, x_axes, d𝜎, d𝜎_axes, x_axes)
    return xd𝜎.sum(axis=axis, keepdims=keepdims)


def centered_vertical_advection(
    w,
    x,
    coordinates,
    axis=-3,
    w_boundary_values=None,
    dx_dsigma_boundary_values=None,
):
    if w_boundary_values is None:
        w_slc_shape = _slice_shape_along_axis(w, axis)
        w_boundary_values = (
            jnp.zeros(w_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
            jnp.zeros(w_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(w.dtype)),
        )
    if dx_dsigma_boundary_values is None:
        x_slc_shape = _slice_shape_along_axis(x, axis)
        dx_dsigma_boundary_values = (
            jnp.zeros(x_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
            jnp.zeros(x_slc_shape,
                      dtype=jax.dtypes.canonicalize_dtype(x.dtype)),
        )
    w_boundary_top, w_boundary_bot = w_boundary_values
    w = jnp.concatenate([w_boundary_top, w, w_boundary_bot], axis=axis)
    x_diff = centered_difference(x, coordinates, axis)
    x_diff_boundary_top, x_diff_boundary_bot = dx_dsigma_boundary_values
    x_diff = jnp.concatenate(
        [x_diff_boundary_top, x_diff, x_diff_boundary_bot], axis=axis)
    w_times_x_diff = w * x_diff
    return -0.5 * (lax.slice_in_dim(w_times_x_diff, 1, None, axis=axis) +
                   lax.slice_in_dim(w_times_x_diff, 0, -1, axis=axis))


def _evaluate_rhombus(n_l, n_m, x):
    y = np.sqrt(1 - x * x)
    p = np.zeros((n_l, n_m, len(x)))
    p[0, 0] = p[0, 0] + 1 / np.sqrt(2)
    for m in range(1, n_m):
        p[0, m] = -np.sqrt(1 + 1 / (2 * m)) * y * p[0, m - 1]
    m_max = n_m
    for k in range(1, n_l):
        m_max = min(n_m, n_l - k)
        m = np.arange(m_max).reshape((-1, 1))
        m2 = np.square(m)
        mk2 = np.square(m + k)
        mkp2 = np.square(m + k - 1)
        a = np.sqrt((4 * mk2 - 1) / (mk2 - m2))
        b = np.sqrt((mkp2 - m2) / (4 * mkp2 - 1))
        p[k, :m_max] = a * (x * p[k - 1, :m_max] - b * p[k - 2, :m_max])
    return p


def evaluate(n_m, n_l, x):
    r = np.transpose(_evaluate_rhombus(n_l=n_l, n_m=n_m, x=x), (1, 2, 0))
    p = np.zeros((n_m, len(x), n_l))
    for m in range(n_m):
        p[m, :, m:n_l] = r[m, :, 0:n_l - m]
    return p


def real_basis(wavenumbers, nodes):
    dft = scipy.linalg.dft(nodes)[:, :wavenumbers] / np.sqrt(np.pi)
    cos = np.real(dft[:, 1:])
    sin = -np.imag(dft[:, 1:])
    f = np.empty(shape=[nodes, 2 * wavenumbers - 1], dtype=np.float64)
    f[:, 0] = 1 / np.sqrt(2 * np.pi)
    f[:, 1::2] = cos
    f[:, 2::2] = sin
    return f


def real_basis_derivative(u, /, axis=-1):
    i = jnp.arange(u.shape[axis]).reshape((-1, ) + (1, ) * (-1 - axis))
    j = (i + 1) // 2
    u_down = shift(u, -1, axis)
    u_up = shift(u, +1, axis)
    return j * jnp.where(i % 2, u_down, -u_up)


def quadrature_nodes(nodes):
    xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    weights = 2 * np.pi / nodes
    return xs, weights


@dataclasses.dataclass
class _SphericalHarmonicBasis:
    f: np.ndarray
    p: np.ndarray
    w: np.ndarray


@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonics:
    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0

    @functools.cached_property
    def nodal_axes(self):
        longitude, _ = quadrature_nodes(self.longitude_nodes)
        sin_latitude, _ = sps.roots_legendre(self.latitude_nodes)
        return longitude, sin_latitude

    @functools.cached_property
    def nodal_shape(self):
        return (self.longitude_nodes, self.latitude_nodes)

    @functools.cached_property
    def modal_axes(self):
        m_pos = np.arange(1, self.longitude_wavenumbers)
        m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
        lon_wavenumbers = np.concatenate([[0], m_pos_neg])
        tot_wavenumbers = np.arange(self.total_wavenumbers)
        return lon_wavenumbers, tot_wavenumbers

    @functools.cached_property
    def modal_shape(self):
        return (2 * self.longitude_wavenumbers - 1, self.total_wavenumbers)

    @functools.cached_property
    def modal_padding(self):
        return (0, 0)

    @functools.cached_property
    def mask(self):
        m, l = np.meshgrid(*self.modal_axes, indexing="ij")
        return abs(m) <= l

    @functools.cached_property
    def basis(self):
        f = real_basis(
            wavenumbers=self.longitude_wavenumbers,
            nodes=self.longitude_nodes,
        )
        _, wf = quadrature_nodes(self.longitude_nodes)
        x, wp = sps.roots_legendre(self.latitude_nodes)
        w = wf * wp
        p = evaluate(n_m=self.longitude_wavenumbers,
                     n_l=self.total_wavenumbers,
                     x=x)
        p = np.repeat(p, 2, axis=0)
        p = p[1:]
        return _SphericalHarmonicBasis(f=f, p=p, w=w)

    def inverse_transform(self, x):
        p = self.basis.p
        f = self.basis.f
        px = einsum("mjl,...ml->...mj", p, x)
        fpx = einsum("im,...mj->...ij", f, px)
        return fpx

    def transform(self, x):
        w = self.basis.w
        f = self.basis.f
        p = self.basis.p
        wx = w * x
        fwx = einsum("im,...ij->...mj", f, wx)
        pfwx = einsum("mjl,...mj->...ml", p, fwx)
        return pfwx

    def longitudinal_derivative(self, x):
        return real_basis_derivative(x, axis=-2)


@dataclasses.dataclass(frozen=True)
class Grid:
    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    longitude_offset: float = 0.0
    radius: float | None = None

    def __post_init__(self):
        if self.radius is None:
            object.__setattr__(self, "radius", 1.0)

    @classmethod
    def construct(
        cls,
        max_wavenumber: int,
        gaussian_nodes: int,
        longitude_offset: float = 0.0,
        radius: float | None = None,
    ):
        return cls(
            longitude_wavenumbers=max_wavenumber + 1,
            total_wavenumbers=max_wavenumber + 2,
            longitude_nodes=4 * gaussian_nodes,
            latitude_nodes=2 * gaussian_nodes,
            longitude_offset=longitude_offset,
            radius=radius,
        )

    @classmethod
    def T42(cls, **kwargs):
        return cls.construct(max_wavenumber=42, gaussian_nodes=32, **kwargs)

    @classmethod
    def T170(cls, **kwargs):
        return cls.construct(max_wavenumber=170, gaussian_nodes=128, **kwargs)

    def asdict(self):
        items = dataclasses.asdict(self)
        return items

    @functools.cached_property
    def spherical_harmonics(self):
        kwargs = dict()
        return RealSphericalHarmonics(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes,
            **kwargs,
        )

    @functools.cached_property
    def nodal_axes(self):
        lon, sin_lat = self.spherical_harmonics.nodal_axes
        return lon + self.longitude_offset, sin_lat

    @functools.cached_property
    def nodal_shape(self):
        return self.spherical_harmonics.nodal_shape

    @functools.cached_property
    def nodal_mesh(self):
        return np.meshgrid(*self.nodal_axes, indexing="ij")

    @functools.cached_property
    def modal_axes(self):
        return self.spherical_harmonics.modal_axes

    @functools.cached_property
    def modal_shape(self):
        return self.spherical_harmonics.modal_shape

    @functools.cached_property
    def modal_padding(self):
        return self.spherical_harmonics.modal_padding

    @functools.cached_property
    def mask(self):
        return self.spherical_harmonics.mask

    @functools.cached_property
    def modal_mesh(self):
        return np.meshgrid(*self.spherical_harmonics.modal_axes, indexing="ij")

    @functools.cached_property
    def cos_lat(self):
        _, sin_lat = self.nodal_axes
        return np.sqrt(1 - sin_lat**2)

    @functools.cached_property
    def sec2_lat(self):
        _, sin_lat = self.nodal_axes
        return 1 / (1 - sin_lat**2)

    @functools.cached_property
    def laplacian_eigenvalues(self):
        _, l = self.modal_axes
        return -l * (l + 1) / (self.radius**2)

    def to_nodal(self, x):
        f = self.spherical_harmonics.inverse_transform
        return tree_map_over_nonscalars(f, x)

    def to_modal(self, z):
        f = self.spherical_harmonics.transform
        return tree_map_over_nonscalars(f, z)

    def laplacian(self, x):
        return x * self.laplacian_eigenvalues

    def inverse_laplacian(self, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_eigenvalues = 1 / self.laplacian_eigenvalues
        inverse_eigenvalues[0] = 0
        inverse_eigenvalues[self.total_wavenumbers:] = 0
        assert not np.isnan(inverse_eigenvalues).any()
        return x * inverse_eigenvalues

    def clip_wavenumbers(self, x, n: int = 1):

        def clip(x):
            num_zeros = n + self.modal_padding[-1]
            mask = jnp.ones(self.modal_shape[-1],
                            x.dtype).at[-num_zeros:].set(0)
            return x * mask

        return tree_map_over_nonscalars(clip, x)

    @functools.cached_property
    def _derivative_recurrence_weights(self):
        m, l = self.modal_mesh
        a = np.sqrt(self.mask * (l**2 - m**2) / (4 * l**2 - 1))
        a[:, 0] = 0
        b = np.sqrt(self.mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
        b[:, -1] = 0
        return a, b

    def d_dlon(self, x):
        return self.spherical_harmonics.longitudinal_derivative(x)

    def cos_lat_d_dlat(self, x):
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = shift(((l + 1) * a) * x, -1, axis=-1)
        x_lp1 = shift((-l * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def sec_lat_d_dlat_cos2(self, x):
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = shift(((l - 1) * a) * x, -1, axis=-1)
        x_lp1 = shift((-(l + 2) * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def cos_lat_grad(self, x, clip: bool = True):
        raw = self.d_dlon(x) / self.radius, self.cos_lat_d_dlat(
            x) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def k_cross(self, v):
        return -v[1], v[0]

    def div_cos_lat(
        self,
        v,
        clip: bool = True,
    ):
        raw = (self.d_dlon(v[0]) +
               self.sec_lat_d_dlat_cos2(v[1])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def curl_cos_lat(
        self,
        v,
        clip: bool = True,
    ):
        raw = (self.d_dlon(v[1]) -
               self.sec_lat_d_dlat_cos2(v[0])) / self.radius
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def integrate(self, z):
        w = self.spherical_harmonics.basis.w * self.radius**2
        return einsum("y,...xy->...", w, z)


def add_constant(x: jnp.ndarray, c):
    return x.at[..., 0, 0].add(_CONSTANT_NORMALIZATION_FACTOR * c)


def get_cos_lat_vector(vorticity, divergence, grid, clip=True):
    stream_function = grid.inverse_laplacian(vorticity)
    velocity_potential = grid.inverse_laplacian(divergence)
    return jax.tree_util.tree_map(
        lambda x, y: x + y,
        grid.cos_lat_grad(velocity_potential, clip=clip),
        grid.k_cross(grid.cos_lat_grad(stream_function, clip=clip)),
    )


@functools.partial(jax.jit, static_argnames=("grid", "clip"))
def uv_nodal_to_vor_div_modal(
    grid: Grid,
    u_nodal,
    v_nodal,
    clip: bool = True,
):
    u_over_cos_lat = grid.to_modal(u_nodal / grid.cos_lat)
    v_over_cos_lat = grid.to_modal(v_nodal / grid.cos_lat)
    vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    return vorticity, divergence


@functools.partial(jax.jit, static_argnames=("grid", "clip"))
def vor_div_to_uv_nodal(grid, vorticity, divergence):
    u_cos_lat, v_cos_lat = get_cos_lat_vector(vorticity,
                                              divergence,
                                              grid,
                                              clip=True)
    u_nodal = grid.to_nodal(u_cos_lat) / grid.cos_lat
    v_nodal = grid.to_nodal(v_cos_lat) / grid.cos_lat
    return u_nodal, v_nodal


@dataclasses.dataclass(frozen=True)
class CoordinateSystem:
    horizontal: Any
    vertical: Any

    def asdict(self):
        out = {**self.horizontal.asdict(), **self.vertical.asdict()}
        out["horizontal_grid_type"] = type(self.horizontal).__name__
        out["vertical_grid_type"] = type(self.vertical).__name__
        return out

    @property
    def surface_nodal_shape(self):
        return (1, ) + self.horizontal.nodal_shape


def get_nodal_shapes(
    inputs,
    coords,
):
    nodal_shape = coords.horizontal.nodal_shape
    array_shape_fn = lambda x: np.asarray(x.shape[:-2] + nodal_shape)
    scalar_shape_fn = lambda x: np.array([], dtype=int)
    return tree_map_over_nonscalars(array_shape_fn,
                                    inputs,
                                    scalar_fn=scalar_shape_fn)


def maybe_to_nodal(
    fields,
    coords,
):
    nodal_shapes = get_nodal_shapes(fields, coords)

    def to_nodal_fn(x):
        return coords.horizontal.to_nodal(x)

    fn = lambda x, nodal: x if x.shape == tuple(nodal) else to_nodal_fn(x)
    return jax.tree_util.tree_map(fn, fields, nodal_shapes)


class ImplicitExplicitODE:

    @classmethod
    def from_functions(
        cls,
        explicit_terms,
        implicit_terms,
        implicit_inverse,
    ):
        explicit_implicit_ode = cls()
        explicit_implicit_ode.explicit_terms = explicit_terms
        explicit_implicit_ode.implicit_terms = implicit_terms
        explicit_implicit_ode.implicit_inverse = implicit_inverse
        return explicit_implicit_ode


@dataclasses.dataclass
class TimeReversedImExODE(ImplicitExplicitODE):
    forward_eq: ImplicitExplicitODE

    def explicit_terms(self, state):
        forward_term = self.forward_eq.explicit_terms(state)
        return tree_map(jnp.negative, forward_term)

    def implicit_terms(self, state):
        forward_term = self.forward_eq.implicit_terms(state)
        return tree_map(jnp.negative, forward_term)

    def implicit_inverse(
        self,
        state,
        step_size: float,
    ):
        return self.forward_eq.implicit_inverse(state, -step_size)


def compose_equations(equations):
    implicit_explicit_eqs = list(
        filter(lambda x: isinstance(x, ImplicitExplicitODE), equations))
    (implicit_explicit_equation, ) = implicit_explicit_eqs
    assert isinstance(implicit_explicit_equation, ImplicitExplicitODE)

    def explicit_fn(x):
        explicit_tendencies = [fn.explicit_terms(x) for fn in equations]
        return tree_map(lambda *args: sum([x for x in args if x is not None]),
                        *explicit_tendencies)

    return ImplicitExplicitODE.from_functions(
        explicit_fn,
        implicit_explicit_equation.implicit_terms,
        implicit_explicit_equation.implicit_inverse,
    )


@dataclasses.dataclass
class ImExButcherTableau:
    a_ex: Any
    a_im: Any
    b_ex: Any
    b_im: Any


def imex_runge_kutta(
    tableau,
    equation,
    time_step,
):
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    G = tree_math.unwrap(equation.implicit_terms)
    G_inv = tree_math.unwrap(equation.implicit_inverse, vector_argnums=0)
    a_ex = tableau.a_ex
    a_im = tableau.a_im
    b_ex = tableau.b_ex
    b_im = tableau.b_im
    num_steps = len(b_ex)

    @tree_math.wrap
    def step_fn(y0):
        f = [None] * num_steps
        g = [None] * num_steps
        f[0] = F(y0)
        g[0] = G(y0)
        for i in range(1, num_steps):
            ex_terms = dt * sum(a_ex[i - 1][j] * f[j]
                                for j in range(i) if a_ex[i - 1][j])
            im_terms = dt * sum(a_im[i - 1][j] * g[j]
                                for j in range(i) if a_im[i - 1][j])
            Y_star = y0 + ex_terms + im_terms
            Y = G_inv(Y_star, dt * a_im[i - 1][i])
            if any(a_ex[j][i] for j in range(i, num_steps - 1)) or b_ex[i]:
                f[i] = F(Y)
            if any(a_im[j][i] for j in range(i, num_steps - 1)) or b_im[i]:
                g[i] = G(Y)
        ex_terms = dt * sum(b_ex[j] * f[j]
                            for j in range(num_steps) if b_ex[j])
        im_terms = dt * sum(b_im[j] * g[j]
                            for j in range(num_steps) if b_im[j])
        y_next = y0 + ex_terms + im_terms
        return y_next

    return step_fn


def imex_rk_sil3(
    equation,
    time_step,
):
    return imex_runge_kutta(
        tableau=ImExButcherTableau(
            a_ex=[[1 / 3], [1 / 6, 1 / 2], [1 / 2, -1 / 2, 1]],
            a_im=[[1 / 6, 1 / 6], [1 / 3, 0, 1 / 3], [3 / 8, 0, 3 / 8, 1 / 4]],
            b_ex=[1 / 2, -1 / 2, 1, 0],
            b_im=[3 / 8, 0, 3 / 8, 1 / 4],
        ),
        equation=equation,
        time_step=time_step,
    )


def runge_kutta_step_filter(state_filter):

    def _filter(u, u_next):
        del u
        return state_filter(u_next)

    return _filter


def exponential_step_filter(
    grid,
    dt,
    tau=0.010938,
    order=18,
    cutoff=0,
):
    filter_fn = exponential_filter(grid, dt / tau, order, cutoff)
    return runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_step_filter(
    grid,
    dt,
    tau,
    order=1,
):
    eigenvalues = grid.laplacian_eigenvalues
    scale = dt / (tau * abs(eigenvalues[-1])**order)
    filter_fn = horizontal_diffusion_filter(grid, scale, order)
    return runge_kutta_step_filter(filter_fn)


def step_with_filters(
    step_fn,
    filters,
):

    def _step_fn(u):
        u_next = step_fn(u)
        for filter_fn in filters:
            u_next = filter_fn(u, u_next)
        return u_next

    return _step_fn


def repeated(fn, steps, scan_fn=jax.lax.scan):

    def f_repeated(x_initial):
        g = lambda x, _: (fn(x), None)
        x_final, _ = scan_fn(g, x_initial, xs=None, length=steps)
        return x_final

    return f_repeated


def trajectory_from_step(
    step_fn,
    outer_steps,
    inner_steps,
    *,
    start_with_input=False,
    post_process_fn=lambda x: x,
    outer_scan_fn=jax.lax.scan,
    inner_scan_fn=jax.lax.scan,
):
    if inner_steps != 1:
        step_fn = repeated(step_fn, inner_steps, inner_scan_fn)

    def step(carry_in, _):
        carry_out = step_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, post_process_fn(frame)

    def multistep(x):
        return outer_scan_fn(step, x, xs=None, length=outer_steps)

    return multistep


def accumulate_repeated(
    step_fn,
    weights,
    state,
    scan_fn=jax.lax.scan,
):

    def f(carry, weight):
        state, averaged = carry
        state = step_fn(state)
        averaged = tree_map(lambda s, a: a + weight * s, state, averaged)
        return (state, averaged), None

    zeros = tree_map(jnp.zeros_like, state)
    init = (state, zeros)
    (_, averaged), _ = scan_fn(f, init, weights)
    return averaged


def _dfi_lanczos_weights(
    time_span,
    cutoff_period,
    dt,
):
    N = round(time_span / (2 * dt))
    n = np.arange(1, N + 1)
    w = np.sinc(n / (N + 1)) * np.sinc(n * time_span / (cutoff_period * N))
    return w


def digital_filter_initialization(
    equation,
    ode_solver,
    filters,
    time_span,
    cutoff_period,
    dt,
):

    def f(state):
        forward_step = step_with_filters(ode_solver(equation, dt), filters)
        backward_step = step_with_filters(
            ode_solver(TimeReversedImExODE(equation), dt), filters)
        weights = _dfi_lanczos_weights(time_span, cutoff_period, dt)
        init_weight = 1.0
        total_weight = init_weight + 2 * weights.sum()
        init_weight /= total_weight
        weights /= total_weight
        init_term = tree_map(lambda x: x * init_weight, state)
        forward_term = accumulate_repeated(forward_step, weights, state)
        backward_term = accumulate_repeated(backward_step, weights, state)
        return tree_map(lambda *xs: sum(xs), init_term, forward_term,
                        backward_term)

    return f


@tree_math.struct
class State:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    log_surface_pressure: Any
    tracers: Any = dataclasses.field(default_factory=dict)
    sim_time: Any = None


def _asdict(state: State):
    return {
        field.name: getattr(state, field.name)
        for field in state.fields
        if field.name != "sim_time" or state.sim_time is not None
    }


State.asdict = _asdict


@tree_math.struct
class DiagnosticState:
    vorticity: Any
    divergence: Any
    temperature_variation: Any
    cos_lat_u: Any
    sigma_dot_explicit: Any
    sigma_dot_full: Any
    cos_lat_grad_log_sp: Any
    u_dot_grad_log_sp: Any
    tracers: Any


def compute_diagnostic_state(
    state,
    coords,
):

    def to_nodal_fn(x):
        return coords.horizontal.to_nodal(x)

    nodal_vorticity = to_nodal_fn(state.vorticity)
    nodal_divergence = to_nodal_fn(state.divergence)
    nodal_temperature_variation = to_nodal_fn(state.temperature_variation)
    tracers = to_nodal_fn(state.tracers)
    nodal_cos_lat_u = jax.tree_util.tree_map(
        to_nodal_fn,
        get_cos_lat_vector(state.vorticity,
                           state.divergence,
                           coords.horizontal,
                           clip=False),
    )
    cos_lat_grad_log_sp = coords.horizontal.cos_lat_grad(
        state.log_surface_pressure, clip=False)
    nodal_cos_lat_grad_log_sp = to_nodal_fn(cos_lat_grad_log_sp)
    nodal_u_dot_grad_log_sp = sum(
        jax.tree_util.tree_map(
            lambda x, y: x * y * coords.horizontal.sec2_lat,
            nodal_cos_lat_u,
            nodal_cos_lat_grad_log_sp,
        ))
    f_explicit = cumulative_sigma_integral(nodal_u_dot_grad_log_sp,
                                           coords.vertical)
    f_full = cumulative_sigma_integral(
        nodal_divergence + nodal_u_dot_grad_log_sp, coords.vertical)
    sum_𝜎 = np.cumsum(coords.vertical.layer_thickness)[:, np.newaxis,
                                                       np.newaxis]
    sigma_dot_explicit = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_explicit, -1, None) - f_explicit, 0, -1)
    sigma_dot_full = lax.slice_in_dim(
        sum_𝜎 * lax.slice_in_dim(f_full, -1, None) - f_full, 0, -1)
    return DiagnosticState(
        vorticity=nodal_vorticity,
        divergence=nodal_divergence,
        temperature_variation=nodal_temperature_variation,
        cos_lat_u=nodal_cos_lat_u,
        sigma_dot_explicit=sigma_dot_explicit,
        sigma_dot_full=sigma_dot_full,
        cos_lat_grad_log_sp=nodal_cos_lat_grad_log_sp,
        u_dot_grad_log_sp=nodal_u_dot_grad_log_sp,
        tracers=tracers,
    )


def compute_vertical_velocity(state, coords):
    sigma_dot_boundaries = compute_diagnostic_state(state,
                                                    coords).sigma_dot_full
    assert sigma_dot_boundaries.ndim == 3
    sigma_dot_padded = jnp.pad(sigma_dot_boundaries, [(1, 1), (0, 0), (0, 0)])
    return 0.5 * (sigma_dot_padded[1:] + sigma_dot_padded[:-1])


def get_sigma_ratios(coordinates):
    alpha = np.diff(np.log(coordinates.centers), append=0) / 2
    alpha[-1] = -np.log(coordinates.centers[-1])
    return alpha


def get_geopotential_weights(coordinates, ):
    alpha = get_sigma_ratios(coordinates)
    weights = np.zeros([coordinates.layers, coordinates.layers])
    for j in range(coordinates.layers):
        weights[j, j] = alpha[j]
        for k in range(j + 1, coordinates.layers):
            weights[j, k] = alpha[k] + alpha[k - 1]
    return ideal_gas_constant * weights


def get_geopotential_diff(temperature, coordinates):
    weights = get_geopotential_weights(coordinates)
    return _vertical_matvec(weights, temperature)


def get_geopotential(
    temperature_variation,
    reference_temperature,
    orography,
    coordinates,
):
    surface_geopotential = orography * gravity_acceleration
    temperature = add_constant(temperature_variation, reference_temperature)
    geopotential_diff = get_geopotential_diff(temperature, coordinates)
    return surface_geopotential + geopotential_diff


def get_temperature_implicit_weights(
    coordinates,
    reference_temperature,
):
    p = np.tril(np.ones([coordinates.layers, coordinates.layers]))
    alpha = get_sigma_ratios(coordinates)[..., np.newaxis]
    p_alpha = p * alpha
    p_alpha_shifted = np.roll(p_alpha, 1, axis=0)
    p_alpha_shifted[0] = 0
    h0 = (kappa * reference_temperature[..., np.newaxis] *
          (p_alpha + p_alpha_shifted) /
          coordinates.layer_thickness[..., np.newaxis])
    temp_diff = np.diff(reference_temperature)
    thickness_sum = coordinates.layer_thickness[:
                                                -1] + coordinates.layer_thickness[
                                                    1:]
    k0 = np.concatenate((temp_diff / thickness_sum, [0]), axis=0)[...,
                                                                  np.newaxis]
    thickness_cumulative = np.cumsum(coordinates.layer_thickness)[...,
                                                                  np.newaxis]
    k1 = p - thickness_cumulative
    k = k0 * k1
    k_shifted = np.roll(k, 1, axis=0)
    k_shifted[0] = 0
    return (h0 - k - k_shifted) * coordinates.layer_thickness


def get_temperature_implicit(
    divergence,
    coordinates,
    reference_temperature,
):
    weights = -get_temperature_implicit_weights(coordinates,
                                                reference_temperature)
    return _vertical_matvec(weights, divergence)


def _vertical_matvec(a, x):
    return einsum("gh,...hml->...gml", a, x)


def _vertical_matvec_per_wavenumber(a, x):
    return einsum("lgh,...hml->...gml", a, x)


def _get_implicit_term_matrix(eta, coords, reference_temperature):
    eye = np.eye(coords.vertical.layers)[np.newaxis]
    lam = coords.horizontal.laplacian_eigenvalues
    g = get_geopotential_weights(coords.vertical)
    r = ideal_gas_constant
    h = get_temperature_implicit_weights(coords.vertical,
                                         reference_temperature)
    t = reference_temperature[:, np.newaxis]
    thickness = coords.vertical.layer_thickness[np.newaxis, np.newaxis, :]
    l = coords.horizontal.modal_shape[1]
    j = k = coords.vertical.layers
    row0 = np.concatenate(
        [
            np.broadcast_to(eye, [l, j, k]),
            eta * np.einsum("l,jk->ljk", lam, g),
            eta * r * np.einsum("l,jo->ljo", lam, t),
        ],
        axis=2,
    )
    row1 = np.concatenate(
        [
            eta * np.broadcast_to(h[np.newaxis], [l, j, k]),
            np.broadcast_to(eye, [l, j, k]),
            np.zeros([l, j, 1]),
        ],
        axis=2,
    )
    row2 = np.concatenate(
        [
            np.broadcast_to(eta * thickness, [l, 1, k]),
            np.zeros([l, 1, k]),
            np.ones([l, 1, 1]),
        ],
        axis=2,
    )
    return np.concatenate((row0, row1, row2), axis=1)


def div_sec_lat(m_component, n_component, grid: Grid):
    m_component = grid.to_modal(m_component * grid.sec2_lat)
    n_component = grid.to_modal(n_component * grid.sec2_lat)
    return grid.div_cos_lat((m_component, n_component), clip=False)


def truncated_modal_orography(
    orography,
    coords,
    wavenumbers_to_clip: int = 1,
):
    grid = coords.horizontal
    return grid.clip_wavenumbers(grid.to_modal(orography),
                                 n=wavenumbers_to_clip)


@dataclasses.dataclass
class PrimitiveEquations(ImplicitExplicitODE):
    reference_temperature: np.ndarray
    orography: Any
    coords: Any
    vertical_matmul_method: Any = dataclasses.field(default=None)
    implicit_inverse_method: Any = dataclasses.field(default="split")
    vertical_advection: Any = dataclasses.field(
        default=centered_vertical_advection)
    include_vertical_advection: Any = dataclasses.field(default=True)

    @property
    def coriolis_parameter(self):
        _, sin_lat = self.coords.horizontal.nodal_mesh
        return 2 * angular_velocity * sin_lat

    @property
    def T_ref(self):
        return self.reference_temperature[..., np.newaxis, np.newaxis]

    def _vertical_tendency(self, w, x):
        return self.vertical_advection(w, x, self.coords.vertical)

    def _t_omega_over_sigma_sp(self, temperature_field, g_term,
                               v_dot_grad_log_sp):
        f = cumulative_sigma_integral(g_term, self.coords.vertical)
        alpha = get_sigma_ratios(self.coords.vertical)
        alpha = alpha[:, np.newaxis, np.newaxis]
        del_𝜎 = self.coords.vertical.layer_thickness[:, np.newaxis, np.newaxis]
        padding = [(1, 0), (0, 0), (0, 0)]
        g_part = (alpha * f + jnp.pad(alpha * f, padding)[:-1, ...]) / del_𝜎
        return temperature_field * (v_dot_grad_log_sp - g_part)

    def kinetic_energy_tendency(self, aux_state: DiagnosticState):
        nodal_cos_lat_u2 = jnp.stack(aux_state.cos_lat_u)**2
        kinetic = nodal_cos_lat_u2.sum(0) * self.coords.horizontal.sec2_lat / 2
        return -self.coords.horizontal.laplacian(
            self.coords.horizontal.to_modal(kinetic))

    def orography_tendency(self):
        return -gravity_acceleration * self.coords.horizontal.laplacian(
            self.orography)

    def curl_and_div_tendencies(
        self,
        aux_state: DiagnosticState,
    ):
        sec2_lat = self.coords.horizontal.sec2_lat
        u, v = aux_state.cos_lat_u
        total_vorticity = aux_state.vorticity + self.coriolis_parameter
        nodal_vorticity_u = -v * total_vorticity * sec2_lat
        nodal_vorticity_v = u * total_vorticity * sec2_lat
        d𝜎_dt = aux_state.sigma_dot_full
        sigma_dot_u = -self._vertical_tendency(d𝜎_dt, u)
        sigma_dot_v = -self._vertical_tendency(d𝜎_dt, v)
        rt = ideal_gas_constant * aux_state.temperature_variation
        grad_log_ps_u, grad_log_ps_v = aux_state.cos_lat_grad_log_sp
        vertical_term_u = (sigma_dot_u + rt * grad_log_ps_u) * sec2_lat
        vertical_term_v = (sigma_dot_v + rt * grad_log_ps_v) * sec2_lat
        combined_u = self.coords.horizontal.to_modal(nodal_vorticity_u +
                                                     vertical_term_u)
        combined_v = self.coords.horizontal.to_modal(nodal_vorticity_v +
                                                     vertical_term_v)
        dζ_dt = -self.coords.horizontal.curl_cos_lat(
            (combined_u, combined_v), clip=False)
        d𝛅_dt = -self.coords.horizontal.div_cos_lat(
            (combined_u, combined_v), clip=False)
        return (dζ_dt, d𝛅_dt)

    def nodal_temperature_vertical_tendency(
        self,
        aux_state: DiagnosticState,
    ):
        sigma_dot_explicit = aux_state.sigma_dot_explicit
        sigma_dot_full = aux_state.sigma_dot_full
        temperature_variation = aux_state.temperature_variation
        tendency = self._vertical_tendency(sigma_dot_full,
                                           temperature_variation)
        if np.unique(self.T_ref.ravel()).size > 1:
            tendency += self._vertical_tendency(sigma_dot_explicit, self.T_ref)
        return tendency

    def horizontal_scalar_advection(
        self,
        scalar,
        aux_state,
    ):
        u, v = aux_state.cos_lat_u
        nodal_terms = scalar * aux_state.divergence
        modal_terms = -div_sec_lat(u * scalar, v * scalar,
                                   self.coords.horizontal)
        return nodal_terms, modal_terms

    def nodal_temperature_adiabatic_tendency(self, aux_state: DiagnosticState):
        g_explicit = aux_state.u_dot_grad_log_sp
        g_full = g_explicit + aux_state.divergence
        mean_t_part = self._t_omega_over_sigma_sp(self.T_ref, g_explicit,
                                                  aux_state.u_dot_grad_log_sp)
        variation_t_part = self._t_omega_over_sigma_sp(
            aux_state.temperature_variation, g_full,
            aux_state.u_dot_grad_log_sp)
        return kappa * (mean_t_part + variation_t_part)

    def nodal_log_pressure_tendency(self, aux_state: DiagnosticState):
        g = aux_state.u_dot_grad_log_sp
        return -sigma_integral(g, self.coords.vertical)

    def explicit_terms(self, state: State):
        aux_state = compute_diagnostic_state(state, self.coords)
        vorticity_tendency, divergence_dot = self.curl_and_div_tendencies(
            aux_state)
        kinetic_energy_tendency = self.kinetic_energy_tendency(aux_state)
        orography_tendency = self.orography_tendency()
        horizontal_tendency_fn = functools.partial(
            self.horizontal_scalar_advection, aux_state=aux_state)
        dT_dt_horizontal_nodal, dT_dt_horizontal_modal = horizontal_tendency_fn(
            aux_state.temperature_variation)
        tracers_horizontal_nodal_and_modal = jax.tree_util.tree_map(
            horizontal_tendency_fn, aux_state.tracers)
        dT_dt_vertical = self.nodal_temperature_vertical_tendency(aux_state)
        dT_dt_adiabatic = self.nodal_temperature_adiabatic_tendency(aux_state)
        log_sp_tendency = self.nodal_log_pressure_tendency(aux_state)
        sigma_dot_full = aux_state.sigma_dot_full
        vertical_tendency_fn = functools.partial(self._vertical_tendency,
                                                 sigma_dot_full)
        tracers_vertical_nodal = jax.tree_util.tree_map(
            vertical_tendency_fn, aux_state.tracers)
        to_modal_fn = self.coords.horizontal.to_modal
        divergence_tendency = (divergence_dot + kinetic_energy_tendency +
                               orography_tendency)
        temperature_tendency = (to_modal_fn(dT_dt_horizontal_nodal +
                                            dT_dt_vertical + dT_dt_adiabatic) +
                                dT_dt_horizontal_modal)
        log_surface_pressure_tendency = to_modal_fn(log_sp_tendency)
        tracers_tendency = jax.tree_util.tree_map(
            lambda x, y_z: to_modal_fn(x + y_z[0]) + y_z[1],
            tracers_vertical_nodal,
            tracers_horizontal_nodal_and_modal,
        )
        tendency = State(
            vorticity=vorticity_tendency,
            divergence=divergence_tendency,
            temperature_variation=temperature_tendency,
            log_surface_pressure=log_surface_pressure_tendency,
            tracers=tracers_tendency,
            sim_time=None if state.sim_time is None else 1.0,
        )
        return self.coords.horizontal.clip_wavenumbers(tendency)

    def implicit_terms(self, state: State):
        geopotential_diff = get_geopotential_diff(state.temperature_variation,
                                                  self.coords.vertical)
        rt_log_p = (ideal_gas_constant * self.T_ref *
                    state.log_surface_pressure)
        vorticity_implicit = jnp.zeros_like(state.vorticity)
        divergence_implicit = -self.coords.horizontal.laplacian(
            geopotential_diff + rt_log_p)
        temperature_variation_implicit = get_temperature_implicit(
            state.divergence,
            self.coords.vertical,
            self.reference_temperature,
        )
        log_surface_pressure_implicit = -_vertical_matvec(
            self.coords.vertical.layer_thickness[np.newaxis], state.divergence)
        tracers_implicit = jax.tree_util.tree_map(jnp.zeros_like,
                                                  state.tracers)
        return State(
            vorticity=vorticity_implicit,
            divergence=divergence_implicit,
            temperature_variation=temperature_variation_implicit,
            log_surface_pressure=log_surface_pressure_implicit,
            tracers=tracers_implicit,
            sim_time=None if state.sim_time is None else 0.0,
        )

    def implicit_inverse(self, state, step_size):
        implicit_matrix = _get_implicit_term_matrix(
            step_size,
            self.coords,
            self.reference_temperature,
        )
        assert implicit_matrix.dtype == np.float64
        layers = self.coords.vertical.layers
        div = slice(0, layers)
        temp = slice(layers, 2 * layers)
        logp = slice(2 * layers, 2 * layers + 1)

        def named_vertical_matvec(name):
            return jax.named_call(_vertical_matvec_per_wavenumber, name=name)

        inverse = np.linalg.inv(implicit_matrix)
        assert not np.isnan(inverse).any()
        inverted_divergence = (
            named_vertical_matvec("div_from_div")(inverse[:, div, div],
                                                  state.divergence) +
            named_vertical_matvec("div_from_temp")(
                inverse[:, div, temp], state.temperature_variation) +
            named_vertical_matvec("div_from_logp")(inverse[:, div, logp],
                                                   state.log_surface_pressure))
        inverted_temperature_variation = (
            named_vertical_matvec("temp_from_div")(inverse[:, temp, div],
                                                   state.divergence) +
            named_vertical_matvec("temp_from_temp")(
                inverse[:, temp, temp], state.temperature_variation) +
            named_vertical_matvec("temp_from_logp")(
                inverse[:, temp, logp], state.log_surface_pressure))
        inverted_log_surface_pressure = (
            named_vertical_matvec("logp_from_div")(inverse[:, logp, div],
                                                   state.divergence) +
            named_vertical_matvec("logp_from_temp")(
                inverse[:, logp, temp], state.temperature_variation) +
            named_vertical_matvec("logp_from_logp")(
                inverse[:, logp, logp], state.log_surface_pressure))
        inverted_vorticity = state.vorticity
        inverted_tracers = state.tracers
        return State(
            inverted_vorticity,
            inverted_divergence,
            inverted_temperature_variation,
            inverted_log_surface_pressure,
            inverted_tracers,
            sim_time=state.sim_time,
        )


def isothermal_rest_atmosphere(
    coords,
    tref=288.0 * units.degK,
    p0=1e5 * units.pascal,
    p1=0.0 * units.pascal,
):
    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)
    tref = DEFAULT_SCALE.nondimensionalize(units.Quantity(tref))
    p0 = DEFAULT_SCALE.nondimensionalize(units.Quantity(p0))
    p1 = DEFAULT_SCALE.nondimensionalize(units.Quantity(p1))
    orography = np.zeros_like(lat)

    def _get_vorticity(sigma, lon, lat):
        del sigma, lon
        return jnp.zeros_like(lat)

    def _get_surface_pressure(lon, lat, rng_key):

        def relative_pressure(altitude_m):
            g = 9.80665
            cp = 1004.68506
            T0 = 288.16
            M = 0.02896968
            R0 = 8.314462618
            return (1 - g * altitude_m / (cp * T0))**(cp * M / R0)

        altitude_m = DEFAULT_SCALE.dimensionalize(orography,
                                                  units.meter).magnitude
        surface_pressure = (p0 * np.ones(coords.surface_nodal_shape) *
                            relative_pressure(altitude_m))
        keys = jax.random.split(rng_key, 2)
        lon0 = jax.random.uniform(keys[1],
                                  minval=np.pi / 2,
                                  maxval=3 * np.pi / 2)
        lat0 = jax.random.uniform(keys[0], minval=-np.pi / 4, maxval=np.pi / 4)
        stddev = np.pi / 20
        k = 4
        perturbation = (jnp.exp(-((lon - lon0)**2) / (2 * stddev**2)) *
                        jnp.exp(-((lat - lat0)**2) /
                                (2 * stddev**2)) * jnp.sin(k * (lon - lon0)))
        return surface_pressure + p1 * perturbation

    def random_state_fn(rng_key: jnp.ndarray):
        nodal_vorticity = jnp.stack([
            _get_vorticity(sigma, lon, lat)
            for sigma in coords.vertical.centers
        ])
        modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
        nodal_surface_pressure = _get_surface_pressure(lon, lat, rng_key)
        return State(
            vorticity=modal_vorticity,
            divergence=jnp.zeros_like(modal_vorticity),
            temperature_variation=jnp.zeros_like(modal_vorticity),
            log_surface_pressure=(coords.horizontal.to_modal(
                jnp.log(nodal_surface_pressure))),
        )

    aux_features = {
        "orography": orography,
        "ref_temperatures": np.full((coords.vertical.layers, ), tref),
    }
    return random_state_fn, aux_features


def steady_state_jw(
    coords,
    u0=35.0 * units.m / units.s,
    p0=1e5 * units.pascal,
    t0=288.0 * units.degK,
    delta_t=4.8e5 * units.degK,
    gamma=0.005 * units.degK / units.m,
    sigma_tropo: float = 0.2,
    sigma0: float = 0.252,
):
    u0 = DEFAULT_SCALE.nondimensionalize(u0)
    t0 = DEFAULT_SCALE.nondimensionalize(t0)
    delta_t = DEFAULT_SCALE.nondimensionalize(delta_t)
    p0 = DEFAULT_SCALE.nondimensionalize(p0)
    gamma = DEFAULT_SCALE.nondimensionalize(gamma)
    a = radius
    g = gravity_acceleration
    r_gas = ideal_gas_constant
    omega = angular_velocity

    def _get_reference_temperature(sigma):
        top_mean_t = t0 * sigma**(r_gas * gamma / g)
        if sigma < sigma_tropo:
            return top_mean_t + delta_t * (sigma_tropo - sigma)**5
        else:
            return top_mean_t

    def _get_reference_geopotential(sigma):
        top_mean_potential = (t0 * g / gamma) * (1 -
                                                 sigma**(r_gas * gamma / g))
        if sigma < sigma_tropo:
            return top_mean_potential - r_gas * delta_t * (
                (np.log(sigma / sigma_tropo) + 137 / 60) * sigma_tropo**5 -
                5 * sigma * sigma_tropo**4 + 5 * (sigma**2) *
                (sigma_tropo**3) - (10 / 3) * (sigma_tropo**2) * sigma**3 +
                (5 / 4) * sigma_tropo * sigma**4 - (sigma**5) / 5)
        else:
            return top_mean_potential

    def _get_geopotential(lat, lon, sigma):
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return _get_reference_geopotential(
            sigma) + u0 * np.cos(sigma_nu)**1.5 * (
                ((-2 * np.sin(lat)**6 *
                  (np.cos(lat)**2 + 1 / 3) + 10 / 63) * u0 * np.cos(sigma_nu)**
                 (3 / 2)) +
                ((1.6 * (np.cos(lat)**3) *
                  (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega))

    def _get_temperature_variation(lat, lon, sigma):
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        cos_𝜎ν = np.cos(sigma_nu)
        sin_𝜎ν = np.sin(sigma_nu)
        return (0.75 * (sigma * np.pi * u0 / r_gas) * sin_𝜎ν *
                np.sqrt(cos_𝜎ν) *
                (((-2 * (np.cos(lat)**2 + 1 / 3) * np.sin(lat)**6 + 10 / 63) *
                  2 * u0 * cos_𝜎ν**(3 / 2)) +
                 ((1.6 * (np.cos(lat)**3) *
                   (np.sin(lat)**2 + 2 / 3) - np.pi / 4) * a * omega)))

    def _get_vorticity(lat, lon, sigma):
        del lon
        sigma_nu = (sigma - sigma0) * np.pi / 2
        return ((-4 * u0 / a) * (np.cos(sigma_nu)**(3 / 2)) * np.sin(lat) *
                np.cos(lat) * (2 - 5 * np.sin(lat)**2))

    def _get_surface_pressure(
        lat,
        lon,
    ):
        del lon
        return p0 * np.ones(lat.shape)[np.newaxis, ...]

    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)

    def initial_state_fn(rng_key=None):
        del rng_key
        nodal_vorticity = np.stack([
            _get_vorticity(lat, lon, sigma)
            for sigma in coords.vertical.centers
        ])
        modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
        nodal_temperature_variation = np.stack([
            _get_temperature_variation(lat, lon, sigma)
            for sigma in coords.vertical.centers
        ])
        log_nodal_surface_pressure = np.log(_get_surface_pressure(lat, lon))
        state = State(
            vorticity=modal_vorticity,
            divergence=np.zeros_like(modal_vorticity),
            temperature_variation=coords.horizontal.to_modal(
                nodal_temperature_variation),
            log_surface_pressure=coords.horizontal.to_modal(
                log_nodal_surface_pressure),
        )
        return state

    orography = _get_geopotential(lat, lon, 1.0) / g
    geopotential = np.stack([
        _get_geopotential(lat, lon, sigma) for sigma in coords.vertical.centers
    ])
    reference_temperatures = np.stack([
        _get_reference_temperature(sigma) for sigma in coords.vertical.centers
    ])
    aux_features = {
        "geopotential": geopotential,
        "orography": orography,
        "ref_temperatures": reference_temperatures,
    }
    return initial_state_fn, aux_features


def baroclinic_perturbation_jw(
    coords,
    u_perturb=1.0 * units.m / units.s,
    lon_location=np.pi / 9,
    lat_location=2 * np.pi / 9,
    perturbation_radius=0.1,
):
    u_p = DEFAULT_SCALE.nondimensionalize(u_perturb)
    a = radius

    def _get_vorticity_perturbation(lat, lon, sigma):
        del sigma
        x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
            lat) * np.cos(lon - lon_location)
        r = a * np.arccos(x)
        R = a * perturbation_radius
        return ((u_p / a) * np.exp(-((r / R)**2)) *
                (np.tan(lat) - (2 * ((a / R)**2) * np.arccos(x)) *
                 (np.sin(lat_location) * np.cos(lat) - np.cos(lat_location) *
                  np.sin(lat) * np.cos(lon - lon_location)) /
                 (np.sqrt(1 - x**2))))

    def _get_divergence_perturbation(lat, lon, sigma):
        del sigma
        x = np.sin(lat_location) * np.sin(lat) + np.cos(lat_location) * np.cos(
            lat) * np.cos(lon - lon_location)
        r = a * np.arccos(x)
        R = a * perturbation_radius
        return ((-2 * u_p * a /
                 (R**2)) * np.exp(-((r / R)**2)) * np.arccos(x) *
                ((np.cos(lat_location) * np.sin(lon - lon_location)) /
                 (np.sqrt(1 - x**2))))

    lon, sin_lat = coords.horizontal.nodal_mesh
    lat = np.arcsin(sin_lat)
    nodal_vorticity = np.stack([
        _get_vorticity_perturbation(lat, lon, sigma)
        for sigma in coords.vertical.centers
    ])
    nodal_divergence = np.stack([
        _get_divergence_perturbation(lat, lon, sigma)
        for sigma in coords.vertical.centers
    ])
    modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
    modal_divergence = coords.horizontal.to_modal(nodal_divergence)
    state = State(
        vorticity=modal_vorticity,
        divergence=modal_divergence,
        temperature_variation=np.zeros_like(modal_vorticity),
        log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),
    )
    return state


def _preserves_shape(target, scaling):
    target_shape = np.shape(target)
    return target_shape == np.broadcast_shapes(target_shape, scaling.shape)


def _make_filter_fn(scaling, name=None):
    rescale = lambda x: scaling * x if _preserves_shape(x, scaling) else x
    return functools.partial(jax.tree_util.tree_map,
                             jax.named_call(rescale, name=name))


def exponential_filter(
    grid,
    attenuation=16,
    order=18,
    cutoff=0,
):
    _, total_wavenumber = grid.modal_axes
    k = total_wavenumber / total_wavenumber.max()
    a = attenuation
    c = cutoff
    p = order
    scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c))**(2 * p))))
    return _make_filter_fn(scaling, "exponential_filter")


def horizontal_diffusion_filter(
    grid,
    scale,
    order=1,
):
    eigenvalues = grid.laplacian_eigenvalues
    scaling = jnp.exp(-scale * (-eigenvalues)**order)
    return _make_filter_fn(scaling, "horizontal_diffusion_filter")


class HeldSuarezForcing:

    def __init__(
        self,
        coords,
        reference_temperature,
        p0=1e5 * units.pascal,
        sigma_b=0.7,
        kf=1 / (1 * units.day),
        ka=1 / (40 * units.day),
        ks=1 / (4 * units.day),
        minT=200 * units.degK,
        maxT=315 * units.degK,
        dTy=60 * units.degK,
        dThz=10 * units.degK,
    ):
        self.coords = coords
        self.reference_temperature = reference_temperature
        self.p0 = DEFAULT_SCALE.nondimensionalize(p0)
        self.sigma_b = sigma_b
        self.kf = DEFAULT_SCALE.nondimensionalize(kf)
        self.ka = DEFAULT_SCALE.nondimensionalize(ka)
        self.ks = DEFAULT_SCALE.nondimensionalize(ks)
        self.minT = DEFAULT_SCALE.nondimensionalize(minT)
        self.maxT = DEFAULT_SCALE.nondimensionalize(maxT)
        self.dTy = DEFAULT_SCALE.nondimensionalize(dTy)
        self.dThz = DEFAULT_SCALE.nondimensionalize(dThz)
        self.sigma = self.coords.vertical.centers
        _, sin_lat = self.coords.horizontal.nodal_mesh
        self.lat = np.arcsin(sin_lat)

    def kv(self):
        kv_coeff = self.kf * (np.maximum(0, (self.sigma - self.sigma_b) /
                                         (1 - self.sigma_b)))
        return kv_coeff[:, np.newaxis, np.newaxis]

    def kt(self):
        cutoff = np.maximum(0,
                            (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        return self.ka + (self.ks - self.ka) * (
            cutoff[:, np.newaxis, np.newaxis] * np.cos(self.lat)**4)

    def equilibrium_temperature(self, nodal_surface_pressure):
        p_over_p0 = (self.sigma[:, np.newaxis, np.newaxis] *
                     nodal_surface_pressure / self.p0)
        temperature = p_over_p0**kappa * (
            self.maxT - self.dTy * np.sin(self.lat)**2 -
            self.dThz * jnp.log(p_over_p0) * np.cos(self.lat)**2)
        return jnp.maximum(self.minT, temperature)

    def explicit_terms(self, state: State):
        aux_state = compute_diagnostic_state(state=state, coords=self.coords)
        nodal_velocity_tendency = jax.tree.map(
            lambda x: -self.kv() * x / self.coords.horizontal.cos_lat**2,
            aux_state.cos_lat_u,
        )
        nodal_temperature = (
            self.reference_temperature[:, np.newaxis, np.newaxis] +
            aux_state.temperature_variation)
        nodal_log_surface_pressure = self.coords.horizontal.to_nodal(
            state.log_surface_pressure)
        nodal_surface_pressure = jnp.exp(nodal_log_surface_pressure)
        Teq = self.equilibrium_temperature(nodal_surface_pressure)
        nodal_temperature_tendency = -self.kt() * (nodal_temperature - Teq)
        temperature_tendency = self.coords.horizontal.to_modal(
            nodal_temperature_tendency)
        velocity_tendency = self.coords.horizontal.to_modal(
            nodal_velocity_tendency)
        vorticity_tendency = self.coords.horizontal.curl_cos_lat(
            velocity_tendency)
        divergence_tendency = self.coords.horizontal.div_cos_lat(
            velocity_tendency)
        log_surface_pressure_tendency = jnp.zeros_like(
            state.log_surface_pressure)
        return State(
            vorticity=vorticity_tendency,
            divergence=divergence_tendency,
            temperature_variation=temperature_tendency,
            log_surface_pressure=log_surface_pressure_tendency,
        )


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
    a_boundaries: np.ndarray
    b_boundaries: np.ndarray

    @classmethod
    def ECMWF137(cls):
        a_in_pa, b = np.loadtxt("ecmwf137_hybrid_levels.csv",
                                skiprows=1,
                                usecols=(1, 2),
                                delimiter="\t").T
        a = a_in_pa / 100
        return cls(a_boundaries=a, b_boundaries=b)

    def __hash__(self):
        return hash((tuple(self.a_boundaries.tolist()),
                     tuple(self.b_boundaries.tolist())))

    def get_sigma_boundaries(self, surface_pressure):
        return self.a_boundaries / surface_pressure + self.b_boundaries


def _interval_overlap(source_bounds, target_bounds):
    upper = jnp.minimum(target_bounds[1:, jnp.newaxis],
                        source_bounds[jnp.newaxis, 1:])
    lower = jnp.maximum(target_bounds[:-1, jnp.newaxis],
                        source_bounds[jnp.newaxis, :-1])
    return jnp.maximum(upper - lower, 0)


def conservative_regrid_weights(source_bounds, target_bounds):
    weights = _interval_overlap(source_bounds, target_bounds)
    weights /= jnp.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (target_bounds.size - 1, source_bounds.size - 1)
    return weights


@functools.partial(jax.jit, static_argnums=(1, 2))
def regrid_hybrid_to_sigma(
    fields,
    hybrid_coords,
    sigma_coords,
    surface_pressure,
):

    @jax.jit
    @functools.partial(jnp.vectorize, signature="(x,y),(a),(b,x,y)->(c,x,y)")
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    @functools.partial(jax.vmap, in_axes=(-1, None, -1), out_axes=-1)
    def regrid(surface_pressure, sigma_bounds, field):
        assert sigma_bounds.shape == (sigma_coords.layers + 1, )
        hybrid_bounds = hybrid_coords.get_sigma_boundaries(surface_pressure)
        weights = conservative_regrid_weights(hybrid_bounds, sigma_bounds)
        result = jnp.einsum("ab,b->a", weights, field, precision="float32")
        assert result.shape[0] == sigma_coords.layers
        return result

    return tree_map_over_nonscalars(
        lambda x: regrid(surface_pressure, sigma_coords.boundaries, x), fields)


def _maybe_update_shape_and_dim_with_realization_time_sample(
    shape,
    dims,
    times,
    sample_ids,
    include_realization,
):
    if times is not None:
        shape = times.shape + shape
        dims = ("time", ) + dims
    return shape, dims


def _infer_dims_shape_and_coords(
    coords,
    times,
    sample_ids,
    additional_coords,
):
    lon_k, lat_k = coords.horizontal.modal_axes
    lon, sin_lat = coords.horizontal.nodal_axes
    all_xr_coords = {
        "lon": lon * 180 / np.pi,
        "lat": np.arcsin(sin_lat) * 180 / np.pi,
        "longitudinal_mode": lon_k,
        "total_wavenumber": lat_k,
        "level": coords.vertical.centers,
        **additional_coords,
    }
    if times is not None:
        all_xr_coords["time"] = times
    basic_shape_to_dims = {}
    basic_shape_to_dims[tuple()] = tuple()
    modal_shape = coords.horizontal.modal_shape
    nodal_shape = coords.horizontal.nodal_shape
    basic_shape_to_dims[(coords.vertical.layers, ) +
                        modal_shape] = ("level", "longitudinal_mode",
                                        "total_wavenumber")
    basic_shape_to_dims[(coords.vertical.layers, ) + nodal_shape] = ("level",
                                                                     "lon",
                                                                     "lat")
    basic_shape_to_dims[nodal_shape] = "lon", "lat"
    basic_shape_to_dims[modal_shape] = "longitudinal_mode", "total_wavenumber"
    basic_shape_to_dims[coords.surface_nodal_shape] = "lon", "lat"
    for dim, value in additional_coords.items():
        basic_shape_to_dims[value.shape + modal_shape] = (dim,
                                                          "longitudinal_mode",
                                                          "total_wavenumber")
        basic_shape_to_dims[value.shape + nodal_shape] = (dim, "lon", "lat")
        basic_shape_to_dims[value.shape] = (dim, )
    update_shape_dims_fn = functools.partial(
        _maybe_update_shape_and_dim_with_realization_time_sample,
        times=times,
        sample_ids=sample_ids,
        include_realization="realization" in additional_coords,
    )
    shape_to_dims = {}
    for shape, dims in basic_shape_to_dims.items():
        full_shape, full_dims = update_shape_dims_fn(shape, dims)
        shape_to_dims[full_shape] = full_dims
    return all_xr_coords, shape_to_dims


def data_to_xarray(
    data,
    *,
    coords,
    times,
    serialize_coords_to_attrs=True,
):
    # assert serialize_coords_to_attrs is None
    prognostic_keys = set(data.keys()) - {"tracers"} - {"diagnostics"}
    additional_coords = {}
    if coords.vertical.layers != 1:
        additional_coords["surface"] = np.ones(1)
    all_coords, shape_to_dims = _infer_dims_shape_and_coords(
        coords, times, None, additional_coords)
    dims_in_state = set()
    data_vars = {}
    for key in prognostic_keys:
        value = data[key]
        dims = shape_to_dims[value.shape]
        data_vars[key] = (dims, value)
        dims_in_state.update(set(dims))
    dataset_attrs = coords.asdict() if serialize_coords_to_attrs else {}
    coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
    return xarray.Dataset(data_vars, coords, attrs=dataset_attrs)


def temperature_variation_to_absolute(
    temperature_variation,
    ref_temperature,
):
    return temperature_variation + ref_temperature[:, np.newaxis, np.newaxis]
