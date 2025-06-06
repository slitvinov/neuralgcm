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

tree_map = jax.tree_util.tree_map
einsum = functools.partial(jnp.einsum, precision=lax.Precision.HIGHEST)
units = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
Unit = units.Unit
GRAVITY_ACCELERATION = 9.80616 * units.m / units.s**2
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
    6.37122e6 * units.m,
    1 / 2 / 7.292e-5 * units.s,
    1 * units.kilogram,
    1 * units.degK,
)

gravity_acceleration = DEFAULT_SCALE.nondimensionalize(GRAVITY_ACCELERATION)
kappa = 2 / 7
ideal_gas_constant = DEFAULT_SCALE.nondimensionalize(
    kappa * 1004 * units.J / units.kilogram / units.degK)


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


def _slice_shape_along_axis(x, axis):
    x_shape = list(x.shape)
    x_shape[axis] = 1
    return tuple(x_shape)


def _with_f64_math(f):
    return lambda x: f(x.astype(np.float64)).astype(x.dtype)


class SigmaCoordinates:

    def __init__(self, boundaries):
        self.boundaries = boundaries

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


@dataclasses.dataclass(frozen=True)
class RealSphericalHarmonics:
    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0

    @functools.cached_property
    def nodal_axes(self):
        longitude = np.linspace(0,
                                2 * np.pi,
                                self.longitude_nodes,
                                endpoint=False)
        sin_latitude, _ = sps.roots_legendre(self.latitude_nodes)
        return longitude, sin_latitude

    @functools.cached_property
    def nodal_shape(self):
        return self.longitude_nodes, self.latitude_nodes

    @functools.cached_property
    def modal_axes(self):
        m_pos = np.arange(1, self.longitude_wavenumbers)
        m_pos_neg = np.stack([m_pos, -m_pos], axis=1).ravel()
        lon_wavenumbers = np.concatenate([[0], m_pos_neg])
        tot_wavenumbers = np.arange(self.total_wavenumbers)
        return lon_wavenumbers, tot_wavenumbers

    @functools.cached_property
    def modal_shape(self):
        return 2 * self.longitude_wavenumbers - 1, self.total_wavenumbers

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
        wf = 2 * np.pi / self.longitude_nodes
        x, wp = sps.roots_legendre(self.latitude_nodes)
        w = wf * wp
        p = evaluate(n_m=self.longitude_wavenumbers,
                     n_l=self.total_wavenumbers,
                     x=x)
        p = np.repeat(p, 2, axis=0)
        p = p[1:]
        return f, p, w

    def inverse_transform(self, x):
        f, p, w = self.basis
        px = einsum("mjl,...ml->...mj", p, x)
        fpx = einsum("im,...mj->...ij", f, px)
        return fpx

    def transform(self, x):
        f, p, w = self.basis
        wx = w * x
        fwx = einsum("im,...ij->...mj", f, wx)
        pfwx = einsum("mjl,...mj->...ml", p, fwx)
        return pfwx

    def longitudinal_derivative(self, x):
        return real_basis_derivative(x, axis=-2)


@dataclasses.dataclass(frozen=True)
class Grid:
    longitude_wavenumbers: int
    total_wavenumbers: int
    longitude_nodes: int
    latitude_nodes: int

    @functools.cached_property
    def spherical_harmonics(self):
        return RealSphericalHarmonics(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes)

    @functools.cached_property
    def nodal_axes(self):
        return self.spherical_harmonics.nodal_axes

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
        return -l * (l + 1) / (1.0**2)

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
            mask = jnp.ones(self.modal_shape[-1], x.dtype).at[-n:].set(0)
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
        raw = self.d_dlon(x) / 1.0, self.cos_lat_d_dlat(x) / 1.0
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
        raw = (self.d_dlon(v[0]) + self.sec_lat_d_dlat_cos2(v[1])) / 1.0
        if clip:
            return self.clip_wavenumbers(raw)
        return raw

    def curl_cos_lat(
        self,
        v,
        clip: bool = True,
    ):
        raw = (self.d_dlon(v[1]) - self.sec_lat_d_dlat_cos2(v[0])) / 1.0
        if clip:
            return self.clip_wavenumbers(raw)
        return raw


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
def uv_nodal_to_vor_div_modal(grid, u_nodal, v_nodal, clip=True):
    u_over_cos_lat = grid.to_modal(u_nodal / grid.cos_lat)
    v_over_cos_lat = grid.to_modal(v_nodal / grid.cos_lat)
    vorticity = grid.curl_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    divergence = grid.div_cos_lat((u_over_cos_lat, v_over_cos_lat), clip=clip)
    return vorticity, divergence


@functools.partial(jax.jit, static_argnames=("grid", ))
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


def maybe_to_nodal(fields, coords):
    array_shape_fn = lambda x: np.asarray(x.shape[:-2] + coords.horizontal.
                                          nodal_shape)
    scalar_shape_fn = lambda x: np.array([], dtype=int)
    nodal_shapes = tree_map_over_nonscalars(array_shape_fn,
                                            fields,
                                            scalar_fn=scalar_shape_fn)
    fn = lambda x, nodal: x if x.shape == tuple(
        nodal) else coords.horizontal.to_nodal(x)
    return jax.tree_util.tree_map(fn, fields, nodal_shapes)


class ImplicitExplicitODE:

    def __init__(self, explicit_terms, implicit_terms, implicit_inverse):
        self.explicit_terms = explicit_terms
        self.implicit_terms = implicit_terms
        self.implicit_inverse = implicit_inverse


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


def _asdict(state):
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


def div_sec_lat(m_component, n_component, grid):
    m_component = grid.to_modal(m_component * grid.sec2_lat)
    n_component = grid.to_modal(n_component * grid.sec2_lat)
    return grid.div_cos_lat((m_component, n_component), clip=False)


def truncated_modal_orography(orography, coords, wavenumbers_to_clip=1):
    return coords.horizontal.clip_wavenumbers(coords.horizontal.to_modal(orography),
                                              n=wavenumbers_to_clip)


class PrimitiveEquations:

    def __init__(self, reference_temperature, orography, coords):
        self.reference_temperature = reference_temperature
        self.orography = orography
        self.coords = coords

    @property
    def coriolis_parameter(self):
        _, sin_lat = self.coords.horizontal.nodal_mesh
        return sin_lat

    @property
    def T_ref(self):
        return self.reference_temperature[..., np.newaxis, np.newaxis]

    def _vertical_tendency(self, w, x):
        return centered_vertical_advection(w, x, self.coords.vertical)

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


def _preserves_shape(target, scaling):
    target_shape = np.shape(target)
    return target_shape == np.broadcast_shapes(target_shape, scaling.shape)


def _make_filter_fn(scaling):
    rescale = lambda x: scaling * x if _preserves_shape(x, scaling) else x
    return functools.partial(jax.tree_util.tree_map, rescale)


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
    return _make_filter_fn(scaling)


def horizontal_diffusion_filter(
    grid,
    scale,
    order=1,
):
    eigenvalues = grid.laplacian_eigenvalues
    scaling = jnp.exp(-scale * (-eigenvalues)**order)
    return _make_filter_fn(scaling)


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
