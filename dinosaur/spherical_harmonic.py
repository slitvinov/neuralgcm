from __future__ import annotations
import dataclasses
import functools
from typing import Any, Callable
from dinosaur import jax_numpy_utils
from dinosaur import pytree_utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.special as sps


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


def gauss_legendre_nodes(n):
    return sps.roots_legendre(n)


einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)
LATITUDE_SPACINGS = dict(gauss=gauss_legendre_nodes, )


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
    u_down = jax_numpy_utils.shift(u, -1, axis)
    u_up = jax_numpy_utils.shift(u, +1, axis)
    return j * jnp.where(i % 2, u_down, -u_up)


def quadrature_nodes(nodes):
    xs = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    weights = 2 * np.pi / nodes
    return xs, weights


def get_latitude_nodes(n: int, spacing: str):
    get_nodes = LATITUDE_SPACINGS.get(spacing)
    return get_nodes(n)


@dataclasses.dataclass
class _SphericalHarmonicBasis:
    f: np.ndarray
    p: np.ndarray
    w: np.ndarray


@dataclasses.dataclass(frozen=True)
class SphericalHarmonics:
    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"


class RealSphericalHarmonics(SphericalHarmonics):

    @functools.cached_property
    def nodal_axes(self):
        longitude, _ = quadrature_nodes(self.longitude_nodes)
        sin_latitude, _ = get_latitude_nodes(self.latitude_nodes,
                                             self.latitude_spacing)
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
        x, wp = get_latitude_nodes(self.latitude_nodes, self.latitude_spacing)
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
        px = jax.named_call(einsum, name="inv_legendre")("mjl,...ml->...mj", p,
                                                         x)
        fpx = jax.named_call(einsum, name="inv_fourier")("im,...mj->...ij", f,
                                                         px)
        return fpx

    def transform(self, x):
        w = self.basis.w
        f = self.basis.f
        p = self.basis.p
        wx = w * x
        fwx = jax.named_call(einsum, name="fwd_fourier")("im,...ij->...mj", f,
                                                         wx)
        pfwx = jax.named_call(einsum, name="fwd_legendre")("mjl,...mj->...ml",
                                                           p, fwx)
        return pfwx

    def longitudinal_derivative(self, x):
        return real_basis_derivative(x, axis=-2)


P = jax.sharding.PartitionSpec


def _vertical_pad(field, mesh):
    return field, None


def _vertical_crop(field):
    return field


def _with_vertical_padding(f, mesh):

    def g(x):
        x, padding = _vertical_pad(x, mesh)
        return _vertical_crop(f(x), padding)

    return g


SphericalHarmonicsImpl = Callable[..., SphericalHarmonics]


@dataclasses.dataclass(frozen=True)
class Grid:
    longitude_wavenumbers: int = 0
    total_wavenumbers: int = 0
    longitude_nodes: int = 0
    latitude_nodes: int = 0
    latitude_spacing: str = "gauss"
    longitude_offset: float = 0.0
    radius: float | None = None
    spherical_harmonics_impl: SphericalHarmonicsImpl = RealSphericalHarmonics

    def __post_init__(self):
        if self.radius is None:
            object.__setattr__(self, "radius", 1.0)

    @classmethod
    def construct(
        cls,
        max_wavenumber: int,
        gaussian_nodes: int,
        latitude_spacing: str = "gauss",
        longitude_offset: float = 0.0,
        radius: float | None = None,
        spherical_harmonics_impl:
        SphericalHarmonicsImpl = RealSphericalHarmonics,
    ):
        return cls(
            longitude_wavenumbers=max_wavenumber + 1,
            total_wavenumbers=max_wavenumber + 2,
            longitude_nodes=4 * gaussian_nodes,
            latitude_nodes=2 * gaussian_nodes,
            latitude_spacing=latitude_spacing,
            longitude_offset=longitude_offset,
            spherical_harmonics_impl=spherical_harmonics_impl,
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
        items[
            'spherical_harmonics_impl'] = self.spherical_harmonics_impl.__name__
        return items

    @functools.cached_property
    def spherical_harmonics(self):
        kwargs = dict()
        return self.spherical_harmonics_impl(
            longitude_wavenumbers=self.longitude_wavenumbers,
            total_wavenumbers=self.total_wavenumbers,
            longitude_nodes=self.longitude_nodes,
            latitude_nodes=self.latitude_nodes,
            latitude_spacing=self.latitude_spacing,
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
        f = _with_vertical_padding(self.spherical_harmonics.inverse_transform, None)
        return pytree_utils.tree_map_over_nonscalars(f, x)

    def to_modal(self, z):
        f = _with_vertical_padding(self.spherical_harmonics.transform, None)
        return pytree_utils.tree_map_over_nonscalars(f, z)

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

        return pytree_utils.tree_map_over_nonscalars(clip, x)

    @functools.cached_property
    def _derivative_recurrence_weights(self):
        m, l = self.modal_mesh
        a = np.sqrt(self.mask * (l**2 - m**2) / (4 * l**2 - 1))
        a[:, 0] = 0
        b = np.sqrt(self.mask * ((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
        b[:, -1] = 0
        return a, b

    def d_dlon(self, x):
        return _with_vertical_padding(
            self.spherical_harmonics.longitudinal_derivative,
            None)(x)

    def cos_lat_d_dlat(self, x):
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = jax_numpy_utils.shift(((l + 1) * a) * x, -1, axis=-1)
        x_lp1 = jax_numpy_utils.shift((-l * b) * x, +1, axis=-1)
        return x_lm1 + x_lp1

    def sec_lat_d_dlat_cos2(self, x):
        _, l = self.modal_mesh
        a, b = self._derivative_recurrence_weights
        x_lm1 = jax_numpy_utils.shift(((l - 1) * a) * x, -1, axis=-1)
        x_lp1 = jax_numpy_utils.shift((-(l + 2) * b) * x, +1, axis=-1)
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


_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


def add_constant(x: jnp.ndarray, c):
    return x.at[..., 0, 0].add(_CONSTANT_NORMALIZATION_FACTOR * c)


def get_cos_lat_vector(
    vorticity,
    divergence,
    grid: Grid,
    clip: bool = True,
):
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
def vor_div_to_uv_nodal(
    grid: Grid,
    vorticity,
    divergence,
    clip: bool = True,
):
    u_cos_lat, v_cos_lat = get_cos_lat_vector(vorticity,
                                              divergence,
                                              grid,
                                              clip=clip)
    u_nodal = grid.to_nodal(u_cos_lat) / grid.cos_lat
    v_nodal = grid.to_nodal(v_cos_lat) / grid.cos_lat
    return u_nodal, v_nodal
