from __future__ import annotations
import dataclasses
import math
from dinosaur import filtering
from dinosaur import spherical_harmonic
from dinosaur import typing
import jax
import jax.numpy as jnp
import numpy as np
import tree_math

tree_map = jax.tree_util.tree_map


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
        state: PyTreeState,
        step_size: float,
    ):
        return self.forward_eq.implicit_inverse(state, -step_size)


def compose_equations(equations: Sequence[Union[ImplicitExplicitODE,
                                                ExplicitODE]], ):
    implicit_explicit_eqs = list(
        filter(lambda x: isinstance(x, ImplicitExplicitODE), equations))
    if len(implicit_explicit_eqs) != 1:
        raise ValueError(
            'compose_equations supports at most 1 ImplicitExplicitODE '
            f'got {len(implicit_explicit_eqs)}')
    (implicit_explicit_equation, ) = implicit_explicit_eqs
    assert isinstance(implicit_explicit_equation, ImplicitExplicitODE)

    def explicit_fn(x: PyTreeState):
        explicit_tendencies = [fn.explicit_terms(x) for fn in equations]
        return tree_map(lambda *args: sum([x for x in args if x is not None]),
                        *explicit_tendencies)

    return ImplicitExplicitODE.from_functions(
        explicit_fn, implicit_explicit_equation.implicit_terms,
        implicit_explicit_equation.implicit_inverse)


@dataclasses.dataclass
class ImExButcherTableau:
    a_ex: Sequence[Sequence[float]]
    a_im: Sequence[Sequence[float]]
    b_ex: Sequence[float]
    b_im: Sequence[float]


def imex_runge_kutta(
    tableau: ImExButcherTableau,
    equation: ImplicitExplicitODE,
    time_step: float,
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
    equation: ImplicitExplicitODE,
    time_step: float,
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


def runge_kutta_step_filter(state_filter: PyTreeTermsFn, ):

    def _filter(u: PyTreeState, u_next: PyTreeState):
        del u  # unused
        return state_filter(u_next)

    return _filter


def exponential_step_filter(
    grid,
    dt,
    tau=0.010938,
    order=18,
    cutoff=0,
):
    filter_fn = filtering.exponential_filter(grid, dt / tau, order, cutoff)
    return runge_kutta_step_filter(filter_fn)


def horizontal_diffusion_step_filter(
    grid,
    dt,
    tau,
    order=1,
):
    eigenvalues = grid.laplacian_eigenvalues
    scale = dt / (tau * abs(eigenvalues[-1])**order)
    filter_fn = filtering.horizontal_diffusion_filter(grid, scale, order)
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
    if steps == 1:
        return fn

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
        init_weight = 1.0  # for time=0
        total_weight = init_weight + 2 * weights.sum()
        init_weight /= total_weight
        weights /= total_weight
        init_term = tree_map(lambda x: x * init_weight, state)
        forward_term = accumulate_repeated(forward_step, weights, state)
        backward_term = accumulate_repeated(backward_step, weights, state)
        return tree_map(lambda *xs: sum(xs), init_term, forward_term,
                        backward_term)

    return f
