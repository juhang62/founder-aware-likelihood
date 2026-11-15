"""Opt-in compiled extinction driver for the validated Mconst extension.

The original solver remains available to reproduce prior deliveries. The
context applies the same solver to every method in a matched timing run.
"""

from contextlib import contextmanager
from functools import lru_cache
import warnings
import numpy as np
from numba import njit
from scipy.integrate import ode
from . import model
from .extinction import ExtinctionSettings, extinction, rhs, smallest_fixed_point

SOLVER = 'scipy-ode-dop853-numba-rhs-v1'


@njit(cache=True)
def compiled_rhs(q, birth, death, switching):
    return death-(birth+death+switching)*q+birth*q*q+switching*q[::-1]


@lru_cache(maxsize=8192)
def compiled_extinction(birth, death, switching, times, settings=ExtinctionSettings()):
    b, d, s = (np.asarray(x, float) for x in (birth, death, switching))
    rho, iterations, converged = smallest_fixed_point(b, d, s, settings.iteration_tolerance, settings.max_iterations)
    residual = float(np.max(np.abs(rhs(rho, b, d, s))))
    if not converged or residual > 10*settings.iteration_tolerance*max(1., np.max(b+d+s)):
        raise ArithmeticError('Eventual-extinction iteration failed')
    ordered = np.unique(times)
    if len(ordered) == 0 or ordered[0] < 0:
        raise ValueError('Nonnegative observation times are required')
    values = np.zeros((2, len(ordered)))
    if np.any(d) and ordered[-1] > 0:
        solver = ode(lambda t, q: compiled_rhs(q, b, d, s)).set_integrator(
            'dop853', rtol=settings.rtol, atol=settings.atol, nsteps=10000)
        solver.set_initial_value(np.zeros(2), 0.)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for j, t in enumerate(ordered):
                if t > 0:
                    values[:, j] = solver.integrate(t)
                    if not solver.successful():
                        raise ArithmeticError(f'Compiled DOP853 failed: {solver.get_return_code()}')
    tolerance = 20*(settings.atol+settings.iteration_tolerance)
    if (np.any(values < -tolerance) or np.any(values > rho[:, None]+tolerance)
            or np.any(np.diff(values, axis=1) < -tolerance)):
        raise ArithmeticError('Invalid finite-time extinction probabilities')
    return dict(rho=rho, q={float(t): values[:, j] for j, t in enumerate(ordered)},
                iterations=iterations, residual=residual)


def clear_caches():
    model.candidate.cache_clear()
    extinction.cache_clear()
    compiled_extinction.cache_clear()


@contextmanager
def backend():
    previous = model.extinction
    clear_caches()
    model.extinction = compiled_extinction
    try:
        yield
    finally:
        model.extinction = previous
        clear_caches()
