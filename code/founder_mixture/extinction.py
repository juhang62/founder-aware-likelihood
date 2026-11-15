"""Candidate-specific finite-time and eventual extinction, without subtraction of rounded weights."""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ExtinctionSettings:
    rtol: float = 1e-10
    atol: float = 1e-12
    iteration_tolerance: float = 1e-13
    max_iterations: int = 10000


def rhs(q, birth, death, switching):
    return death-(birth+death+switching)*q+birth*q*q+switching*q[::-1]


@njit(cache=True)
def smallest_fixed_point(birth, death, switching, tolerance, maximum):
    q = np.zeros(2)
    for iteration in range(maximum):
        new = (death+birth*q*q+switching*q[::-1])/(birth+death+switching)
        if np.max(np.abs(new-q)) <= tolerance:
            return new, iteration+1, True
        q = new
    return q, maximum, False


@lru_cache(maxsize=8192)
def extinction(birth, death, switching, times, settings=ExtinctionSettings()):
    b, d, s = (np.asarray(x, float) for x in (birth, death, switching))
    rho, iterations, converged = smallest_fixed_point(b, d, s, settings.iteration_tolerance, settings.max_iterations)
    residual = float(np.max(np.abs(rhs(rho, b, d, s))))
    if not converged or residual > 10*settings.iteration_tolerance*max(1., np.max(b+d+s)):
        raise ArithmeticError("Eventual-extinction iteration failed")
    ordered = np.unique(times)
    if np.all(d == 0) or max(ordered) == 0:
        values = np.zeros((2, len(ordered)))
    else:
        solution = solve_ivp(lambda t, q: rhs(q, b, d, s), (0., max(ordered)), np.zeros(2),
                             t_eval=ordered, rtol=settings.rtol, atol=settings.atol, method="DOP853")
        if not solution.success:
            raise ArithmeticError(solution.message)
        values = solution.y
    tolerance = 20*(settings.atol+settings.iteration_tolerance)
    if np.any(values < -tolerance) or np.any(values > rho[:, None]+tolerance) or np.any(np.diff(values, axis=1) < -tolerance):
        raise ArithmeticError("Invalid finite-time extinction probabilities")
    return dict(rho=rho, q={float(t): values[:, j] for j, t in enumerate(ordered)},
                iterations=iterations, residual=residual)


def log1mexp(x):
    if x == -np.inf:
        return 0.
    if x > 0:
        raise ValueError("A log probability must be nonpositive")
    return float(np.log1p(-np.exp(x)) if x < -np.log(2.) else np.log(-np.expm1(x)))


def log_weights(probabilities, z0):
    z0 = np.asarray(z0)
    if np.any(z0 < 0) or np.any(z0 != np.floor(z0)) or not z0.sum():
        raise ValueError("A nonzero integer founder state is required")
    selected = np.asarray(probabilities)[z0 > 0]
    empty = -np.inf if np.any(selected == 0) else float(z0[z0 > 0] @ np.log(selected))
    return empty, log1mexp(empty)
