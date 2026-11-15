"""Normalized observation likelihoods with adaptive, scaled amplitude integration.

The production integrator uses embedded 16/32-point Gauss-Legendre rules on
log amplitude. Observation-specific peak splits prevent missing narrow M0
integrands. An independent SciPy calculation integrates in linear amplitude.
Truncated Gamma mass is never renormalized and probabilities are never floored.
"""

from dataclasses import dataclass
from time import perf_counter
import math
import warnings
import numpy as np
from numba import njit
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gammaln, gammaincinv, gammainccinv
from .model import from_config, candidate

X16, W16 = np.polynomial.legendre.leggauss(16)
X32, W32 = np.polynomial.legendre.leggauss(32)


class IntegrationError(ArithmeticError):
    pass


@dataclass(frozen=True)
class Settings:
    tail_probability: float = 1e-10
    rtol: float = 2e-7
    atol: float = 1e-10
    max_depth: int = 18


@njit(cache=True)
def log_kernel(u, y, h, T, variance, k, tau, constant):
    """Log Gaussian times Gamma density times dw/du, with w=exp(u)."""
    w = math.exp(u)
    c11, c12, c22 = variance+w*T[0, 0], w*T[0, 1], variance+w*T[1, 1]
    det = c11*c22-c12*c12
    d1, d2 = y[0]-w*h[0], y[1]-w*h[1]
    quadratic = (c22*d1*d1-2*c12*d1*d2+c11*d2*d2)/det
    return -math.log(2*math.pi)-.5*(math.log(det)+quadratic)+k*u-w/tau-constant


@njit(cache=True)
def splits(y, h, T, variance, k, tau, constant, low, high, dense=False):
    # Analytic log-amplitude mode for M0 also seeds a robust search for M.
    q = np.dot(h, h)/variance
    d = np.dot(y, h)/variance-1/tau
    root = math.sqrt(d*d+4*q*k)
    w0 = (d+root)/(2*q) if d >= 0 else 2*k/(root-d)
    # Gamma probability cutoffs alone can miss almost the entire likelihood
    # for an observation unlikely under a corner parameter. Enlarge them to
    # include an observation-based peak and numerically negligible tails.
    u0 = math.log(w0)
    low, high = min(low, u0-2), max(high, u0+2)
    drop = 55. if high-low < 100 else 65.
    anchor = max(log_kernel(u0, y, h, T, variance, k, tau, constant),
                 log_kernel(math.log(k*tau), y, h, T, variance, k, tau, constant))
    for _ in range(80):
        if log_kernel(low, y, h, T, variance, k, tau, constant) < anchor-drop:
            break
        low -= 2.
    for _ in range(80):
        if log_kernel(high, y, h, T, variance, k, tau, constant) < anchor-drop:
            break
        high += 2.
    points = np.empty(35)
    points[:33] = np.linspace(low, high, 33)
    points[33], points[34] = u0, min(high, max(low, math.log(k*tau)))
    points.sort()
    values = np.array([log_kernel(u, y, h, T, variance, k, tau, constant) for u in points])
    idx = np.argmax(values)
    a, b = points[max(0, idx-1)], points[min(34, idx+1)]
    # Bracketed golden-section maximization. Keep brackets around every
    # sampled local peak, so a second peak is not discarded by this search.
    ratio = .6180339887498949
    c, e = b-ratio*(b-a), a+ratio*(b-a)
    fc = log_kernel(c, y, h, T, variance, k, tau, constant)
    fe = log_kernel(e, y, h, T, variance, k, tau, constant)
    for _ in range(65):
        if b-a < 1e-12:
            break
        if fc > fe:
            b, e, fe = e, c, fc
            c = b-ratio*(b-a)
            fc = log_kernel(c, y, h, T, variance, k, tau, constant)
        else:
            a, c, fc = c, e, fe
            e = a+ratio*(b-a)
            fe = log_kernel(e, y, h, T, variance, k, tau, constant)
    mode = (a+b)/2
    shift = log_kernel(mode, y, h, T, variance, k, tau, constant)
    step = 1e-4
    curvature = -(log_kernel(mode+step, y, h, T, variance, k, tau, constant)
                  -2*shift+log_kernel(mode-step, y, h, T, variance, k, tau, constant))/step**2
    width = min(2., 1/math.sqrt(max(curvature, .25)))
    extras = np.array([-8., -4., -2., -1., -.25, 0., .25, 1., 2., 4., 8.])*width+mode
    if dense:
        retained = points
    else:
        retained_list = [low, high, u0, min(high, max(low, math.log(k*tau)))]
        for i in range(1, len(points)-1):
            if values[i] >= values[i-1] and values[i] >= values[i+1]:
                retained_list.extend([points[i-1], points[i], points[i+1]])
        retained = np.array(retained_list)
    out = np.concatenate((retained, np.minimum(high, np.maximum(low, extras))))
    out.sort()
    return out, shift, mode, width


@njit(cache=True)
def integrate_one(y, h, T, variance, k, tau, constant, low, high, rtol, atol, max_depth):
    points, shift, _, _ = splits(y, h, T, variance, k, tau, constant, low, high)
    low, high = points[0], points[-1]
    total, error, evaluations = 0., 0., 0
    left, right = np.empty(4096), np.empty(4096)
    depths = np.empty(4096, dtype=np.int64)
    size = 0
    for i in range(len(points)-1):
        if points[i+1] > points[i]:
            left[size], right[size], depths[size] = points[i], points[i+1], 0
            size += 1
    failed = False
    while size:
        size -= 1
        a, b, depth = left[size], right[size], depths[size]
        mid, half = (a+b)/2, (b-a)/2
        v16, v32 = 0., 0.
        for i in range(16):
            v16 += W16[i]*math.exp(log_kernel(mid+half*X16[i], y, h, T, variance, k, tau, constant)-shift)
        for i in range(32):
            v32 += W32[i]*math.exp(log_kernel(mid+half*X32[i], y, h, T, variance, k, tau, constant)-shift)
        v16, v32 = half*v16, half*v32
        err = abs(v32-v16)
        evaluations += 48
        if err <= atol*(b-a)/(high-low)+rtol*abs(v32):
            total += v32
            error += err
        elif depth >= max_depth or size+2 >= len(left):
            failed = True
            total += v32
            error += err
        else:
            left[size], right[size], depths[size] = a, mid, depth+1
            left[size+1], right[size+1], depths[size+1] = mid, b, depth+1
            size += 2
    if total <= 0 or not math.isfinite(total):
        return -math.inf, math.inf, evaluations, True
    return shift+math.log(total), error/total, evaluations, failed


@njit(cache=True)
def integrate_batch(ys, h, T, variance, k, tau, constant, low, high, rtol, atol, max_depth):
    values, errors = np.empty(len(ys)), np.empty(len(ys))
    evaluations, failures = 0, 0
    for i in range(len(ys)):
        values[i], errors[i], count, fail = integrate_one(
            ys[i], h, T, variance, k, tau, constant, low, high, rtol, atol, max_depth)
        evaluations += count
        failures += int(fail)
    return values, errors, evaluations, failures


def amplitude_limits(k, tau, settings):
    p = settings.tail_probability
    return np.log(tau*gammaincinv(k, p)), np.log(tau*gammainccinv(k, p))


def adaptive_reference(y, h, T, variance, k, tau, settings):
    """Independent QUADPACK integration in w, with peak-centered splits."""
    low, high = amplitude_limits(k, tau, settings)
    constant = gammaln(k)+k*np.log(tau)
    points, shift, mode, width = splits(y, h, T, variance, k, tau, constant, low, high, dense=True)
    raw_ws = np.unique(np.exp(points))
    # Merge almost coincident splits without deleting any integration area.
    ws = [raw_ws[0]]
    for w in raw_ws[1:-1]:
        if w-ws[-1] > 1e-11*max(w, 1e-100):
            ws.append(w)
    ws.append(raw_ws[-1])
    ws = np.array(ws)
    # Scaling cancels the different Jacobian and keeps the integrals O(1).
    scale = shift-mode
    def fun(w):
        u = np.log(w)
        return np.exp(log_kernel(u, y, h, T, variance, k, tau, constant)-u-scale)
    total, error = 0., 0.
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        for a, b in zip(ws[:-1], ws[1:]):
            value, err = quad(fun, a, b, epsabs=1e-12/len(ws), epsrel=2e-9, limit=250)
            total += value
            error += err
    if total <= 0 or error/total > 1e-7:
        raise IntegrationError(f"Reference quadrature error {error}, value {total}")
    return scale+np.log(total)


class Likelihood:
    def __init__(self, config, settings=None):
        self.config = config
        self.settings = settings or Settings(**config["integration"])
        self.timing = dict(construction_seconds=0., integration_seconds=0., evaluations=0,
                           kernel_evaluations=0, max_relative_error=0., moment_seconds=0., extinction_seconds=0.,
                           spectral_amplitude_seconds=0., positive_amplitude_seconds=0.)

    def logpdf(self, method, theta, y, t, z0, reference=False, survivor=False, residual_variance=0.):
        if method not in ("G", "EG", "M", "M0", "Mconst"):
            raise ValueError(f"Unknown method {method}")
        if not np.isfinite(residual_variance) or residual_variance < 0:
            raise ValueError("Residual variance must be finite and nonnegative")
        if method != "Mconst" and residual_variance != 0:
            raise ValueError("Only Mconst has a fitted residual variance")
        y = np.atleast_2d(y).astype(float)
        tick = perf_counter()
        misses = candidate.cache_info().misses
        c = from_config(theta, self.config)
        if candidate.cache_info().misses > misses:
            for key, value in c.construction_times.items():
                self.timing[key] += value
        m, V = c.at(t, z0)
        variance = self.config["measurement_variance"]
        weights = None
        if method != 'G':
            extinction_ready = 'extinction' in c.__dict__
            weights = c.survival(t, z0)
            if not extinction_ready:
                self.timing['extinction_seconds'] += c.construction_times['extinction_seconds']
        empty = -np.log(2*np.pi*variance)-.5*np.sum(y*y, axis=1)/variance
        def combine(living):
            if survivor or method == "G":
                return living
            return np.logaddexp(weights["log_empty"]+empty, weights["log_survival"]+living)
        if method in ("G", "EG"):
            if method == "EG":
                m, V = c.survivor_moments(t, z0)
            C = V+variance*np.eye(2)
            factor = np.linalg.cholesky(C)
            standardized = np.linalg.solve(factor, (y-m).T)
            out = -np.log(2*np.pi)-np.log(np.diag(factor)).sum()-.5*(standardized**2).sum(axis=0)
            self.timing["construction_seconds"] += perf_counter()-tick
            return combine(out)
        amplitude_started = perf_counter()
        a = c.amplitude(z0)[0]
        _, _, k, tau = c.positive_amplitude(z0)
        self.timing["positive_amplitude_seconds"] += perf_counter()-amplitude_started
        h = m/a
        T = np.exp(c.L*t)*c.B if method == "M" else np.zeros((2, 2))
        if method == "Mconst":
            # The empty density above retains the known measurement variance.
            variance += residual_variance
        log_offset = np.zeros(len(y))
        if method in ("M0", "Mconst"):
            # R = variance*I: rotate into the mean direction and its perpendicular.
            # The perpendicular Gaussian factor does not depend on amplitude.
            # Remove its potentially enormous log density before quadrature so
            # subtracting the log-integrand peak does not lose small variations.
            # This is an exact coordinate change, not an approximation to M0.
            length = np.linalg.norm(h)
            direction = h/length
            perpendicular = -direction[1]*y[:, 0]+direction[0]*y[:, 1]
            log_offset = -.5*perpendicular**2/variance
            y = np.column_stack((y @ direction, np.zeros(len(y))))
            h = np.array([length, 0.])
        low, high = amplitude_limits(k, tau, self.settings)
        self.timing["construction_seconds"] += perf_counter()-tick
        tick = perf_counter()
        if reference:
            out = np.array([adaptive_reference(yi, h, T, variance, k, tau, self.settings) for yi in y])
        else:
            s = self.settings
            out, errors, count, failures = integrate_batch(
                y, h, T, variance, k, tau, gammaln(k)+k*np.log(tau), low, high,
                s.rtol, s.atol, s.max_depth)
            self.timing["kernel_evaluations"] += count
            self.timing["max_relative_error"] = max(self.timing["max_relative_error"], float(errors.max()))
            if failures or not np.all(np.isfinite(out)) or np.max(errors) > 10*s.rtol:
                raise IntegrationError(f"{failures} unresolved integrals; maximum relative error {errors.max()}")
        self.timing["integration_seconds"] += perf_counter()-tick
        return combine(out+log_offset)

    def loglik(self, method, theta, data):
        # Only observed y and known design enter inference; no counts or W.
        residual_variance = 0.
        if method == "Mconst":
            if len(theta) != 3 or theta[2] < 0:
                raise ValueError("Mconst expects two birth rates and u >= 0")
            residual_variance = self.config["measurement_variance"]*np.expm1(theta[2])
            theta = theta[:2]
        total = 0.
        for cell in np.unique(data["cell"]):
            ids = np.flatnonzero(data["cell"] == cell)
            first = ids[0]
            total += self.logpdf(method, theta, data["y"][ids], float(data["time"][first]), data["z0"][first],
                                 residual_variance=residual_variance).sum()
        self.timing["evaluations"] += 1
        return float(total)
