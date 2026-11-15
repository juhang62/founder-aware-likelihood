"""Candidate-specific spectral quantities and exact linear branching moments."""

from dataclasses import dataclass, field
from time import perf_counter
from functools import lru_cache, cached_property
import numpy as np
from scipy.linalg import expm
from .extinction import extinction, ExtinctionSettings, log_weights


def symmetric_psd(matrix, name="covariance"):
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] < -1e-10 * max(1.0, np.linalg.norm(matrix)):
        raise ValueError(f"Materially indefinite {name}: {eigenvalues}")
    return matrix


def noise(z, birth, switching, death=(0., 0.)):
    """Infinitesimal covariance D(z); z may be a real mean or eigenvector."""
    switch = np.dot(switching, z)
    return np.diag((np.asarray(birth)+np.asarray(death)) * z) + switch * np.array([[1., -1.], [-1., 1.]])


@dataclass
class Candidate:
    birth: np.ndarray
    switching: np.ndarray
    death: np.ndarray
    extinction_settings: ExtinctionSettings
    A: np.ndarray
    L: float
    lambda2: float
    r: np.ndarray
    ell: np.ndarray
    v: np.ndarray
    B: np.ndarray
    u2: np.ndarray
    u3: np.ndarray
    moments: dict
    construction_times: dict
    state_cache: dict = field(default_factory=dict)

    @cached_property
    def extinction(self):
        """Construct survival quantities only when a method needs them."""
        tick = perf_counter()
        result = extinction(tuple(self.birth), tuple(self.death), tuple(self.switching),
                            tuple(self.moments), self.extinction_settings)
        self.construction_times['extinction_seconds'] = perf_counter()-tick
        return result

    def amplitude(self, z0):
        a, v0 = float(self.ell @ z0), float(self.v @ z0)
        if min(a, v0) <= 0:
            raise ValueError("Amplitude mean and variance must be positive")
        return a, v0, a*a/v0, v0/a

    def survival(self, t, z0):
        key = ("survival", float(t), tuple(z0))
        if key in self.state_cache:
            return self.state_cache[key]
        le, ls = log_weights(self.extinction["q"][float(t)], z0)
        lei, lsi = log_weights(self.extinction["rho"], z0)
        result = dict(log_empty=le, log_survival=ls, log_eventual_empty=lei,
                    log_eventual_survival=lsi, pi=float(np.exp(ls)), pi_infinity=float(np.exp(lsi)),
                    epsilon=float(-np.expm1(lsi-ls)))
        self.state_cache[key] = result
        return result

    def positive_amplitude(self, z0):
        key = ("positive_amplitude", tuple(z0))
        if key in self.state_cache:
            return self.state_cache[key]
        a, v, _, _ = self.amplitude(z0)
        _, ls = log_weights(self.extinction["rho"], z0)
        pi = np.exp(ls)
        positive_mean = a/pi
        positive_variance = (v+a*a)/pi-positive_mean**2
        values = (positive_mean, positive_variance, positive_mean**2/positive_variance,
                  positive_variance/positive_mean)
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ArithmeticError("Invalid full-state positive-amplitude moments")
        self.state_cache[key] = values
        return values

    def survivor_moments(self, t, z0):
        m, V = self.at(t, z0)
        weights = self.survival(t, z0)
        pi = weights["pi"]
        # Use the stored empty weight, even if pi rounds to one.
        return m/pi, symmetric_psd(V/pi-np.exp(weights["log_empty"])*np.outer(m, m)/pi**2)

    def mixture_moments(self, t, z0, residual=True):
        m, _ = self.at(t, z0)
        a = self.amplitude(z0)[0]
        ap, vp, _, _ = self.positive_amplitude(z0)
        weights = self.survival(t, z0)
        mu = ap*m/a
        C = vp*np.outer(m, m)/a**2
        if residual:
            C += ap*np.exp(self.L*t)*self.B
        return weights["pi"]*mu, symmetric_psd(weights["pi"]*(C+np.exp(weights["log_empty"])*np.outer(mu, mu)))

    def at(self, t, z0):
        means, covariances = self.moments[float(t)]
        return means @ z0, symmetric_psd(np.einsum("i,ijk->jk", z0, covariances))

    def mixture_covariances(self, t, z0):
        return self.mixture_moments(t, z0)[1], self.mixture_moments(t, z0, False)[1]


@lru_cache(maxsize=8192)
def candidate(birth, switching=(0.6, 0.4), times=(5., 7.), death=(0., 0.), extinction_settings=ExtinctionSettings()):
    """Cache only parameter/design calculations, never simulator truth."""
    started = perf_counter()
    birth, switching = np.asarray(birth, float), np.asarray(switching, float)
    if np.any(birth <= 0) or np.any(switching <= 0):
        raise ValueError("The two-type pilot requires positive birth and switching rates")
    death = np.asarray(death, float)
    if death.shape != (2,) or np.any(death < 0) or not np.isfinite(death).all():
        raise ValueError("Two finite nonnegative death rates are required")
    b1, b2 = birth
    s12, s21 = switching
    A = np.array([[b1-death[0]-s12, s21], [s12, b2-death[1]-s21]])
    gap = np.hypot(A[0, 0]-A[1, 1], 2*np.sqrt(s12*s21))
    L, lambda2 = (np.trace(A)+gap)/2, (np.trace(A)-gap)/2
    if not (L > 0 and lambda2 < L/2):
        raise ValueError(f"Spectral condition violated: L={L}, lambda2={lambda2}")
    r = np.array([(L-A[1, 1])/s12, 1.])
    ell = np.array([s12, L-A[0, 0]])
    ell /= ell @ r
    if np.any(r <= 0) or np.any(ell <= 0):
        raise ValueError("Perron eigenvectors must be positive")
    if not (np.allclose(A @ r, L*r) and np.allclose(ell @ A, L*ell)):
        raise ArithmeticError("Eigenvector identity failed")
    D = np.array([noise(e, birth, switching, death) for e in np.eye(2)])
    c = np.einsum("i,kij,j->k", ell, D, ell)
    v = np.linalg.solve(2*L*np.eye(2)-A.T, c)
    u2 = np.linalg.solve(2*L*np.eye(2)-A.T, 2*birth*ell**2)
    u3 = np.linalg.solve(3*L*np.eye(2)-A.T, 6*birth*ell*u2)
    if not np.allclose(u2-ell**2, v, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("Independent amplitude variance formulas disagree")
    P = np.outer(r, ell)
    S = np.eye(2)-P
    Dr = noise(r, birth, switching, death)
    B = symmetric_psd(P @ Dr @ P.T/L + S @ Dr @ S.T/(L-2*lambda2), "B")
    spectral_amplitude_seconds = perf_counter()-started
    moment_started = perf_counter()
    # The block linear system evolves (m1,m2,V11,V12,V22) for both
    # one-founder states at once. No covariance subtraction is needed.
    K = np.zeros((5, 5))
    K[:2, :2] = A
    K[2:, :2] = D[:, (0, 0, 1), (0, 1, 1)].T
    aa, ab, ac, ad = A.ravel()
    K[2:, 2:] = [[2*aa, 2*ab, 0], [ac, aa+ad, ab], [0, 2*ac, 2*ad]]
    initial = np.zeros((5, 2))
    initial[:2] = np.eye(2)
    moments = {}
    for t in times:
        if t < 0:
            raise ValueError("Negative observation time")
        out = expm(K*t) @ initial
        covs = np.array([[[out[2, i], out[3, i]], [out[3, i], out[4, i]]] for i in range(2)])
        moments[float(t)] = (out[:2], np.array([symmetric_psd(V) for V in covs]))
    moment_seconds = perf_counter()-moment_started
    return Candidate(birth, switching, death, extinction_settings, A, L, lambda2, r, ell, v, B, u2, u3, moments, dict(spectral_amplitude_seconds=spectral_amplitude_seconds,
        moment_seconds=moment_seconds, extinction_seconds=0.))


def from_config(theta, config):
    return candidate(tuple(float(x) for x in theta), tuple(config["switching"]), tuple(config["times"]),
                     tuple(config.get("death", (0., 0.))), ExtinctionSettings(**config.get("extinction", {})))


def yule_coefficients(birth, founders):
    """Scalar specialization of the same spectral/amplitude formulas."""
    L = birth
    variance = birth/(2*L-birth)
    B = birth/L
    return dict(L=L, v=variance, B=B, shape=founders/variance, scale=variance)
