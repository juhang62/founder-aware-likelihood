"""Exact scalar count reference and deterministic approximation comparisons.

No simulation or fitted parameters enter this module. Count truncation is not
renormalized. Scalar densities reuse the validated amplitude integrator by
embedding one scalar Gaussian in two dimensions and removing the constant
second Gaussian factor. CDFs use independent adaptive integration.
"""

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq
from scipy.special import ndtr, gammaln, logsumexp
from scipy.stats import binom, nbinom, gamma, norm
from .likelihood import Settings, amplitude_limits, integrate_one, adaptive_reference

METHODS = ("G", "EG", "M", "M0")
PROBABILITIES = np.array([.001, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .999])


def scalar_parameters(b, d, n, t):
    if not b > d >= 0 or n < 1 or n != int(n) or t < 0:
        raise ValueError("Require b>d>=0, a positive integer founder count, and t>=0")
    g = b-d
    E = np.exp(-g*t)
    q = d*(-np.expm1(-g*t))/(b-d*E)
    eta = b*(-np.expm1(-g*t))/(b-d*E)
    mean = n/E
    variance = n*(b+d)/g*(1-E)/E**2
    pi = -np.expm1(n*np.log(q)) if q > 0 else 1.
    pi_inf = -np.expm1(n*np.log(d/b)) if d > 0 else 1.
    ap = n/pi_inf
    vp = (n*(b+d)/g+n*n)/pi_inf-ap*ap
    return dict(q=q, eta=eta, mean=mean, variance=variance, pi=pi, pi_inf=pi_inf,
                ap=ap, vp=vp, k=ap*ap/vp, tau=vp/ap, growth=1/E, B=(b+d)/g)


@dataclass
class CountReference:
    b: float
    d: float
    n: int
    t: float
    tail: float = 1e-12

    def __post_init__(self):
        self.parameters = p = scalar_parameters(self.b, self.d, self.n, self.t)
        if self.t == 0:
            self.counts = np.arange(self.n+1)
            self.mass = np.eye(1, self.n+1, self.n).ravel()
            self.omitted = 0.
        else:
            ks = np.arange(1, self.n+1)
            weights = binom.pmf(ks, self.n, 1-p["q"])
            success = 1-p["eta"]
            upper = max(self.n, int(np.ceil(p["mean"])))
            def remainder(limit):
                return float(weights @ nbinom.sf(limit-ks, ks, success))
            while remainder(upper) > self.tail:
                upper *= 2
            self.counts = np.arange(upper+1)
            self.mass = np.zeros(upper+1)
            self.mass[0] = p["q"]**self.n
            for k, weight in zip(ks, weights):
                self.mass[k:] += weight*nbinom.pmf(self.counts[k:]-k, k, success)
            self.omitted = remainder(upper)
        self.cumulative = np.cumsum(self.mass)

    def cdf(self, x, noisy=True):
        if not noisy:
            j = int(np.floor(x))
            return 0. if j < 0 else float(self.cumulative[min(j, len(self.mass)-1)])
        return float(self.mass @ ndtr((x-self.counts)/2))

    def logpdf(self, x):
        positive = self.mass > 0
        return float(logsumexp(np.log(self.mass[positive])+norm.logpdf(x, self.counts[positive], 2)))

    def quantile(self, u, noisy=True, xtol=1e-8):
        if not noisy:
            return float(np.searchsorted(self.cumulative, u))
        return inverse_continuous(lambda x: self.cdf(x), u, self.parameters["mean"],
                                  np.sqrt(self.parameters["variance"]+4), xtol)


def inverse_continuous(cdf, u, mean, sd, xtol):
    low, high = mean-8*max(sd, 1.), mean+8*max(sd, 1.)
    while cdf(low) > u:
        low -= 2*(high-low)
    while cdf(high) < u:
        high += 2*(high-low)
    return float(brentq(lambda x: cdf(x)-u, low, high, xtol=xtol, rtol=1e-14))


class ScalarApproximation:
    def __init__(self, b, d, n, t, settings=Settings()):
        self.p = scalar_parameters(b, d, n, t)
        self.settings = settings
        self.n = n

    def gaussian(self, method, noisy):
        p = self.p
        m, v = p["mean"], p["variance"]
        if method == "EG":
            v = v/p["pi"]-(1-p["pi"])*m*m/p["pi"]**2
            m /= p["pi"]
        return m, v+(4 if noisy else 0)

    def cdf(self, method, x, noisy=True):
        p = self.p
        if method in ("G", "EG"):
            m, v = self.gaussian(method, noisy)
            value = ndtr((x-m)/np.sqrt(v)) if v > 0 else float(x >= m)
        elif method == "M0" and not noisy:
            value = gamma.cdf(x/p["growth"], p["k"], scale=p["tau"]) if x >= 0 else 0.
        else:
            low, high = amplitude_limits(p["k"], p["tau"], self.settings)
            constant = gammaln(p["k"])+p["k"]*np.log(p["tau"])
            residual = p["B"]*p["growth"] if method == "M" else 0.
            def integrand(u):
                w = np.exp(u)
                sd = np.sqrt((4 if noisy else 0)+w*residual)
                return ndtr((x-w*p["growth"])/sd)*np.exp(p["k"]*u-w/p["tau"]-constant)
            # Retain a coarse partition plus the sharp measurement transition.
            points = list(np.linspace(low, high, 25))
            if x > 0:
                center = np.log(x/p["growth"])
                width = np.sqrt((4 if noisy else 0)+x*p["B"]*(method == "M"))/x
                points.extend(np.clip(center+np.array([-8, -2, 0, 2, 8])*width, low, high))
            points = np.unique(points)
            value = 0.
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                for a, b in zip(points[:-1], points[1:]):
                    val, error = quad(integrand, a, b, epsabs=self.settings.atol/len(points),
                                      epsrel=self.settings.rtol, limit=150)
                    value += val
        if method == "G":
            return float(value)
        empty = ndtr(x/2) if noisy else float(x >= 0)
        return float((1-p["pi"])*empty+p["pi"]*value)

    def logpdf(self, method, x, reference=False):
        p = self.p
        if method in ("G", "EG"):
            m, v = self.gaussian(method, True)
            living = norm.logpdf(x, m, np.sqrt(v))
        else:
            h = np.array([p["growth"], 0.])
            T = np.diag([p["growth"]*p["B"] if method == "M" else 0., 0.])
            y = np.array([x, 0.])
            low, high = amplitude_limits(p["k"], p["tau"], self.settings)
            if reference:
                living = adaptive_reference(y, h, T, 4., p["k"], p["tau"], self.settings)
            else:
                s = self.settings
                living, error, _, failed = integrate_one(y, h, T, 4., p["k"], p["tau"],
                    gammaln(p["k"])+p["k"]*np.log(p["tau"]), low, high, s.rtol, s.atol, s.max_depth)
                if failed or error > 10*s.rtol or not np.isfinite(living):
                    raise ArithmeticError("Unresolved scalar density integral")
            living += .5*np.log(8*np.pi)
        if method == "G" or p["pi"] == 1:
            return float(living)
        return float(np.logaddexp(np.log1p(-p["pi"])+norm.logpdf(x, 0, 2), np.log(p["pi"])+living))

    def quantile(self, method, u, noisy=True, xtol=1e-8):
        if not noisy and method != "G":
            right = self.cdf(method, 0., False)
            left = right-(1-self.p["pi"])
            if left <= u <= right:
                return 0.
        return inverse_continuous(lambda x: self.cdf(method, x, noisy), u, self.p["mean"],
                                  np.sqrt(self.p["variance"]+4*noisy), xtol)


def comparisons(config):
    from time import perf_counter
    start = perf_counter()
    quantiles, densities, refinements, moments, validations = [], [], [], [], []
    for d in (0., .1):
        for n in (1, 5):
            for t in (.5, 2., 5.):
                key = dict(b=.8, d=d, n=n, time=t)
                exact = CountReference(.8, d, n, t)
                tighter = CountReference(.8, d, n, t, 1e-14)
                base = ScalarApproximation(.8, d, n, t, Settings(**config["integration"]))
                tight = ScalarApproximation(.8, d, n, t, Settings(**config["tight_integration"]))
                p = exact.parameters
                # A separate finite mixture computes positive-amplitude moments.
                ks = np.arange(1, n+1)
                weights = binom.pmf(ks, n, 1-d/.8)/p["pi_inf"]
                scale = .8/(.8-d)
                ap = float(weights @ (ks*scale))
                vp = float(weights @ (ks*(ks+1)*scale**2)-ap*ap)
                validations.append(dict(**key, mass_error=abs(exact.mass.sum()+exact.omitted-1),
                    mean_error=abs(exact.mass @ exact.counts-p["mean"]),
                    variance_error=abs(exact.mass @ (exact.counts-p["mean"])**2-p["variance"]),
                    positive_moment_error=max(abs(ap-p["ap"]),abs(vp-p["vp"])),
                    omitted=exact.omitted, tight_omitted=tighter.omitted, count_limit=int(exact.counts[-1])))
                for noisy in (False, True):
                    max_change = {m: 0. for m in ("exact", *METHODS)}
                    for u in PROBABILITIES:
                        original = exact.quantile(u, noisy)
                        refined = tighter.quantile(u, noisy, 1e-10)
                        max_change["exact"] = max(max_change["exact"], abs(original-refined))
                        for method in METHODS:
                            value = base.quantile(method, u, noisy)
                            refined_value = tight.quantile(method, u, noisy, 1e-10)
                            max_change[method] = max(max_change[method], abs(value-refined_value))
                            quantiles.append(dict(**key, noisy=noisy, probability=u, method=method,
                                exact=original, approximate=value, tight_exact=refined, tight_approximate=refined_value))
                    refinements.append(dict(**key, calculation="noisy_quantiles" if noisy else "population_quantiles",
                                            maximum_changes=max_change, passed=max(max_change.values()) < 1e-5))
                xs = np.unique([-2., 0., 2., *[exact.quantile(u) for u in (.01, .1, .5, .9, .99)]])
                for x in xs:
                    reference_log = exact.logpdf(x)
                    tight_reference = tighter.logpdf(x)
                    for method in METHODS:
                        logp = base.logpdf(method, x)
                        logtight = tight.logpdf(method, x)
                        independent = tight.logpdf(method, x, reference=True)
                        densities.append(dict(**key, x=x, method=method, exact_density=np.exp(reference_log),
                            approximate_density=np.exp(logp), log_difference=logp-reference_log,
                            reference_log_change=tight_reference-reference_log, integration_log_change=logtight-logp,
                            independent_log_change=independent-logtight))
                print(f"one-type d={d} n={n} t={t}: comparisons saved in memory", flush=True)
    for t in (.5, 2., 5.):
        for w in (.1, 1., 3.):
            exact_v = w*np.expm1(.8*t)
            moments.append(dict(time=t, w=w, exact_mean=1+exact_v, exact_variance=exact_v,
                approximate_mean=w*np.exp(.8*t), approximate_variance=w*np.exp(.8*t),
                mean_error=w-1, variance_error=w, scaled_mean_error=(w-1)/np.sqrt(exact_v+4),
                relative_variance_error=1/np.expm1(.8*t)))
    max_density_change = max(abs(r[k]) for r in densities for k in
                            ("reference_log_change", "integration_log_change", "independent_log_change"))
    passed = (all(r["passed"] for r in refinements) and max_density_change < 1e-3
              and max(r["positive_moment_error"] for r in validations) < 1e-10
              and max(r["mass_error"] for r in validations) < 1e-12)
    return dict(quantiles=quantiles, densities=densities, conditional_moments=moments, refinements=refinements,
                reference_checks=validations, max_log_density_numerical_change=max_density_change,
                probabilities=PROBABILITIES, passed=passed, seconds=perf_counter()-start)
