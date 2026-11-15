"""Common multistart bounded fits and inspected profile-likelihood sets."""

from time import perf_counter
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from .likelihood import Likelihood, Settings


def select_best(results, tolerance=1e-7):
    """Prefer a converged solution in an objective tie at numerical precision."""
    best = min(results, key=lambda r: r["objective"])
    successful = [r for r in results if r["success"]]
    if successful:
        converged = min(successful, key=lambda r: r["objective"])
        if converged["objective"] <= best["objective"]+tolerance:
            return converged
    return best


def fit(method, data, config, tight=False):
    if tight and "tight_extinction" in config:
        config = dict(config, extinction=config["tight_extinction"])
    settings = config["tight_optimization" if tight else "optimization"]
    likelihood = Likelihood(config, Settings(**config["tight_integration" if tight else "integration"]))
    bounds = np.asarray(config["bounds"])
    start = perf_counter()
    failures = []
    objectives = {}

    def objective(theta):
        key = tuple(float(x) for x in theta)
        if key not in objectives:
            objectives[key] = -likelihood.loglik(method, theta, data)
        return objectives[key]

    grid = np.array(np.meshgrid(*[np.linspace(a, b, settings["grid_size"]) for a, b in bounds])).reshape(2, -1).T
    try:
        values = np.array([objective(theta) for theta in grid])
        grid_seconds = perf_counter()-start
        starts = grid[np.argsort(values)[:settings["starts"]]]
        results = []
        for theta in starts:
            result = minimize(objective, theta, method="L-BFGS-B", bounds=bounds,
                              options={k: settings[k] for k in ["ftol", "gtol", "maxiter"]})
            results.append(dict(theta=result.x.tolist(), objective=float(result.fun),
                                success=bool(result.success), message=str(result.message),
                                start=theta.tolist(), iterations=int(result.nit)))
        # Explicitly search each closed edge, including its best sampled local
        # peaks and both endpoints. This also verifies a reported boundary MLE.
        edges = []
        for fixed in range(2):
            free = 1-fixed
            xs = np.linspace(*bounds[free], settings["grid_size"])
            for bound in bounds[fixed]:
                def edge(x):
                    theta = np.empty(2)
                    theta[fixed], theta[free] = bound, x
                    return objective(theta)
                edge_values = np.array([edge(x) for x in xs])
                for j in range(1, len(xs)-1):
                    if edge_values[j] <= min(edge_values[j-1], edge_values[j+1]):
                        result = minimize_scalar(edge, bounds=(xs[j-1], xs[j+1]), method="bounded",
                                                 options={"xatol": 1e-9 if tight else 1e-7})
                        theta = np.empty(2)
                        theta[fixed], theta[free] = bound, result.x
                        edges.append(dict(theta=theta.tolist(), objective=float(result.fun),
                                          success=bool(result.success), message=str(result.message)))
                for j in (0, len(xs)-1):
                    theta = np.empty(2)
                    theta[fixed], theta[free] = bound, xs[j]
                    edges.append(dict(theta=theta.tolist(), objective=float(edge_values[j]), success=True, message="corner"))
        preliminary = select_best(results+edges)
        # Very sharp likelihoods can stop L-BFGS-B through its relative
        # objective tolerance or numerical gradient error while a profile
        # still finds a higher value. Polish the best candidate for every
        # method using absolute objective and parameter tolerances.
        polished = minimize(objective, preliminary["theta"], method="Nelder-Mead", bounds=bounds,
                            options={"xatol": 1e-10 if tight else 1e-9,
                                     "fatol": 1e-9 if tight else 1e-8,
                                     "maxiter": 1000})
        polish = dict(theta=polished.x.tolist(), objective=float(polished.fun),
                      success=bool(polished.success), message=str(polished.message),
                      start=preliminary["theta"], iterations=int(polished.nit),
                      evaluations=int(polished.nfev), method="Nelder-Mead")
        best = select_best(results+edges+[polish])
        # A materially better failed run is not hidden by choosing a worse
        # successful run. Failed individual starts are retained in the artifact.
        success = best["success"] and any(r["success"] for r in results)
        estimate = np.array(best["theta"])
        elapsed = perf_counter()-start
        timings = dict(likelihood.timing)
        tick = perf_counter()
        warm_value = likelihood.loglik(method, estimate, data)
        warm_seconds = perf_counter()-tick
        return dict(method=method, tight=tight, success=bool(success), estimate=estimate.tolist(),
                    loglik=-best["objective"], boundary=np.any(np.isclose(estimate[:, None], bounds, atol=1e-5, rtol=0), axis=1).tolist(),
                    starts=results, edges=edges, polish=polish,
                    start_objective_range=float(np.ptp([r["objective"] for r in results])),
                    failed_starts=sum(not r["success"] for r in results), seconds=elapsed,
                    grid_seconds=grid_seconds, warm_loglik_seconds=warm_seconds,
                    warm_loglik=warm_value, timings=timings, unique_objectives=len(objectives), failures=failures)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return dict(method=method, tight=tight, success=False, estimate=None, loglik=None,
                    seconds=perf_counter()-start, timings=likelihood.timing,
                    failures=[f"{type(error).__name__}: {error}"])


def profile_sets(method, data, config, fitted, tight=False):
    """Return unions of intervals; coverage uses set membership, not their span.

    Nuisance maxima are searched from every sampled local peak. Finite grids
    cannot rule out arbitrarily narrow components; refinement doubles both
    grids. The stored profiles allow inspection of this limitation.
    """
    start = perf_counter()
    if not fitted["success"]:
        return dict(success=False, reason="Unresolved point estimate", seconds=0.)
    if tight and "tight_extinction" in config:
        config = dict(config, extinction=config["tight_extinction"])
    likelihood = Likelihood(config, Settings(**config["tight_integration" if tight else "integration"]))
    options = config["profiles"]
    bounds = np.asarray(config["bounds"])
    grid_size = 2*options["grid_size"]-1 if tight else options["grid_size"]
    nuisance_size = 2*options["nuisance_grid_size"]-1 if tight else options["nuisance_grid_size"]
    xtol = options["xtol"]/(10 if tight else 1)
    threshold = fitted["loglik"]-options["cutoff"]/2
    outputs = []
    try:
        for interest in range(2):
            nuisance = 1-interest
            memo = {}
            nuisance_estimates = {}
            nuisance_grid = np.linspace(*bounds[nuisance], nuisance_size)
            def profile(x):
                if x in memo:
                    return memo[x]
                def objective(v):
                    theta = np.empty(2)
                    theta[interest], theta[nuisance] = x, v
                    return -likelihood.loglik(method, theta, data)
                vals = np.array([objective(v) for v in nuisance_grid])
                candidates = [(float(value), float(v)) for value, v in zip(vals, nuisance_grid)]
                for j in range(1, nuisance_size-1):
                    if vals[j] <= min(vals[j-1], vals[j+1]):
                        result = minimize_scalar(objective, bounds=(nuisance_grid[j-1], nuisance_grid[j+1]),
                                                 method="bounded", options={"xatol": xtol/10})
                        if not result.success:
                            raise ArithmeticError("Nuisance profile optimization failed")
                        candidates.append((float(result.fun), float(result.x)))
                best_value, best_nuisance = min(candidates)
                value = -best_value
                if value > fitted["loglik"]+1e-4:
                    raise ArithmeticError("Profile exceeds the fitted maximum; refit required")
                memo[x] = value
                nuisance_estimates[x] = best_nuisance
                return value
            xs = np.unique(np.r_[np.linspace(*bounds[interest], grid_size), fitted["estimate"][interest]])
            ys = np.array([profile(float(x)) for x in xs])
            roots = []
            for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
                if (ya-threshold)*(yb-threshold) < 0:
                    roots.append(brentq(lambda x: profile(x)-threshold, a, b, xtol=xtol))
            roots.extend(xs[np.isclose(ys, threshold, atol=1e-10, rtol=0)].tolist())
            cuts = np.unique(np.r_[bounds[interest, 0], roots, bounds[interest, 1]])
            intervals = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                if profile((a+b)/2) >= threshold:
                    intervals.append([float(a), float(b)])
            if not intervals:
                raise ArithmeticError("Profile set does not contain a resolved interval around the estimate")
            outputs.append(dict(rate=interest, intervals=intervals, length=sum(b-a for a, b in intervals),
                                disconnected=len(intervals)>1, flat=float(np.ptp(ys))<options["cutoff"]/2,
                                bound_truncated=[profile(float(b)) >= threshold for b in bounds[interest]],
                                x=sorted(memo), loglik=[memo[x] for x in sorted(memo)],
                                nuisance_estimates=[nuisance_estimates[x] for x in sorted(memo)],
                                threshold=threshold, crossing_tolerance=xtol,
                                grid_size=grid_size, nuisance_grid_size=nuisance_size))
        return dict(success=True, rates=outputs, seconds=perf_counter()-start, timings=getattr(likelihood, "timing", {}))
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return dict(success=False, reason=str(error), rates=outputs, seconds=perf_counter()-start, timings=getattr(likelihood, "timing", {}))
