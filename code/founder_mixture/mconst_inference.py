"""Constant living-component residual variance: bounded searches and profiles.

The variance cap is computational only. Every fitted or constrained optimum
must exclude larger variances with an analytic likelihood bound, checked with
both production and tighter integration at the feasible candidate.
"""

from itertools import product
from time import perf_counter
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from .likelihood import Likelihood, Settings
from .inference import select_best
from .mconst_checkpoint import operation


def variance_limit(data, variance=4.):
    return max(variance, float(np.mean(np.sum(data['y']**2, axis=1))/2))


def tail_bound(data, cap, variance=4.):
    empty = -np.log(2*np.pi*variance)-np.sum(data['y']**2, axis=1)/(2*variance)
    return float(np.maximum(empty, -np.log(2*np.pi*(variance+cap))).sum())


def search(objective, bounds, sizes, settings, tight=False, extra=(), checkpoint=None, prefix=''):
    """Refine grid minima and every closed face, edge, and corner.

    The objective is minimized. All full-grid values are reused on faces;
    independent searches on faces allow a genuine boundary optimum.
    """
    bounds = np.asarray(bounds, float)
    axes = [np.linspace(a, b, size) for (a, b), size in zip(bounds, sizes)]
    shape = tuple(sizes)
    def grid():
        values = np.empty(shape)
        for idx in np.ndindex(shape):
            values[idx] = objective(np.array([axes[j][i] for j, i in enumerate(idx)]))
        return values.tolist()
    values = np.asarray(operation(checkpoint, prefix+'grid', grid))
    results = []
    # 0 = free, -1/+1 = lower/upper boundary. This includes corners.
    for state in product((0, -1, 1), repeat=len(bounds)):
        free = [j for j, s in enumerate(state) if s == 0]
        fixed = np.array([bounds[j, int(s > 0)] if s else 0. for j, s in enumerate(state)])
        if not free:
            results.append(operation(checkpoint, prefix+f'corner:{state}', lambda: dict(
                theta=fixed.tolist(), objective=float(objective(fixed)), success=True, message='corner', face=list(state))))
            continue
        face_axes = [axes[j] for j in free]
        face_values = values[tuple(slice(None) if s == 0 else (0 if s < 0 else -1) for s in state)]
        local = []
        for idx in np.ndindex(face_values.shape):
            val = face_values[idx]
            neighbors = []
            for j in range(len(idx)):
                for step in (-1, 1):
                    other = list(idx)
                    other[j] += step
                    if 0 <= other[j] < face_values.shape[j]:
                        neighbors.append(face_values[tuple(other)])
            if all(val <= other for other in neighbors):
                local.append(idx)
        top = settings['starts'] if len(free) == len(bounds) else 1
        local.extend(np.unravel_index(i, face_values.shape) for i in np.argsort(face_values, axis=None)[:top])
        starts = [np.array([face_axes[j][i] for j, i in enumerate(idx)]) for idx in dict.fromkeys(local)]
        if len(free) == len(bounds):
            starts.extend(np.clip(np.asarray(x, float), bounds[:, 0], bounds[:, 1]) for x in extra)
        free_bounds = bounds[free]
        def expand(x):
            theta = fixed.copy()
            theta[free] = x
            return theta
        def fun(x):
            return objective(expand(x))
        if len(free) == 1:
            # Search every sampled one-dimensional local bracket. Endpoints
            # are also retained by the explicit corner enumeration below.
            for idx in dict.fromkeys(local):
                j=idx[0]
                lo,hi=face_axes[0][max(0,j-1)],face_axes[0][min(len(face_axes[0])-1,j+1)]
                def scalar():
                    result=minimize_scalar(lambda x:fun(np.array([x])),bounds=(lo,hi),method='bounded',
                                           options=dict(xatol=1e-10 if tight else 1e-9))
                    return dict(theta=expand([result.x]).tolist(),objective=float(result.fun),
                        success=bool(result.success),message=str(result.message),face=list(state),
                        start=expand([face_axes[0][j]]).tolist(),iterations=int(result.nfev),method='bounded scalar')
                results.append(operation(checkpoint, prefix+f'face:{state}:bracket:{j}', scalar))
            continue
        for index, start in enumerate(starts):
            def optimize():
                first = minimize(fun, start, method='L-BFGS-B', bounds=free_bounds,
                                 options={k: settings[k] for k in ('ftol', 'gtol', 'maxiter')})
                return dict(theta=expand(first.x).tolist(), objective=float(first.fun),
                    success=bool(first.success), message=str(first.message), face=list(state),
                    start=expand(start).tolist(), iterations=int(first.nit), method='L-BFGS-B')
            results.append(operation(checkpoint, prefix+f'face:{state}:start:{index}', optimize))
    # The existing fitting procedure performs one final refinement after its
    # multistart and boundary searches. Repeating it after every start can
    # repeatedly polish the same optimum without adding starting points.
    preliminary=select_best(results)
    def polish():
        polished=minimize(objective,preliminary['theta'],method='Nelder-Mead',bounds=bounds,
                          options=dict(xatol=1e-10 if tight else 1e-9,
                                       fatol=1e-9 if tight else 1e-8,maxiter=1000))
        return dict(theta=polished.x.tolist(),objective=float(polished.fun),success=bool(polished.success),
            message=str(polished.message),face=[0]*len(bounds),start=preliminary['theta'],
            iterations=int(polished.nit),method='final Nelder-Mead')
    results.append(operation(checkpoint, prefix+'final Nelder-Mead', polish))
    best = select_best(results)
    return dict(best=best, searches=results, grid_shape=list(shape),
                grid_min=float(values.min()), start_objective_range=float(np.ptp([r['objective'] for r in results])),
                failed_starts=sum(not r['success'] for r in results))


class Problem:
    def __init__(self, data, config, tight=False):
        # Only observations and known information are retained by inference.
        self.data = {k: data[k] for k in ('y', 'time', 'z0', 'cell')}
        self.config = {k: v for k, v in config.items() if k != 'truth'}
        self.tight = tight
        self.settings = config['tight_optimization' if tight else 'optimization']
        cfg = dict(self.config, extinction=config['tight_extinction']) if tight else self.config
        self.likelihood = Likelihood(cfg, Settings(**config['tight_integration' if tight else 'integration']))
        self.validation = Likelihood(dict(self.config, extinction=config['tight_extinction']),
                                     Settings(**config['tight_integration']))
        self.objectives = {}
        self.objective_count = 0
        self.checkpoint = None
        self.saved_objectives = {}
        self.cap = variance_limit(self.data, config['measurement_variance'])

    def objective(self, theta):
        key = tuple(float(x) for x in theta)
        if key not in self.objectives:
            self.objectives[key] = -self.likelihood.loglik('Mconst', theta, self.data)
            self.objective_count += 1
        if self.checkpoint is not None:
            self.saved_objectives[key] = self.objectives[key]
        return self.objectives[key]

    def solve(self, fixed=None, extra=()):
        fixed = fixed or {}
        free = [i for i in range(3) if i not in fixed]
        scope = None
        if self.checkpoint is not None:
            key = 'fit' if not fixed else 'profile:'+repr(sorted(fixed.items()))
            scope = self.checkpoint.scope(key, dict(fixed={str(k): v for k,v in fixed.items()},
                                                   extra=[list(x) for x in extra], initial_cap=self.cap,
                                                   profile_context=getattr(self,'profile_context',None)))
            self.saved_objectives = {tuple(t): v for t,v in scope.state['objectives']}
            if 'timings' in scope.state:
                self.likelihood.timing = dict(scope.state['timings'])
                self.validation.timing = dict(scope.state['validation_timings'])
                self.objective_count = scope.state['objective_count']
            if scope.state['result'] is not None:
                self.checkpoint.reused_steps += len(scope.state['steps'])
                self.cap = scope.state['result']['variance_cap']
                return scope.state['result']
            self.objectives.update(self.saved_objectives)
            def capture(state):
                state.update(objectives=[[list(k),v] for k,v in self.saved_objectives.items()],
                             timings=dict(self.likelihood.timing), validation_timings=dict(self.validation.timing),
                             objective_count=self.objective_count, current_cap=self.cap)
            scope.capture = capture
        history = []
        while True:
            if not np.isfinite(self.cap):
                raise ArithmeticError('Variance search exceeded floating-point range')
            bounds = np.array([*self.config['bounds'], [0., np.log1p(self.cap/4)]])
            sizes = [self.settings['grid_size']]*2+[self.config['mconst']['tight_variance_grid_size' if self.tight else 'variance_grid_size']]
            if fixed:
                sizes[:2] = [2*self.config['profiles']['nuisance_grid_size']-1 if self.tight else self.config['profiles']['nuisance_grid_size']]*2
            def expand(x):
                theta = np.empty(3)
                for j, value in fixed.items():
                    theta[j] = value
                theta[free] = x
                return theta
            result = search(lambda x: self.objective(expand(x)), bounds[free], [sizes[j] for j in free],
                            self.settings, self.tight, extra=[np.array(x)[free] for x in extra],
                            checkpoint=scope, prefix=f'expansion:{len(history)}:')
            best = result['best']
            theta = expand(best['theta'])
            value = -best['objective']
            def certificate():
                checked = self.validation.loglik('Mconst', theta, self.data)
                upper = tail_bound(self.data, self.cap)
                certified = upper < min(value, checked)-self.config['mconst']['variance_bound_margin']
                return dict(cap=self.cap, upper_bound=upper, loglik=value, tight_loglik=checked,
                            certified=bool(certified), estimate=theta.tolist(), **result)
            history.append(operation(scope, f'expansion:{len(history)}:certificate', certificate))
            checked, certified = history[-1]['tight_loglik'], history[-1]['certified']
            if abs(value-checked) >= self.config['refinement']['dataset_loglik_target']:
                raise ArithmeticError('Integration changes feasible likelihood by at least 0.05')
            if certified:
                if not best['success']:
                    raise ArithmeticError('Best constrained search did not converge')
                answer = dict(theta=theta.tolist(), loglik=value, variance_cap=self.cap, expansions=history)
                return scope.finish(answer) if scope else answer
            self.cap *= self.config['mconst']['variance_expansion_factor']
            if scope:
                scope.state['expansion_history'] = history
                scope.save()


def fit(data, config, tight=False, m0=None, extra=(), checkpoint=None):
    start = perf_counter()
    problem = Problem(data, config, tight)
    problem.checkpoint = checkpoint
    try:
        seeds = list(extra)
        if m0 and m0['success']:
            seeds.append([*m0['estimate'], 0.])
            nested = -problem.objective(seeds[-1])
            if abs(nested-m0['loglik']) > config['refinement']['dataset_loglik_target']:
                raise ArithmeticError('Retained M0 maximum is numerically incompatible')
        result = problem.solve(extra=seeds)
        theta = np.array(result['theta'])
        if m0 and m0['success'] and result['loglik'] < nested-1e-4:
            raise ArithmeticError('Fit is below feasible M0 maximum')
        tick = perf_counter()
        warm = problem.likelihood.loglik('Mconst', theta, problem.data)
        warm_seconds = perf_counter()-tick
        bounds = np.asarray(config['bounds'])
        return dict(method='Mconst', success=True, tight=tight, estimate=theta[:2].tolist(),
            u=float(theta[2]), residual_variance=float(4*np.expm1(theta[2])), variance_boundary=bool(theta[2]<1e-7),
            boundary=np.any(np.isclose(theta[:2,None], bounds, atol=1e-5, rtol=0), axis=1).tolist(),
            seconds=perf_counter()-start, timings=problem.likelihood.timing, validation_timings=problem.validation.timing,
            warm_loglik=warm, warm_loglik_seconds=warm_seconds, unique_objectives=problem.objective_count,
            failed_starts=sum(h['failed_starts'] for h in result['expansions']), **result)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return dict(method='Mconst', success=False, tight=tight, estimate=None, loglik=None,
                    reason=str(error), seconds=perf_counter()-start, timings=problem.likelihood.timing)


def profile_sets(data, config, fitted, tight=False, checkpoint=None):
    start = perf_counter()
    if not fitted['success']:
        return dict(success=False, reason='Unresolved point estimate', seconds=0., rates=[])
    problem = Problem(data, config, tight)
    problem.checkpoint = checkpoint
    problem.cap = max(problem.cap, fitted['variance_cap'])
    options = config['profiles']
    size = 2*options['grid_size']-1 if tight else options['grid_size']
    xtol = options['xtol']/(10 if tight else 1)
    threshold = fitted['loglik']-options['cutoff']/2
    outputs = []
    evaluated = {}
    try:
        for interest in range(2):
            memo = {}
            problem.profile_context = dict(rate=interest, phase='grid', threshold=threshold)
            def profile(x):
                x = float(x)
                if x not in memo:
                    seeds = [fitted['theta']]
                    if memo:
                        seeds.append(memo[min(memo, key=lambda y: abs(y-x))]['theta'])
                    memo[x] = problem.solve(fixed={interest: x}, extra=seeds)
                    evaluated[f'{interest}:{x:.17g}'] = memo[x]
                    if memo[x]['loglik'] > fitted['loglik']+1e-4:
                        raise ArithmeticError('Profile exceeds the fitted maximum; refit required')
                return memo[x]['loglik']
            bounds = config['bounds'][interest]
            xs = np.unique(np.r_[np.linspace(*bounds, size), fitted['estimate'][interest]])
            ys = np.array([profile(x) for x in xs])
            roots = []
            for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
                if (ya-threshold)*(yb-threshold) < 0:
                    problem.profile_context = dict(rate=interest, phase='crossing', threshold=threshold,
                                                   bracket=[float(a),float(b)], completed_roots=list(roots))
                    roots.append(brentq(lambda x: profile(x)-threshold, a, b, xtol=xtol))
            roots.extend(xs[np.isclose(ys, threshold, atol=1e-10, rtol=0)].tolist())
            cuts = np.unique(np.r_[bounds[0], roots, bounds[1]])
            problem.profile_context = dict(rate=interest, phase='interval classification', threshold=threshold,
                                           cuts=cuts.tolist(), completed_roots=list(roots))
            intervals = [[float(a), float(b)] for a, b in zip(cuts[:-1], cuts[1:]) if profile((a+b)/2) >= threshold]
            if not intervals:
                raise ArithmeticError('No resolved interval around the fitted estimate')
            outputs.append(dict(rate=interest, intervals=intervals, length=sum(b-a for a,b in intervals),
                disconnected=len(intervals)>1, flat=float(np.ptp(ys))<options['cutoff']/2,
                bound_truncated=[profile(b)>=threshold for b in bounds], x=sorted(memo),
                loglik=[memo[x]['loglik'] for x in sorted(memo)],
                nuisance_estimates=[[memo[x]['theta'][1-interest], memo[x]['theta'][2]] for x in sorted(memo)],
                threshold=threshold, crossing_tolerance=xtol, grid_size=size,
                nuisance_grid_size=2*options['nuisance_grid_size']-1 if tight else options['nuisance_grid_size']))
            if checkpoint:
                checkpoint.scope(f'rate:{interest}', dict(threshold=threshold, bounds=bounds)).finish(outputs[-1])
        return dict(success=True, method='Mconst', tight=tight, rates=outputs, evaluated=evaluated,
                    seconds=perf_counter()-start, timings=problem.likelihood.timing,
                    validation_timings=problem.validation.timing, unique_objectives=problem.objective_count)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return dict(success=False, method='Mconst', tight=tight, reason=str(error), rates=outputs,
                    evaluated=evaluated, seconds=perf_counter()-start, timings=problem.likelihood.timing)
