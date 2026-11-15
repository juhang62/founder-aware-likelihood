"""Opt-in efficiency experiments; never launches or replaces study results.

Compare the current extinction solver with SciPy's compiled DOP853 driver.
The driver uses the same equations, tolerances, and fixed-point calculation,
but its adaptive steps differ. It must pass numerical comparisons before use.
All patches are local to this single-process benchmark and restored afterward.
"""

import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
from itertools import product
import os
from pathlib import Path
import resource
import signal
import sys
from time import perf_counter
import warnings

import numpy as np
from numba import njit
from scipy.integrate import ode

from . import ROOT, model, mconst_inference
from .extinction import ExtinctionSettings, extinction, rhs, smallest_fixed_point
from .likelihood import Likelihood, Settings
from .storage import digest, provenance, read_json, write_json


@njit(cache=True)
def compiled_rhs(q, birth, death, switching):
    return death-(birth+death+switching)*q+birth*q*q+switching*q[::-1]


def compiled_driver(birth, death, switching, times, settings, use_compiled_rhs=True):
    """Experimental implementation, with the original probability checks."""
    b, d, s = (np.asarray(x, float) for x in (birth, death, switching))
    rho, iterations, converged = smallest_fixed_point(
        b, d, s, settings.iteration_tolerance, settings.max_iterations)
    residual = float(np.max(np.abs(rhs(rho, b, d, s))))
    if not converged or residual > 10*settings.iteration_tolerance*max(1., np.max(b+d+s)):
        raise ArithmeticError('Eventual-extinction iteration failed')
    ordered = np.unique(times)
    if len(ordered) == 0 or ordered[0] < 0:
        raise ValueError('Nonnegative observation times are required')
    values = np.zeros((2, len(ordered)))
    if np.any(d) and ordered[-1] > 0:
        fun = compiled_rhs if use_compiled_rhs else rhs
        solver = ode(lambda t, q: fun(q, b, d, s)).set_integrator(
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


@lru_cache(maxsize=8192)
def experimental_extinction(birth, death, switching, times, settings=ExtinctionSettings()):
    return compiled_driver(birth, death, switching, times, settings)


@contextmanager
def backend(name):
    original = model.extinction
    model.candidate.cache_clear()
    extinction.cache_clear()
    experimental_extinction.cache_clear()
    model.extinction = extinction if name == 'original' else experimental_extinction
    try:
        yield
    finally:
        model.extinction = original
        model.candidate.cache_clear()
        extinction.cache_clear()
        experimental_extinction.cache_clear()


def cache_statistics():
    return dict(candidate=model.candidate.cache_info()._asdict(),
                extinction=model.extinction.cache_info()._asdict())


def dataset(n):
    path = ROOT/f'data/founder-mixture-birth-death/observations/inference/n{n}/rep000.npz'
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ('y', 'time', 'z0', 'cell')}


def saved(kind, n, tight=False):
    folder = kind+('-refined' if tight else '')
    return read_json(ROOT/f'data/founder-mixture-birth-death/mconst/{folder}/n{n}/rep000.json')['result']


def old_fit(n, method='M0'):
    return read_json(ROOT/f'data/founder-mixture-birth-death/fits/inference/n{n}/rep000-{method}.json')['result']


def parameter_points(config):
    points = list(product(*[np.linspace(a, b, 13) for a, b in config['bounds']]))
    points.append(tuple(config['truth']))
    for n in config['founders']:
        points.extend(tuple(old_fit(n, m)['estimate']) for m in config['methods'])
        for tight in (False, True):
            points.append(tuple(saved('fits', n, tight)['estimate']))
        profile = saved('profiles', n)
        for i, rate in enumerate(profile['rates']):
            values = [v for key, v in profile['evaluated'].items() if int(key.split(':')[0]) == i]
            for interval in rate['intervals']:
                for x in interval:
                    points.append(tuple(min(values, key=lambda v: abs(v['theta'][i]-x))['theta'][:2]))
    return sorted(set(points))


def kernels(config):
    points = parameter_points(config)
    args = [(tuple(theta), tuple(config['death']), tuple(config['switching']), tuple(config['times']))
            for theta in points]
    tick = perf_counter()
    experimental_extinction(*args[0], ExtinctionSettings(**config['extinction']))
    startup = perf_counter()-tick
    rows = []
    for tight in (False, True):
        settings = ExtinctionSettings(**config['tight_extinction' if tight else 'extinction'])
        reference = [extinction.__wrapped__(*a, settings) for a in args]
        variants = [('original', lambda a: extinction.__wrapped__(*a, settings)),
                    ('compiled_driver_python_rhs', lambda a: compiled_driver(*a, settings, False)),
                    ('compiled_driver', lambda a: compiled_driver(*a, settings))]
        for repeat in range(3):
            for name, fun in variants if repeat % 2 == 0 else variants[::-1]:
                tick = perf_counter()
                values = [fun(a) for a in args]
                elapsed = perf_counter()-tick
                q_error = max(float(np.max(abs(v['q'][t]-r['q'][t])))
                              for v, r in zip(values, reference) for t in config['times'])
                rho_error = max(float(np.max(abs(v['rho']-r['rho']))) for v, r in zip(values, reference))
                rows.append(dict(backend=name, tight=tight, repeat=repeat, seconds=elapsed,
                                 pairs=len(args), max_q_change=q_error, max_rho_change=rho_error))
    special = []
    for death in ((0., 0.), tuple(config['death'])):
        a = ((.8, .5), death, tuple(config['switching']), (11., 0., 9., 9.))
        settings = ExtinctionSettings(**config['tight_extinction'])
        original = extinction.__wrapped__(*a, settings)
        revised = compiled_driver(*a, settings)
        error = max(float(np.max(abs(original['q'][t]-revised['q'][t]))) for t in set(a[-1]))
        special.append(dict(death=death, times=a[-1], max_q_change=error))
    return dict(rows=rows, points=points, startup_seconds=startup, special_cases=special,
                passed=all(r['max_q_change'] < (1e-11 if r['tight'] else 1e-9)
                           and r['max_rho_change'] == 0 for r in rows)
                       and all(r['max_q_change'] < 1e-11 for r in special))


def likelihoods(config):
    points = parameter_points(config)
    rows = []
    for tight in (False, True):
        cfg = dict(config, extinction=config['tight_extinction']) if tight else config
        settings = Settings(**cfg['tight_integration' if tight else 'integration'])
        for n in config['founders']:
            data = dataset(n)
            f = saved('fits', n, tight)
            variances = [0., 4., f['residual_variance'], f['variance_cap']]
            outputs = {}
            for name in ('original', 'compiled_driver'):
                with backend(name):
                    like = Likelihood(cfg, settings)
                    tick = perf_counter()
                    values = []
                    for theta in points:
                        values.extend(like.loglik(m, theta, data) for m in config['methods'])
                        values.extend(like.loglik('Mconst', [*theta, np.log1p(v/4)], data) for v in variances)
                    outputs[name] = np.array(values)
                    rows.append(dict(backend=name, tight=tight, founders=n, seconds=perf_counter()-tick,
                                     count=len(values), caches=cache_statistics(), timing=like.timing))
            error = abs(outputs['original']-outputs['compiled_driver'])
            rows[-1].update(max_dataset_loglik_change=float(error.max()), passed=bool(error.max() < 1e-6))
    return dict(rows=rows, points=points, passed=all(r.get('passed', True) for r in rows))


def refinement_checks(config):
    """Compare with tighter integration and an exact scalar extinction law."""
    rows = []
    production = ExtinctionSettings(**config['extinction'])
    tighter = ExtinctionSettings(**config['tight_extinction'])
    for theta in parameter_points(config):
        args = (tuple(theta), tuple(config['death']), tuple(config['switching']), tuple(config['times']))
        original_tight = extinction.__wrapped__(*args, tighter)
        for settings, label in ((production, 'production'), (tighter, 'tight')):
            revised = compiled_driver(*args, settings)
            change = max(float(np.max(abs(revised['q'][t]-original_tight['q'][t]))) for t in config['times'])
            rows.append(dict(check='tighter_reference', theta=theta, accuracy=label,
                             maximum_change=change, passed=change < (1e-9 if label == 'production' else 1e-11)))
    # With equal birth and death rates for both types, switching cancels in
    # the extinction equations. The one-type birth-death law is then exact.
    for birth in (.3, .7, 1.1):
        death = .2
        times = (0., 9., 11.)
        for settings, label in ((production, 'production'), (tighter, 'tight')):
            revised = compiled_driver((birth, birth), (death, death), tuple(config['switching']), times, settings)
            ratio = death/birth
            changes = [float(np.max(abs(revised['q'][t]-ratio*(-np.expm1(-(birth-death)*t))
                                       /(1-ratio*np.exp(-(birth-death)*t))))) for t in times]
            eventual_change = float(np.max(abs(revised['rho']-ratio)))
            rows.append(dict(check='exact_scalar_law', birth=birth, death=death, accuracy=label,
                             maximum_change=max(changes), eventual_change=eventual_change,
                             passed=max(changes) < (1e-9 if label == 'production' else 1e-11)
                                    and eventual_change < (1e-11 if label == 'production' else 1e-12)))
    return dict(rows=rows, passed=all(r['passed'] for r in rows))


def searches(config, output):
    """Unchanged full fits and selected nuisance searches, with fresh caches."""
    rows = []
    for tight in (False, True):
        for n in config['founders']:
            data = dataset(n)
            fitted = saved('fits', n, tight)
            profile = saved('profiles', n)
            # Complete fits; constrained searches at two saved interval ends
            # and one corner. No full profile or scientific result is written.
            tasks = [('fit', None, None)]
            for interest in range(2):
                endpoints = profile['rates'][interest]['intervals']
                x = endpoints[0][0] if interest == 0 else endpoints[-1][-1]
                tasks.append((f'rate{interest}-crossing', interest, x))
            tasks.append(('rate0-lower-bound', 0, config['bounds'][0][0]))
            for task_index, (label, interest, x) in enumerate(tasks):
                results = {}
                order = ('original', 'compiled_driver') if (task_index+int(tight)+n) % 2 else ('compiled_driver', 'original')
                for name in order:
                    with backend(name):
                        tick = perf_counter()
                        if interest is None:
                            result = mconst_inference.fit(data, config, tight, old_fit(n))
                        else:
                            problem = mconst_inference.Problem(data, config, tight)
                            problem.cap = max(problem.cap, fitted['variance_cap'])
                            result = problem.solve(fixed={interest: x}, extra=[fitted['theta']])
                            result.update(success=True, unique_objectives=len(problem.objectives),
                                          timings=problem.likelihood.timing,
                                          validation_timings=problem.validation.timing)
                        elapsed = perf_counter()-tick
                        info = dict(backend=name, tight=tight, founders=n, task=label, fixed=x,
                                    seconds=elapsed, caches=cache_statistics(), result=result,
                                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        results[name] = info
                        write_json(output/f'search-{n}-{int(tight)}-{label}-{name}.json', info)
                        print(f'{label}: n={n}, tight={tight}, {name}, {elapsed:.2f}s', flush=True)
                a, b = (results[name]['result'] for name in ('original', 'compiled_driver'))
                change = np.abs(np.array(a['theta'])-b['theta']) if a['success'] and b['success'] else np.full(3, np.inf)
                row = dict(founders=n, tight=tight, task=label, fixed=x,
                           original_seconds=results['original']['seconds'],
                           revised_seconds=results['compiled_driver']['seconds'],
                           loglik_change=abs(a['loglik']-b['loglik']), birth_change=float(change[:2].max()),
                           u_change=float(change[2]),
                           objective_evaluations={name: v['result']['unique_objectives'] for name, v in results.items()},
                           extinction_misses={name: v['caches']['extinction']['misses'] for name, v in results.items()})
                row['passed'] = bool(row['loglik_change'] < 1e-4 and row['birth_change'] < 1e-5 and row['u_change'] < 1e-3)
                rows.append(row)
                write_json(output/'search-progress.json', dict(rows=rows, complete=False))
    return dict(rows=rows, passed=all(r['passed'] for r in rows))


def summary(output):
    records = {name: read_json(output/f'{name}.json')
               for name in ('kernels', 'likelihoods', 'refinement_checks', 'searches')}
    rows = []
    for tight in (False, True):
        times = {name: float(np.median([r['seconds'] for r in records['kernels']['rows']
                                       if r['tight'] == tight and r['backend'] == name]))
                 for name in ('original', 'compiled_driver')}
        rows.append(dict(task='extinction, median of three batches', tight=tight,
                         parameter_pairs=len(records['kernels']['points']), **times,
                         speedup=times['original']/times['compiled_driver']))
        for fits in (True, False):
            selected = [r for r in records['searches']['rows'] if r['tight'] == tight and (r['task'] == 'fit') == fits]
            original = sum(r['original_seconds'] for r in selected)
            revised = sum(r['revised_seconds'] for r in selected)
            rows.append(dict(task='complete fits' if fits else 'selected nuisance optimizations', tight=tight,
                             paired_cases=len(selected), original=original, compiled_driver=revised,
                             speedup=original/revised,
                             individual_speedup_range=[min(r['original_seconds']/r['revised_seconds'] for r in selected),
                                                       max(r['original_seconds']/r['revised_seconds'] for r in selected)]))
    protected = read_json(output/'protected-files.json')
    changed = [name for name, expected in protected.items()
               if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected]
    allowed = ['code/founder_mixture/README.md', 'reports/founder-mixture-birth-death/README.md']
    unexpected = [name for name in changed if name not in allowed]
    likelihood_rows = records['likelihoods']['rows']
    search_rows = records['searches']['rows']
    return dict(rows=rows, passed=all(r['passed'] for r in records.values()) and not unexpected,
                validation=dict(likelihood_pairs=sum(r['count'] for r in likelihood_rows if r['backend'] == 'original'),
                    maximum_dataset_loglik_change=max(r.get('max_dataset_loglik_change', 0.) for r in likelihood_rows),
                    optimization_pairs=len(search_rows),
                    maximum_optimized_loglik_change=max(r['loglik_change'] for r in search_rows),
                    maximum_optimized_birth_change=max(r['birth_change'] for r in search_rows),
                    maximum_optimized_u_change=max(r['u_change'] for r in search_rows)),
                preservation=dict(protected_files=len(protected), changed=changed, unexpected_changes=unexpected),
                observations='Only the saved replicate-0 datasets for N0=1 and N0=50; no new data.',
                scientific_settings_changed=False, production_implementation_changed=False,
                full_study_resumed=False, complete_profile_calculations=0,
                limitations=['Each fit or nuisance optimization has one paired timing run; wall time can fluctuate.',
                             'Complete profile runtimes, profile-set topology, and all crossing locations were not retested.',
                             'Benchmark memory usage does not establish the cause of the earlier computer freeze.'],
                benchmark_peak_rss_mib=max(r['process_peak_rss_kib'] for r in records.values())/1024,
                records={name: str(output/f'{name}.json') for name in records},
                recommendation='The compiled extinction driver merits adoption after complete profile validation; retain all prescribed grids and accuracy checks.')


def protected_inventory():
    """Freeze inputs and preexisting results before the first benchmark."""
    paths = set((ROOT/'code/founder_mixture').glob('*.py'))
    paths.add(ROOT/'code/founder_mixture/README.md')
    paths.update((ROOT/'reports/founder-mixture-birth-death').rglob('*'))
    paths.update((ROOT/'configs').glob('*.json'))
    paths.update(ROOT.glob('*.md'))
    for folder in ('observations', 'fits', 'profiles', 'mconst/fits', 'mconst/fits-refined',
                   'mconst/profiles', 'mconst/profiles-refined'):
        paths.update((ROOT/'data/founder-mixture-birth-death'/folder).rglob('*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and p != Path(__file__).resolve()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('kernels', 'likelihoods', 'refinement_checks', 'searches', 'summary'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Limit each invocation to one numerical thread, 3 GiB, and 30 minutes.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        if os.environ.get(name) != '1':
            parser.error(f'Set {name}=1 before running this benchmark')
    resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    signal.alarm(1800)
    os.nice(10)
    output = Path(args.output)
    if (output/f'{args.stage}.json').exists():
        parser.error('Choose a fresh output directory; existing benchmark records are preserved')
    if not (output/'protected-files.json').exists():
        if args.stage == 'summary':
            parser.error('The summary requires the inventory recorded before the benchmarks')
        write_json(output/'protected-files.json', protected_inventory())
    extension = read_json(ROOT/'configs/founder-mixture-birth-death-mconst.json')
    if hashlib.sha256((ROOT/extension['baseline_config']).read_bytes()).hexdigest() != extension['baseline_sha256']:
        raise ValueError('The baseline configuration changed')
    config = dict(read_json(ROOT/extension['baseline_config']), mconst=extension)
    meta = provenance(config)
    meta.update(workers=1, memory_limit_gib=3, invocation_limit_seconds=1800,
                execution_command=getattr(sys, 'orig_argv', [sys.executable, *sys.argv]),
                input_hashes={str(n): digest(dataset(n)) for n in config['founders']},
                comparison='Same saved data, likelihood, grids, starts, tolerances, and cleared caches; alternate DOP853 driver only.',
                production_adoption=False)
    for filename, checksum in meta['source_sha256'].items():
        target = output/'source-versions'/f'{checksum}.py'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/filename).read_bytes())
    write_json(output/f'{args.stage}-provenance.json', meta)
    # Warm both implementations and the amplitude integration kernel before
    # paired measurements; scientific candidate/ODE caches are cleared next.
    startup = {}
    for name in (() if args.stage == 'summary' else ('original', 'compiled_driver')):
        with backend(name):
            tick = perf_counter()
            Likelihood(config).loglik('Mconst', [.7, .6, 1.], dataset(1))
            startup[name] = perf_counter()-tick
    if args.stage == 'summary':
        result = summary(output)
    elif args.stage == 'searches':
        result = searches(config, output)
    else:
        result = globals()[args.stage](config)
    result.update(startup_seconds=startup,
                  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(output/f'{args.stage}.json', result)
    print(dict(stage=args.stage, passed=result['passed'], rows=len(result['rows']),
               peak_rss_mib=result['process_peak_rss_kib']/1024), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
