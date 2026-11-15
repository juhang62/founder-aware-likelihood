"""Delivery 3: preserve the first 20 fits, measure costs, and checkpoint profiles."""

import hashlib
import ast
import importlib.util
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
import numpy as np
from .storage import Store, read_json, write_json, digest
from .jobs import run_jobs
from .model import candidate
from .likelihood import Likelihood
from .inference import fit, profile_sets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExtendedStore(Store):
    """Explicit, byte-verified compatibility exception for frozen D1/D2 evidence."""
    def __init__(self, root, config, retry=False):
        super().__init__(root, config, retry)
        baseline = self.root/'delivery3/baseline/manifest.json'
        if not baseline.exists():
            raise ValueError('Create the Delivery 2 baseline snapshot with snapshot() before Delivery 3')
        self.baseline = read_json(baseline)

    def result(self, path):
        path = str(path)
        old_hash = self.baseline['files'].get(path)
        if old_hash is not None:
            full = self.root/path
            if sha(full) != old_hash:
                raise ValueError(f'Frozen Delivery 1/2 artifact changed: {path}')
            record = read_json(full)
            if record['provenance']['config_digest'] != digest(self.config):
                raise ValueError(f'Historical configuration mismatch: {path}')
            return record['result']
        full = self.root/path
        if full.exists() and path.startswith('delivery3/'):
            record = read_json(full)
            # A cache-policy correction to the separate cost benchmark must not
            # erase numerical profiles or retained fits. Compare exact function
            # sources against a hash-verified snapshot of the prior module.
            for name in ('birth_death_extended.py',):
                key = 'code/founder_mixture/'+name
                if record['provenance']['source_sha256'].get(key) != self.provenance['source_sha256'][key]:
                    previous_hash = record['provenance']['source_sha256'].get(key)
                    previous = self.root/'delivery3/source-versions'/f'{previous_hash}.py'
                    dependencies = ['observed_inputs']
                    if path.startswith('delivery3/cost/'):
                        dependencies += ['cost_task', 'legacy_modules']
                    elif path == 'delivery3/checks.json':
                        dependencies += ['checks', 'legacy_modules']
                    elif 'profile' in path:
                        dependencies += ['profile_task', 'profile_comparison', 'refinements', 'run_profiles']
                    else:
                        dependencies += ['run_delivery']
                    def functions(source):
                        return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                    if previous.exists() and sha(previous) == previous_hash:
                        old, new = functions(previous.read_text()), functions(Path(__file__).read_text())
                        if all(old.get(function) == new.get(function) for function in dependencies):
                            return super().result(path)
                    if not self.retry:
                        raise ValueError(f'Delivery 3 source changed: {name}; inspect and use --retry')
                    archive = self.root/'attempts'/self.provenance['timestamp_utc'].replace(':', '-')/path
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    full.replace(archive)
                    write_json(archive.with_suffix('.retry.json'), dict(original_path=path,
                        changed_source=name, old_sha256=previous_hash,
                        new_sha256=self.provenance['source_sha256'][key],
                        reason='Explicit retry after a change to a computational dependency; prior attempt preserved.'))
                    return None
        return super().result(path)


def snapshot(root):
    """Freeze completed prior artifacts and their actual numerical sources once."""
    import shutil
    import subprocess
    root = Path(root)
    dest = root/'delivery3/baseline'
    if (dest/'manifest.json').exists():
        return
    for name in ('inference', 'refits', 'integration'):
        if not read_json(root/f'delivery2/{name}.json')['result']['passed']:
            raise ValueError('Complete Delivery 2 before snapshotting it')
    source_record = read_json(root/'fits/inference/n1/rep000-G.json')['provenance']
    sources = source_record['source_sha256']
    dest.mkdir(parents=True, exist_ok=True)
    legacy = dest/'legacy_code'
    legacy.mkdir(exist_ok=True)
    # The reviewed Delivery 2 commit contains the same numerical sources as
    # its pre-commit provenance. Validate bytes instead of trusting the label.
    revision = '6e38982'
    for path, expected in sources.items():
        content = subprocess.check_output(['git', 'show', f'{revision}:{path}'])
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Historical source mismatch: {path}')
        (legacy/Path(path).name).write_bytes(content)
    files = {}
    for folder in ('fits', 'observations', 'delivery2'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file():
                files[str(path.relative_to(root))] = sha(path)
    for path in root.glob('*.json'):
        files[path.name] = sha(path)
    write_json(dest/'manifest.json', dict(git_revision=revision, files=files, source_sha256=sources,
        note='Frozen prior artifacts; retained after checked numerical equivalence. Original timings are historical.'))


def observed_inputs(data, config):
    return ({key: data[key] for key in ('y', 'time', 'z0', 'cell')},
            {key: value for key, value in config.items() if key != 'truth'})


def legacy_modules(root):
    name = '_birth_death_delivery2'
    if name not in sys.modules:
        path = Path(root)/'delivery3/baseline/legacy_code/__init__.py'
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return tuple(importlib.import_module(name+'.'+part) for part in ('model', 'likelihood', 'inference'))


def fit_task(method, data, config):
    data, config = observed_inputs(data, config)
    candidate.cache_clear()
    return fit(method, data, config)


def profile_task(method, data, config, fitted, tight=False):
    data, config = observed_inputs(data, config)
    candidate.cache_clear()
    result = profile_sets(method, data, config, fitted, tight=tight)
    result.update(method=method, tight=tight, fitted_estimate=fitted['estimate'], fitted_loglik=fitted['loglik'])
    return result


def cost_task(root, method, data, config, replicate):
    """Paired actual executions, identical warm-JIT / empty-candidate-cache policy."""
    old_model, old_likelihood, old_inference = legacy_modules(root)
    data, config = observed_inputs(data, config)
    theta = [.7, .6]  # Common deterministic warmup; no generating-rate input.
    from .model import extinction
    versions = {'legacy': (old_model.candidate, old_model.extinction, old_likelihood.Likelihood, old_inference.fit),
                'revised': (candidate, extinction, Likelihood, fit)}
    startup = {}
    for label, (cache, extinction_cache, cls, fitter) in versions.items():
        cache.cache_clear()
        extinction_cache.cache_clear()
        tick = perf_counter()
        cls(config).loglik(method, theta, data)
        startup[label] = perf_counter()-tick
    results = {}
    order = ['legacy', 'revised'] if replicate % 2 == 0 else ['revised', 'legacy']
    for label in order:
        cache, extinction_cache, cls, fitter = versions[label]
        cold, warm, values = [], [], []
        for _ in range(9):
            cache.cache_clear()
            extinction_cache.cache_clear()
            assert cache.cache_info().currsize == extinction_cache.cache_info().currsize == 0
            likelihood = cls(config)
            tick = perf_counter()
            values.append(likelihood.loglik(method, theta, data))
            cold.append(perf_counter()-tick)
            tick = perf_counter()
            for repeat in range(10):
                likelihood.loglik(method, theta, data)
            warm.append((perf_counter()-tick)/10)
        cache.cache_clear()
        extinction_cache.cache_clear()
        cache_before_fit = dict(candidate=cache.cache_info()._asdict(), extinction=extinction_cache.cache_info()._asdict())
        fitted = fitter(method, data, config)
        results[label] = dict(fit=fitted, cold_seconds=cold, warm_seconds=warm, values=values,
                              cache_before_fit=cache_before_fit, cold_cache_resets=9)
    a, b = results['legacy']['fit'], results['revised']['fit']
    valid = a['success'] and b['success']
    change = float(np.max(abs(np.array(a['estimate'])-b['estimate']))) if valid else None
    llchange = abs(a['loglik']-b['loglik']) if valid else None
    controlled_profile = None
    profile_agreement = None
    if replicate == 0 and b['success']:
        # A representative profile cost also starts with BOTH caches empty;
        # throughput timings from the full production queue are kept separately.
        candidate.cache_clear()
        extinction.cache_clear()
        controlled_profile = profile_sets(method, data, config, b)
        n = int(data['z0'][0].sum())
        previous = read_json(Path(root)/f'delivery3/profiles/n{n}/rep000-{method}.json')['result']
        profile_agreement = profile_comparison(previous, controlled_profile, config)
    profile_ok = replicate != 0 or (controlled_profile and controlled_profile['success'] and all(r['passed'] for r in profile_agreement))
    return dict(success=bool(valid and change<1e-7 and llchange<1e-7 and profile_ok), versions=results,
                estimate_change=change, loglik_change=llchange, startup_seconds=startup, order=order,
                controlled_profile=controlled_profile, profile_agreement=profile_agreement,
                seconds=sum(r['fit']['seconds'] for r in results.values())+(controlled_profile or {}).get('seconds',0.),
                policy='One BLAS thread, four workers, JIT/cache load warmup recorded separately; clear BOTH candidate and extinction caches before each cold evaluation and fit; warm evaluations repeat identical parameters ten times; nine cold/warm batches. Order alternates by replicate.')


def checks(store):
    from unittest.mock import patch
    from .checks import run_unit_checks
    old_model, old_likelihood, _ = legacy_modules(store.root)
    rows = []
    for n in store.config['founders']:
        for rep in (0, 4, 19):
            data = store.dataset('inference', n, rep, store.config['cultures_per_cell'])
            observed, known = observed_inputs(data, store.config)
            parameters = [*product(*known['bounds']), (.7, .6)]
            parameters += [store.result(f'fits/inference/n{n}/rep{rep:03d}-{m}.json')['estimate'] for m in known['methods']]
            for method in known['methods']:
                a, b = old_likelihood.Likelihood(known), Likelihood(known)
                changes = [abs(a.loglik(method, theta, observed)-b.loglik(method, theta, observed)) for theta in parameters]
                rows.append(dict(founders=n, replicate=rep, method=method, parameter_count=len(parameters), maximum_change=max(changes)))
    candidate.cache_clear()
    with patch('founder_mixture.model.extinction', side_effect=AssertionError('G requested extinction')):
        Likelihood(known).loglik('G', (.7, .6), observed)
    # Check the actual cache avoids extinction until EG/M/M0 requests it.
    model = candidate((.7, .6), tuple(known['switching']), tuple(known['times']), tuple(known['death']))
    assert 'extinction' not in model.__dict__
    unit = run_unit_checks()
    original = []
    for n in store.config['founders']:
        for rep in range(20):
            data = store.dataset('inference', n, rep, store.config['cultures_per_cell'])
            obs, cfg = observed_inputs(data, store.config)
            for method in cfg['methods']:
                f = store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')
                actual = Likelihood(cfg).loglik(method, f['estimate'], obs)
                original.append(dict(founders=n, replicate=rep, method=method, absolute_change=abs(actual-f['loglik'])))
    return dict(passed=unit['passed'] and max(r['maximum_change'] for r in rows)<1e-9 and max(r['absolute_change'] for r in original)<1e-9,
                unit=unit, comparisons=rows, retained_maxima=original, gaussian_extinction_not_called=True,
                source_policy='Only lazy extinction and profile tolerance/diagnostic changes; unchanged densities verified on saved observations. All original results retained byte-for-byte.')


def profile_comparison(base, tight, config):
    rows = []
    for rate in range(2):
        row = dict(rate=rate, passed=False)
        if base and tight and base['success'] and tight['success']:
            a, b = base['rates'][rate], tight['rates'][rate]
            topology = len(a['intervals']) == len(b['intervals'])
            truncation = a['bound_truncated'] == b['bound_truncated']
            change = float(np.max(abs(np.array(a['intervals'])-b['intervals']))) if topology else None
            row.update(topology_agrees=topology, truncation_agrees=truncation, endpoint_change=change,
                       production_intervals=a['intervals'], refined_intervals=b['intervals'],
                       passed=bool(topology and truncation and change<config['refinement']['estimate_target']))
        else:
            row['reason'] = 'Unresolved production or refined profile'
        rows.append(row)
    return rows


def run_profiles(store, replicates, workers, tight=False):
    jobs = []
    for n in store.config['founders']:
        for rep in replicates:
            data = store.dataset('inference', n, rep, store.config['cultures_per_cell'])
            for method in store.config['methods']:
                folder = 'inference-refined' if tight else 'inference'
                fitted = store.result(f'fits/{folder}/n{n}/rep{rep:03d}-{method}.json')
                if fitted is None:
                    raise ValueError('Missing point estimate before profiling')
                path = f'delivery3/profiles{"-refined" if tight else ""}/n{n}/rep{rep:03d}-{method}.json'
                jobs.append((path, (method, data, store.config, fitted, tight)))
    run_jobs(store, jobs, profile_task, workers)


def refinements(store, replicates):
    rows = []
    for n in store.config['founders']:
        for rep in replicates:
            for method in store.config['methods']:
                base = store.result(f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
                tight = store.result(f'delivery3/profiles-refined/n{n}/rep{rep:03d}-{method}.json')
                rows.extend(dict(founders=n, replicate=rep, method=method, **r) for r in profile_comparison(base, tight, store.config))
    return dict(passed=all(r['passed'] for r in rows), rows=rows)


def run_delivery(args, config):
    if args.workers > config['execution']['max_workers']:
        raise ValueError('At most four workers')
    if Path(args.output).resolve().name != config['experiment']:
        raise ValueError('Use the birth-death output directory')
    snapshot(args.output)
    store = ExtendedStore(args.output, config, retry=args.retry)
    start = args.invocation_started
    store.deadline = start+config['execution']['stage_wall_hours']*3600
    stage = f'delivery3/stages/{args.command}.json'
    existing = store.root/stage
    if existing.exists() and read_json(existing)['result']['status'] != 'complete':
        if not args.retry:
            raise ValueError('Incomplete stage retained; inspect and use --retry')
        dest = store.root/'attempts'/store.provenance['timestamp_utc'].replace(':', '-')/stage
        dest.parent.mkdir(parents=True, exist_ok=True)
        existing.replace(dest)
    store.save_result(stage, dict(status='running', workers=args.workers))
    complete, passed = False, False
    try:
        if args.command == 'checks':
            result = store.result('delivery3/checks.json')
            if result is None:
                result = checks(store)
                store.save_result('delivery3/checks.json', result)
            passed = result['passed']
        else:
            checked = store.result('delivery3/checks.json')
            if not checked or not checked['passed']:
                raise ValueError('Pass Delivery 3 equivalence checks first')
            if args.command == 'cost':
                jobs = []
                for n in config['founders']:
                    for rep in range(5):
                        data = store.dataset('inference', n, rep, config['cultures_per_cell'])
                        for method in config['methods']:
                            jobs.append((f'delivery3/cost/n{n}/rep{rep:03d}-{method}.json', (str(store.root), method, data, config, rep)))
                run_jobs(store, jobs, cost_task, args.workers)
                passed = all(store.result(p)['success'] for p, _ in jobs)
            elif args.command == 'inference':
                jobs = []
                for n in config['founders']:
                    for rep in range(20, 100):
                        data = store.dataset('inference', n, rep, config['cultures_per_cell'])
                        for method in config['methods']:
                            jobs.append((f'fits/inference/n{n}/rep{rep:03d}-{method}.json', (method, data, config)))
                run_jobs(store, jobs, fit_task, args.workers)
                passed = all(store.result(p)['success'] for p, _ in jobs)
            elif args.command == 'profile-check':
                run_profiles(store, range(1), args.workers)
                run_profiles(store, range(1), args.workers, tight=True)
                result = refinements(store, range(1))
                store.save_result('delivery3/profile-check.json', result)
                passed = result['passed']
            elif args.command in ('profiles', 'refine'):
                pilot = store.result('delivery3/profile-check.json')
                if not pilot or not pilot['passed']:
                    raise ValueError('Replicate 0 profiles must pass before expanding')
                run_profiles(store, range(100) if args.command=='profiles' else range(5), args.workers, tight=args.command=='refine')
                if args.command=='refine':
                    result = refinements(store, range(5))
                    store.save_result('delivery3/profile-refinement.json', result)
                    passed = result['passed']
                else:
                    passed = all(store.result(f'delivery3/profiles/n{n}/rep{r:03d}-{m}.json')['success'] for n in config['founders'] for r in range(100) for m in config['methods'])
            elif args.command == 'report':
                from .birth_death_extended_report import export
                export(store, args.report_dir)
                passed = True
            else:
                raise ValueError('Delivery 3 commands: checks, cost, inference, profile-check, profiles, refine, report')
        complete = True
    finally:
        store.save_result(stage, dict(status='complete' if complete else 'incomplete', passed=passed, workers=args.workers))
        store.save_result(f"timings/delivery3-{args.command}-{store.provenance['timestamp_utc'].replace(':', '-')}.json",
            dict(command=args.command, delivery=3, seconds=perf_counter()-start, workers=args.workers,
                 passed=passed, complete=complete, stage_budget_hours=config['execution']['stage_wall_hours']))
    return 0 if passed else 1
