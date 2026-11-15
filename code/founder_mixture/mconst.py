"""Reproduce the Mconst addendum on the frozen birth-death observations."""

import argparse
import hashlib
import sys
import os
import resource
from pathlib import Path
from time import perf_counter
import numpy as np
from . import ROOT
from .storage import read_json, write_json, provenance, digest
from .jobs import run_jobs
from .model import candidate, extinction
from .likelihood import Likelihood
from . import mconst_inference as inference
from .mconst_checkpoint import Checkpoint, durable_json
from .mconst_solver import backend, clear_caches, SOLVER

NUMERICAL_SOURCES = ['mconst.py', 'mconst_inference.py', 'mconst_checkpoint.py', 'mconst_solver.py',
                     'likelihood.py', 'model.py', 'extinction.py', 'inference.py', 'jobs.py', 'storage.py',
                     'mconst_coverage.py']


def execution_limits():
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError(f'Set {name}=1 before starting the extension')
    from threadpoolctl import threadpool_limits
    from numba import set_num_threads
    threadpool_limits(limits=1)
    set_num_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_identity(fitted):
    return digest({key:fitted.get(key) for key in ('success','theta','loglik','variance_cap')}) if fitted is not None else None


class Store:
    def __init__(self, config_path, output, retry=False):
        extension = read_json(config_path)
        if extension['extension']!='Mconst' or extension['replicates']!=100 or extension['bootstrap_stage']!=61 or extension['bootstrap_repetitions']!=2000:
            raise ValueError('The addendum fixes Mconst, 100 replicates, and the specified bootstrap stream and count')
        base = ROOT/extension['baseline_config']
        if sha(base) != extension['baseline_sha256']:
            raise ValueError('Original configuration changed')
        self.config = dict(read_json(base), mconst=extension)
        self.config['execution'] = dict(self.config.get('execution', {}), **extension['execution'])
        if extension['execution']['max_workers'] != 2 or extension['execution']['numerical_threads'] != 1:
            raise ValueError('The execution amendment requires two workers and one numerical thread each')
        self.root = Path(output).resolve()
        self.baseline = read_json(self.root/'baseline/manifest.json')
        self.retry = retry
        self.provenance = provenance(self.config)
        from importlib.metadata import version
        self.provenance['versions']['threadpoolctl'] = version('threadpoolctl')
        self.provenance['execution_command'] = getattr(sys,'orig_argv',[sys.executable,*sys.argv])
        self.provenance['extension_config_sha256'] = sha(config_path)
        self.provenance['baseline_manifest_sha256'] = sha(self.root/'baseline/manifest.json')
        self.provenance['extinction_solver'] = SOLVER
        self.provenance['checkpoint_policy'] = 'Durable grid, local search, certificate, profile point, and individual rate; deterministic replay'
        self.provenance['thread_environment'].update({k: os.environ.get(k) for k in ('MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
        self.provenance['stage_ids'] = dict(self.provenance['stage_ids'], mconst=61)
        self.root.mkdir(parents=True, exist_ok=True)
        archive=self.root/'source-versions'
        archive.mkdir(exist_ok=True)
        for name,expected in self.provenance['source_sha256'].items():
            content=(ROOT/name).read_bytes()
            if hashlib.sha256(content).hexdigest()!=expected:
                raise ValueError('Source changed while recording provenance')
            target=archive/(expected+'.py')
            if not target.exists():target.write_bytes(content)
        self.provenance['source_archive']=str(archive.relative_to(ROOT))
        self.save_result('runs/'+self.provenance['timestamp_utc'].replace(':', '-')+'.json', {'status': 'started'})

    def baseline_path(self, path):
        path = Path(path)
        key = str(path.relative_to(ROOT))
        if key not in self.baseline['files'] or sha(path) != self.baseline['files'][key]:
            raise ValueError(f'Baseline artifact absent or changed: {path}')
        return path

    def dataset(self, n, rep):
        path = self.baseline_path(self.root.parent/f'observations/inference/n{n}/rep{rep:03d}.npz')
        with np.load(path, allow_pickle=False) as saved:
            if not saved['complete'].all():
                raise ValueError('Incomplete original dataset')
            return {key: saved[key] for key in ('y', 'time', 'z0', 'cell')}

    def old_fit(self, n, rep, method='M0'):
        return read_json(self.baseline_path(self.root.parent/f'fits/inference/n{n}/rep{rep:03d}-{method}.json'))['result']

    def save_result(self, path, result):
        durable_json(self.root/path, dict(result=result, provenance=self.provenance))

    def checkpoint(self, path, n, rep, kind, tight=False, fitted=None, config=None):
        data = self.dataset(n, rep)
        metadata = dict(founders=n, replicate=rep, method='Mconst', calculation=kind,
                        tight=tight, observed_inputs_sha256=digest(data),
                        config_digest=digest(config or self.config), extinction_solver=SOLVER,
                        numerical_sources={name:self.provenance['source_sha256'].get('code/founder_mixture/'+name)
                                           for name in NUMERICAL_SOURCES},
                        fitted_digest=fit_identity(fitted))
        return dict(root=str(self.root/'checkpoints'/path), metadata=metadata)

    def result(self, path):
        full = self.root/path
        if not full.exists():
            return None
        record = read_json(full)
        config_changed = record['provenance']['config_digest'] != digest(self.config)
        if config_changed and not self.retry:
            raise ValueError('Extension configuration changed; preserve and inspect the existing result')
        names = list(NUMERICAL_SOURCES)
        if str(path).startswith(('checks','analytic-checks','density-checks','pilot')):
            names.append('mconst_checks.py')
        if str(path).startswith('coverage'):
            names.append('mconst_coverage.py')
        old = record['provenance']['source_sha256']
        changed = [n for n in names if old.get('code/founder_mixture/'+n) != self.provenance['source_sha256'].get('code/founder_mixture/'+n)]
        result = record['result']
        failed = result.get('passed', result.get('success', True)) is False
        if self.retry and (changed or config_changed):
            archive = self.root/'attempts'/self.provenance['timestamp_utc'].replace(':', '-')/path
            archive.parent.mkdir(parents=True, exist_ok=True)
            full.replace(archive)
            durable_json(archive.with_suffix('.retry.json'), dict(changed_sources=changed, prior_failed=failed, config_changed=config_changed,
                reason='Explicit retry on the same frozen observations; earlier attempt retained'))
            return None
        if changed:
            raise ValueError(f'Computational source changed: {changed}; use explicit --retry after inspection')
        return result


def calculate(kind, data, config, fitted=None, tight=False, m0=None, checkpoint=None):
    execution_limits()
    config = {k: v for k, v in config.items() if k != 'truth'}
    cp = Checkpoint(**checkpoint) if checkpoint else None
    final = cp.scope('complete', dict(kind=kind)) if cp else None
    if final and final.state['result'] is not None:
        result = final.state['result']
        result['execution_cost'] = cp.cost()
        result['seconds'] = result['execution_cost']['seconds']
        return result
    with backend():
        if kind == 'fit':
            result=inference.fit(data, config, tight=tight, m0=m0, checkpoint=cp)
        elif kind == 'profile':
            result=inference.profile_sets(data, config, fitted, tight=tight, checkpoint=cp)
            result.update(fitted_theta=fitted.get('theta'),fitted_loglik=fitted.get('loglik'))
        else:
            raise ValueError(kind)
    result['observed_inputs_sha256']=digest({k:data[k] for k in ('y','time','z0','cell')})
    result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result['extinction_solver'] = SOLVER
    if cp:
        result['execution_cost'] = cp.cost()
        result['seconds'] = result['execution_cost']['seconds']
        final.finish(result)
        result['execution_cost'] = cp.cost()
        result['seconds'] = result['execution_cost']['seconds']
    return result


def compute(store, kind, reps, workers, tight=False):
    folder = ('fits' if kind == 'fit' else 'profiles')+('-refined' if tight else '')
    jobs = []
    for n in store.config['founders']:
        for rep in reps:
            fitted = store.result(f'fits{"-refined" if tight else ""}/n{n}/rep{rep:03d}.json') if kind == 'profile' else None
            if kind == 'profile' and fitted is None:
                raise ValueError(f'Missing point estimate n{n} rep{rep}')
            path = f'{folder}/n{n}/rep{rep:03d}.json'
            jobs.append((path, (kind, store.dataset(n, rep), store.config, fitted, tight, store.old_fit(n, rep),
                               store.checkpoint(path, n, rep, kind, tight, fitted))))
    run_jobs(store, jobs, calculate, workers)


def refinement(store, reps):
    rows = []
    for n in store.config['founders']:
        for rep in reps:
            a, b = [store.result(f'fits{suffix}/n{n}/rep{rep:03d}.json') for suffix in ('', '-refined')]
            p, q = [store.result(f'profiles{suffix}/n{n}/rep{rep:03d}.json') for suffix in ('', '-refined')]
            row = dict(founders=n, replicate=rep, passed=False, rates=[])
            if a and b and p and q and all(x['success'] for x in (a,b,p,q)):
                row.update(estimate_change=float(np.max(abs(np.array(a['estimate'])-b['estimate']))),
                           loglik_change=abs(a['loglik']-b['loglik']), u_change=abs(a['u']-b['u']))
                for i, (s, t) in enumerate(zip(p['rates'], q['rates'])):
                    topology = len(s['intervals']) == len(t['intervals'])
                    distance = float(np.max(abs(np.array(s['intervals'])-t['intervals']))) if topology else None
                    cover = [any(lo<=store.config['truth'][i]<=hi for lo, hi in r['intervals']) for r in (s,t)]
                    row['rates'].append(dict(rate=i, endpoint_change=distance, topology_agrees=topology,
                        coverage_agrees=cover[0]==cover[1], truncation_agrees=s['bound_truncated']==t['bound_truncated'],
                        passed=bool(topology and distance<.005 and cover[0]==cover[1] and s['bound_truncated']==t['bound_truncated'])))
                row['passed'] = bool(row['estimate_change']<.005 and row['loglik_change']<.05 and row['u_change']<.001 and all(x['passed'] for x in row['rates']))
            rows.append(row)
    return dict(passed=all(r['passed'] for r in rows), rows=rows)


def cost_task(data, config, m0, rep, checkpoint=None):
    execution_limits()
    with backend():
        return _cost_task(data, config, m0, rep, checkpoint)


def _cost_task(data, config, m0, rep, checkpoint):
    from .inference import fit, profile_sets
    config = {k: v for k, v in config.items() if k != 'truth'}
    results = {}
    order = ['M', 'Mconst'] if rep % 2 == 0 else ['Mconst', 'M']
    start = perf_counter()
    for method in order:
        cp = None
        clear_caches()
        like = Likelihood(config)
        theta = [.7, .6, 0.] if method == 'Mconst' else [.7, .6]
        tick = perf_counter()
        like.loglik(method, theta, data)
        startup = perf_counter()-tick
        clear_caches()
        from .mconst_solver import compiled_extinction
        state = dict(candidate=candidate.cache_info()._asdict(), extinction=compiled_extinction.cache_info()._asdict())
        tick=perf_counter()
        if method == 'Mconst' and checkpoint:
            cp = Checkpoint(root=checkpoint['root']+'/fit', metadata=dict(checkpoint['metadata'], calculation='cost_fit'))
            if cp.resumed:
                raise ValueError('Interrupted timing attempt: archive it and start a fresh matched run')
        f = inference.fit(data, config, m0=m0, checkpoint=cp) if method == 'Mconst' else fit(method, data, config)
        if cp:
            f['execution_cost'] = cp.cost()
            cp.scope('complete', {}).finish(f)
            f['execution_cost'] = cp.cost()
        fit_call_seconds=perf_counter()-tick
        if checkpoint:
            durable_json(Path(checkpoint['root'])/(method+'-fit.json'), dict(metadata=checkpoint['metadata'],
                         result=f, fit_call_seconds=fit_call_seconds, startup_seconds=startup, completed=True))
        p = None
        profile_call_seconds=None
        if rep == 0:
            clear_caches()
            tick=perf_counter()
            if cp:
                cp = Checkpoint(root=checkpoint['root']+'/profile', metadata=dict(checkpoint['metadata'], calculation='cost_profile', fitted_digest=fit_identity(f)))
                if cp.resumed:
                    raise ValueError('Interrupted timing attempt: archive it and start a fresh matched run')
            p = inference.profile_sets(data, config, f, checkpoint=cp) if method == 'Mconst' else profile_sets(method, data, config, f)
            if cp:
                p['execution_cost'] = cp.cost()
                cp.scope('complete', {}).finish(p)
                p['execution_cost'] = cp.cost()
            profile_call_seconds=perf_counter()-tick
            if checkpoint:
                durable_json(Path(checkpoint['root'])/(method+'-profile.json'), dict(metadata=checkpoint['metadata'],
                             result=p, profile_call_seconds=profile_call_seconds, completed=True))
        results[method] = dict(fit=f, profile=p, startup_seconds=startup, cache_before_fit=state,
                               fit_call_seconds=fit_call_seconds,profile_call_seconds=profile_call_seconds)
    return dict(success=all(r['fit']['success'] and (r['profile'] is None or r['profile']['success']) for r in results.values()),
                methods=results, order=order, seconds=perf_counter()-start, workers=1, extinction_solver=SOLVER,
                checkpoint_policy='Mconst: each grid/local search/profile point; M: completed fit/profile. Both include their actual storage costs.',
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['checks', 'pilot', 'inference', 'profiles', 'refine', 'cost', 'report', 'audit'])
    parser.add_argument('--config', default='configs/founder-mixture-birth-death-mconst.json')
    parser.add_argument('--output', default='data/founder-mixture-birth-death/mconst')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    if not 1<=args.workers<=2:
        parser.error('Use one or two workers')
    execution_limits()
    store = Store(args.config, args.output, args.retry)
    started = perf_counter()
    store.deadline = started+86400
    success = False
    try:
        if args.stage == 'checks':
            from .mconst_checks import run_checks
            result = run_checks(store)
            from .mconst_recovery_checks import run_checks as recovery_checks
            recovery = recovery_checks(store.root/'recovery-tests')
            store.save_result('recovery-checks.json', recovery)
            from .mconst_solver_checks import baseline_checks
            solver = baseline_checks(store)
            store.save_result('solver-checks.json', solver)
            result['passed'] = bool(result['passed'] and recovery['passed'] and solver['passed'])
            store.save_result('checks.json', result)
            success = result['passed']
        elif args.stage == 'pilot':
            if not (store.result('checks.json') or {}).get('passed'):
                raise ValueError('Complete implementation checks before the pilot')
            # The two founder counts and two accuracy settings give four
            # independent fits, then four profiles, within the worker limit.
            for kind in ('fit','profile'):
                jobs=[]
                for tight in (False,True):
                    suffix='-refined' if tight else ''
                    for n in store.config['founders']:
                        fitted=store.result(f'fits{suffix}/n{n}/rep000.json') if kind=='profile' else None
                        folder=('fits' if kind=='fit' else 'profiles')+suffix
                        path = f'{folder}/n{n}/rep000.json'
                        jobs.append((path,(kind,store.dataset(n,0),store.config,fitted,tight,store.old_fit(n,0),
                                           store.checkpoint(path,n,0,kind,tight,fitted))))
                run_jobs(store,jobs,calculate,args.workers)
            from .mconst_checks import density_checks
            densities = density_checks(store)
            store.save_result('density-checks.json', densities)
            result = refinement(store, [0])
            from .mconst_solver_checks import profile_checks
            compatibility = profile_checks(store)
            store.save_result('solver-compatibility.json', compatibility)
            result['passed'] = bool(result['passed'] and densities['passed'] and compatibility['passed'])
            store.save_result('pilot.json', result)
            success = result['passed']
        elif args.stage in ('inference', 'profiles', 'refine', 'cost'):
            if not (store.result('pilot.json') or {}).get('passed'):
                raise ValueError('Complete and pass the numerical pilot first')
            if args.stage == 'inference':
                compute(store, 'fit', range(100), args.workers)
            elif args.stage == 'profiles':
                compute(store, 'profile', range(100), args.workers)
            elif args.stage == 'refine':
                compute(store, 'fit', range(5), args.workers, True)
                compute(store, 'profile', range(5), args.workers, True)
                store.save_result('refinement.json', refinement(store, range(5)))
            else:
                jobs = []
                for n in store.config['founders']:
                    for r in range(5):
                        path = f'cost/n{n}/rep{r:03d}.json'
                        cp = store.checkpoint(path,n,r,'cost')
                        partial = Path(cp['root'])
                        if store.result(path) is None and partial.exists():
                            if not store.retry:
                                raise ValueError('Interrupted timing attempt: use --retry to preserve it and measure a fresh run')
                            archive = store.root/'attempts'/store.provenance['timestamp_utc'].replace(':','-')/'timing-checkpoints'/path
                            archive.parent.mkdir(parents=True,exist_ok=True)
                            partial.replace(archive)
                        jobs.append((path,(store.dataset(n,r),store.config,store.old_fit(n,r),r,cp)))
                run_jobs(store, jobs, cost_task, 1)
            success = True
        else:
            from .mconst_report import report, audit
            result = report(store) if args.stage == 'report' else audit(store)
            success = result.get('passed', True)
        print(dict(stage=args.stage, success=success, seconds=perf_counter()-started), flush=True)
    finally:
        store.save_result('timings/'+args.stage+'-'+store.provenance['timestamp_utc'].replace(':','-')+'.json',
                          dict(success=success, seconds=perf_counter()-started, workers=1 if args.stage=='cost' else args.workers,
                               allowance_hours=24))
    return 0 if success else 1


def main():
    with backend():
        return _main()


if __name__ == '__main__':
    raise SystemExit(main())
