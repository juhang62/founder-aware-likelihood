"""Resolve coverage classifications close to a numerical profile endpoint.

The generating rates select numerical checks only. They are never supplied to
the fitter or profile calculation. Original fits and profiles remain unchanged.
"""

import argparse
from time import perf_counter
import numpy as np
from . import ROOT
from .birth_death_extended import ExtendedStore, observed_inputs, profile_comparison
from .inference import fit, profile_sets
from .jobs import run_jobs
from .model import candidate
from .storage import read_json, write_json


def calculate(method, data, config):
    observations, known = observed_inputs(data, config)
    candidate.cache_clear()
    start = perf_counter()
    fitted = fit(method, observations, known, tight=True)
    # profile_sets divides this tolerance by ten in its tighter branch.
    known = dict(known, profiles=dict(known['profiles'], xtol=1e-7))
    profile = profile_sets(method, observations, known, fitted, tight=True)
    return dict(success=bool(fitted['success'] and profile['success']), fit=fitted,
                profile=profile, seconds=perf_counter()-start,
                crossing_tolerance=1e-8, nuisance_tolerance=1e-9,
                numerical_config=known)


def run(store, workers):
    selected, jobs = [], []
    for n in store.config['founders']:
        for rep in range(100):
            for method in store.config['methods']:
                original = store.result(f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
                if not original or not original['success']:
                    continue
                distances = [min(abs(x-store.config['truth'][rate]) for interval in q['intervals'] for x in interval)
                             for rate, q in enumerate(original['rates'])]
                if min(distances) > store.config['profiles']['xtol']:
                    continue
                path = f'delivery3/coverage-refinement/n{n}/rep{rep:03d}-{method}.json'
                existing = store.result(path)
                if existing:
                    source = read_json(store.root/path)['provenance']['source_sha256']
                    name = 'code/founder_mixture/birth_death_coverage_refinement.py'
                    if source.get(name) != store.provenance['source_sha256'][name]:
                        raise ValueError('Coverage-check source changed; preserve and inspect the existing result')
                data = store.dataset('inference', n, rep, store.config['cultures_per_cell'])
                jobs.append((path, (method, data, store.config)))
                selected.append(dict(founders=n, replicate=rep, method=method, path=path,
                                     original_minimum_distances=distances))
    run_jobs(store, jobs, calculate, workers)
    rows = []
    for item in selected:
        n, rep, method = item['founders'], item['replicate'], item['method']
        checked = store.result(item['path'])
        original = store.result(f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
        fitted = store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')
        comparisons = profile_comparison(original, checked['profile'], store.config)
        estimate_change = float(np.max(abs(np.array(checked['fit']['estimate'])-fitted['estimate']))) if checked['fit']['success'] else None
        for row in comparisons:
            if row['passed']:
                truth = store.config['truth'][row['rate']]
                row['original_covered'] = any(a<=truth<=b for a,b in row['production_intervals'])
                row['refined_covered'] = any(a<=truth<=b for a,b in row['refined_intervals'])
                row['coverage_agrees'] = row['original_covered']==row['refined_covered']
                row['minimum_refined_distance'] = min(abs(x-truth) for interval in row['refined_intervals'] for x in interval)
                row['passed'] = bool(row['coverage_agrees'] and row['minimum_refined_distance']>1e-7 and
                                     estimate_change<store.config['refinement']['estimate_target'])
            rows.append(dict(item, estimate_change=estimate_change, **row))
    result = dict(passed=all(r['passed'] for r in rows), selected=selected, rows=rows,
                  additional_tight_fits=len(jobs), additional_two_rate_profiles=len(jobs),
                  selection='Every successful production profile with an endpoint within 1e-5/day of its generating rate; no outcome-dependent exclusion or replacement.',
                  interpretation='Checks only: original scientific fits and profiles remain the reported records when coverage agrees.')
    store.save_result('delivery3/coverage-refinement.json', result)
    write_json(ROOT/'reports/founder-mixture-birth-death/tables/delivery3-coverage-refinement.json',
               dict(result=result, provenance=store.provenance))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--retry',action='store_true',help='Archive failed or interrupted checks and reuse the same observations')
    args=parser.parse_args()
    config=read_json(ROOT/'configs/founder-mixture-birth-death.json')
    if not 1<=args.workers<=config['execution']['max_workers']:
        parser.error('Use one to four workers')
    store=ExtendedStore(ROOT/'data/founder-mixture-birth-death',config,retry=args.retry)
    started=perf_counter()
    store.deadline=started+config['execution']['stage_wall_hours']*3600
    result=run(store,args.workers)
    store.save_result('delivery3/coverage-refinement-timing.json',dict(seconds=perf_counter()-started,workers=args.workers,
                      stage_budget_hours=config['execution']['stage_wall_hours']))
    print({k:result[k] for k in ('passed','additional_tight_fits','additional_two_rate_profiles')},flush=True)
    raise SystemExit(0 if result['passed'] else 1)
