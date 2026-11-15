"""Recompute saved maxima and check Delivery 3 datasets and profile artifacts."""

from pathlib import Path
from time import perf_counter
import numpy as np
from . import ROOT
from .birth_death_extended import ExtendedStore, observed_inputs, sha
from .storage import read_json, write_json
from .likelihood import Likelihood, Settings
from .model import from_config
from .simulation import generate


def audit(store):
    start = perf_counter()
    config = store.config
    fits, refined_fits, profiles, endpoints, survival, regenerated = [], [], [], [], [], []
    for n in config['founders']:
        for rep in range(100):
            data = store.dataset('inference',n,rep,config['cultures_per_cell'])
            obs,known = observed_inputs(data,config)
            assert len(obs['y'])==40
            assert np.array_equal(obs['cell'],np.repeat(np.arange(4),10))
            assert np.array_equal(obs['time'],np.repeat(config['times']*2,10))
            assert np.array_equal(obs['z0'],np.repeat(np.eye(2,dtype=int)*n,20,axis=0))
            if rep in (20,99):
                again=generate(config,'inference',n,rep,config['cultures_per_cell'])
                equal={key:bool(np.array_equal(value,again[key])) for key,value in data.items()}
                regenerated.append(dict(founders=n,replicate=rep,arrays_equal=equal,passed=all(equal.values())))
            for method in config['methods']:
                if rep<5:
                    refined=store.result(f'fits/inference-refined/n{n}/rep{rep:03d}-{method}.json')
                    tight=Likelihood(dict(known,extinction=known['tight_extinction']),Settings(**known['tight_integration']))
                    change=abs(tight.loglik(method,refined['estimate'],obs)-refined['loglik']) if refined['success'] else None
                    refined_fits.append(dict(founders=n,replicate=rep,method=method,change=change,
                        passed=bool(refined['success'] and change<1e-7)))
                f=store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')
                if f and f['success']:
                    value=Likelihood(known).loglik(method,f['estimate'],obs)
                    change=abs(value-f['loglik'])
                    fits.append(dict(founders=n,replicate=rep,method=method,change=change,passed=change<1e-7,
                                     distinct_starts=len({tuple(s['start']) for s in f['starts']}),failed_starts=f['failed_starts']))
                    c=from_config(f['estimate'],known)
                    maximum=max(c.survival(t,z0)['epsilon'] for t in known['times'] for z0 in np.eye(2,dtype=int)*n)
                    survival.append(dict(founders=n,replicate=rep,method=method,maximum_epsilon=maximum))
                p=store.result(f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
                if not p or not p['success']:
                    profiles.append(dict(founders=n,replicate=rep,method=method,passed=False,reason=(p or {}).get('reason','missing')))
                    continue
                for rate,q in enumerate(p['rates']):
                    intervals=np.array(q['intervals'])
                    contains=any(a<=f['estimate'][rate]<=b for a,b in intervals)
                    valid=bool(contains and np.all(intervals[:,0]<=intervals[:,1]) and np.all(intervals>=.3) and np.all(intervals<=1.1)
                        and np.all(intervals[1:,0]>intervals[:-1,1]) and abs(np.sum(intervals[:,1]-intervals[:,0])-q['length'])<1e-12
                        and q['grid_size']==81 and q['nuisance_grid_size']==15 and max(q['loglik'])<=f['loglik']+1e-4)
                    profiles.append(dict(founders=n,replicate=rep,method=method,rate=rate,passed=valid,
                                         contains_estimate=contains,maximum_above_fit=max(q['loglik'])-f['loglik']))
                    for component,(a,b) in enumerate(intervals):
                        for side,x in enumerate((a,b)):
                            idx=int(np.argmin(abs(np.array(q['x'])-x)))
                            assert abs(q['x'][idx]-x)<1e-12
                            theta=np.empty(2);theta[rate]=x;theta[1-rate]=q['nuisance_estimates'][idx]
                            value=Likelihood(known).loglik(method,theta,obs)
                            change=abs(value-q['loglik'][idx])
                            c=from_config(theta,known)
                            epsilon=max(c.survival(t,z0)['epsilon'] for t in known['times'] for z0 in np.eye(2,dtype=int)*n)
                            endpoints.append(dict(founders=n,replicate=rep,method=method,rate=rate,component=component,side=side,
                                theta=theta,loglik_change=change,threshold_difference=value-q['threshold'],
                                truth_distance=abs(x-config['truth'][rate]),
                                maximum_epsilon=epsilon,passed=change<1e-7))
            if rep%20==19: print(f'audit N={n}: {rep+1}/100 datasets',flush=True)
    changes=[p for p,h in store.baseline['files'].items() if sha(store.root/p)!=h]
    result=dict(passed=not changes and all(r['passed'] for r in fits+refined_fits+profiles+endpoints+regenerated),
        fit_checks=fits,refined_fit_checks=refined_fits,profile_checks=profiles,endpoint_checks=endpoints,fitted_survival=survival,seed_checks=regenerated,
        baseline_changes=changes,maximum_fit_loglik_change=max(r['change'] for r in fits),
        maximum_endpoint_loglik_change=max(r['loglik_change'] for r in endpoints),
        maximum_refined_fit_loglik_change=max(r['change'] for r in refined_fits if r['change'] is not None),
        maximum_endpoint_threshold_difference=max(abs(r['threshold_difference']) for r in endpoints),
        maximum_fitted_epsilon=max(r['maximum_epsilon'] for r in survival),
        maximum_profile_endpoint_epsilon=max(r['maximum_epsilon'] for r in endpoints),seconds=perf_counter()-start)
    result['minimum_truth_endpoint_distance']=min(r['truth_distance'] for r in endpoints)
    result['near_truth_endpoints']=[r for r in endpoints if r['truth_distance']<=config['profiles']['xtol']]
    return result


if __name__=='__main__':
    config=read_json(ROOT/'configs/founder-mixture-birth-death.json')
    store=ExtendedStore(ROOT/'data/founder-mixture-birth-death',config)
    result=audit(store)
    store.save_result('delivery3/final-audit.json',result)
    write_json(ROOT/'reports/founder-mixture-birth-death/tables/delivery3-final-audit.json',dict(result=result,provenance=store.provenance))
    print({k:v for k,v in result.items() if k.startswith('maximum') or k in ('passed','seconds')},flush=True)
    raise SystemExit(0 if result['passed'] else 1)
