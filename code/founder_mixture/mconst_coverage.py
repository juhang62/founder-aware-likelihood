"""Additional numerical checks for Mconst set boundaries close to truth."""

import argparse
from time import perf_counter
import numpy as np
from .mconst import Store, calculate, execution_limits, fit_identity
from .jobs import run_jobs


def check(data, config, m0, original_fit, original_profile, checkpoints=None):
    start=perf_counter()
    known=dict(config,profiles=dict(config['profiles'],xtol=1e-7))
    fitted=calculate('fit',data,known,tight=True,m0=m0,checkpoint=checkpoints[0] if checkpoints else None)
    if checkpoints:
        checkpoints[1]['metadata']['fitted_digest'] = fit_identity(fitted)
    profile=calculate('profile',data,known,fitted,tight=True,checkpoint=checkpoints[1] if checkpoints else None)
    rows=[]
    if fitted['success'] and profile['success']:
        for i,(a,b) in enumerate(zip(original_profile['rates'],profile['rates'])):
            topology=len(a['intervals'])==len(b['intervals'])
            delta=float(np.max(abs(np.array(a['intervals'])-b['intervals']))) if topology else None
            truth=config['truth'][i]
            covered=[any(lo<=truth<=hi for lo,hi in p['intervals']) for p in (a,b)]
            distance=min(abs(x-truth) for pair in b['intervals'] for x in pair)
            rows.append(dict(rate=i,endpoint_change=delta,coverage_agrees=covered[0]==covered[1],
                minimum_refined_distance=distance,passed=bool(topology and delta<.005 and covered[0]==covered[1] and distance>1e-7)))
    estimate_change=float(np.max(abs(np.array(fitted['estimate'])-original_fit['estimate']))) if fitted['success'] else None
    return dict(success=bool(len(rows)==2 and all(r['passed'] for r in rows) and estimate_change<.005),
        passed=bool(len(rows)==2 and all(r['passed'] for r in rows) and estimate_change<.005),rows=rows,fit=fitted,profile=profile,
        estimate_change=estimate_change,seconds=perf_counter()-start,crossing_tolerance=1e-8,
        truth_usage='Select numerical checks and assess coverage only; absent from optimization inputs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/founder-mixture-birth-death-mconst.json')
    parser.add_argument('--output',default='data/founder-mixture-birth-death/mconst')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--retry',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error('Use one or two workers')
    execution_limits()
    store=Store(args.config,args.output,args.retry);start=perf_counter();store.deadline=start+86400
    jobs=[];selected=[]
    for n in store.config['founders']:
        for rep in range(100):
            p=store.result(f'profiles/n{n}/rep{rep:03d}.json')
            if not p or not p['success']:continue
            distances=[min(abs(x-store.config['truth'][i]) for pair in r['intervals'] for x in pair) for i,r in enumerate(p['rates'])]
            if min(distances)>1e-5:continue
            path=f'coverage/n{n}/rep{rep:03d}.json'
            known=dict(store.config,profiles=dict(store.config['profiles'],xtol=1e-7))
            checkpoints=[store.checkpoint(path+'/'+kind,n,rep,kind,True,config=known) for kind in ('fit','profile')]
            jobs.append((path,(store.dataset(n,rep),store.config,store.old_fit(n,rep),store.result(f'fits/n{n}/rep{rep:03d}.json'),p,checkpoints)))
            selected.append(dict(founders=n,replicate=rep,path=path,original_distances=distances))
    run_jobs(store,jobs,check,args.workers)
    result=dict(passed=all(store.result(r['path'])['passed'] for r in selected),selected=selected,seconds=perf_counter()-start)
    store.save_result('coverage.json',result);print(result,flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
