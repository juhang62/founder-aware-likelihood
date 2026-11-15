"""Independent checks of saved extension results and matched timing repeats."""

import argparse
from itertools import product
from time import perf_counter
import numpy as np
from . import ROOT
from .likelihood import Likelihood, Settings
from .mconst import Store
from .mconst_inference import Problem
from .storage import digest, write_json


def pointwise_identity(store):
    rows=[]
    for n in store.config['founders']:
        data=store.dataset(n,0)
        points=list(product(*[np.linspace(a,b,7) for a,b in store.config['bounds']]))
        points += [store.old_fit(n,0,m)['estimate'] for m in store.config['methods']]
        likelihood=Likelihood(store.config)
        changes=[]
        for theta in points:
            for group in np.unique(data['cell']):
                ids=np.flatnonzero(data['cell']==group); first=ids[0]
                args=(theta,data['y'][ids],float(data['time'][first]),data['z0'][first])
                for living_only in (False,True):
                    a=likelihood.logpdf('Mconst',*args,survivor=living_only,residual_variance=0.)
                    b=likelihood.logpdf('M0',*args,survivor=living_only)
                    changes.extend(abs(a-b))
        rows.append(dict(founders=n,parameter_points=len(points),log_density_pairs=len(changes),
                         maximum_change=float(max(changes)),passed=bool(max(changes)<1e-8)))
        contaminated=dict(data,counts=np.full_like(data['y'],999.),survived=np.zeros(len(data['y']),bool))
        ordinary=Problem(data,store.config)
        modified=Problem(contaminated,dict(store.config,truth=[99.,99.]))
        absent=all(k not in modified.data for k in ('counts','survived')) and 'truth' not in modified.config
        change=abs(ordinary.objective([.7,.6,1.])-modified.objective([.7,.6,1.]))
        rows[-1].update(latent_fit_inputs_excluded=absent,contaminated_objective_change=change,
                        passed=bool(rows[-1]['passed'] and absent and change==0.))
    return dict(passed=all(r['passed'] for r in rows),rows=rows)


def inspect_saved(store):
    rows=[]
    for suffix in ('','-refined'):
        for n in store.config['founders']:
            for path in sorted((store.root/f'fits{suffix}/n{n}').glob('rep*.json')):
                rep=int(path.stem[3:]); data=store.dataset(n,rep)
                fitted=store.result(str(path.relative_to(store.root)))
                profile=store.result(f'profiles{suffix}/n{n}/rep{rep:03d}.json')
                cfg=store.config
                if suffix:cfg=dict(cfg,extinction=cfg['tight_extinction'])
                likelihood=Likelihood(cfg,Settings(**cfg['tight_integration' if suffix else 'integration']))
                row=dict(founders=n,replicate=rep,tight=bool(suffix),fit_success=fitted['success'],
                         profile_success=bool(profile and profile['success']),passed=True)
                if fitted['success']:
                    change=abs(likelihood.loglik('Mconst',fitted['theta'],data)-fitted['loglik'])
                    identity=digest({k:data[k] for k in ('y','time','z0','cell')})
                    final=fitted['expansions'][-1]
                    row.update(fitted_loglik_change=change,observed_inputs_match=fitted['observed_inputs_sha256']==identity,
                               variance_cap_certified=final['certified'],
                               mean_parameter_change=float(np.max(abs(np.array(fitted['theta'][:2])-fitted['estimate']))))
                    row['passed']=bool(change<1e-7 and row['observed_inputs_match'] and final['certified'] and row['mean_parameter_change']==0.)
                if profile and profile['success']:
                    endpoints=[]; certificates=[]
                    for value in profile['evaluated'].values():
                        final=value['expansions'][-1]
                        certificates.append(final['certified'] and final['upper_bound']<min(final['loglik'],final['tight_loglik'])-.05)
                    for i,rate in enumerate(profile['rates']):
                        for interval in rate['intervals']:
                            for x in interval:
                                if x in store.config['bounds'][i]:continue
                                candidates=[v for key,v in profile['evaluated'].items() if int(key.split(':')[0])==i]
                                nearest=min(candidates,key=lambda v:abs(v['theta'][i]-x))
                                check=likelihood.loglik('Mconst',nearest['theta'],data)
                                endpoints.append(dict(rate=i,boundary=x,stored_rate=nearest['theta'][i],
                                    loglik_change=abs(check-nearest['loglik']),threshold_residual=check-rate['threshold']))
                    row.update(endpoints=endpoints,profile_certificates=all(certificates),
                               profile_fitted_loglik_change=abs(profile['fitted_loglik']-fitted['loglik']))
                    row['passed']=bool(row['passed'] and all(certificates) and row['profile_fitted_loglik_change']<1e-7
                        and all(r['loglik_change']<1e-7 and abs(r['stored_rate']-r['boundary'])<1e-8 for r in endpoints))
                rows.append(row)
    return dict(passed=all(r['passed'] for r in rows),rows=rows,
                policy='Reevaluate saved maxima and interval boundaries; do not rerun or replace the scientific optimizations.')


def matched_repeats(store):
    rows=[]
    for n in store.config['founders']:
        for rep in range(5):
            timed=store.result(f'cost/n{n}/rep{rep:03d}.json')
            if not timed:continue
            for method,record in timed['methods'].items():
                scientific=store.result(f'fits/n{n}/rep{rep:03d}.json') if method=='Mconst' else store.old_fit(n,rep,method)
                fitted=record['fit']
                row=dict(founders=n,replicate=rep,method=method,passed=True,
                         comparable_fits=bool(fitted['success'] and scientific['success']))
                if fitted['success'] and scientific['success']:
                    row.update(estimate_change=float(np.max(abs(np.array(fitted['estimate'])-scientific['estimate']))),
                               loglik_change=abs(fitted['loglik']-scientific['loglik']))
                    row['passed']=row['estimate_change']<.005 and row['loglik_change']<.05
                if rep==0:
                    if method=='Mconst':profile=store.result(f'profiles/n{n}/rep000.json')
                    else:
                        from .mconst_report import old_profile
                        profile=old_profile(store,n,rep,method)
                    repeat=record['profile']; endpoints=[]
                    if repeat and repeat['success'] and profile['success']:
                        for a,b in zip(repeat['rates'],profile['rates']):
                            endpoints.append(float(np.max(abs(np.array(a['intervals'])-b['intervals']))) if len(a['intervals'])==len(b['intervals']) else None)
                    row['profile_endpoint_changes']=endpoints
                    row['comparable_profiles']=bool(repeat and repeat['success'] and profile['success'])
                    if row['comparable_profiles']:
                        row['passed']=bool(row['passed'] and len(endpoints)==2 and all(v is not None and v<.005 for v in endpoints))
                rows.append(row)
    return dict(passed=all(r['passed'] for r in rows),rows=rows)


def run(store):
    start=perf_counter()
    sections=dict(pointwise_identity=pointwise_identity(store),saved_results=inspect_saved(store),matched_repeats=matched_repeats(store))
    result=dict(passed=all(r['passed'] for r in sections.values()),seconds=perf_counter()-start,**sections)
    store.save_result('validation.json',result)
    write_json(ROOT/'reports/founder-mixture-birth-death/tables/mconst-validation.json',dict(result=result,provenance=store.provenance))
    print(dict(passed=result['passed'],seconds=result['seconds'],fits_checked=len(result['saved_results']['rows']),
               timing_fits_checked=len(result['matched_repeats']['rows'])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/founder-mixture-birth-death-mconst.json')
    parser.add_argument('--output',default='data/founder-mixture-birth-death/mconst')
    args=parser.parse_args()
    from .mconst import execution_limits
    from .mconst_solver import backend
    execution_limits()
    with backend():
        result=run(Store(args.config,args.output))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
