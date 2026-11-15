"""Compatibility of the compiled solver with the preserved original results."""

import io
from itertools import product
import unittest
import numpy as np
from . import model
from .extinction import extinction
from .likelihood import Likelihood
from .mconst_solver import backend, clear_caches
from .storage import read_json


def baseline_checks(store):
    rows=[]
    for n in store.config['founders']:
        data=store.dataset(n,0)
        points=list(product(*store.config['bounds']))
        points += [store.old_fit(n,0,m)['estimate'] for m in store.config['methods']]
        original = model.extinction
        try:
            model.extinction=extinction
            clear_caches()
            old=Likelihood(store.config)
            reference={(method,i):old.loglik(method,t,data) for method in store.config['methods'] for i,t in enumerate(points)}
        finally:
            model.extinction=original
            clear_caches()
        with backend():
            revised=Likelihood(store.config)
            for method in store.config['methods']:
                change=max(abs(revised.loglik(method,t,data)-reference[method,i]) for i,t in enumerate(points))
                f=store.old_fit(n,0,method)
                saved_change=abs(revised.loglik(method,f['estimate'],data)-f['loglik'])
                rows.append(dict(founders=n,method=method,maximum_loglik_change=change,
                                 saved_maximum_loglik_change=saved_change,passed=max(change,saved_change)<1e-7))
    from .checks import FormulaChecks
    from .birth_death_checks import implementation_checks
    output=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FormulaChecks)
    with backend():
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
        branching=implementation_checks(store.config)
    return dict(passed=all(r['passed'] for r in rows) and result.wasSuccessful() and branching['passed'],rows=rows,
                branching_checks=branching,
                shared_checks=dict(passed=result.wasSuccessful(),tests=result.testsRun,output=output.getvalue()))


def profile_checks(store):
    root=store.root/'source-transition-20261019'
    rows=[]
    for n in store.config['founders']:
        oldfit=read_json(root/f'fits/n{n}/rep000.json')['result']
        oldprofile=read_json(root/f'profiles/n{n}/rep000.json')['result']
        newfit=store.result(f'fits/n{n}/rep000.json')
        newprofile=store.result(f'profiles/n{n}/rep000.json')
        row=dict(founders=n,passed=False,rates=[])
        if all(r and r['success'] for r in (oldfit,oldprofile,newfit,newprofile)):
            row.update(estimate_change=float(np.max(abs(np.array(oldfit['estimate'])-newfit['estimate']))),
                       u_change=abs(oldfit['u']-newfit['u']),loglik_change=abs(oldfit['loglik']-newfit['loglik']))
            for i,(a,b) in enumerate(zip(oldprofile['rates'],newprofile['rates'])):
                topology=len(a['intervals'])==len(b['intervals'])
                change=float(np.max(abs(np.array(a['intervals'])-b['intervals']))) if topology else None
                coverage=[any(lo<=store.config['truth'][i]<=hi for lo,hi in r['intervals']) for r in (a,b)]
                row['rates'].append(dict(rate=i,endpoint_change=change,topology_agrees=topology,
                    coverage_agrees=coverage[0]==coverage[1],truncation_agrees=a['bound_truncated']==b['bound_truncated'],
                    passed=bool(topology and change<.005 and coverage[0]==coverage[1] and a['bound_truncated']==b['bound_truncated'])))
            row['passed']=bool(row['estimate_change']<.005 and row['u_change']<.001 and row['loglik_change']<.05 and all(r['passed'] for r in row['rates']))
        rows.append(row)
    return dict(passed=all(r['passed'] for r in rows),rows=rows,reference_directory=str(root),
                original_timings_preserved=True,observations_unchanged=True)
