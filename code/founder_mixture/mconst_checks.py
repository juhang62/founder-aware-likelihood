"""Independent and analytic checks for the constant-variance extension."""

import importlib.util
import io
import sys
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from . import ROOT
from .storage import read_json
from .likelihood import Likelihood, Settings
from .model import from_config
from . import mconst_inference as inference


class AnalyticChecks(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT/'configs/founder-mixture-birth-death.json')
        self.config['mconst'] = read_json(ROOT/'configs/founder-mixture-birth-death-mconst.json')
        # Small grids are sufficient for these analytic test functions only.
        for key in ('optimization', 'tight_optimization'):
            self.config[key] = dict(self.config[key], grid_size=5, starts=3)
        self.config['mconst'] = dict(self.config['mconst'], variance_grid_size=5, tight_variance_grid_size=7)
        self.config['profiles'] = dict(self.config['profiles'], grid_size=17, nuisance_grid_size=5)
        self.data = dict(y=np.array([[10., 10.]]), time=np.array([9.]), z0=np.array([[1.,0.]]), cell=np.array([0]))

    def exercise(self, fun, expected, intervals, tight=False):
        class Quadratic:
            def __init__(self, *args): self.timing = {}
            def loglik(self, method, theta, data): return fun(np.asarray(theta))
        with patch.object(inference, 'Likelihood', Quadratic), patch.object(inference, 'tail_bound', return_value=-1e6):
            f = inference.fit(self.data, self.config, tight=tight)
            self.assertTrue(f['success'], f)
            assert_allclose(f['theta'], expected, atol=3e-5)
            p = inference.profile_sets(self.data, self.config, f, tight=tight)
            self.assertTrue(p['success'], p.get('reason'))
            for rate, desired in enumerate(intervals):
                assert_allclose(p['rates'][rate]['intervals'], desired, atol=2e-5)
            bad = dict(f, loglik=f['loglik']-.1)
            self.assertFalse(inference.profile_sets(self.data, self.config, bad, tight)['success'])
            return f, p

    def test_correlated_quadratic(self):
        half = .02*np.sqrt(self.config['profiles']['cutoff'])
        def fun(t):
            x,y,u=t
            return -.5*((x-.8)**2+(y-.6-.4*(x-.8))**2+(u-1-.3*(x-.8)-.2*(y-.6))**2)/.02**2
        for tight in (False,True):
            self.exercise(fun, [.8,.6,1.], [[[.8-half,.8+half]],
                          [[.6-half*np.sqrt(1.16),.6+half*np.sqrt(1.16)]]], tight)

    def test_zero_variance_and_truncation(self):
        half = np.sqrt(self.config['profiles']['cutoff']/200)
        f,p=self.exercise(lambda t:-100*((t[0]-.3)**2+(t[1]-.7)**2+t[2]**2),
                         [.3,.7,0.], [[[.3,.3+half]],[[.7-half,.7+half]]])
        self.assertTrue(f['variance_boundary'])
        self.assertEqual(p['rates'][0]['bound_truncated'],[True,False])

    def test_multiple_maxima_disconnected_sets(self):
        def fun(t):
            return -1000*min((t[0]-.5)**2,(t[0]-.9)**2)-1000*(t[1]-.7)**2-(t[2]-1)**2
        class Multi:
            def __init__(self,*args): self.timing={}
            def loglik(self,method,theta,data): return fun(theta)
        with patch.object(inference,'Likelihood',Multi), patch.object(inference,'tail_bound',return_value=-1e6):
            f=inference.fit(self.data,self.config)
            self.assertTrue(f['success'])
            p=inference.profile_sets(self.data,self.config,f)
        self.assertTrue(p['success'],p.get('reason'))
        h=np.sqrt(self.config['profiles']['cutoff']/2000)
        assert_allclose(p['rates'][0]['intervals'],[[.5-h,.5+h],[.9-h,.9+h]],atol=2e-5)
        self.assertTrue(p['rates'][0]['disconnected'])
        self.assertAlmostEqual(p['rates'][0]['length'],4*h,places=4)

    def test_flat_profile(self):
        class Flat:
            def __init__(self,*args): self.timing={}
            def loglik(self,method,theta,data): return -100*(theta[1]-.7)**2-(theta[2]-1)**2
        with patch.object(inference,'Likelihood',Flat), patch.object(inference,'tail_bound',return_value=-1e6):
            fitted=dict(success=True,estimate=[.7,.7],theta=[.7,.7,1.],loglik=0.,variance_cap=100.)
            p=inference.profile_sets(self.data,self.config,fitted)
        self.assertTrue(p['success'],p.get('reason'))
        self.assertTrue(p['rates'][0]['flat'])
        assert_allclose(p['rates'][0]['intervals'],[[.3,1.1]])

    def test_tail_bound_and_mixture_moments(self):
        ys=np.array([[0.,0.],[-2.,1.],[10.,30.],[1000.,2000.]])
        weights=np.array([.2,.3,.5]); means=np.array([[1.,2.],[4.,3.],[5.,8.]])
        for pi, v, cap in product((.01,.5,.999),(4.,20.,100.),(0.,4.)):
            if v<cap: continue
            empty=multivariate_normal.logpdf(ys,mean=[0,0],cov=4*np.eye(2))
            living=logsumexp([np.log(w)+multivariate_normal.logpdf(ys,mean=m,cov=(4+v)*np.eye(2)) for w,m in zip(weights,means)],axis=0)
            total=np.logaddexp(np.log1p(-pi)+empty,np.log(pi)+living).sum()
            self.assertLessEqual(total,inference.tail_bound(dict(y=ys),cap)+1e-10)
            component_weights=np.r_[1-pi,pi*weights]
            component_means=np.vstack(([0,0],means))
            mu=component_weights@component_means
            base=sum(w*(4*np.eye(2)+np.outer(m-mu,m-mu)) for w,m in zip(component_weights,component_means))
            added=sum(w*((4+(v if j else 0))*np.eye(2)+np.outer(m-mu,m-mu)) for j,(w,m) in enumerate(zip(component_weights,component_means)))
            assert_allclose(added-base,pi*v*np.eye(2),atol=1e-12)


def run_checks(store):
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalyticChecks))
    store.save_result('analytic-checks.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=output.getvalue()))
    print(output.getvalue(), flush=True)
    path=store.root/'baseline/likelihood.py'
    spec=importlib.util.spec_from_file_location('founder_mixture._mconst_baseline_likelihood',path)
    legacy=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=legacy
    spec.loader.exec_module(legacy)
    rows=[]
    for n in store.config['founders']:
        data=store.dataset(n,0)
        cfg={k:v for k,v in store.config.items() if k!='truth'}
        like,old=Likelihood(cfg),legacy.Likelihood(cfg)
        points=list(product(*[np.linspace(a,b,7) for a,b in cfg['bounds']]))
        points += [store.old_fit(n,0,m)['estimate'] for m in cfg['methods']]
        zero=max(abs(like.loglik('Mconst',list(t)+[0.],data)-like.loglik('M0',t,data)) for t in points)
        for method in cfg['methods']:
            maximum=max(abs(like.loglik(method,t,data)-old.loglik(method,t,data)) for t in points)
            rows.append(dict(founders=n,method=method,parameters=len(points),baseline_max_change=maximum,zero_variance_max_change=zero))
        contaminated=dict(data,counts=np.zeros_like(data['y']),survived=np.ones(len(data['y']),bool),truth=np.array([99.,99.]))
        assert like.loglik('Mconst',[.7,.6,1.],data)==like.loglik('Mconst',[.7,.6,1.],contaminated)
        ys=np.array([[0.,0.],[-2.,1.],[10.,30.]])
        theta=[.7,.6]; z0=np.array([n,0]); t=9.
        weights=from_config(theta,cfg).survival(t,z0)
        empty=multivariate_normal.logpdf(ys,mean=[0.,0.],cov=4*np.eye(2))
        for v in (0.,4.,1e4):
            living=like.logpdf('Mconst',theta,ys,t,z0,survivor=True,residual_variance=v)
            independent=Likelihood(dict(cfg,measurement_variance=4+v)).logpdf('M0',theta,ys,t,z0,survivor=True)
            assert_allclose(living,independent,atol=1e-10,rtol=0)
            full=like.logpdf('Mconst',theta,ys,t,z0,residual_variance=v)
            expected=np.logaddexp(weights['log_empty']+empty,weights['log_survival']+living)
            assert_allclose(full,expected,atol=1e-10,rtol=0)
        for t in points:
            c=from_config(t,cfg)
            for z0,time in ((np.array([n,0]),9.),(np.array([0,n]),11.)):
                s=c.survival(time,z0)
                assert_allclose(np.exp(s['log_empty'])+np.exp(s['log_survival']),1.,atol=1e-14)
                _,_,k,tau=c.positive_amplitude(z0)
                assert k>0 and tau>0
    return dict(passed=result.wasSuccessful() and all(r['baseline_max_change']<1e-9 and r['zero_variance_max_change']<1e-8 for r in rows),
                analytic_checks=dict(passed=result.wasSuccessful(),tests=result.testsRun,output=output.getvalue()),comparisons=rows,
                latent_labels_excluded=True,normalization='Positive Gaussian covariance; normalized positive Gamma; weights sum to one.')


def density_checks(store):
    rows=[]
    for n in store.config['founders']:
        data=store.dataset(n,0); cfg=store.config
        fitted=store.result(f'fits/n{n}/rep000.json')
        if not fitted or not fitted['success']:
            return dict(passed=False,reason='Missing successful pilot fit',rows=rows)
        parameters=[cfg['truth'],*product(*cfg['bounds']),fitted['estimate']]
        variances=[0.,4.,fitted['residual_variance'],fitted['variance_cap']]
        regular=Likelihood(cfg)
        tighter=Likelihood(dict(cfg,extinction=cfg['tight_extinction']),Settings(**cfg['tight_integration']))
        for group in np.unique(data['cell']):
            ids=np.flatnonzero(data['cell']==group)
            ordered=ids[np.argsort(data['y'][ids].sum(axis=1),kind='stable')]
            picked=ordered[np.linspace(0,len(ids)-1,5).round().astype(int)]
            ys=np.vstack((data['y'][picked],[[0.,0.],[-2.,1.]]))
            first=ids[0]; t=float(data['time'][first]); z0=data['z0'][first]
            for theta,v in product(parameters,variances):
                for survivor in (True,False):
                    a=regular.logpdf('Mconst',theta,ys,t,z0,survivor=survivor,residual_variance=v)
                    b=tighter.logpdf('Mconst',theta,ys,t,z0,survivor=survivor,residual_variance=v)
                    c=tighter.logpdf('Mconst',theta,ys,t,z0,reference=True,survivor=survivor,residual_variance=v)
                    rows.append(dict(founders=n,group=int(group),theta=list(theta),variance=v,survivor=survivor,
                        production_tight_change=float(np.max(abs(a-b))),independent_change=float(np.max(abs(b-c)))))
    return dict(passed=all(max(r['production_tight_change'],r['independent_change'])<1e-3 for r in rows),rows=rows)
