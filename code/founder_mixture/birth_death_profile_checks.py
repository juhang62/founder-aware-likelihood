"""Analytic checks of profile crossings, nuisance searches, and failure flags."""

import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from .inference import profile_sets
from .storage import read_json
from . import ROOT


class ProfileChecks(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT/'configs/founder-mixture-birth-death.json')

    def test_correlated_quadratic_and_refinement_settings(self):
        configs = []
        class Quadratic:
            def __init__(self, config, settings):
                configs.append((config, settings))
            def loglik(self, method, theta, data):
                x, y = theta
                return -.5*((x-.8)**2+(y-.6-.4*(x-.8))**2)/.02**2
        for tight in (False,True):
            with patch('founder_mixture.inference.Likelihood',Quadratic):
                result = profile_sets('G',{},self.config,dict(success=True,estimate=[.8,.6],loglik=0.),tight=tight)
            self.assertTrue(result['success'])
            half = .02*np.sqrt(self.config['profiles']['cutoff'])
            assert_allclose(result['rates'][0]['intervals'],[[.8-half,.8+half]],atol=1e-5)
            assert_allclose(result['rates'][1]['intervals'],[[.6-half*np.sqrt(1.16),.6+half*np.sqrt(1.16)]],atol=1e-5)
            cfg, settings = configs[-1]
            self.assertEqual(cfg['extinction'], self.config['tight_extinction' if tight else 'extinction'])
            self.assertEqual(settings.rtol,self.config['tight_integration' if tight else 'integration']['rtol'])
            for r in result['rates']:
                self.assertEqual(r['grid_size'],161 if tight else 81)
                self.assertEqual(r['nuisance_grid_size'],29 if tight else 15)

    def test_higher_profile_marks_fit_unresolved(self):
        class Quadratic:
            def __init__(self,*args): pass
            def loglik(self,method,theta,data):
                return -100*np.sum((np.array(theta)-.7)**2)
        with patch('founder_mixture.inference.Likelihood',Quadratic):
            result=profile_sets('G',{},self.config,dict(success=True,estimate=[.65,.7],loglik=-.25))
        self.assertFalse(result['success'])
        self.assertIn('exceeds the fitted maximum',result['reason'])

    def test_boundary_truncation_and_failed_point_estimate(self):
        class Boundary:
            def __init__(self,*args): pass
            def loglik(self,method,theta,data):
                x,y=theta
                return -100*(x-.3)**2-100*(y-.7)**2
        with patch('founder_mixture.inference.Likelihood',Boundary):
            result=profile_sets('G',{},self.config,dict(success=True,estimate=[.3,.7],loglik=0.))
        self.assertTrue(result['success'])
        self.assertEqual(result['rates'][0]['bound_truncated'],[True,False])
        half=np.sqrt(self.config['profiles']['cutoff']/200)
        assert_allclose(result['rates'][0]['intervals'],[[.3,.3+half]],atol=1e-5)
        failed=profile_sets('G',{},self.config,dict(success=False))
        self.assertFalse(failed['success'])
        self.assertIn('Unresolved',failed['reason'])
