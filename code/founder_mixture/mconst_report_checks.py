"""Check paired summaries and unresolved-case bounds against known examples."""

import re
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from .mconst_report import collect


class FakeStore:
    root=Path('/tmp/mconst-report-check-nonexistent')
    config=dict(truth=[.8,.5],founders=[1,50],master_seed=20261017)

    def baseline_path(self,path):return path

    def old_fit(self,n,rep,method='M0'):
        scale={'M':2.,'M0':.5,'G':3.,'EG':3.,'Mconst':1.}[method]
        return dict(success=True,estimate=[.8+scale*.01,.5+scale*.02],loglik=-1.,boundary=[False,False],
                    residual_variance=4.,u=np.log(2),variance_boundary=False,variance_cap=100.,expansions=[{}],
                    failed_starts=0,seconds=1.)

    def result(self,path):
        if path in ('refinement.json','pilot.json'):return dict(rows=[])
        matched=re.search(r'rep(\d+)',path)
        if not matched:return None
        rep=int(matched[1])
        if path.startswith('fits/'):
            if rep==99:return dict(success=False,estimate=None,loglik=None,reason='Deliberate failed fit',seconds=1.)
            return self.old_fit(1,rep,'Mconst')
        if path.startswith('profiles/'):
            if rep==99:return dict(success=False,reason='Unresolved point estimate',seconds=0.)
            return synthetic_profile(self,1,rep,'Mconst')
        return None


def synthetic_profile(store,n,rep,method):
    rates=[]
    for rate,truth in enumerate(store.config['truth']):
        covers=rep<(90 if method=='Mconst' else 80) if rate==0 else True
        half=.01 if method=='Mconst' else .02
        center=truth if covers else truth-.2
        rates.append(dict(intervals=[[center-half,center+half]],length=2*half,disconnected=False,flat=False,bound_truncated=[False,False]))
    return dict(success=True,rates=rates,seconds=1.)


class SummaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('founder_mixture.mconst_report.read_json',return_value=dict(metrics=[])), \
             patch('founder_mixture.mconst_report.old_profile',side_effect=synthetic_profile):
            cls.result=collect(FakeStore())

    def test_paired_constant_ratios_and_exclusions(self):
        for r in self.result['paired']:
            expected=-1. if r['comparison']=='M-Mconst' else .5
            self.assertAlmostEqual(r['rmse_reduction'],expected)
            assert_allclose(r['rmse_reduction_ci95'],[expected,expected],atol=1e-12)
            self.assertEqual(r['excluded_replicates'],[99])
            self.assertEqual(r['paired_successful'],99)
            self.assertEqual(r['paired_profiles'],99)

    def test_unresolved_coverage_bounds(self):
        for r in self.result['metrics']:
            self.assertEqual(r['profile_resolved'],99)
            if r['rate']=='b1':
                self.assertEqual(r['covered'],90)
                self.assertAlmostEqual(r['coverage'],90/99)
                assert_allclose(r['coverage_bounds'],[.9,.91])
                self.assertGreater(r['coverage_mcse'],0.)
            else:
                self.assertEqual(r['coverage'],1.)
                self.assertEqual(r['coverage_mcse'],0.)
                self.assertLess(r['coverage_ci95'][0],1.)
                assert_allclose(r['coverage_bounds'],[.99,1.])
        for r in self.result['paired']:
            if r['rate']=='b1':
                assert_allclose(r['coverage_difference_bounds'],[-.11,-.1])
                self.assertAlmostEqual(r['coverage_difference'],-10/99)
            else:
                assert_allclose(r['coverage_difference_bounds'],[0.,.01])

    def test_whole_dataset_resampling_and_lengths(self):
        for r in self.result['paired']:
            self.assertAlmostEqual(r['length_difference'],.02)
            assert_allclose(r['length_difference_ci95'],[.02,.02],atol=1e-12)
        for record in self.result['bootstrap'].values():
            self.assertEqual(record['indices'].shape,(2000,100))
            self.assertEqual(record['indices'].min(),0)
            self.assertEqual(record['indices'].max(),99)


if __name__=='__main__':unittest.main()
