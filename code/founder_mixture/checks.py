"""Independent formula, simulator, likelihood, and reproducibility checks."""

import unittest
from pathlib import Path
import json
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp, quad
from scipy.linalg import expm
from scipy.stats import exponnorm, norm, poisson, geom
from . import ROOT
from .model import candidate, from_config, noise, yule_coefficients
from .simulation import generate, simulate_counts, stream
from .likelihood import Likelihood, Settings, integrate_one
from .inference import fit, select_best, profile_sets


def moment_summary(samples, mean, covariance):
    """MC standard errors use independent cultures and covariance influences."""
    n = len(samples)
    empirical_mean = samples.mean(axis=0)
    empirical_covariance = np.cov(samples, rowvar=False, ddof=1)
    centered = samples-empirical_mean
    products = centered[:, :, None]*centered[:, None, :]
    mean_se = samples.std(axis=0, ddof=1)/np.sqrt(n)
    covariance_se = products.std(axis=0, ddof=1)/np.sqrt(n)*n/(n-1)
    return dict(n=n, mean=empirical_mean, exact_mean=mean, mean_mcse=mean_se,
                mean_z=(empirical_mean-mean)/mean_se, covariance=empirical_covariance,
                exact_covariance=covariance, covariance_mcse=covariance_se,
                covariance_z=(empirical_covariance-covariance)/covariance_se)


def moment_solver_refinement(config):
    """Compare matrix-exponential moments with two independent ODE accuracies."""
    from itertools import product
    parameters = [config["truth"], *product(*config["bounds"])]
    rows = []
    for theta in parameters:
        c = from_config(theta, config)
        for founders in config["founders"]:
            for founder_type in range(2):
                z0 = np.zeros(2)
                z0[founder_type] = founders
                def ode(t, state):
                    mean, V = state[:2], state[2:].reshape(2, 2)
                    return np.r_[c.A @ mean, (c.A @ V+V @ c.A.T+noise(mean, c.birth, c.switching)).ravel()]
                initial = np.r_[z0, np.zeros(4)]
                times = sorted(config["times"])
                ordinary = solve_ivp(ode, (0, max(times)), initial, t_eval=times, rtol=1e-9, atol=1e-11)
                tight = solve_ivp(ode, (0, max(times)), initial, t_eval=times, rtol=1e-12, atol=1e-13)
                if not ordinary.success or not tight.success:
                    raise ArithmeticError("Independent moment ODE failed")
                for j, t in enumerate(times):
                    mean, V = c.at(t, z0)
                    exact = np.r_[mean, V.ravel()]
                    relative_change = np.max(np.abs(ordinary.y[:, j]-tight.y[:, j])/np.maximum(1., abs(exact)))
                    agreement = np.max(np.abs(tight.y[:, j]-exact)/np.maximum(1., abs(exact)))
                    rows.append(dict(theta=theta, z0=z0, time=t, maximum_scaled_solver_change=float(relative_change),
                                     maximum_scaled_matrix_agreement=float(agreement), passed=bool(max(relative_change, agreement)<1e-8)))
    return dict(rows=rows, ordinary_rtol=1e-9, ordinary_atol=1e-11, tight_rtol=1e-12, tight_atol=1e-13,
                scale="Each component divided by max(1, abs(matrix-exponential moment))", passed=all(r["passed"] for r in rows))


def simulation_checks(store):
    config = store.config
    rows = []
    model = from_config(config["truth"], config)
    for founders in config["founders"]:
        data = store.dataset("checks", founders, 0, config["moment_check_cultures"])
        for cell in np.unique(data["cell"]):
            ids = np.flatnonzero(data["cell"] == cell)
            t, z0 = float(data["time"][ids[0]]), data["z0"][ids[0]]
            mean, covariance = model.at(t, z0)
            row = moment_summary(data["counts"][ids], mean, covariance)
            row.update(founders=founders, cell=int(cell), time=t, z0=z0)
            rows.append(row)
    yule = []
    for founders in (1, 50):
        birth, t, count = .8, 3., config["moment_check_cultures"]
        rng = stream(config["master_seed"], "checks", founders, 0, 0, 1, "yule")
        states, complete = simulate_counts(rng, count, np.array([founders, 0]), t,
                                          np.array([birth, 0.]), np.zeros(2), config["population_cap"])
        if not complete.all():
            raise RuntimeError("Incomplete Yule diagnostic simulation")
        x = states[:, 0].astype(float)
        E = np.exp(birth*t)
        variance = founders*E*(E-1)
        variance_se = np.std((x-x.mean())**2, ddof=1)/np.sqrt(count)*count/(count-1)
        yule.append(dict(founders=founders, n=count, mean=float(x.mean()), exact_mean=founders*E,
                         mean_mcse=float(x.std(ddof=1)/np.sqrt(count)), variance=float(x.var(ddof=1)),
                         exact_variance=variance, variance_mcse=float(variance_se)))
    maximum = max(max(np.abs(r["mean_z"]).max(), np.abs(r["covariance_z"]).max()) for r in rows)
    return dict(two_type=rows, yule=yule, max_absolute_two_type_z=float(maximum),
                passed=bool(maximum < 5),
                interpretation="Monte Carlo comparisons with estimated standard errors; a 5-SE flag is a diagnostic, not a theorem.")


class FormulaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/"configs/founder-mixture-pilot.json").read_text())

    def test_spectrum_box_and_normalization(self):
        for b1 in np.linspace(.3, 1.1, 17):
            for b2 in np.linspace(.3, 1.1, 17):
                c = candidate((b1, b2))
                self.assertGreater(c.L, 0)
                self.assertGreater(c.L/2-c.lambda2, 0)
                assert_allclose(c.ell @ c.r, 1.)
                assert_allclose(c.A @ c.r, c.L*c.r, atol=1e-13)
                assert_allclose(c.ell @ c.A, c.L*c.ell, atol=1e-13)
                self.assertGreaterEqual(np.linalg.eigvalsh(c.B).min(), 0)
        c = candidate((.8, .5))
        assert_allclose(c.L, .6424428900898053)
        assert_allclose([c.moments[t][0].sum(axis=0) for t in (5., 7.)],
                        [[28.77797146, 21.26840420], [104.10071935, 76.78826316]], rtol=1e-9)

    def test_moments_independent_ode_and_additivity(self):
        c = candidate((.8, .5))
        z0 = np.array([3, 7])
        def ode(t, state):
            m, V = state[:2], state[2:].reshape(2, 2)
            return np.r_[c.A @ m, (c.A @ V+V @ c.A.T+noise(m, c.birth, c.switching)).ravel()]
        initial = np.r_[z0, np.zeros(4)]
        loose = solve_ivp(ode, (0, 7), initial, t_eval=[5, 7], rtol=1e-9, atol=1e-11)
        tight = solve_ivp(ode, (0, 7), initial, t_eval=[5, 7], rtol=1e-12, atol=1e-13)
        for j, t in enumerate((5, 7)):
            m, V = c.at(t, z0)
            assert_allclose(m, expm(c.A*t) @ z0)
            assert_allclose(m, tight.y[:2, j], rtol=2e-11)
            assert_allclose(V, tight.y[2:, j].reshape(2, 2), rtol=2e-11)
            assert_allclose(loose.y[:, j], tight.y[:, j], rtol=1e-8)
            assert_allclose(V, 3*c.at(t, [1, 0])[1]+7*c.at(t, [0, 1])[1])
        for z0 in ([1, 0], [0, 1]):
            a, v, k, tau = c.amplitude(z0)
            a50, v50, k50, tau50 = c.amplitude(np.array(z0)*50)
            assert_allclose([a50, v50, k50, tau50], [50*a, 50*v, 50*k, tau])

    def test_yule_exact_benchmarks(self):
        for founders in (1, 50):
            out = yule_coefficients(.8, founders)
            assert_allclose([out["v"], out["B"], out["shape"], out["scale"]], [1, 1, founders, 1])
        for t in (.5, 2.):
            E = np.exp(.8*t)
            for n in (1, 2, 5, 12):
                integrated = quad(lambda w: poisson.pmf(n-1, w*(E-1))*np.exp(-w),
                                  0, np.inf, epsabs=1e-12)[0]
                assert_allclose(integrated, geom.pmf(n, 1/E), rtol=1e-8)
            # A geometric count mixed with its future Gamma amplitude has the
            # exact joint law; W is not defined to cancel an endpoint residual.
            rng = np.random.default_rng(90210)
            counts = rng.geometric(1/E, 50000)
            W = rng.gamma(counts, 1/E)
            self.assertLess(abs(W.mean()-1), .025)
            self.assertLess(abs(W.var()-1), .06)
            assert_allclose(1+(E-1), E)
            assert_allclose((E-1)+(E-1)**2, E*(E-1))

    def test_m0_against_analytic_exponential_convolution(self):
        h, T, variance, tau = np.array([17., 23.]), np.zeros((2, 2)), 4., .7
        s = Settings(tail_probability=1e-12)
        for y in (np.array([-4., -2.]), np.array([10., 20.]), np.array([1000., 2000.])):
            val, err, _, failed = integrate_one(y, h, T, variance, 1., tau, np.log(tau),
                                                np.log(tau*1e-12), np.log(-tau*np.log(1e-12)), s.rtol, s.atol, s.max_depth)
            unit = h/np.linalg.norm(h)
            x = y @ unit
            transverse = y @ np.array([-unit[1], unit[0]])
            expected = norm.logpdf(transverse, scale=2)+exponnorm.logpdf(x, np.linalg.norm(h)*tau/2, scale=2)
            self.assertFalse(failed)
            assert_allclose(val, expected, atol=1e-7, rtol=0)

    def test_narrow_integrals_against_independent_reference(self):
        config = self.config
        base, tight = Likelihood(config), Likelihood(config, Settings(**config["tight_integration"]))
        for founders in (1, 50):
            for theta in ([.8, .5], [.3, .3], [1.1, 1.1], [.3, 1.1], [1.1, .3]):
                z0 = [founders, 0]
                for method in ("M", "M0"):
                    ys = np.array([[-2., 1.], [50., 45.]])*founders
                    v = base.logpdf(method, theta, ys, 7., z0)
                    w = tight.logpdf(method, theta, ys, 7., z0, reference=True)
                    assert_allclose(v, w, atol=1e-3, rtol=0)

    def test_reproducibility_no_truth_in_inference_and_cap(self):
        cfg = self.config
        one = generate(cfg, "smoke", 1, 0, 3)
        two = generate(cfg, "smoke", 1, 0, 5)
        for cell in range(4):
            for key in ("y", "counts"):
                assert_array_equal(one[key][one["cell"] == cell], two[key][two["cell"] == cell][:3])
        ll = Likelihood(cfg)
        val = ll.loglik("M", [.7, .6], one)
        one["counts"][:] = -100000
        assert_allclose(ll.loglik("M", [.7, .6], one), val)
        counts, complete = simulate_counts(np.random.default_rng(1), 5, np.array([1, 0]), 100.,
                                          np.array([.8, .5]), np.array([.6, .4]), 2)
        self.assertFalse(complete.any())
        self.assertTrue((counts.sum(axis=1) == 2).all())

    def test_m0_large_perpendicular_readout_against_exact_convolution(self):
        # Equal birth rates give an exact exponential amplitude for one founder.
        # Large deviations perpendicular to the mean direction contribute only
        # a constant Gaussian log density and must not spoil amplitude integration.
        config = dict(self.config, times=[9., 11.])
        theta, z0, t = (1.1, 1.1), np.array([1, 0]), 11.
        model = from_config(theta, config)
        a, _, k, tau = model.amplitude(z0)
        assert_allclose(k, 1., rtol=1e-13)
        h = model.at(t, z0)[0]/a
        length = np.linalg.norm(h)
        along = h/length
        across = np.array([-along[1], along[0]])
        sd = np.sqrt(config["measurement_variance"])
        y = np.array([u*along+v*across for u, v in
                      [(-5., 10.), (100000., 0.), (100000., 50000.), (200000., 100000.)]])
        expected = (exponnorm.logpdf(y @ along, length*tau/sd, scale=sd)
                    + norm.logpdf(y @ across, scale=sd))
        settings = Settings(tail_probability=1e-12, rtol=2e-10, atol=1e-13, max_depth=8)
        likelihood = Likelihood(config, settings)
        for reference in (False, True):
            actual = likelihood.logpdf("M0", theta, y, t, z0, reference=reference)
            assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_sharp_m0_fit_reaches_known_better_candidate(self):
        # A later-time many-founder likelihood exposed premature convergence:
        # a profile search found this candidate above the reported maximum.
        config = dict(self.config, times=[9., 11.], master_seed=20261013,
                      population_cap=50000000)
        data = generate(config, "inference", 50, 0, 10)
        likelihood = Likelihood(config, Settings(**config["tight_integration"]))
        candidate_loglik = likelihood.loglik("M0", [.8001229683, .4978321830], data)
        for tight in (False, True):
            result = fit("M0", data, config, tight=tight)
            self.assertTrue(result["success"])
            self.assertGreaterEqual(result["loglik"], candidate_loglik-1e-5)

    def test_optimizer_tie_and_disconnected_profile_sets(self):
        from unittest.mock import patch
        good = dict(objective=10., success=True)
        tied = dict(objective=10.-1e-12, success=False)
        better = dict(objective=9., success=False)
        self.assertIs(select_best([good, tied]), good)
        self.assertIs(select_best([good, better]), better)
        # An independent smooth objective has two separated, known level sets.
        class DoublePeak:
            def __init__(self, *args, **kwargs):
                pass
            def loglik(self, method, theta, data):
                x, y = theta
                return -20000*((x-.5)*(x-.9))**2-100*(y-.7)**2
        fitted = dict(success=True, estimate=[.5, .7], loglik=0.)
        with patch("founder_mixture.inference.Likelihood", DoublePeak):
            result = profile_sets("G", {}, self.config, fitted)
        self.assertTrue(result["success"])
        self.assertTrue(result["rates"][0]["disconnected"])
        self.assertEqual(len(result["rates"][0]["intervals"]), 2)
        for a, b in result["rates"][0]["intervals"]:
            for x in (a, b):
                assert_allclose(-2*DoublePeak().loglik("G", [x, .7], {}), 2.705543, atol=.005)
        half_width = np.sqrt(2.705543/200)
        assert_allclose(result["rates"][1]["intervals"], [[.7-half_width, .7+half_width]], atol=2e-5)
        class Flat:
            def __init__(self, *args, **kwargs):
                pass
            def loglik(self, *args):
                return 0.
        with patch("founder_mixture.inference.Likelihood", Flat):
            flat = profile_sets("G", {}, self.config, dict(success=True, estimate=[.7, .7], loglik=0.))
        self.assertTrue(flat["success"])
        for curve in flat["rates"]:
            self.assertTrue(curve["flat"])
            self.assertEqual(curve["bound_truncated"], [True, True])
            assert_allclose(curve["intervals"], [[.3, 1.1]])


def run_unit_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FormulaChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=[text for _, text in result.failures], errors=[text for _, text in result.errors])
