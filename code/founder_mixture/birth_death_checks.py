"""Delivery 1 deterministic checks; no large two-type Monte Carlo samples."""

from itertools import product
from time import perf_counter
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from .model import from_config, noise
from .extinction import rhs
from .one_type import CountReference, scalar_parameters
from .likelihood import Likelihood
from .simulation import generate, simulate_birth_death_counts, stream
from .inference import fit


def tight_config(config):
    return dict(config, extinction=config["tight_extinction"], integration=config["tight_integration"])


def preflight(config):
    start = perf_counter()
    spectrum, states, endpoints, refinement = [], [], [], []
    grid = product(*[np.linspace(*bound, 7) for bound in config["bounds"]])
    parameters = [("truth", config["truth"]), *[(f"grid{i:02d}", x) for i, x in enumerate(grid)]]
    for label, theta in parameters:
        c, ct = from_config(theta, config), from_config(theta, tight_config(config))
        spectrum.append(dict(parameter=label, theta=theta, L=c.L, lambda2=c.lambda2,
                             margin=c.L/2-c.lambda2, r=c.r, ell=c.ell))
        for n in config["founders"]:
            for i in range(2):
                z0 = np.eye(2, dtype=int)[i]*n
                ap, vp, k, tau = c.positive_amplitude(z0)
                apt, vpt, _, _ = ct.positive_amplitude(z0)
                w = c.survival(config["times"][0], z0)
                states.append(dict(parameter=label, theta=theta, z0=z0, founders=n, founder_type=i,
                    pi_infinity=w["pi_infinity"], a_positive=ap, v_positive=vp, shape=k, scale=tau))
                for t in config["times"]:
                    m, V = c.at(t, z0)
                    w, wt = c.survival(t, z0), ct.survival(t, z0)
                    # JSON has no infinity; preserve zero probabilities explicitly.
                    log_weights = {key: (value if np.isfinite(value) else None) for key, value in w.items() if key.startswith("log_")}
                    endpoints.append(dict(parameter=label, theta=theta, z0=z0, founders=n, founder_type=i,
                        time=t, mean=m, covariance=V, pi=w["pi"], epsilon=w["epsilon"],
                        survivor_mean_relative_difference=w["pi"]/w["pi_infinity"]-1, **log_weights))
                    changes = dict(pi=abs(w["pi"]-wt["pi"]), pi_infinity=abs(w["pi_infinity"]-wt["pi_infinity"]),
                                   a_positive=abs(ap-apt), v_positive=abs(vp-vpt))
                    refinement.append(dict(parameter=label, z0=z0, time=t, changes=changes,
                                           passed=max(changes["pi"], changes["pi_infinity"]) < 1e-9))
    # Analytic bounds apply to this fixed-rate box, not just its grid.
    b = np.asarray(config["bounds"])
    lower_L = float(np.min(b[:, 0]-config["death"]))
    max_trace = float(np.sum(b[:, 1]-config["death"])-sum(config["switching"]))
    min_gap = 2*np.sqrt(np.prod(config["switching"]))
    margin_bound = (3*min_gap-max_trace)/4
    return dict(spectrum=spectrum, founder_states=states, endpoints=endpoints, extinction_refinement=refinement,
        analytic_L_lower_bound=lower_L, analytic_margin_lower_bound=margin_bound,
        observation_times=config["times"], time_decision="Retain days 9 and 11: expected sizes agree with the prespecified design; survival substitution is small at truth, while its larger lower-corner discrepancy remains an explicit limitation.",
        passed=bool(lower_L > 0 and margin_bound > 0 and all(r["passed"] for r in refinement)), seconds=perf_counter()-start)


def implementation_checks(config):
    from unittest.mock import patch
    start = perf_counter()
    outcomes, moment_rows, extinction_rows = [], [], []
    def check(name, action):
        tick = perf_counter()
        try:
            action()
            outcomes.append(dict(name=name, passed=True, seconds=perf_counter()-tick))
        except Exception as error:
            outcomes.append(dict(name=name, passed=False, failure=f"{type(error).__name__}: {error}", seconds=perf_counter()-tick))

    def deterministic():
        events = np.array([[1,0],[0,1],[-1,0],[0,-1],[-1,1],[1,-1]])
        for theta in [config["truth"], *product(*config["bounds"])]:
            c = from_config(theta, config)
            assert_allclose(c.A @ c.r, c.L*c.r, atol=1e-13)
            assert_allclose(c.ell @ c.A, c.L*c.ell, atol=1e-13)
            assert_allclose(c.ell @ c.r, 1., atol=1e-14)
            assert_allclose(c.u2-c.ell*c.ell, c.v, atol=1e-12)
            for z0 in ([1,0],[0,1],[50,0],[0,50],[3,7]):
                z0 = np.array(z0)
                rates = np.r_[c.birth*z0, c.death*z0, c.switching*z0]
                assert_allclose(rates @ events, c.A @ z0, atol=1e-13)
                assert_allclose(np.einsum("k,ki,kj->ij", rates, events, events), noise(z0,c.birth,c.switching,c.death))
                def ode(t, state):
                    m, V = state[:2], state[2:].reshape(2,2)
                    return np.r_[c.A @ m,(c.A @ V+V @ c.A.T+noise(m,c.birth,c.switching,c.death)).ravel()]
                solutions = [solve_ivp(ode,(0,max(config["times"])),np.r_[z0,np.zeros(4)],
                    t_eval=config["times"],rtol=r,atol=a) for r,a in ((1e-9,1e-11),(1e-12,1e-13))]
                assert all(s.success for s in solutions)
                for j,t in enumerate(config["times"]):
                    m,V = c.at(t,z0)
                    exact = np.r_[m,V.ravel()]
                    error = np.max(abs(solutions[1].y[:,j]-exact)/np.maximum(1,abs(exact)))
                    change = np.max(abs(solutions[0].y[:,j]-solutions[1].y[:,j])/np.maximum(1,abs(exact)))
                    assert max(error,change) < 1e-8
                    assert np.linalg.eigvalsh(V).min() >= -1e-10
                    assert_allclose(m,z0[0]*c.at(t,[1,0])[0]+z0[1]*c.at(t,[0,1])[0])
                    assert_allclose(V,z0[0]*c.at(t,[1,0])[1]+z0[1]*c.at(t,[0,1])[1])
                    mp,Vp = c.survivor_moments(t,z0)
                    w=c.survival(t,z0)
                    assert_allclose(w["pi"]*mp,m)
                    assert_allclose(w["pi"]*(Vp+np.exp(w["log_empty"])*np.outer(mp,mp)),V,rtol=1e-12)
                    assert_allclose(np.logaddexp(w["log_empty"],w["log_survival"]),0,atol=1e-14)
                    moment_rows.append(dict(theta=theta,z0=z0,time=t,scaled_ode_change=change,scaled_matrix_agreement=error))
            horizon= max(100.,40/c.L)
            solution=solve_ivp(lambda t,q:rhs(q,c.birth,c.death,c.switching),(0,2*horizon),[0.,0.],
                t_eval=[*config["times"],horizon,2*horizon],rtol=1e-12,atol=1e-14)
            assert solution.success
            assert np.min(np.diff(solution.y,axis=1)) > -1e-11
            assert_allclose(solution.y[:,-2:],np.tile(c.extinction["rho"],(2,1)).T,atol=1e-11,rtol=0)
            assert np.max(solution.y-c.extinction["rho"][:,None]) < 1e-11
            extinction_rows.append(dict(theta=theta,horizon=horizon,maximum_long_time_error=float(np.max(abs(solution.y[:,-2:]-c.extinction["rho"][:,None]))),residual=c.extinction["residual"]))
    check("six events, moments, eigenvectors, additivity, survivor moments, normalization, long-time extinction",deterministic)

    def equal_rates():
        c=from_config([.8,.8],config)
        assert_allclose(c.ell,[.6,.6])
        for z0 in ([1,0],[0,1],[50,0],[0,50],[3,7]):
            for t in config["times"]:
                p=scalar_parameters(.8,.1,sum(z0),t)
                m,V=c.at(t,z0)
                assert_allclose([m.sum(),V.sum()],[p["mean"],p["variance"]],rtol=1e-10)
                assert_allclose(c.extinction["q"][t],p["q"],atol=1e-11,rtol=0)
                assert_allclose(c.positive_amplitude(z0)[:2],[.6*p["ap"],.36*p["vp"]],rtol=1e-11)
    check("equal rates agree with the scalar total-population law and amplitude normalization",equal_rates)

    def no_death():
        cfg={k:v for k,v in config.items() if k != "death"}
        zero=dict(cfg,death=[0.,0.])
        c=from_config(config["truth"],cfg)
        likelihood=Likelihood(zero)
        for n in (1,50):
            z0=[n,0]
            assert_allclose(c.positive_amplitude(z0),c.amplitude(z0),rtol=1e-13)
            assert c.survival(9.,z0)["log_empty"] == -np.inf
            for method in ("G","M","M0"):
                assert_allclose(likelihood.logpdf(method,config["truth"],[[0,0],[100,150]],9.,z0),
                                Likelihood(cfg).logpdf(method,config["truth"],[[0,0],[100,150]],9.,z0),atol=0,rtol=0)
            assert_allclose(likelihood.logpdf("G",config["truth"],[[0,0],[100,150]],9.,z0),
                            likelihood.logpdf("EG",config["truth"],[[0,0],[100,150]],9.,z0),atol=0,rtol=0)
        for key,value in generate(cfg,"smoke",1,0,2).items():
            assert_array_equal(value,generate(zero,"smoke",1,0,2)[key])
    check("zero death recovers amplitude, likelihood, and seeded simulation",no_death)

    def software():
        from .__main__ import _fit_task
        # These small checks use recorded keys 900--903, separate from later checks.
        small=dict(config,times=[.1,.2])
        data=generate(small,"checks",1,900,3)
        original={k:v.copy() for k,v in data.items()}
        extended=generate(small,"checks",1,901,3)
        repeated=generate(small,"checks",1,900,5)
        for group in range(4):
            for key in ("y","counts"):
                assert_array_equal(original[key][original["cell"]==group],repeated[key][repeated["cell"]==group][:3])
        assert not np.array_equal(data["y"],extended["y"])
        assert (data["counts"]>=0).all()
        for method in config["methods"]:
            before=Likelihood(small).loglik(method,[.7,.6],data)
            data["counts"][:]=-123
            data["survived"]=np.zeros(len(data["y"]))
            assert_allclose(Likelihood(small).loglik(method,[.7,.6],data),before,rtol=0,atol=0)
            def spy(m,observed,known):
                assert set(observed)=={"y","time","z0","cell"}
                assert "truth" not in known
                return {k: observed[k].tolist() for k in observed}
            with patch("founder_mixture.__main__.fit",spy):
                assert _fit_task(method,data,small)==_fit_task(method,original,small)
        def rng(key):
            return stream(config["master_seed"],"checks",1,0,0,key,"ctmc")
        states,done=simulate_birth_death_counts(rng(902),5,np.array([0,0]),10.,np.array([.8,.5]),np.array([.1,.1]),np.array([.6,.4]),100)
        assert not states.any() and done.all()
        states,done=simulate_birth_death_counts(rng(903),5,np.array([1,0]),100.,np.array([.8,.5]),np.zeros(2),np.array([.6,.4]),2)
        assert not done.any() and (states.sum(axis=1)==2).all()
        # Actual bounded optimization is invariant to arbitrary latent fields.
        known={k:v for k,v in small.items() if k != "truth"}
        a=fit("EG",original,known)
        b=fit("EG",data,known)
        assert a["success"] and b["success"]
        assert_allclose(a["estimate"],b["estimate"],atol=0,rtol=0)
        assert a["loglik"]==b["loglik"]
    check("absorption, caps, deterministic streams, dataset extension, latent-data exclusion",software)

    def tiny_extinction():
        c=from_config(config["truth"],config)
        for z0 in ([50,0],[0,50]):
            for t in config["times"]:
                weights=c.survival(t,z0)
                assert weights["pi"] == 1 and np.isfinite(weights["log_empty"])
                for method in ("EG","M","M0"):
                    ll=Likelihood(config)
                    living=ll.logpdf(method,config["truth"],[[0,0],[-2,1]],t,z0,survivor=True)
                    empty=-np.log(8*np.pi)-np.array([0.,5.])/8
                    full=ll.logpdf(method,config["truth"],[[0,0],[-2,1]],t,z0)
                    assert_allclose(full,np.logaddexp(weights["log_empty"]+empty,weights["log_survival"]+living),atol=1e-12,rtol=0)
    check("nonzero tiny extinction log weights remain in every near-zero observation density",tiny_extinction)

    def references():
        for n in (1,5):
            for d in (0.,.1):
                initial=CountReference(.8,d,n,0.)
                assert initial.mass[n] == 1 and initial.mass.sum()==1
                for t in (.5,2.,5.):
                    ref=CountReference(.8,d,n,t,1e-14)
                    p=ref.parameters
                    assert abs(ref.mass.sum()+ref.omitted-1)<1e-12
                    assert_allclose(ref.mass @ ref.counts,p["mean"],rtol=1e-10)
                    assert_allclose(ref.mass @ (ref.counts-p["mean"])**2,p["variance"],rtol=1e-9)
                    if d==0:
                        assert_allclose([p["k"],p["tau"]],[n,1.])
    check("scalar limiting distributions, normalization, analytic moments, and Yule Gamma law",references)
    return dict(checks=outcomes,moments=moment_rows,extinction=extinction_rows,
                small_check_replicate_keys=[900,901,902,903],passed=all(r["passed"] for r in outcomes),seconds=perf_counter()-start)
