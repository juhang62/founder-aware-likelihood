"""Paired predictive scores, covariance diagnostics, and numerical refinement."""

from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .model import from_config
from .likelihood import Likelihood, Settings
from .simulation import stream
from .inference import fit, profile_sets

METHODS = ("G", "M", "M0")


def paired_summary(values):
    values = np.asarray(values, float)
    return dict(n=len(values), mean=float(values.mean()),
                se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)


def screen(store):
    if any(store.config.get("death", (0., 0.))):
        raise NotImplementedError("This no-death diagnostic is not a Delivery 1 birth-death stage; use the birth-death smoke refinement command")
    config = store.config
    model = from_config(config["truth"], config)
    likelihood = Likelihood(config)
    rows, predictions = [], {}
    start = perf_counter()
    for founders in config["founders"]:
        data = store.dataset("screen", founders, 0, config["screen_cultures_per_cell"])
        for cell in np.unique(data["cell"]):
            ids = np.flatnonzero(data["cell"] == cell)
            t, z0 = float(data["time"][ids[0]]), data["z0"][ids[0]]
            y, counts = data["y"][ids], data["counts"][ids]
            scores = {method: likelihood.logpdf(method, config["truth"], y, t, z0) for method in METHODS}
            m, V = model.at(t, z0)
            VM, VM0 = model.mixture_covariances(t, z0)
            a, v, k, tau = model.amplitude(z0)
            covariances = dict(G=V, M=VM, M0=VM0)
            projection_vectors = {"total": np.array([1., 1.]), "contrast": np.array([1., -1.])}
            projections = {}
            for name, vector in projection_vectors.items():
                projections[name] = dict(mean=float(vector @ m),
                                         population_variances={method: float(vector @ covariances[method] @ vector) for method in METHODS})
            rows.append(dict(founders=founders, cell=int(cell), z0=z0, time=t,
                             M_minus_G=paired_summary(scores["M"]-scores["G"]),
                             M_minus_M0=paired_summary(scores["M"]-scores["M0"]),
                             mean=m, population_covariances=covariances,
                             covariance_relative_frobenius={method: float(np.linalg.norm(covariances[method]-V)/np.linalg.norm(V)) for method in METHODS},
                             count_quantile_probabilities=[.05, .25, .5, .75, .95, .99],
                             count_quantiles=np.quantile(np.c_[counts, counts.sum(axis=1)], [.05, .25, .5, .75, .95, .99], axis=0),
                             projections=projections, gamma_shape=k, gamma_scale=tau))
            prefix = f"n{founders}_cell{cell}"
            predictions[prefix+"_observed"] = y
            for method_index, method in enumerate(METHODS):
                rng = stream(config["master_seed"], "diagnostics", founders, int(cell)//2,
                             int(cell)%2, method_index, "predictions")
                size = 20000
                if method == "G":
                    predicted = rng.multivariate_normal(m, V+config["measurement_variance"]*np.eye(2), size=size)
                else:
                    w = rng.gamma(k, tau, size=size)
                    predicted = w[:, None]*m/a
                    if method == "M":
                        residual = rng.multivariate_normal([0., 0.], model.B, size=size)
                        predicted += np.sqrt(w*np.exp(model.L*t))[:, None]*residual
                    predicted += rng.normal(0, np.sqrt(config["measurement_variance"]), size=(size, 2))
                predictions[prefix+"_"+method] = predicted
                predictions[prefix+"_logpdf_"+method] = scores[method]
    path = store.root/"screen-predictions.npz"
    np.savez_compressed(path, **predictions)
    central3 = model.u3-3*model.ell*model.u2+2*model.ell**3
    gamma3 = 2*model.v**2/model.ell
    return dict(cells=rows, L=model.L, lambda2=model.lambda2,
                spectral_margin=model.L/2-model.lambda2, r=model.r, ell=model.ell,
                exact_amplitude_third_central=central3, gamma_amplitude_third_central=gamma3,
                gamma_third_relative_error=(gamma3-central3)/central3,
                predictions_path=str(path), seconds=perf_counter()-start, timings=likelihood.timing)


def integration_refinement(store):
    if any(store.config.get("death", (0., 0.))):
        raise NotImplementedError("This no-death diagnostic is not a Delivery 1 birth-death stage; use the birth-death smoke refinement command")
    config = store.config
    rows, datasets = [], []
    base = Likelihood(config)
    tight = Likelihood(config, Settings(**config["tight_integration"]))
    near_corners = [[.3001, .3001], [.3001, 1.0999], [1.0999, .3001], [1.0999, 1.0999]]
    for founders in config["founders"]:
        data = store.dataset("screen", founders, 0, config["screen_cultures_per_cell"])
        parameters = [("truth", config["truth"])]+[(f"corner{i}", v) for i, v in enumerate(near_corners)]
        for method in METHODS:
            result = store.result(f"fits/inference/n{founders}/rep000-{method}.json")
            if result and result["success"]:
                parameters.append((f"fit_{method}", result["estimate"]))
        for cell in np.unique(data["cell"]):
            ids = np.flatnonzero(data["cell"] == cell)
            sorted_ids = ids[np.argsort(data["y"][ids].sum(axis=1))]
            number = int(np.ceil(config["refinement"]["observations_per_founder"]/4))
            chosen = sorted_ids[np.linspace(0, len(ids)-1, number).astype(int)]
            t, z0 = float(data["time"][ids[0]]), data["z0"][ids[0]]
            for label, theta in parameters:
                for method in ("M", "M0"):
                    try:
                        default = base.logpdf(method, theta, data["y"][chosen], t, z0)
                        refined = tight.logpdf(method, theta, data["y"][chosen], t, z0)
                        reference = tight.logpdf(method, theta, data["y"][chosen], t, z0, reference=True)
                        change = float(np.max(abs(default-refined)))
                        independent = float(np.max(abs(refined-reference)))
                        rows.append(dict(founders=founders, cell=int(cell), parameter=label, theta=theta,
                                         method=method, observations=len(chosen), observation_indices=chosen,
                                         max_logpdf_change=change, max_reference_change=independent,
                                         passed=max(change, independent)<config["refinement"]["log_density_target"]))
                    except (ArithmeticError, ValueError, Warning) as error:
                        rows.append(dict(founders=founders, cell=int(cell), parameter=label, theta=theta,
                                         method=method, passed=False, failure=str(error)))
        for replicate in range(config["refinement"]["fits_per_founder"]):
            data_fit = store.dataset("inference", founders, replicate, config["cultures_per_cell"])
            for method in METHODS:
                result = store.result(f"fits/inference/n{founders}/rep{replicate:03d}-{method}.json")
                if not result or not result["success"]:
                    continue
                theta = result["estimate"]
                ll_base = {m: base.loglik(m, theta, data_fit) for m in METHODS}
                ll_tight = {m: tight.loglik(m, theta, data_fit) for m in METHODS}
                change = {m: ll_tight[m]-ll_base[m] for m in METHODS}
                paired_change = {m: change["M"]-change[m] for m in ("G", "M0")}
                datasets.append(dict(founders=founders, replicate=replicate, fitted_method=method,
                                     loglik_change=change, paired_loglik_change=paired_change,
                                     passed=max(abs(x) for x in list(change.values())+list(paired_change.values()))<config["refinement"]["dataset_loglik_target"]))
    return dict(observations=rows, datasets=datasets, passed=all(r["passed"] for r in rows+datasets))


def _refinement_job(data, config, method, original, refined, original_profile, refined_profile):
    data = {key: data[key] for key in ("y", "time", "z0", "cell")}
    config = {key: value for key, value in config.items() if key != "truth"}
    if refined is None:
        refined = fit(method, data, config, tight=True)
    row = dict(passed=False, original_success=original["success"], refined_success=refined["success"])
    if original["success"] and refined["success"]:
        change = np.abs(np.array(original["estimate"])-refined["estimate"])
        row.update(absolute_estimate_change=change, passed=bool(change.max()<config["refinement"]["estimate_target"]))
    if original_profile:
        if refined_profile is None:
            refined_profile = profile_sets(method, data, config, refined, tight=True)
        profile_pass = False
        if original_profile["success"] and refined_profile["success"]:
            old = [r["intervals"] for r in original_profile["rates"]]
            new = [r["intervals"] for r in refined_profile["rates"]]
            if all(len(a) == len(b) for a, b in zip(old, new)):
                max_change = max(float(np.max(abs(np.array(a)-b))) for a, b in zip(old, new))
                row["max_interval_endpoint_change"] = max_change
                profile_pass = max_change < config["refinement"]["estimate_target"]
        row["profile_refinement_passed"] = profile_pass
        row["passed"] = row["passed"] and profile_pass
    return row, refined, refined_profile


def fit_refinement(store, workers=1):
    if any(store.config.get("death", (0., 0.))):
        raise NotImplementedError("This no-death diagnostic is not a Delivery 1 birth-death stage; use the birth-death smoke refinement command")
    config, rows, jobs = store.config, [], []
    for founders in config["founders"]:
        for replicate in range(config["refinement"]["fits_per_founder"]):
            data = store.dataset("inference", founders, replicate, config["cultures_per_cell"])
            for method in METHODS:
                original = store.result(f"fits/inference/n{founders}/rep{replicate:03d}-{method}.json")
                if not original:
                    rows.append(dict(founders=founders, replicate=replicate, method=method, passed=False, failure="Original fit absent"))
                    continue
                path = f"fits/refined/n{founders}/rep{replicate:03d}-{method}.json"
                refined = store.result(path)
                original_profile = store.result(f"profiles/n{founders}/rep{replicate:03d}-{method}.json")
                profile_path = f"profiles/refined/n{founders}/rep{replicate:03d}-{method}.json"
                refined_profile = store.result(profile_path)
                jobs.append((founders, replicate, method, path, profile_path,
                             (data, config, method, original, refined, original_profile, refined_profile),
                             refined is None, original_profile is not None and refined_profile is None))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_refinement_job, *job[5]): job for job in jobs}
        for future in as_completed(pending):
            founders, replicate, method, path, profile_path, _, save_fit, save_profile = pending[future]
            row, refined, refined_profile = future.result()
            if save_fit:
                store.save_result(path, refined)
            if save_profile:
                store.save_result(profile_path, refined_profile)
            row.update(founders=founders, replicate=replicate, method=method)
            rows.append(row)
            print(f"refinement N={founders} replicate={replicate} {method}: {row['passed']}", flush=True)
    rows.sort(key=lambda row: (row["founders"], row["replicate"], row["method"]))
    return dict(fits=rows, passed=all(r["passed"] for r in rows))
