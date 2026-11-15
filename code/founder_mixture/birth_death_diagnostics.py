"""Four-method scientific comparisons and numerical checks for Delivery 2."""

from itertools import product
from time import perf_counter
import numpy as np
from .birth_death_checks import tight_config
from .birth_death_delivery import _tight_fit
from .diagnostics import paired_summary
from .jobs import run_jobs
from .likelihood import Likelihood
from .model import from_config
from .simulation import stream


def predict(method, model, t, z0, variance, size, rng):
    """Draw the specified observation law, including noise on empty cultures."""
    mean, covariance = model.at(t, z0)
    if method == 'G':
        return rng.multivariate_normal(mean, covariance+variance*np.eye(2), size=size)
    weights = model.survival(t, z0)
    living = rng.random(size) >= np.exp(weights['log_empty'])
    count = int(living.sum())
    y = rng.normal(0., np.sqrt(variance), (size, 2))
    if method == 'EG':
        m, V = model.survivor_moments(t, z0)
        y[living] += rng.multivariate_normal(m, V, size=count)
    elif method in ('M', 'M0'):
        a = model.amplitude(z0)[0]
        _, _, k, tau = model.positive_amplitude(z0)
        w = rng.gamma(k, tau, size=count)
        y[living] += w[:, None]*mean/a
        if method == 'M':
            residual = rng.multivariate_normal([0., 0.], model.B, size=count)
            y[living] += np.sqrt(w*np.exp(model.L*t))[:, None]*residual
    else:
        raise ValueError(method)
    return y


def screen(store):
    config = store.config
    model = from_config(config['truth'], config)
    likelihood = Likelihood(config)
    rows, predictions = [], {}
    start = perf_counter()
    for n in config['founders']:
        data = store.dataset('screen', n, 0, config['screen_cultures_per_cell'])
        for group in range(4):
            ids = np.flatnonzero(data['cell'] == group)
            t, z0 = float(data['time'][ids[0]]), data['z0'][ids[0]]
            y, counts = data['y'][ids], data['counts'][ids]
            scores = {m: likelihood.logpdf(m, config['truth'], y, t, z0) for m in config['methods']}
            mean, V = model.at(t, z0)
            weights = model.survival(t, z0)
            moments = {m: (mean, V) if m in ('G', 'EG') else model.mixture_moments(t, z0, m == 'M') for m in config['methods']}
            discrepancies = {m: dict(mean=mh, population_covariance=vh,
                              mean_relative_difference=float(np.linalg.norm(mh-mean)/np.linalg.norm(mean)),
                              covariance_discrepancy_percent=float(100*np.linalg.norm(vh-V)/np.linalg.norm(V)))
                             for m, (mh, vh) in moments.items()}
            row = dict(founders=n, combination=group, founder_type=group//2+1, time=t, z0=z0, cultures=len(ids),
                       scores={f'M-{m}': paired_summary(scores['M']-scores[m]) for m in ('G', 'EG', 'M0')},
                       exact_mean=mean, exact_population_covariance=V, moments=discrepancies,
                       survival=weights, survivor_mean_relative_difference=weights['pi']/weights['pi_infinity']-1,
                       count_quantile_probabilities=[.05, .25, .5, .75, .95, .99],
                       count_quantiles=np.quantile(np.c_[counts, counts.sum(axis=1)], [.05, .25, .5, .75, .95, .99], axis=0),
                       empty_count=int(np.sum(counts.sum(axis=1)==0)), empty_fraction=float(np.mean(counts.sum(axis=1)==0)),
                       negative_reading_components=int(np.sum(y<0)), maximum_population=int(counts.sum(axis=1).max()),
                       passed=all(np.isfinite(x).all() for x in scores.values()))
            rows.append(row)
            prefix = f'n{n}_combination{group}'
            predictions[prefix+'_observed'] = y
            for index, method in enumerate(config['methods']):
                rng = stream(config['master_seed'], 'diagnostics', n, group//2, group%2, index, 'predictions')
                predictions[prefix+'_'+method] = predict(method, model, t, z0, config['measurement_variance'], 20000, rng)
                predictions[prefix+'_logpdf_'+method] = scores[method]
            print(f'screen N={n} combination={group}: complete', flush=True)
            store.save_result('delivery2/screen-partial.json', dict(rows=rows, complete=False))
    path = store.root/'delivery2/screen-predictions.npz'
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **predictions)
    temporary.replace(path)
    return dict(rows=rows, complete=True, passed=all(r['passed'] for r in rows),
                seconds=perf_counter()-start, timings=likelihood.timing, predictions_path=str(path),
                prediction_size=20000, prediction_keys='diagnostics stage, predictions purpose, replicate=method index in [G, EG, M, M0]')


def inference_summary(store):
    config = store.config
    truth = np.array(config['truth'])
    metrics, paired, records, survival = [], [], [], []
    for n in config['founders']:
        fits = {}
        for method in config['methods']:
            fits[method] = {r: store.result(f'fits/inference/n{n}/rep{r:03d}-{method}.json') for r in range(20)}
            valid = {r: f for r, f in fits[method].items() if f and f['success']}
            for r, f in fits[method].items():
                records.append(dict(founders=n, replicate=r, method=method, result=f))
                if f and f['success']:
                    model = from_config(f['estimate'], config)
                    tight = from_config(f['estimate'], tight_config(config))
                    for group in range(4):
                        z0 = np.eye(2, dtype=int)[group//2]*n
                        t = config['times'][group%2]
                        a, b = model.survival(t, z0), tight.survival(t, z0)
                        ap, vp, _, _ = model.positive_amplitude(z0)
                        apt, vpt, _, _ = tight.positive_amplitude(z0)
                        survival.append(dict(founders=n, replicate=r, method=method, combination=group, time=t,
                                             theta=f['estimate'], **a,
                                             pi_change=abs(a['pi']-b['pi']), pi_infinity_change=abs(a['pi_infinity']-b['pi_infinity']),
                                             positive_mean_change=abs(ap-apt), positive_variance_change=abs(vp-vpt),
                                             survivor_mean_relative_difference=a['pi']/a['pi_infinity']-1))
            for rate in range(2):
                row = dict(founders=n, method=method, rate=f'b{rate+1}', attempted=20, successful=len(valid),
                           failures=20-len(valid), excluded_replicates=[r for r in range(20) if r not in valid])
                if valid:
                    errors = np.array([f['estimate'][rate] for f in valid.values()])-truth[rate]
                    row.update(bias=float(errors.mean()), bias_mcse=float(errors.std(ddof=1)/np.sqrt(len(errors))) if len(errors)>1 else None,
                               rmse=float(np.sqrt(np.mean(errors**2))), boundary_count=sum(f['boundary'][rate] for f in valid.values()),
                               boundary_frequency=float(np.mean([f['boundary'][rate] for f in valid.values()])),
                               mean_fit_seconds=float(np.mean([f['seconds'] for f in fits[method].values() if f])),
                               total_fit_seconds=float(sum(f['seconds'] for f in fits[method].values() if f)),
                               mean_warm_loglik_seconds=float(np.mean([f['warm_loglik_seconds'] for f in valid.values()])),
                               failed_starts=sum(f['failed_starts'] for f in fits[method].values() if f),
                               mean_grid_seconds=float(np.mean([f['grid_seconds'] for f in fits[method].values() if f])),
                               mean_components={key: float(np.mean([f['timings'][key] for f in fits[method].values() if f])) for key in
                                   ('construction_seconds', 'moment_seconds', 'extinction_seconds', 'spectral_amplitude_seconds', 'positive_amplitude_seconds', 'integration_seconds')})
                metrics.append(row)
        for index, comparator in enumerate(('G', 'EG', 'M0')):
            ids = [r for r in range(20) if fits['M'][r] and fits[comparator][r] and fits['M'][r]['success'] and fits[comparator][r]['success']]
            # One set of whole-dataset resamples for both rates of this pair.
            rng = stream(config['master_seed'], 'diagnostics', n, 0, 0, index, 'bootstrap')
            draws = rng.integers(0, len(ids), (2000, len(ids))) if ids else None
            for rate in range(2):
                entry = dict(founders=n, comparison=f'M-{comparator}', rate=f'b{rate+1}', paired_successful=len(ids),
                             included_replicates=ids, excluded_replicates=[r for r in range(20) if r not in ids], bootstrap_repetitions=2000)
                if len(ids)>1:
                    left = np.array([fits['M'][r]['estimate'][rate] for r in ids])-truth[rate]
                    right = np.array([fits[comparator][r]['estimate'][rate] for r in ids])-truth[rate]
                    r1, r2 = np.sqrt(np.mean(left**2)), np.sqrt(np.mean(right**2))
                    b1, b2 = np.sqrt(np.mean(left[draws]**2, axis=1)), np.sqrt(np.mean(right[draws]**2, axis=1))
                    entry.update(rmse_difference=float(r1-r2), rmse_reduction=float(1-r1/r2),
                                 rmse_difference_ci95=np.quantile(b1-b2, [.025, .975]),
                                 rmse_reduction_ci95=np.quantile(1-b1/b2, [.025, .975]),
                                 paired_mse=paired_summary(left**2-right**2),
                                 paired_fit_seconds=paired_summary([fits['M'][r]['seconds']-fits[comparator][r]['seconds'] for r in ids]))
                paired.append(entry)
    return dict(metrics=metrics, paired=paired, fits=records, survival=survival, profiles_run=False,
                expected_fits=160, available_fits=sum(r['result'] is not None for r in records),
                successful_fits=sum(bool(r['result'] and r['result']['success']) for r in records),
                bootstrap_seed_keys='diagnostics, founder count, founder_type=0, time_index=0, replicate=comparator index [G, EG, M0], bootstrap purpose',
                complete=all(r['result'] is not None for r in records),
                passed=all(r['result'] and r['result']['success'] for r in records))


def integration_refinement(store):
    config = store.config
    base, tight = Likelihood(config), Likelihood(tight_config(config))
    rows, selection, missing = [], [], []
    start = perf_counter()
    for n in config['founders']:
        data = store.dataset('screen', n, 0, config['screen_cultures_per_cell'])
        parameters = [('truth', config['truth']), *[(f'corner{i}', theta) for i, theta in enumerate(product(*config['bounds']))]]
        for method in config['methods']:
            first = next(((r, f) for r in range(20)
                          if (f := store.result(f'fits/inference/n{n}/rep{r:03d}-{method}.json')) and f['success']), None)
            if first:
                parameters.append((f'fit_{method}_rep{first[0]:03d}', first[1]['estimate']))
            else:
                missing.append(dict(founders=n, method=method, reason='No successful fit for density refinement'))
        for group in range(4):
            ids = np.flatnonzero(data['cell']==group)
            ranked = ids[np.argsort(data['y'][ids].sum(axis=1), kind='stable')]
            number = max(5, int(np.ceil(config['refinement']['observations_per_founder']/4)))
            chosen = list(ranked[np.linspace(0, len(ids)-1, number).astype(int)])
            artificial = []
            norms = np.linalg.norm(data['y'][ids], axis=1)
            closest = ids[np.argmin(norms)]
            chosen.append(closest)
            if norms.min() > 4:
                artificial.append([0., 0.])
            negatives = ids[np.any(data['y'][ids] < 0, axis=1)]
            if len(negatives):
                chosen.append(negatives[np.argmin(np.linalg.norm(data['y'][negatives], axis=1))])
            else:
                artificial.append([-2., 1.])
            chosen = np.array(sorted(set(chosen)), dtype=int)
            ys = np.vstack((data['y'][chosen], np.array(artificial).reshape(-1, 2)))
            t, z0 = float(data['time'][ids[0]]), data['z0'][ids[0]]
            selection.append(dict(founders=n, combination=group, indices=chosen, artificial=artificial, readings=ys))
            for label, theta in parameters:
                for method in config['methods']:
                    row = dict(founders=n, combination=group, parameter=label, theta=theta, method=method, passed=False)
                    try:
                        values = {}
                        for survivor in ([False] if method=='G' else [False, True]):
                            p = base.logpdf(method, theta, ys, t, z0, survivor=survivor)
                            q = tight.logpdf(method, theta, ys, t, z0, survivor=survivor)
                            ref = tight.logpdf(method, theta, ys, t, z0, survivor=survivor, reference=True)
                            values['survivor' if survivor else 'full'] = dict(production=p, tight=q, reference=ref,
                                maximum_change=float(np.max(abs(p-q))), maximum_reference_change=float(np.max(abs(q-ref))))
                        maximum = max(max(v['maximum_change'], v['maximum_reference_change']) for v in values.values())
                        row.update(values=values, maximum_change=maximum, passed=maximum<config['refinement']['log_density_target'])
                    except (ArithmeticError, ValueError, Warning) as error:
                        row['failure'] = f'{type(error).__name__}: {error}'
                    rows.append(row)
            print(f'screen density refinement N={n} combination={group}: complete', flush=True)
            store.save_result('delivery2/integration-partial.json', dict(rows=rows, selection=selection, complete=False))
    return dict(rows=rows, selection=selection, missing=missing, complete=True, seconds=perf_counter()-start,
                passed=not missing and all(r['passed'] for r in rows))


def fit_refinement(store, workers):
    config = store.config
    jobs = []
    start = perf_counter()
    for n in config['founders']:
        for r in range(5):
            data = store.dataset('inference', n, r, config['cultures_per_cell'])
            for method in config['methods']:
                jobs.append((f'fits/inference-refined/n{n}/rep{r:03d}-{method}.json', (method, data, config)))
    run_jobs(store, jobs, _tight_fit, workers)
    rows = []
    base, tight = Likelihood(config), Likelihood(tight_config(config))
    for n in config['founders']:
        for r in range(5):
            data = store.dataset('inference', n, r, config['cultures_per_cell'])
            for method in config['methods']:
                original = store.result(f'fits/inference/n{n}/rep{r:03d}-{method}.json')
                refined = store.result(f'fits/inference-refined/n{n}/rep{r:03d}-{method}.json')
                row = dict(founders=n, replicate=r, method=method, passed=False)
                if original and refined and original['success'] and refined['success']:
                    fixed = [tight.loglik(method, theta, data)-base.loglik(method, theta, data)
                             for theta in (original['estimate'], refined['estimate'])]
                    rate_change = abs(np.array(original['estimate'])-refined['estimate'])
                    maximum_change = abs(original['loglik']-refined['loglik'])
                    row.update(original=original['estimate'], refined=refined['estimate'], estimate_changes=rate_change,
                               maximum_loglik_change=maximum_change, fixed_parameter_loglik_changes=fixed,
                               seconds=refined['seconds'],
                               passed=float(rate_change.max())<config['refinement']['estimate_target'] and
                               max(maximum_change, *map(abs, fixed))<config['refinement']['dataset_loglik_target'])
                else:
                    row['failure'] = 'Missing or unsuccessful production/refined fit; no replacement data'
                rows.append(row)
    return dict(rows=rows, complete=True, passed=all(r['passed'] for r in rows), profiles_run=False, seconds=perf_counter()-start)
