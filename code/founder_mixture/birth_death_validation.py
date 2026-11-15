"""Delivery 2 simulator validation and independent published-formula audit."""

import hashlib
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import beta, multivariate_normal
from .checks import moment_summary
from .model import from_config
from .likelihood import Likelihood


def published_comparator(config):
    """Evaluate Gunnarsson et al. (2023), equations (8), (15), (16)."""
    start = perf_counter()
    # Optional local copy of the published article, recorded by hash for provenance only.
    pdf = Path(os.environ.get('GUNNARSSON2023_PDF', 'references/2023-Gunnarsson-cell-proliferation-phenotypic-switching-cancer.pdf'))
    b = np.array(config['truth'])
    d = np.array(config['death'])
    s12, s21 = config['switching']
    # Construct their row-convention generator independently of Candidate.A.
    Arow = np.array([[b[0]-d[0]-s12, s12], [s21, b[1]-d[1]-s21]])
    model = from_config(config['truth'], config)
    likelihood = Likelihood(config)
    rows = []
    for t in config['times']:
        transition = expm(t*Arow)
        single_covariances = []
        single_errors = []
        for j in range(2):
            def integrand(tau):
                m = expm(tau*Arow)[j]
                propagated = expm((t-tau)*Arow)
                return 2*propagated.T @ np.diag(b*m) @ propagated
            integral, error = quad_vec(integrand, 0, t, epsabs=1e-10, epsrel=1e-12)
            m = transition[j]
            single_covariances.append(integral + np.diag(m) - np.outer(m, m))
            single_errors.append(error)
        for z0 in ([1, 0], [0, 1], [50, 0], [0, 50], [3, 7]):
            z0 = np.array(z0)
            mean = z0 @ transition
            covariance = np.einsum('j,jkl->kl', z0, single_covariances)
            actual_mean, actual_covariance = model.at(t, z0)
            readings = np.vstack(([0., 0.], mean, mean + np.array([1., -1.])))
            published_logpdf = multivariate_normal.logpdf(readings, mean, covariance + config['measurement_variance']*np.eye(2))
            actual_logpdf = likelihood.logpdf('G', config['truth'], readings, t, z0)
            mean_change = float(np.max(abs(mean-actual_mean)/np.maximum(1., abs(mean))))
            covariance_change = float(np.linalg.norm(covariance-actual_covariance)/np.linalg.norm(covariance))
            density_change = float(np.max(abs(published_logpdf-actual_logpdf)))
            rows.append(dict(time=t, z0=z0, mean=mean, population_covariance=covariance,
                             mean_scaled_difference=mean_change, covariance_relative_difference=covariance_change,
                             quadrature_error_bound=float(z0 @ single_errors), readings=readings,
                             published_logpdf=published_logpdf, implemented_logpdf=actual_logpdf,
                             maximum_logpdf_difference=density_change,
                             passed=max(mean_change, covariance_change, density_change)<1e-8))
    return dict(rows=rows, passed=all(r['passed'] for r in rows), seconds=perf_counter()-start,
                source=str(pdf) if pdf.is_file() else None,
                source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.is_file() else None,
                doi='10.1016/j.jtbi.2023.111497', sections=['3.3', 'A.2'], equations=[8, 15, 16],
                conventions='A_published=A_code.T; published row means are transposes of code column means; covariances are unchanged; lambda_i=b_i-d_i; nu_ij=s_ij',
                measurement='Published E_num is set to known R=4 I_2; G already includes demographic covariance; EG is our adaptation.',
                scope='Independent formula agreement at matched inputs, not reproduction of a published experiment.')


def simulation_validation(store):
    config = store.config
    model = from_config(config['truth'], config)
    start = perf_counter()
    rows = []
    for n in config['founders']:
        size = config['moment_check_cultures'] if n == 1 else config['moment_check_cultures_many_founders']
        data = store.dataset('checks', n, 0, size)
        for group in range(4):
            ids = np.flatnonzero(data['cell'] == group)
            t, z0 = float(data['time'][ids[0]]), data['z0'][ids[0]]
            counts = data['counts'][ids].astype(float)
            mean, covariance = model.at(t, z0)
            row = moment_summary(counts, mean, covariance)
            empty = int(np.sum(counts.sum(axis=1) == 0))
            p = float(np.exp(model.survival(t, z0)['log_empty']))
            interval = [float(beta.ppf(.025, empty, size-empty+1)) if empty else 0.,
                        float(beta.ppf(.975, empty+1, size-empty)) if empty < size else 1.]
            # Analytic binomial SE remains nonzero after zero observed events.
            se = float(np.sqrt(p*(1-p)/size))
            z = (empty/size-p)/se
            flags = []
            if np.max(abs(row['mean_z'])) > 5:
                flags.append('mean discrepancy exceeds five estimated standard errors')
            if np.max(abs(row['covariance_z'])) > 5:
                flags.append('covariance discrepancy exceeds five estimated standard errors')
            if abs(z) > 5:
                flags.append('empty proportion discrepancy exceeds five analytic binomial standard errors')
            row.update(founders=n, combination=group, founder_type=group//2+1, time=t, z0=z0,
                       empty_count=empty, empty_proportion=empty/size, exact_empty_probability=p,
                       empty_ci95=interval, empty_analytic_mcse=se, empty_z=z,
                       max_population=int(counts.sum(axis=1).max()), incomplete=0, flags=flags,
                       passed=not flags)
            rows.append(row)
            print(f'validation N={n} combination={group}: flags={flags}', flush=True)
            store.save_result('delivery2/simulation-partial.json', dict(rows=rows, complete=False))
    return dict(rows=rows, complete=True, passed=all(r['passed'] for r in rows),
                cultures=sum(r['n'] for r in rows), seconds=perf_counter()-start,
                empty_interval='95% equal-tailed Clopper-Pearson interval',
                covariance_mcse='Sample SD of centered cross products / sqrt(n), multiplied by n/(n-1)',
                seed_keys='checks stage, replicate 0; founder type and time indices follow the recorded scheme')
