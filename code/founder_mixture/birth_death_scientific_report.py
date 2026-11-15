"""Export Delivery 2 evidence without regenerating Delivery 1 experiments."""

from pathlib import Path
import numpy as np
from .birth_death_report import csv_table, latex_table
from .checks import moment_summary
from .storage import read_json, write_json


def export(store, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    if destination.name != store.config['experiment']:
        raise ValueError('Use the separate birth-death report directory')
    tables, figures = destination/'tables', destination/'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    names = ('comparator', 'simulation', 'screen', 'inference', 'integration', 'refits')
    evidence = {name: store.result(f'delivery2/{name}.json') for name in names}
    if any(result is None for result in evidence.values()):
        raise ValueError('All Delivery 2 outcomes must be recorded before the complete report export')
    for name, result in evidence.items():
        write_json(tables/f'delivery2-{name}.json', result)
    simulation, screen, inference, integration, refits = [evidence[k] for k in names[1:]]
    for name, rows in (('simulation', simulation['rows']), ('comparator', evidence['comparator']['rows']),
                       ('metrics', inference['metrics']), ('paired', inference['paired']),
                       ('fitted-survival', inference['survival']), ('refits', refits['rows']),
                       ('integration-selection', integration['selection'])):
        csv_table(tables/f'delivery2-{name}.csv', rows)
    csv_table(tables/'delivery2-fits.csv', [dict({'founders':r['founders'], 'replicate':r['replicate'], 'method':r['method']}, **(r['result'] or {})) for r in inference['fits']])
    score_rows, moment_rows = [], []
    for r in screen['rows']:
        for comparison, score in r['scores'].items():
            score_rows.append(dict(founders=r['founders'], founder_type=r['founder_type'], time=r['time'], comparison=comparison, **score))
        for method, m in r['moments'].items():
            moment_rows.append(dict(founders=r['founders'], founder_type=r['founder_type'], time=r['time'], method=method, **m))
    csv_table(tables/'delivery2-scores.csv', score_rows)
    csv_table(tables/'delivery2-moments.csv', moment_rows)

    def f(x, digits=4):
        return f'{x:.{digits}g}'
    def ci(x, scale=1):
        return f'[{f(x[0]*scale)}, {f(x[1]*scale)}]'
    empty = [[r['founders'], r['founder_type'], f(r['time']), f"{r['empty_count']}/{r['n']}",
              f(r['exact_empty_probability']), ci(r['empty_ci95'])] for r in simulation['rows']]
    latex_table(tables/'delivery2-empty.tex', [r'$N_0$', 'Type', '$t$', 'Empty / total', 'Exact probability', r'95\% interval'], empty)
    for n in store.config['founders']:
        rows = []
        for r in simulation['rows']:
            if r['founders'] != n:
                continue
            for i in range(2):
                rows.append([r['founder_type'], f(r['time']), f'$m_{i+1}$', f(r['exact_mean'][i], 6),
                             f(r['mean'][i], 6), f(r['mean_mcse'][i]), f(r['mean_z'][i])])
            for i, j in ((0, 0), (0, 1), (1, 1)):
                rows.append([r['founder_type'], f(r['time']), f'$V_{{{i+1}{j+1}}}$', f(r['exact_covariance'][i][j], 6),
                             f(r['covariance'][i][j], 6), f(r['covariance_mcse'][i][j]), f(r['covariance_z'][i][j])])
        latex_table(tables/f'delivery2-validation-n{n}.tex', ['Type', '$t$', 'Quantity', 'Analytic', 'Empirical', 'MCSE', '$z$'], rows)
    rows = [[r['founders'], r['founder_type'], f(r['time']),
             *[f"{f(r['scores'][c]['mean'])} ({f(r['scores'][c]['se'])})" for c in ('M-G', 'M-EG', 'M-M0')]] for r in screen['rows']]
    latex_table(tables/'delivery2-scores.tex', [r'$N_0$', 'Type', '$t$', '$M-G$', '$M-EG$', '$M-M0$'], rows)
    rows = [[r['founders'], r['founder_type'], f(r['time']),
             f(100*r['moments']['M']['mean_relative_difference']),
             f(r['moments']['M']['covariance_discrepancy_percent']),
             f(r['moments']['M0']['covariance_discrepancy_percent'])] for r in screen['rows']]
    latex_table(tables/'delivery2-moments.tex', [r'$N_0$', 'Type', '$t$', r'Mean error (\%)', r'$M$ cov. (\%)', r'$M0$ cov. (\%)'], rows)
    rows = [[r['founders'], r['method'], '$'+r['rate'][0]+'_'+r['rate'][1]+'$',
             f"{f(r['bias'])} ({f(r['bias_mcse'])})" if 'bias' in r else '--', f(r['rmse']) if 'rmse' in r else '--',
             f"{r.get('boundary_count',0)}/{r['successful']}", f"{r['failures']}/{r['attempted']}"] for r in inference['metrics']]
    latex_table(tables/'delivery2-estimation.tex', [r'$N_0$', 'Method', 'Rate', 'Bias (MCSE)', 'RMSE', 'Boundary', 'Failures'], rows)
    rows = [[r['founders'], '$'+r['rate'][0]+'_'+r['rate'][1]+'$', r['comparison'],
             f(r['rmse_difference']) if 'rmse_difference' in r else '--', ci(r['rmse_difference_ci95']) if 'rmse_difference_ci95' in r else '--',
             f"{f(100*r['rmse_reduction'])} {ci(r['rmse_reduction_ci95'],100)}" if 'rmse_reduction' in r else '--',
             r['paired_successful']] for r in inference['paired']]
    latex_table(tables/'delivery2-paired.tex', [r'$N_0$', 'Rate', 'Pair', r'$\Delta$ RMSE', r'95\% interval', r'Reduction (\%) [95\%]', '$n$'], rows)
    unique = [r for r in inference['metrics'] if r['rate']=='b1' and 'mean_fit_seconds' in r]
    rows = [[r['founders'], r['method'], f(r['total_fit_seconds']), f(r['mean_fit_seconds']),
             f(r['mean_grid_seconds']), f(r['mean_components']['construction_seconds']), f(r['mean_components']['integration_seconds']),
             f(r['mean_warm_loglik_seconds'])] for r in unique]
    latex_table(tables/'delivery2-runtime.tex', [r'$N_0$', 'Method', 'Total', 'Per fit', 'Grid', 'Construct', 'Integrate', 'Warm'], rows)
    rows = [[r['founders'], r['method'], *[f(r['mean_components'][key]) for key in
             ('moment_seconds', 'extinction_seconds', 'spectral_amplitude_seconds', 'positive_amplitude_seconds')]] for r in unique]
    latex_table(tables/'delivery2-components.tex', [r'$N_0$', 'Method', 'Moments', 'Extinction', 'Spectral/amplitude', 'Positive amplitude'], rows)
    rows = []
    for n in store.config['founders']:
        for method in store.config['methods']:
            selected = [r for r in refits['rows'] if r['founders']==n and r['method']==method]
            valid = [r for r in selected if 'estimate_changes' in r]
            rows.append([n, method, f"{sum(r['passed'] for r in selected)}/{len(selected)}",
                         f(max(max(r['estimate_changes']) for r in valid)) if valid else '--',
                         f(max(r['maximum_loglik_change'] for r in valid)) if valid else '--',
                         f(max(max(map(abs,r['fixed_parameter_loglik_changes'])) for r in valid)) if valid else '--'])
    latex_table(tables/'delivery2-refits.tex', [r'$N_0$', 'Method', 'Passed', 'Max. rate change', 'Max. optimum change', 'Max. fixed-rate change'], rows)

    styles = {'G':('#b33c31','-'), 'EG':('#d58b00','--'), 'M':('#126da8','-'), 'M0':('#238151',':')}
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    quantiles, prediction_checks = [], []
    probabilities = np.r_[.001, .01, .025, .05, .1, .25, .5, .75, .9, .95, .975, .99, .999]
    with np.load(screen['predictions_path']) as archive:
        for r in screen['rows']:
            for method, moments in r['moments'].items():
                y = archive[f"n{r['founders']}_combination{r['combination']}_{method}"]
                check = moment_summary(y, np.array(moments['mean']), np.array(moments['population_covariance'])+store.config['measurement_variance']*np.eye(2))
                maximum = max(np.max(abs(check['mean_z'])), np.max(abs(check['covariance_z'])))
                prediction_checks.append(dict(founders=r['founders'], combination=r['combination'], method=method,
                                              maximum_z=float(maximum), passed=bool(maximum<5), **check))
        for n in store.config['founders']:
            fig, axes = plt.subplots(4, 2, figsize=(8.5, 9), layout='constrained')
            for group in range(4):
                prefix = f'n{n}_combination{group}'
                for j, (name, vector) in enumerate((('Total', np.array([1,1])), ('Contrast', np.array([1,-1])))):
                    ax = axes[group,j]
                    for method in ('observed', *store.config['methods']):
                        values = np.sort(archive[prefix+'_'+method] @ vector)
                        index = np.unique(np.r_[0, np.linspace(0,len(values)-1,min(1000,len(values))).astype(int),len(values)-1])
                        if method == 'observed':
                            ax.step(values[index], (index+1)/len(values), where='post', color='black', lw=1.2, label='CTMC + noise')
                        else:
                            color, linestyle = styles[method]
                            ax.step(values[index], (index+1)/len(values), where='post', color=color, linestyle=linestyle, lw=1, label=method)
                        for p, value in zip(probabilities, np.quantile(values, probabilities)):
                            quantiles.append(dict(founders=n, combination=group, projection=name.lower(), method=method,
                                                  probability=p, quantile=float(value), sample_size=len(values)))
                    ax.set(title=f'Type {group//2+1}, day {store.config["times"][group%2]:g}: {name}',
                           xlabel='Readout (cells)', ylabel='Empirical CDF', ylim=(-.02,1.02))
                    ax.grid(alpha=.15)
                    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,4))
            axes[0,0].legend(fontsize=7)
            fig.savefig(figures/f'delivery2-projections-n{n}.png', dpi=180)
            plt.close(fig)
    csv_table(tables/'delivery2-projection-quantiles.csv', quantiles)
    write_json(tables/'delivery2-prediction-check.json', dict(passed=all(r['passed'] for r in prediction_checks), rows=prediction_checks,
               check='Predictive-draw means and covariance entries versus analytic observation moments, 20000 draws per combination and method; five estimated standard errors is an investigation flag.'))
    fig, axes = plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
    for ax, n in zip(axes, store.config['founders']):
        for index, method in enumerate(store.config['methods']):
            rows = [r for r in inference['metrics'] if r['founders']==n and r['method']==method]
            x = np.arange(2)+(index-1.5)*.18
            ax.bar(x,[r.get('rmse',np.nan) for r in rows],width=.17,color=styles[method][0],label=method)
        ax.set(xticks=[0,1],xticklabels=[r'$b_1$',r'$b_2$'],ylabel='RMSE (per day)',title=f'{n} founder'+('s' if n!=1 else ''))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(figures/'delivery2-rmse.png',dpi=180)
    plt.close(fig)

    data_records = []
    for stage in ('checks','screen','inference'):
        for p in sorted((store.root/'observations'/stage).glob('n*/rep*.json')):
            record = read_json(p)
            if stage == 'checks' and record['replicate'] != 0:
                continue
            data_records.append(dict(path=str(p), stage=stage, founders=record['founders'], replicate=record['replicate'],
                                     cultures=record['per_cell']*4, seconds=record['seconds'], incomplete=record['incomplete']))
    timings = [read_json(p)['result'] for p in sorted((store.root/'timings').glob('delivery2-*.json'))]
    summary = dict(delivery=2, complete=all(r.get('complete',True) for r in evidence.values()),
                   passed=all(r['passed'] for r in evidence.values()), validation_cultures=simulation['cultures'],
                   screen_cultures=sum(r['cultures'] for r in screen['rows']), inference_datasets=40,
                   inference_fits=inference['available_fits'], successful_fits=inference['successful_fits'],
                   numerical_refits=len(refits['rows']), successful_numerical_refits=sum(r['passed'] for r in refits['rows']),
                   log_density_batches=len(integration['rows']),
                   maximum_log_density_change=max((r.get('maximum_change',0) for r in integration['rows']),default=0),
                   maximum_estimate_change=max((max(r.get('estimate_changes',[0])) for r in refits['rows']),default=0),
                   maximum_optimum_loglik_change=max((r.get('maximum_loglik_change',0) for r in refits['rows']),default=0),
                   maximum_fixed_parameter_loglik_change=max((max(map(abs,r.get('fixed_parameter_loglik_changes',[0]))) for r in refits['rows']),default=0),
                   maximum_fitted_epsilon=max((r['epsilon'] for r in inference['survival']),default=0),
                   incomplete_paths=sum(r['incomplete'] for r in data_records), datasets=data_records, timings=timings,
                   production_fit_process_seconds=sum(r['result']['seconds'] for r in inference['fits'] if r['result']),
                   refit_process_seconds=sum(r.get('seconds',0) for r in refits['rows']),
                   profiles_run=False, delivery3_run=False, pending=['Delivery 3 decision', '100-dataset extension and profiles if authorized'])
    write_json(tables/'delivery2-summary.json', summary)
    original = destination/'delivery1-provenance.json'
    if not original.exists():
        original.write_bytes((destination/'provenance.json').read_bytes())
    write_json(destination/'provenance.json', dict(report=store.provenance,
               delivery1_provenance='delivery1-provenance.json',
               stage_provenance={name:read_json(store.root/f'delivery2/{name}.json')['provenance'] for name in names},
               timings=timings, config=store.config, execution_delivery=2,
               note='The frozen maximum_delivery=1 configuration field is the original CLI default; --delivery 2 explicitly selects the reviewed stage without changing scientific data keys.'))
    print(f"Delivery 2 report: complete={summary['complete']} passed={summary['passed']}",flush=True)
