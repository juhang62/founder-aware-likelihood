"""Delivery 3 tables and figures; the existing LaTeX report is the narrative."""

from pathlib import Path
import numpy as np
from scipy.stats import beta
from .storage import read_json, write_json
from .birth_death_report import csv_table, latex_table
from .birth_death_extended import sha
from .simulation import stream
from .diagnostics import paired_summary


def estimate_summary(store):
    config = store.config
    metrics, paired, records, profile_rows, costs = [], [], [], [], []
    fits, profiles = {}, {}
    refinement = store.result('delivery3/profile-refinement.json')
    unresolved_refinement = {(r['founders'], r['replicate'], r['method'], r['rate'])
                            for r in (refinement or {}).get('rows', []) if not r['passed'] or
                            (any(a<=config['truth'][r['rate']]<=b for a,b in r['production_intervals']) !=
                             any(a<=config['truth'][r['rate']]<=b for a,b in r['refined_intervals']))}
    coverage_refinement = store.result('delivery3/coverage-refinement.json')
    unresolved_refinement.update((r['founders'],r['replicate'],r['method'],r['rate'])
                                for r in (coverage_refinement or {}).get('rows',[]) if not r['passed'])
    def fit_resolved(fitted, profile):
        return bool(fitted and fitted['success'] and
                    'exceeds the fitted maximum' not in (profile or {}).get('reason',''))
    for n in config['founders']:
        for method_index, method in enumerate(config['methods']):
            key = (n, method)
            fits[key], profiles[key] = {}, {}
            for rep in range(100):
                f = store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')
                p = store.result(f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
                fits[key][rep], profiles[key][rep] = f, p
                records.append(dict(founders=n, replicate=rep, method=method, result=f,
                                    source_delivery=2 if rep<20 else 3, fit_resolved=fit_resolved(f,p)))
                for rate in range(2):
                    resolved = bool(f and f['success'] and p and p['success'] and
                                    (n,rep,method,rate) not in unresolved_refinement)
                    row = dict(founders=n, replicate=rep, method=method, rate=f'b{rate+1}', resolved=resolved)
                    if resolved:
                        q = p['rates'][rate]
                        row.update(intervals=q['intervals'], length=q['length'], disconnected=q['disconnected'],
                                   flat=q['flat'], truncated=any(q['bound_truncated']), bound_truncated=q['bound_truncated'],
                                   covered=any(a<=config['truth'][rate]<=b for a,b in q['intervals']))
                    else:
                        row['reason'] = (p or {}).get('reason','Missing profile or failed numerical refinement')
                    profile_rows.append(row)
            good = {r:f for r,f in fits[key].items() if fit_resolved(f,profiles[key][r])}
            rng = stream(config['master_seed'], 'diagnostics', n, 0, 0, 100+method_index, 'bootstrap')
            draws = rng.integers(0,len(good),(2000,len(good))) if good else None
            for rate in range(2):
                row = dict(founders=n, method=method, rate=f'b{rate+1}', attempted=100, successful=len(good),
                           failures=100-len(good), excluded_replicates=[r for r in range(100) if r not in good])
                if good:
                    error = np.array([f['estimate'][rate]-config['truth'][rate] for f in good.values()])
                    row.update(bias=float(error.mean()), bias_mcse=float(error.std(ddof=1)/np.sqrt(len(error))) if len(error)>1 else None,
                               rmse=float(np.sqrt(np.mean(error**2))), rmse_ci95=np.quantile(np.sqrt(np.mean(error[draws]**2,axis=1)),[.025,.975]),
                               boundary_count=sum(f['boundary'][rate] for f in good.values()),
                               boundary_frequency=float(np.mean([f['boundary'][rate] for f in good.values()])),
                               failed_starts=sum(f['failed_starts'] for f in good.values()))
                sets = [r for r in profile_rows if r['founders']==n and r['method']==method and r['rate']==f'b{rate+1}' and r['resolved']]
                covered = sum(r['covered'] for r in sets)
                row.update(profile_resolved=len(sets), profile_unresolved=100-len(sets), covered=covered,
                           coverage_bounds=[covered/100, (covered+100-len(sets))/100])
                if sets:
                    coverage = covered/len(sets)
                    lengths = np.array([r['length'] for r in sets])
                    row.update(coverage=coverage, coverage_mcse=float(np.sqrt(coverage*(1-coverage)/len(sets))),
                               coverage_ci95=[float(beta.ppf(.025,covered,len(sets)-covered+1)) if covered else 0.,
                                              float(beta.ppf(.975,covered+1,len(sets)-covered)) if covered<len(sets) else 1.],
                               mean_length=float(lengths.mean()), length_mcse=float(lengths.std(ddof=1)/np.sqrt(len(sets))) if len(sets)>1 else None,
                               disconnected_count=sum(r['disconnected'] for r in sets),
                               truncated_count=sum(r['truncated'] for r in sets), flat_count=sum(r['flat'] for r in sets),
                               disconnected_fraction=float(np.mean([r['disconnected'] for r in sets])),
                               truncated_fraction=float(np.mean([r['truncated'] for r in sets])))
                metrics.append(row)
            # Keep original and revised production timing strata separate.
            for label, ids in [('delivery2_original',range(20)), ('delivery3_revised',range(20,100))]:
                values = [fits[key][r] for r in ids if fits[key][r]]
                if values:
                    costs.append(dict(founders=n,method=method,stratum=label,count=len(values),
                        total_seconds=sum(f['seconds'] for f in values), mean_seconds=float(np.mean([f['seconds'] for f in values])),
                        mean_grid_seconds=float(np.mean([f['grid_seconds'] for f in values if f['success']])) if any(f['success'] for f in values) else None,
                        mean_warm_seconds=float(np.mean([f['warm_loglik_seconds'] for f in values if f['success']])) if any(f['success'] for f in values) else None,
                        mean_components={k:float(np.mean([f['timings'][k] for f in values])) for k in
                            ('construction_seconds','moment_seconds','extinction_seconds','spectral_amplitude_seconds','positive_amplitude_seconds','integration_seconds')}))
            values = [p for p in profiles[key].values() if p]
            costs.append(dict(founders=n,method=method,stratum='delivery3_profiles',count=len(values),
                total_seconds=sum(p['seconds'] for p in values), mean_seconds=float(np.mean([p['seconds'] for p in values])) if values else None,
                mean_components={k:float(np.mean([p.get('timings',{}).get(k,0.) for p in values])) for k in
                    ('construction_seconds','moment_seconds','extinction_seconds','spectral_amplitude_seconds','positive_amplitude_seconds','integration_seconds')}))
        for index, comparator in enumerate(('G','EG','M0')):
            left, right = fits[n,'M'], fits[n,comparator]
            ids = [r for r in range(100) if fit_resolved(left[r],profiles[n,'M'][r]) and fit_resolved(right[r],profiles[n,comparator][r])]
            rng = stream(config['master_seed'],'diagnostics',n,0,0,200+index,'bootstrap')
            draws = rng.integers(0,len(ids),(2000,len(ids))) if ids else None
            # Same whole-dataset resampling for both rates in each comparison.
            for rate in range(2):
                row = dict(founders=n,comparison='M-'+comparator,rate=f'b{rate+1}',paired_successful=len(ids),
                           included_replicates=ids,excluded_replicates=[r for r in range(100) if r not in ids])
                if len(ids)>1:
                    a = np.array([left[r]['estimate'][rate]-config['truth'][rate] for r in ids])
                    b = np.array([right[r]['estimate'][rate]-config['truth'][rate] for r in ids])
                    ra, rb = np.sqrt(np.mean(a*a)),np.sqrt(np.mean(b*b))
                    ba, bb = np.sqrt(np.mean(a[draws]**2,axis=1)),np.sqrt(np.mean(b[draws]**2,axis=1))
                    row.update(rmse_difference=float(ra-rb),rmse_difference_ci95=np.quantile(ba-bb,[.025,.975]),
                               rmse_reduction=float(1-ra/rb),rmse_reduction_ci95=np.quantile(1-ba/bb,[.025,.975]))
                ls = {r['replicate']:r for r in profile_rows if r['founders']==n and r['method']=='M' and r['rate']==f'b{rate+1}' and r['resolved']}
                rs = {r['replicate']:r for r in profile_rows if r['founders']==n and r['method']==comparator and r['rate']==f'b{rate+1}' and r['resolved']}
                common = sorted(set(ls)&set(rs))
                row.update(paired_profiles=len(common),excluded_profiles=[r for r in range(100) if r not in common])
                if len(common)>1:
                    rngp = stream(config['master_seed'],'diagnostics',n,0,0,300+index,'bootstrap')
                    boot = rngp.integers(0,len(common),(2000,len(common)))
                    length = np.array([ls[r]['length']-rs[r]['length'] for r in common])
                    coverage = np.array([int(ls[r]['covered'])-int(rs[r]['covered']) for r in common])
                    row.update(length_difference=paired_summary(length),length_difference_ci95=np.quantile(length[boot].mean(axis=1),[.025,.975]),
                               coverage_difference=paired_summary(coverage),coverage_difference_ci95=np.quantile(coverage[boot].mean(axis=1),[.025,.975]))
                paired.append(row)
    return dict(metrics=metrics,paired=paired,fits=records,profiles=profile_rows,costs=costs,
                expected_fits=800,available_fits=sum(r['result'] is not None for r in records),
                successful_fits=sum(r['fit_resolved'] for r in records),
                optimizer_successful_fits=sum(bool(r['result'] and r['result']['success']) for r in records),
                available_two_rate_profiles=sum(p is not None for group in profiles.values() for p in group.values()),
                successful_two_rate_profiles=sum(bool(p and p['success']) for group in profiles.values() for p in group.values()),
                resolved_rate_sets=sum(r['resolved'] for r in profile_rows),bootstrap_repetitions=2000,
                bootstrap_keys='diagnostics, founder count, type=0, time=0, replicate=100+method index for individual RMSE; 200+comparator index for paired RMSE; 300+comparator index for paired coverage and length; bootstrap purpose. Method order [G,EG,M,M0]; comparator order [G,EG,M0].')


def cost_summary(store):
    rows, comparison = [], []
    for n in store.config['founders']:
        for method in store.config['methods']:
            records = [store.result(f'delivery3/cost/n{n}/rep{r:03d}-{method}.json') for r in range(5)]
            if any(r is None for r in records):
                raise ValueError('Complete matched cost measurements before report')
            for version in ('legacy','revised'):
                values = [r['versions'][version] for r in records]
                rows.append(dict(founders=n,method=method,version=version,fits=len(values),
                    mean_startup_seconds=float(np.mean([r['startup_seconds'][version] for r in records])),
                    total_startup_seconds=sum(r['startup_seconds'][version] for r in records),
                    controlled_profile_seconds=(records[0].get('controlled_profile') or {}).get('seconds') if version=='revised' else None,
                    controlled_profile_components=(records[0].get('controlled_profile') or {}).get('timings') if version=='revised' else None,
                    fit_seconds=float(np.mean([r['fit']['seconds'] for r in values])),
                    fit_seconds_se=float(np.std([r['fit']['seconds'] for r in values],ddof=1)/np.sqrt(len(values))),
                    cold_seconds=float(np.mean([np.mean(r['cold_seconds']) for r in values])),
                    warm_seconds=float(np.mean([np.mean(r['warm_seconds']) for r in values])),
                    components={k:float(np.mean([r['fit']['timings'][k] for r in values])) for k in
                        ('construction_seconds','moment_seconds','extinction_seconds','spectral_amplitude_seconds','positive_amplitude_seconds','integration_seconds')}))
            comparison.append(dict(founders=n,method=method,maximum_estimate_change=max(r['estimate_change'] for r in records),
                maximum_loglik_change=max(r['loglik_change'] for r in records),passed=all(r['success'] for r in records),
                maximum_change_from_retained_estimate=max(float(np.max(abs(np.array(record['versions']['revised']['fit']['estimate'])-
                    store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')['estimate']))) for rep,record in enumerate(records)),
                maximum_change_from_retained_loglik=max(abs(record['versions']['revised']['fit']['loglik']-
                    store.result(f'fits/inference/n{n}/rep{rep:03d}-{method}.json')['loglik']) for rep,record in enumerate(records)),
                paired_fit_seconds=paired_summary([r['versions']['revised']['fit']['seconds']-r['versions']['legacy']['fit']['seconds'] for r in records])))
    return dict(rows=rows,comparison=comparison,passed=all(r['passed'] for r in comparison),
                policy=records[0]['policy'],replicates=list(range(5)),production_replacement=False)


def finalize_provenance(root, destination):
    """Hash completed stage records after run_delivery writes its final timing."""
    root, destination = Path(root), Path(destination)
    provenance = read_json(destination/'provenance.json')
    current = provenance['report']['source_sha256']['code/founder_mixture/birth_death_extended.py']
    timings = []
    for path in sorted((root/'timings').glob('delivery3-*.json')):
        record = read_json(path)
        source = record['provenance']['source_sha256'].get('code/founder_mixture/birth_death_extended.py')
        timing = dict(path=str(path.relative_to(root)),source_sha256=source,**record['result'])
        if timing['command']=='cost':
            timing['cost_protocol']='both caches cleared' if source==current else 'candidate cache only; superseded'
        timings.append(timing)
    provenance['timings'] = timings
    provenance['artifacts'] = {str(path.relative_to(root)):sha(path) for folder in ('fits/inference','delivery3')
                               for path in sorted((root/folder).rglob('*.json'))}
    provenance['finalization'] = 'Artifact hashes and invocation timings captured after the report stage completed.'
    write_json(destination/'provenance.json',provenance)
    summary = read_json(destination/'tables/delivery3-summary.json')
    summary['timings'] = timings
    write_json(destination/'tables/delivery3-summary.json',summary)


def export(store, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    tables, figures = destination/'tables',destination/'figures'
    tables.mkdir(exist_ok=True,parents=True); figures.mkdir(exist_ok=True,parents=True)
    summary = estimate_summary(store)
    costs = cost_summary(store)
    refinement = store.result('delivery3/profile-refinement.json')
    coverage_refinement = store.result('delivery3/coverage-refinement.json')
    final_audit = store.result('delivery3/final-audit.json')
    if coverage_refinement is None or final_audit is None:
        raise ValueError('Complete the final artifact audit and coverage-endpoint checks before export')
    for r in refinement['rows']:
        if r['passed']:
            truth = store.config['truth'][r['rate']]
            r['coverage_agrees'] = (any(a<=truth<=b for a,b in r['production_intervals']) ==
                                   any(a<=truth<=b for a,b in r['refined_intervals']))
            r['passed'] = r['coverage_agrees']
    refinement['passed'] = all(r['passed'] for r in refinement['rows'])
    evidence = dict(summary=summary,cost=costs,checks=store.result('delivery3/checks.json'),
                    profile_check=store.result('delivery3/profile-check.json'),profile_refinement=refinement,
                    baseline=store.baseline)
    for name,value in evidence.items():
        write_json(tables/f'delivery3-{name}.json',value)
    for name,rows in [('metrics',summary['metrics']),('paired',summary['paired']),('profile-sets',summary['profiles']),
                      ('production-cost',summary['costs']),('matched-cost',costs['rows']),('profile-refinement',refinement['rows'])]:
        csv_table(tables/f'delivery3-{name}.csv',rows)
    csv_table(tables/'delivery3-fits.csv',[dict(founders=r['founders'],replicate=r['replicate'],source_delivery=r['source_delivery'],fit_resolved=r['fit_resolved'],**(r['result'] or {'method':r['method'],'success':False})) for r in summary['fits']])
    def f(v):
        if v is None:
            return '--'
        value=f'{v:.4g}'
        if 'e' in value:
            mantissa,exponent=value.split('e')
            return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
        return value
    def ci(v,scale=1): return '['+', '.join(f(x*scale if x is not None else None) for x in v)+']'
    def rate(r): return '$b_'+r['rate'][-1]+'$'
    rows = [[r['founders'],r['method'],rate(r),f(r.get('bias'))+' ('+f(r.get('bias_mcse'))+')',f(r.get('rmse')),
             f"{r.get('boundary_count',0)}/{r['successful']}",f"{r['failures']}/100"] for r in summary['metrics']]
    latex_table(tables/'delivery3-estimation.tex',[r'$N_0$','Method','Rate','Bias (MCSE)','RMSE','Boundary','Failures'],rows)
    rows = [[r['founders'],rate(r),r['comparison'],f(r.get('rmse_difference')),ci(r.get('rmse_difference_ci95',[None,None])),
             (f(100*r['rmse_reduction'])+' '+ci(r['rmse_reduction_ci95'],100)) if 'rmse_reduction' in r else '--',r['paired_successful']] for r in summary['paired']]
    latex_table(tables/'delivery3-paired-rmse.tex',[r'$N_0$','Rate','Pair',r'$\Delta$ RMSE',r'95\% interval',r'Reduction (\%) [95\%]','$n$'],rows)
    rows = [[r['founders'],r['method'],rate(r),f"{r['covered']}/{r['profile_resolved']}",f(100*r.get('coverage_mcse',0)),
             f(r.get('mean_length'))+' ('+f(r.get('length_mcse'))+')',ci(r['coverage_bounds'],100),
             f"{r.get('disconnected_count',0)}/{r.get('truncated_count',0)}/{r['profile_unresolved']}"] for r in summary['metrics']]
    latex_table(tables/'delivery3-coverage.tex',[r'$N_0$','Method','Rate','Covered',r'MCSE (pp)','Length (MCSE)',r'Bounds (\%)','D/T/U'],rows)
    rows = [[r['founders'],rate(r),r['comparison'],(f(100*r['coverage_difference']['mean'])+' ('+f(100*r['coverage_difference']['se'])+')') if 'coverage_difference' in r else '--',
             ci(r['coverage_difference_ci95'],100) if 'coverage_difference_ci95' in r else '--',
             f(r.get('length_difference',{}).get('mean')),ci(r['length_difference_ci95']) if 'length_difference_ci95' in r else '--',r['paired_profiles']] for r in summary['paired']]
    latex_table(tables/'delivery3-paired-profile.tex',[r'$N_0$','Rate','Pair',r'$\Delta$ cover (pp)',r'95\% interval',r'$\Delta$ length',r'95\% interval','$n$'],rows)
    rows = [[r['founders'],r['method'],r['version'],f(r['fit_seconds']),f(r['cold_seconds']*1000),f(r['warm_seconds']*1000),
             f(r['components']['extinction_seconds']), f(r['controlled_profile_seconds'])] for r in costs['rows']]
    latex_table(tables/'delivery3-matched-cost.tex',[r'$N_0$','Method','Version','Fit (s)','Cold (ms)','Warm (ms)','Extinct. (s)','Profile (s)'],rows)
    rows = [[r['founders'],r['method'],f(r['components']['moment_seconds']),f(r['components']['spectral_amplitude_seconds']),
             f(r['components']['positive_amplitude_seconds']),f(r['components']['construction_seconds']),f(r['components']['integration_seconds'])] for r in costs['rows'] if r['version']=='revised']
    latex_table(tables/'delivery3-cost-components.tex',[r'$N_0$','Method','Moments','Spectral/amplitude','Positive amplitude','Construction','Integration'],rows)
    rows = []
    for n in store.config['founders']:
        for method in store.config['methods']:
            fitcost = next(r for r in summary['costs'] if r['founders']==n and r['method']==method and r['stratum']=='delivery3_revised')
            profcost = next(r for r in summary['costs'] if r['founders']==n and r['method']==method and r['stratum']=='delivery3_profiles')
            check = [r for r in refinement['rows'] if r['founders']==n and r['method']==method]
            rows.append([n,method,f(fitcost['mean_seconds']),f(profcost['mean_seconds']),f(profcost['total_seconds']),
                         f"{sum(r['passed'] for r in check)}/{len(check)}",f(max((r.get('endpoint_change') or 0 for r in check),default=0))])
    latex_table(tables/'delivery3-profile-cost-refinement.tex',[r'$N_0$','Method','Fit (s)','Profile (s)','Profile total (s)','Checks','Endpoint change'],rows)
    colors = {'G':'#b33c31','EG':'#d58b00','M':'#126da8','M0':'#238151'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(8.5,5.7),layout='constrained')
    for i,n in enumerate(store.config['founders']):
        for j,(key,label) in enumerate([('rmse','RMSE (per day)'),('coverage','Coverage'),('mean_length','Mean total set length (per day)')]):
            ax=axes[i,j]
            for k,m in enumerate(store.config['methods']):
                rows=[r for r in summary['metrics'] if r['founders']==n and r['method']==m]
                x=np.arange(2)+(k-1.5)*.18
                vals=[r.get(key,np.nan) for r in rows]
                err=[r['coverage_mcse'] if key=='coverage' else r['length_mcse'] if key=='mean_length' else 0 for r in rows]
                ax.bar(x,vals,width=.17,color=colors[m],label=m,yerr=err if key!='rmse' else None,capsize=2)
            ax.set(xticks=[0,1],xticklabels=[r'$b_1$',r'$b_2$'],ylabel=label,title=f'{n} founder'+('s' if n!=1 else ''))
            if key=='coverage': ax.axhline(.9,color='black',ls='--',lw=.8);ax.set_ylim(0,1.05)
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.savefig(figures/'delivery3-inference.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8,5.7),layout='constrained')
    for i,n in enumerate(store.config['founders']):
        for j in range(2):
            ax=axes[i,j]
            for m in store.config['methods']:
                p=store.result(f'delivery3/profiles/n{n}/rep000-{m}.json')
                if p and len(p.get('rates',[]))>j:
                    r=p['rates'][j]; x=np.array(r['x']); y=2*(p['fitted_loglik']-np.array(r['loglik']))
                    ax.plot(x,y,label=m,color=colors[m],lw=1.2,ls='--' if m in ('EG','M0') else '-',marker='.',markersize=2)
            ax.axhline(store.config['profiles']['cutoff'],color='black',ls='--',lw=.8)
            ax.set(ylim=(-.05,8),xlabel=f'Birth rate {j+1} (per day)',ylabel='Twice log-likelihood loss',title=f'{n} founder'+('s' if n!=1 else '')+', replicate 0')
            # Common axes per founder/rate show all four accepted sets.
            ranges=[r['intervals'] for r in summary['profiles'] if r['founders']==n and r['replicate']==0 and r['rate']==f'b{j+1}' and r['resolved']]
            ends=np.array([v for ints in ranges for pair in ints for v in pair])
            if len(ends): ax.set_xlim(max(.3,ends.min()-.01),min(1.1,ends.max()+.01))
    axes[0,0].legend(fontsize=9)
    fig.savefig(figures/'delivery3-profiles.png',dpi=180);plt.close(fig)
    timings=[]
    for p in sorted((store.root/'timings').glob('delivery3-*.json')):
        record=read_json(p)
        source=record['provenance']['source_sha256'].get('code/founder_mixture/birth_death_extended.py')
        timing=dict(path=str(p.relative_to(store.root)),source_sha256=source,**record['result'])
        if timing['command']=='cost':
            timing['cost_protocol']='both caches cleared' if source==store.provenance['source_sha256']['code/founder_mixture/birth_death_extended.py'] else 'candidate cache only; superseded'
        timings.append(timing)
    baseline_changed=[p for p,h in store.baseline['files'].items() if sha(store.root/p)!=h]
    data_records=[]
    for n in store.config['founders']:
        for rep in range(100):
            path=store.root/f'observations/inference/n{n}/rep{rep:03d}.npz'
            with np.load(path) as data:
                data_records.append(dict(founders=n,replicate=rep,sha256=sha(path),observations=len(data['y']),
                    incomplete=int(np.sum(~data['complete'])),negative_components=int(np.sum(data['y']<0)),
                    maximum_population=int(data['counts'].sum(axis=1).max())))
    audit=dict(baseline_unchanged=not baseline_changed,changed_baseline=baseline_changed,datasets=data_records,
               observations=sum(r['observations'] for r in data_records),incomplete_paths=sum(r['incomplete'] for r in data_records),
               negative_components=sum(r['negative_components'] for r in data_records),maximum_population=max(r['maximum_population'] for r in data_records))
    write_json(tables/'delivery3-data-audit.json',audit)
    summary.update(complete=summary['available_fits']==800 and summary['available_two_rate_profiles']==800,
                   numerical_passed=bool(evidence['checks']['passed'] and costs['passed'] and refinement['passed'] and
                                         coverage_refinement['passed'] and final_audit['passed']),
                   baseline_unchanged=audit['baseline_unchanged'], observations=audit['observations'],
                   incomplete_paths=audit['incomplete_paths'], negative_components=audit['negative_components'],
                   timings=timings, additional_production_fits=640, retained_production_fits=160,
                   numerical_profile_calculations=40, numerical_rate_sets=80,
                   matched_cost_fits=80, controlled_profile_cost_calculations=8,
                   coverage_refinement=coverage_refinement,
                   coverage_refinement_timing=store.result('delivery3/coverage-refinement-timing.json'),
                   interrupted_refinement=store.result('delivery3/refinement-interruption.json'),
                   review_boundary='Delivery 3 complete; no further experiments authorized')
    summary['superseded_cost_fits']=2*sum(not p.name.endswith('.retry.json') for p in
        (store.root/'attempts').glob('*/delivery3/cost/n*/rep*.json'))
    write_json(tables/'delivery3-summary.json',summary)
    original=destination/'delivery2-provenance.json'
    if not original.exists(): original.write_bytes((destination/'provenance.json').read_bytes())
    write_json(destination/'provenance.json',dict(report=store.provenance,execution_delivery=3,delivery1_provenance='delivery1-provenance.json',
        delivery2_provenance='delivery2-provenance.json',baseline_manifest='tables/delivery3-baseline.json',timings=timings,
        artifacts={str(p.relative_to(store.root)):sha(p) for folder in ('fits/inference','delivery3') for p in sorted((store.root/folder).rglob('*.json'))},
        compatibility='First 20 datasets/founder and all original fits retained byte-for-byte; revised densities checked against frozen sources and original maxima. Revised cost is measured by actual paired executions, not inferred by subtraction.'))
    print({key:summary[key] for key in ('available_fits','successful_fits','available_two_rate_profiles','successful_two_rate_profiles','resolved_rate_sets')},flush=True)
