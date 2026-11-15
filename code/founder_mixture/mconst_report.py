"""Saved-result summaries for Mconst; narrative lives in the existing report."""

from pathlib import Path
import numpy as np
from scipy.stats import beta
from . import ROOT
from .storage import read_json, write_json
from .birth_death_report import csv_table, latex_table
from .mconst import sha


def interval(values):
    values=np.asarray(values); values=values[np.isfinite(values)]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else None


def mean_se(values):
    x=np.asarray(values)
    return float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None


def old_profile(store,n,rep,method):
    p=store.baseline_path(store.root.parent/f'delivery3/profiles/n{n}/rep{rep:03d}-{method}.json')
    return read_json(p)['result']


def collect(store):
    config=store.config
    fits,profiles={},{}
    baseline=read_json(store.baseline_path(ROOT/'reports/founder-mixture-birth-death/tables/delivery3-summary.json'))
    metrics=list(baseline['metrics'])
    records=[]; sets=[]; costs=[]; variances=[]
    refinement=store.result('refinement.json') or store.result('pilot.json') or {}
    unresolved={(r['founders'],r['replicate']) for r in refinement.get('rows',[]) if not r['passed']}
    for n in config['founders']:
        for method in ('G','EG','M','M0','Mconst'):
            fits[n,method]={}; profiles[n,method]={}
            for rep in range(100):
                f=store.result(f'fits/n{n}/rep{rep:03d}.json') if method=='Mconst' else store.old_fit(n,rep,method)
                p=store.result(f'profiles/n{n}/rep{rep:03d}.json') if method=='Mconst' else old_profile(store,n,rep,method)
                fits[n,method][rep]=f; profiles[n,method][rep]=p
                good=bool(f and f['success'] and 'exceeds the fitted maximum' not in (p or {}).get('reason',''))
                records.append(dict(founders=n,method=method,replicate=rep,fit_resolved=good,
                    estimate=f['estimate'] if f else None,loglik=f['loglik'] if f else None,
                    reason=(f or {}).get('reason'),residual_variance=(f or {}).get('residual_variance')))
                for i in range(2):
                    resolved=bool(good and p and p['success'] and (method!='Mconst' or (n,rep) not in unresolved))
                    row=dict(founders=n,method=method,replicate=rep,rate=f'b{i+1}',resolved=resolved)
                    if resolved:
                        q=p['rates'][i]
                        distance=min(abs(x-config['truth'][i]) for pair in q['intervals'] for x in pair)
                        # Near-truth sets require an additional saved accuracy check.
                        if method=='Mconst' and distance<=1e-5:
                            check_path=store.root/f'coverage/n{n}/rep{rep:03d}.json'
                            checked=store.result(f'coverage/n{n}/rep{rep:03d}.json') if check_path.exists() else None
                            if not checked or not checked.get('passed'):
                                row.update(resolved=False,reason='Coverage boundary requires additional refinement')
                        row.update(intervals=q['intervals'],length=q['length'],covered=any(a<=config['truth'][i]<=b for a,b in q['intervals']),
                            disconnected=q['disconnected'],flat=q['flat'],truncated=any(q['bound_truncated']),minimum_truth_distance=distance)
                    else:
                        row['reason']=(p or {}).get('reason','Missing result or unresolved numerical check')
                    sets.append(row)
                if method=='Mconst':
                    for kind,r in (('production_fit',f),('production_profile',p)):
                        if r:
                            costs.append(dict(founders=n,replicate=rep,method=method,kind=kind,success=r['success'],seconds=r['seconds'],
                                timings=r.get('timings'),validation_timings=r.get('validation_timings'),
                                execution_cost=r.get('execution_cost'),peak_rss_mib=r.get('peak_rss_mib'),
                                extinction_solver=r.get('extinction_solver')))
                    if good:
                        variances.append(dict(founders=n,replicate=rep,variance=f['residual_variance'],u=f['u'],boundary=f['variance_boundary'],
                            search_limit=f['variance_cap'],expansions=len(f['expansions'])-1,failed_starts=f['failed_starts']))
            if method!='Mconst': continue
            good_ids=[r['replicate'] for r in records if r['founders']==n and r['method']==method and r['fit_resolved']]
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([config['master_seed'],61,n,0,0,100,4])))
            draws=rng.integers(0,100,(2000,100))
            for i in range(2):
                row=dict(founders=n,method=method,rate=f'b{i+1}',attempted=100,successful=len(good_ids),failures=100-len(good_ids),
                         excluded_replicates=[r for r in range(100) if r not in good_ids])
                error=np.full(100,np.nan)
                for r in good_ids: error[r]=fits[n,method][r]['estimate'][i]-config['truth'][i]
                if good_ids:
                    row.update(bias=float(np.nanmean(error)),bias_mcse=mean_se(error[good_ids]),rmse=float(np.sqrt(np.nanmean(error**2))),
                        rmse_ci95=interval(np.sqrt(np.nanmean(error[draws]**2,axis=1))),
                        boundary_count=sum(fits[n,method][r]['boundary'][i] for r in good_ids),
                        boundary_frequency=float(np.mean([fits[n,method][r]['boundary'][i] for r in good_ids])))
                resolved=[r for r in sets if r['founders']==n and r['method']==method and r['rate']==f'b{i+1}' and r['resolved']]
                s=len(resolved); c=sum(r['covered'] for r in resolved)
                row.update(profile_resolved=s,profile_unresolved=100-s,covered=c,coverage_bounds=[c/100,(c+100-s)/100])
                if s:
                    p=c/s; lengths=[r['length'] for r in resolved]
                    row.update(coverage=p,coverage_mcse=float(np.sqrt(p*(1-p)/s)),
                        coverage_ci95=[float(beta.ppf(.025,c,s-c+1)) if c else 0.,float(beta.ppf(.975,c+1,s-c)) if c<s else 1.],
                        mean_length=float(np.mean(lengths)),length_mcse=mean_se(lengths),
                        disconnected_fraction=float(np.mean([r['disconnected'] for r in resolved])),
                        truncated_fraction=float(np.mean([r['truncated'] for r in resolved])),
                        flat_fraction=float(np.mean([r['flat'] for r in resolved])))
                metrics.append(row)
    paired=[]; bootstrap={}
    index={(r['founders'],r['method'],r['replicate'],r['rate']):r for r in sets}
    resolved_fits={(r['founders'],r['method'],r['replicate']):r['fit_resolved'] for r in records}
    for n in config['founders']:
        for comparison,left in enumerate(('M','M0')):
            key=[config['master_seed'],61,n,0,0,comparison,4]
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
            draws=rng.integers(0,100,(2000,100)); bootstrap[f'n{n}-{left}-Mconst']=dict(seed=key,indices=draws)
            for i in range(2):
                rate=f'b{i+1}'
                fit_ids=[r for r in range(100) if resolved_fits[n,left,r] and resolved_fits[n,'Mconst',r]]
                profile_ids=[r for r in range(100) if index[n,left,r,rate]['resolved'] and index[n,'Mconst',r,rate]['resolved']]
                row=dict(founders=n,comparison=left+'-Mconst',rate=rate,paired_successful=len(fit_ids),included_replicates=fit_ids,
                         excluded_replicates=[r for r in range(100) if r not in fit_ids],paired_profiles=len(profile_ids),
                         excluded_profiles=[r for r in range(100) if r not in profile_ids])
                if fit_ids:
                    a,b=np.full((2,100),np.nan)
                    for r in fit_ids:
                        a[r]=fits[n,left][r]['estimate'][i]-config['truth'][i]
                        b[r]=fits[n,'Mconst'][r]['estimate'][i]-config['truth'][i]
                    ra,rb=np.sqrt(np.nanmean(a*a)),np.sqrt(np.nanmean(b*b))
                    ba,bb=np.sqrt(np.nanmean(a[draws]**2,axis=1)),np.sqrt(np.nanmean(b[draws]**2,axis=1))
                    row.update(rmse_difference=float(ra-rb),rmse_difference_ci95=interval(ba-bb),
                        rmse_reduction=float(1-ra/rb),rmse_reduction_ci95=interval(1-ba/bb),
                        bias_difference=float(np.nanmean(a-b)),bias_difference_ci95=interval(np.nanmean((a-b)[draws],axis=1)))
                if profile_ids:
                    delta,length=np.full((2,100),np.nan)
                    for r in profile_ids:
                        a,b=index[n,left,r,rate],index[n,'Mconst',r,rate]
                        delta[r]=int(a['covered'])-int(b['covered']);length[r]=a['length']-b['length']
                    row.update(coverage_difference=float(np.nanmean(delta)),coverage_difference_mcse=mean_se(delta[profile_ids]),
                        coverage_difference_ci95=interval(np.nanmean(delta[draws],axis=1)),
                        length_difference=float(np.nanmean(length)),length_difference_ci95=interval(np.nanmean(length[draws],axis=1)))
                lo=hi=0
                for r in range(100):
                    a,b=index[n,left,r,rate],index[n,'Mconst',r,rate]
                    av=[int(a['covered'])]*2 if a['resolved'] else [0,1]
                    bv=[int(b['covered'])]*2 if b['resolved'] else [0,1]
                    lo+=av[0]-bv[1];hi+=av[1]-bv[0]
                row['coverage_difference_bounds']=[lo/100,hi/100]
                paired.append(row)
    return dict(metrics=metrics,paired=paired,records=records,sets=sets,costs=costs,variances=variances,bootstrap=bootstrap)


def figures(result,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'G':'G','EG':'EG','M':'M','M0':r'$M_0$','Mconst':r'$M_{\mathrm{const}}$'}
    for quantity,ci,label in [('rmse','rmse_ci95','RMSE (per day)'),('coverage','coverage_ci95','Coverage'),('mean_length',None,'Mean total set length (per day)')]:
        fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for row,n in enumerate((1,50)):
            for col,rate in enumerate(('b1','b2')):
                ax=axes[row,col]
                for j,method in enumerate(labels):
                    r=next(x for x in result['metrics'] if x['founders']==n and x['rate']==rate and x['method']==method)
                    if quantity not in r:continue
                    value=r[quantity]
                    bounds=r.get(ci) if ci else ([max(0,value-1.96*r['length_mcse']),value+1.96*r['length_mcse']] if r.get('length_mcse') is not None else None)
                    uncertainty=np.array([[max(0,value-bounds[0])],[max(0,bounds[1]-value)]]) if bounds else None
                    ax.errorbar(j,value,yerr=uncertainty,fmt='o',capsize=3,
                                color='C2' if method=='Mconst' else ('C0' if method=='M' else '0.45'))
                ax.set_xticks(range(5),labels.values());ax.set_title(f'{n} founder'+('s' if n>1 else '')+r', $b_'+rate[-1]+'$')
                ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
                if quantity=='coverage':ax.axhline(.9,color='0.5',ls='--',lw=1);ax.set_ylim(0,1.02)
        fig.savefig(destination/f'mconst-{quantity}.png',dpi=180);plt.close(fig)


def report(store):
    from .mconst_validation import run
    validation=run(store)
    if not validation['passed']:
        raise ArithmeticError('Saved-result validation failed; inspect validation.json before interpretation')
    destination=ROOT/'reports/founder-mixture-birth-death';tables=destination/'tables'
    result=collect(store)
    for name in ('metrics','paired','records','sets','costs','variances'):
        csv_table(tables/f'mconst-{name}.csv',result[name])
    bootstrap=result.pop('bootstrap')
    index_path=store.root/'bootstrap-indices.npz'
    temporary=index_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**{k:v['indices'] for k,v in bootstrap.items()})
    temporary.replace(index_path)
    write_json(tables/'mconst-bootstrap.json',dict(path=str(index_path.relative_to(ROOT)),sha256=sha(index_path),
        comparisons={k:dict(seed=v['seed'],shape=list(v['indices'].shape)) for k,v in bootstrap.items()},
        individual_metric_seed='SeedSequence([20261017,61,N0,0,0,100,4])'))
    write_json(tables/'mconst-summary.json',dict(result=result,provenance=store.provenance))
    def f(x,d=4):return '--' if x is None else f'{x:.{d}f}'
    def rate_label(rate):return '$b_'+rate[-1]+'$'
    def estimate_interval(row,key,scale=1,digits=4):
        if key not in row:return '--'
        a,b=row[key+'_ci95']
        return f'{scale*row[key]:.{digits}f} [{scale*a:.{digits}f}, {scale*b:.{digits}f}]'
    rows=[]
    for r in result['metrics']:
        if r['method']!='Mconst':continue
        bias=f(r.get('bias'))+' ('+f(r.get('bias_mcse'))+')'
        rows.append([r['founders'],rate_label(r['rate']),r['successful'],bias,estimate_interval(r,'rmse'),r.get('boundary_count','--')])
    latex_table(tables/'mconst-inference.tex',[r'$N_0$','Rate','Fits','Bias (MCSE)',r'RMSE [95\% interval]','Boundary'],rows)
    rows=[]
    for r in result['metrics']:
        if r['method']!='Mconst':continue
        counts='/'.join(str(round(r.get(key,0)*r['profile_resolved'])) for key in ('disconnected_fraction','truncated_fraction'))+'/'+str(r['profile_unresolved'])
        bounds='['+', '.join(f(100*x,1) for x in r['coverage_bounds'])+']'
        rows.append([r['founders'],rate_label(r['rate']),f"{r['covered']}/{r['profile_resolved']}",estimate_interval(r,'coverage',100,1),
                     f(100*r['coverage_mcse'],1) if 'coverage_mcse' in r else '--',
                     f(r.get('mean_length'),6)+' ('+f(r.get('length_mcse'),6)+')',counts,bounds])
    latex_table(tables/'mconst-coverage.tex',[r'$N_0$','Rate','Covered',r'Coverage [95\% interval]','MCSE','Length (MCSE)','D/T/U','Bounds'],rows)
    rows=[]
    rmse_rows=[]
    bias_rows=[]
    bound_rows=[]
    for r in result['paired']:
        def ci(key,scale=1):
            if key not in r:return '--'
            return f'{scale*r[key]:.3f} [{scale*r[key+"_ci95"][0]:.3f}, {scale*r[key+"_ci95"][1]:.3f}]'
        comparison='$'+('M' if r['comparison'].startswith('M-') else 'M_0')+r'-M_{\mathrm{const}}$'
        rows.append([r['founders'],comparison,rate_label(r['rate']),r['paired_profiles'],ci('coverage_difference',100),estimate_interval(r,'length_difference',digits=6)])
        rmse_rows.append([r['founders'],comparison,rate_label(r['rate']),r['paired_successful'],estimate_interval(r,'rmse_difference',digits=6),ci('rmse_reduction',100)])
        bias_rows.append([r['founders'],comparison,rate_label(r['rate']),r['paired_successful'],estimate_interval(r,'bias_difference',digits=7)])
        bound_rows.append([r['founders'],comparison,rate_label(r['rate']),r['paired_profiles'],
                           '['+', '.join(f(100*x,1) for x in r['coverage_difference_bounds'])+']'])
    latex_table(tables/'mconst-paired.tex',[r'$N_0$','Comparison','Rate',r'$n$','Coverage diff. (pp)','Length diff.'],rows)
    latex_table(tables/'mconst-paired-rmse.tex',[r'$N_0$','Comparison','Rate',r'$n$',r'$\Delta$ RMSE',r'RMSE reduction (\%)'],rmse_rows)
    latex_table(tables/'mconst-paired-bias.tex',[r'$N_0$','Comparison','Rate',r'$n$','Bias difference [95\% interval]'],bias_rows)
    latex_table(tables/'mconst-paired-coverage-bounds.tex',[r'$N_0$','Comparison','Rate',r'$n$','Coverage difference bounds (pp)'],bound_rows)
    matched=[]
    for n in store.config['founders']:
        for method in ('M','Mconst'):
            runs=[store.result(f'cost/n{n}/rep{rep:03d}.json') for rep in range(5)]
            runs=[r['methods'][method] for r in runs if r]
            if not runs:continue
            warm=[r['fit']['warm_loglik_seconds'] for r in runs if r['fit']['success']]
            matched.append(dict(founders=n,method=method,count=len(runs),
                successful_fits=sum(r['fit']['success'] for r in runs),
                successful_profiles=sum(bool(r['profile'] and r['profile']['success']) for r in runs),
                mean_fit_seconds=float(np.mean([r['fit_call_seconds'] for r in runs])),
                mean_warm_loglik_seconds=float(np.mean(warm)) if warm else None,
                profile_seconds=next((r['profile_call_seconds'] for r in runs if r['profile']),None),
                mean_fit_checkpoint_seconds=float(np.mean([r['fit'].get('execution_cost',{}).get('checkpoint_seconds',0.) for r in runs])),
                profile_checkpoint_seconds=next((r['profile'].get('execution_cost',{}).get('checkpoint_seconds',0.) for r in runs if r['profile']),None),
                startup_seconds=[r['startup_seconds'] for r in runs],
                initialization='Retained M0 estimate adds a feasible start; its historical fitting cost is excluded.' if method=='Mconst' else 'Original multistart fit.',
                policy='One worker and one numerical thread; same compiled extinction solver; clear all candidate and extinction caches before each fit and profile. Outer call timers include diagnostics, warm evaluation, and Mconst internal checkpoint costs. Completed-result storage is included in invocation time. Mconst saves each grid/local optimization/profile point; M saves completed fit/profile results.'))
    csv_table(tables/'mconst-matched-cost.csv',matched)
    latex_table(tables/'mconst-matched-cost.tex',[r'$N_0$','Method','Fits','Mean fit (s)','Profile (s)'],
                [[r['founders'],r['method'].replace('Mconst',r'$M_{\mathrm{const}}$'),r['count'],f(r['mean_fit_seconds'],2),f(r['profile_seconds'],2)] for r in matched])
    variance_summary=[]
    for n in store.config['founders']:
        vs=[r for r in result['variances'] if r['founders']==n]
        if vs:
            values=[r['variance'] for r in vs]
            saved_fits=[store.result(f'fits/n{n}/rep{rep:03d}.json') for rep in range(100)]
            identification=[]
            for rep,fit in enumerate(saved_fits):
                if not fit or not fit['success']:continue
                near=[r['theta'][2] for h in fit['expansions'] for r in h['searches']
                      if r['success'] and r.get('face')==[0,0,0]
                      and r['objective'] <= -fit['loglik']+1e-4]
                identification.append(dict(replicate=rep,near_optimal_interior_solutions=len(near),
                    u_range=float(np.ptp(near)) if near else None))
            saved_profiles=[store.result(f'profiles/n{n}/rep{rep:03d}.json') for rep in range(100)]
            constrained=[v for p in saved_profiles if p for v in p.get('evaluated',{}).values()]
            variance_summary.append(dict(founders=n,count=len(vs),median=float(np.median(values)),
                quartiles=np.quantile(values,[.25,.75]).tolist(),minimum=min(values),maximum=max(values),
                zero_boundary_count=sum(r['boundary'] for r in vs),zero_boundary_definition='u < 1e-7',
                fit_expansions=sum(r['expansions'] for r in vs),
                profile_expansions=sum(len(v['expansions'])-1 for v in constrained),
                maximum_profile_variance_cap=max((v['variance_cap'] for v in constrained),default=None),
                unresolved_profiles=sum(bool(p and not p['success']) for p in saved_profiles),
                failed_fit_starts=sum(r['failed_starts'] for r in vs),
                failed_profile_starts=sum(h['failed_starts'] for v in constrained for h in v['expansions']),
                numerical_variance_identification=dict(
                    definition='Range of u across successful unconstrained local solutions within 1e-4 log-likelihood units of the saved maximum. A range above 1e-3 flags numerical disagreement; this is not a confidence interval or a test of statistical identification.',
                    maximum_u_range=max((r['u_range'] for r in identification if r['u_range'] is not None),default=None),
                    flagged_replicates=[r['replicate'] for r in identification if r['u_range'] is None or r['u_range']>1e-3],
                    rows=identification)))
    write_json(tables/'mconst-variance-summary.json',variance_summary)
    write_json(tables/'mconst-execution.json',dict(
        numerical_timings=[dict(path=str(p.relative_to(store.root)),record=read_json(p)) for p in sorted((store.root/'timings').glob('*.json'))],
        workflows=[dict(path=str(p.relative_to(store.root)),record=read_json(p)) for p in
                   [*sorted((store.root/'workflow-history').glob('*.json')),*([store.root/'workflow.json'] if (store.root/'workflow.json').exists() else [])]],
        original_solver_results=str(store.root/'source-transition-20261019'),
        partial_time_limit='Accumulated checkpointed costs exclude time paused. Work after the last durable record is unmeasured after abrupt interruption; recorded interrupted totals are lower bounds.',
        storage_policy='Temporary JSON, file fsync, atomic replacement, directory fsync, previous valid generation retained.'))
    for name in ('checks','analytic-checks','density-checks','pilot','refinement','coverage','validation',
                 'recovery-checks','solver-checks','solver-compatibility'):
        path=store.root/(name+'.json')
        if path.exists():write_json(tables/f'mconst-{name}.json',read_json(path))
    figures(result,destination/'figures')
    write_json(destination/'mconst-provenance.json',dict(provenance=store.provenance,
        inputs={str(p.relative_to(ROOT)):sha(p) for folder in ('fits','profiles','fits-refined','profiles-refined','cost','coverage') for p in sorted((store.root/folder).rglob('*.json'))},
        outputs={str(p.relative_to(ROOT)):sha(p) for folder in ('tables','figures') for p in sorted((destination/folder).glob('mconst-*'))}))
    return dict(passed=True,production_fits=sum(r['method']=='Mconst' and r['estimate'] is not None for r in result['records']),
                resolved_rate_sets=sum(r['method']=='Mconst' and r['resolved'] for r in result['sets']))


def audit(store):
    # Narrative and current build metadata are deliberately updated; scientific
    # baseline tables, fit/profile artifacts, and observations remain immutable.
    allowed={'reports/founder-mixture-birth-death/founder-mixture-birth-death.tex',
             'reports/founder-mixture-birth-death/README.md','reports/founder-mixture-birth-death/tables/build-verification.json'}
    changed=[p for p,h in store.baseline['files'].items() if p not in allowed and (not (ROOT/p).exists() or sha(ROOT/p)!=h)]
    counts={folder:len(list((store.root/folder).glob('n*/rep*.json'))) for folder in ('fits','profiles','fits-refined','profiles-refined','cost')}
    checks={name:(store.result(name+'.json') or {}).get('passed',False) for name in
            ('checks','pilot','validation','recovery-checks','solver-checks','solver-compatibility')}
    result=dict(passed=not changed and all(checks.values()) and counts==dict(fits=200,profiles=200,**{'fits-refined':10,'profiles-refined':10,'cost':10}),
                baseline_artifacts=len(store.baseline['files']),unexpected_changes=changed,counts=counts,
                checks=checks,allowed_narrative_updates=sorted(allowed))
    write_json(ROOT/'reports/founder-mixture-birth-death/tables/mconst-audit.json',dict(result=result,provenance=store.provenance))
    return result
