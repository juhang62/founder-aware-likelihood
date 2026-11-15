"""Export Delivery 1 tables, figures, provenance, and machine-readable summary."""

import csv
import json
import re
from pathlib import Path
import numpy as np
from .storage import write_json, read_json, serializable


def csv_table(path, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fields,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,default=serializable) if isinstance(value,(dict,list,np.ndarray)) else value for key,value in row.items()})


def latex_table(path,headers,rows,columns=None):
    lines=[r"\begin{tabular}{"+(columns or "r"*len(headers))+"}",r"\toprule"," & ".join(headers)+r" \\",r"\midrule"]
    def number(value):
        text=str(value)
        if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]*)?e[+-]?[0-9]+",text):
            mantissa,exponent=text.split("e")
            return "$"+mantissa+r"\times10^{"+str(int(exponent))+"}$"
        return text
    lines += [" & ".join(number(value) for value in row)+r" \\" for row in rows]
    lines += [r"\bottomrule",r"\end{tabular}"]
    path.write_text("\n".join(lines)+"\n")


def export(store, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest=Path(destination)
    if dest.resolve().name != store.config["experiment"]:
        raise ValueError("Use a separate report directory named founder-mixture-birth-death")
    tables,figures=dest/"tables",dest/"figures"
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    write_json(dest/"configuration.json",store.config)
    design=store.result("preflight.json")
    one=store.result("one-type.json")
    checks=store.result("checks.json")
    write_json(tables/"job-preservation-check.json",store.result("job-preservation-check.json"))
    smoke=store.result("smoke.json")
    integration=store.result("smoke-integration.json")
    refits=store.result("smoke-refits.json")
    if any(x is None for x in (design,one,checks,smoke,integration,refits)):
        raise ValueError("Delivery 1 stage artifacts are required for the final initial report")
    for key in ("spectrum","founder_states","endpoints","extinction_refinement"):
        csv_table(tables/f"design-{key}.csv",design[key])
    for key in ("quantiles","densities","conditional_moments","refinements","reference_checks"):
        csv_table(tables/f"one-type-{key}.csv",one[key])
    for name,result in (("implementation",checks),("smoke-integration",integration),("smoke-refits",refits)):
        write_json(tables/f"{name}.json",result)
    styles={"G":("#b33c31","o","-"),"EG":("#d58b00","s","--"),"M":("#126da8","^","-"),"M0":("#238151","v",":")}

    def panel(ax,d,n,t,noisy):
        rows=[r for r in one["quantiles"] if (r["d"],r["n"],r["time"],r["noisy"])==(d,n,t,noisy)]
        values=[]
        for method,(color,marker,linestyle) in styles.items():
            selected=[r for r in rows if r["method"]==method]
            x,y=np.array([[r["exact"],r["approximate"]] for r in selected]).T
            ax.plot(x,y,label=method,color=color,marker=marker,linestyle=linestyle,ms=3,lw=1)
            values.extend([*x,*y])
        lo,hi=min(values),max(values)
        pad=.05*(hi-lo)
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color="0.5",lw=.7,zorder=0)
        ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel="Exact quantile",ylabel="Approximate quantile",
               title=f"{'Noisy reading' if noisy else 'Population'}, t = {t:g} days")
        ax.set_aspect("equal",adjustable="box")
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=7)

    fig,axs=plt.subplots(2,2,figsize=(7,6.8),layout="constrained")
    for i,noisy in enumerate((False,True)):
        for j,t in enumerate((.5,5.)):
            panel(axs[i,j],0.,1,t,noisy)
    axs[0,0].legend(fontsize=8)
    fig.savefig(figures/"yule-main.png",dpi=180)
    plt.close(fig)
    for d in (0.,.1):
        for n in (1,5):
            fig,axs=plt.subplots(2,3,figsize=(9,6.6),layout="constrained")
            for i,noisy in enumerate((False,True)):
                for j,t in enumerate((.5,2.,5.)):
                    panel(axs[i,j],d,n,t,noisy)
            axs[0,0].legend(fontsize=8)
            fig.savefig(figures/f"qq-d{d:g}-n{n}.png",dpi=180)
            plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,2.8),layout="constrained")
    for w in (.1,1.,3.):
        rows=[r for r in one["conditional_moments"] if r["w"]==w]
        axs[0].plot([r["time"] for r in rows],[r["scaled_mean_error"] for r in rows],"o-",label=f"w = {w:g}")
    rows=[r for r in one["conditional_moments"] if r["w"]==1.]
    axs[1].plot([r["time"] for r in rows],[r["relative_variance_error"] for r in rows],"o-")
    axs[0].set(xlabel="Time (days)",ylabel="Scaled conditional mean error")
    axs[1].set(xlabel="Time (days)",ylabel="Relative conditional variance error")
    axs[0].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.savefig(figures/"yule-moments.png",dpi=180)
    plt.close(fig)

    truth=[r for r in design["endpoints"] if r["parameter"]=="truth"]
    latex_table(tables/"design-truth.tex",[r"$N_0$","Type",r"$t$",r"$\mathbb E[Z_1+Z_2]$",r"$\pi_t$",r"$\varepsilon_t$"],
        [[r["founders"],r["founder_type"]+1,int(r["time"]),f'{sum(r["mean"]):.2f}',f'{r["pi"]:.8f}',f'{r["epsilon"]:.3g}'] for r in truth])
    latex_table(tables/"conditional-moments.tex",[r"$t$",r"$w$","Mean error","Variance error","Scaled mean error","Relative variance error"],
        [[f'{r[k]:.5g}' for k in ("time","w","mean_error","variance_error","scaled_mean_error","relative_variance_error")] for r in one["conditional_moments"]])
    # All readout values are retained; print one compact eight-row table per case.
    for d in (0.,.1):
        for n in (1,5):
            for t in (.5,2.,5.):
                rows=[r for r in one["densities"] if (r["d"],r["n"],r["time"])==(d,n,t)]
                values=[]
                for x in sorted(set(r["x"] for r in rows)):
                    rr={r["method"]:r for r in rows if r["x"]==x}
                    values.append([f'{x:.5g}',f'{rr["G"]["exact_density"]:.5g}',*[f'{rr[m]["log_difference"]:+.5g}' for m in styles]])
                latex_table(tables/f"density-d{d:g}-n{n}-t{t:g}.tex",[r"$x$",r"$p_{\rm exact}(x)$","G","EG","M","M0"],values)
    approximation=[]
    for d in (0.,.1):
        for n in (1,5):
            for t in (.5,2.,5.):
                for noisy in (False,True):
                    rows=[r for r in one["quantiles"] if (r["d"],r["n"],r["time"],r["noisy"])==(d,n,t,noisy)]
                    approximation.append(dict(d=d,n=n,time=t,noisy=noisy,**{m:max(abs(r["approximate"]-r["exact"]) for r in rows if r["method"]==m) for m in styles}))
    csv_table(tables/"quantile-errors.csv",approximation)
    latex_table(tables/"quantile-errors.tex",[r"$d$",r"$n$",r"$t$","Reading","G","EG","M","M0"],
        [[f'{r["d"]:g}',r["n"],f'{r["time"]:g}',"noisy" if r["noisy"] else "count",*[f'{r[m]:.4g}' for m in styles]] for r in approximation])
    fits=[]
    for n in store.config["founders"]:
        for replicate in range(store.config["smoke_replicates"]):
            for method in store.config["methods"]:
                result=store.result(f"fits/smoke/n{n}/rep{replicate:03d}-{method}.json")
                fits.append(dict(founders=n,replicate=replicate,**result))
    csv_table(tables/"smoke-fits.csv",fits)
    latex_table(tables/"smoke-fits.tex",[r"$N_0$","Rep.","Method",r"$\widehat b_1$",r"$\widehat b_2$","Seconds","Failed starts","Resolved"],
        [[r["founders"],r["replicate"],r["method"],*[f'{v:.6f}' for v in r["estimate"]],f'{r["seconds"]:.2f}',r["failed_starts"],str(r["success"])] for r in fits])
    latex_table(tables/"smoke-refits.tex",[r"$N_0$","Method",r"$\max_i|\Delta\widehat b_i|$",r"$|\Delta\max\mathscr L|$","Fixed-rate change","Passed"],
        [[r["founders"],r["method"],f'{r["estimate_change"]:.3g}',f'{r["maximum_loglik_change"]:.3g}',f'{max(map(abs,r["fixed_parameter_loglik_changes"])):.3g}',str(r["passed"])] for r in refits["rows"]])
    timings=[read_json(p) for p in sorted((store.root/"timings").glob("*.json"))]
    data_meta=[read_json(p) for p in sorted((store.root/"observations/smoke").glob("*/*.json"))]
    fits_seconds=sum(r["seconds"] for r in fits)
    refined_seconds=sum(store.result(f"fits/smoke-refined/n{n}/rep000-{m}.json")["seconds"] for n in store.config["founders"] for m in store.config["methods"])
    simulation_seconds=sum(r["seconds"] for r in data_meta)
    max_count,incomplete=0,0
    for p in (store.root/"observations/smoke").glob("*/*.npz"):
        with np.load(p) as arrays:
            max_count=max(max_count,int(arrays["counts"].sum(axis=1).max()))
            incomplete+=int((~arrays["complete"]).sum())
    max_numerics=max(r.get("maximum_change",0) for r in integration["rows"])
    summary=dict(delivery=1,complete=all(x["passed"] for x in (design,one,checks,smoke,integration,refits)),
        production_fits=len(fits),numerical_refits=len(refits["rows"]),successful_fits=sum(r["success"] for r in fits),
        fit_seconds_sum=fits_seconds,refit_seconds_sum=refined_seconds,simulation_seconds=simulation_seconds,
        maximum_smoke_population=max_count,incomplete_paths=incomplete,
        max_two_type_log_density_change=max_numerics,max_one_type_log_density_change=one["max_log_density_numerical_change"],
        max_quantile_numerical_change=max(max(r["maximum_changes"].values()) for r in one["refinements"]),
        delivery2_projection=dict(fits=160,numerical_refits=40,
            fit_process_seconds=10*fits_seconds,refit_process_seconds=5*refined_seconds,
            four_worker_fit_seconds_ideal=(10*fits_seconds+5*refined_seconds)/4,
            simulation_seconds_conservative=simulation_seconds*(42000+4000+1600)/160,
            note="Smoke extrapolation; simulation allowance uses the mean cost across 160 smoke cultures, although most validation cultures have one founder. Excludes the unmeasured distribution-screen and comparator-audit cost; no coverage profiles are included."),
        pending=["large two-type simulation validation","published-comparator check","distribution screen","20-dataset inference","birth-death profiles"])
    write_json(tables/"summary.json",summary)
    write_json(dest/"provenance.json",dict(report=store.provenance,initial_manifest=read_json(store.root/"manifest.json"),
        stage_provenance={name:read_json(store.root/f"{name}.json")["provenance"] for name in
            ("preflight","checks","one-type","smoke","smoke-integration","smoke-refits")},timings=timings,config=store.config))
    validation=[
        ["Existing automated tests",checks["existing_tests"]["tests_run"],"passed" if checks["existing_tests"]["passed"] else "failed"],
        ["New deterministic/software check groups",len(checks["implementation"]["checks"]),"passed" if checks["implementation"]["passed"] else "failed"],
        ["One-type quantile numerical change",f'{summary["max_quantile_numerical_change"]:.3g}',r"$10^{-5}$"],
        ["One-type log-density numerical change",f'{summary["max_one_type_log_density_change"]:.3g}',r"$10^{-3}$"],
        ["Two-type log-density numerical change",f'{max_numerics:.3g}',r"$10^{-3}$"],
        ["Largest smoke population",max_count,r"cap $5\times10^7$"],
        ["Incomplete smoke paths",incomplete,"0 required"]]
    latex_table(tables/"validation.tex",["Check","Observed","Target / status"],validation,"lrr")
    runtime=[]
    for method in styles:
        rr=[r for r in fits if r["method"]==method]
        runtime.append([method,f'{sum(r["seconds"] for r in rr):.2f}',f'{sum(r["grid_seconds"] for r in rr):.2f}',
            f'{sum(r["timings"]["construction_seconds"] for r in rr):.2f}',f'{sum(r["timings"]["integration_seconds"] for r in rr):.2f}',
            f'{np.mean([r["warm_loglik_seconds"] for r in rr]):.5f}'])
    components=[]
    for method in styles:
        rr=[r for r in fits if r["method"]==method]
        components.append([method,*[f'{sum(r["timings"].get(k,0) for r in rr):.4f}' for k in
            ("moment_seconds","extinction_seconds","spectral_amplitude_seconds","positive_amplitude_seconds")]])
    latex_table(tables/"construction-runtime.tex",["Method","Population moments","Extinction","Spectral/amplitude","Positive amplitude"],components)
    latex_table(tables/"runtime.tex",["Method","Fit total","Grid","Construction","Integration","Warm evaluation"],runtime)
    print(json.dumps(summary,indent=2),flush=True)
