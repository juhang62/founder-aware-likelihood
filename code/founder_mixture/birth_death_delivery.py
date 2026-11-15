"""Explicit Delivery 1 command routing and smoke-only numerical refinement."""

from itertools import product
from pathlib import Path
from time import perf_counter
import numpy as np
from .storage import Store, read_json
from .checks import run_unit_checks
from .birth_death_checks import preflight, implementation_checks, tight_config
from .one_type import comparisons
from .likelihood import Likelihood
from .inference import fit
from .model import from_config


def smoke_integration_refinement(store):
    cfg=store.config
    base,tight=Likelihood(cfg),Likelihood(tight_config(cfg))
    rows=[]
    for n in cfg["founders"]:
        data=store.dataset("smoke",n,0,cfg["smoke_cultures_per_cell"])
        parameters=[("truth",cfg["truth"]),*[(f"corner{i}",theta) for i,theta in enumerate(product(*cfg["bounds"]))]]
        for method in cfg["methods"]:
            result=store.result(f"fits/smoke/n{n}/rep000-{method}.json")
            if result and result["success"]:
                parameters.append((f"fit_{method}",result["estimate"]))
        for group in range(4):
            ids=np.flatnonzero(data["cell"]==group)
            ranked=ids[np.argsort(data["y"][ids].sum(axis=1),kind="stable")]
            selected=ranked[np.linspace(0,len(ids)-1,5).astype(int)]
            ys=np.vstack((data["y"][selected],[[0.,0.],[-2.,1.]]))
            t,z0=float(data["time"][ids[0]]),data["z0"][ids[0]]
            for label,theta in parameters:
                for method in cfg["methods"]:
                    row=dict(founders=n,combination=group,time=t,z0=z0,parameter=label,theta=theta,
                             method=method,indices=selected,readings=ys)
                    try:
                        changes={}
                        for survivor in ([False] if method=="G" else [False,True]):
                            original=base.logpdf(method,theta,ys,t,z0,survivor=survivor)
                            refined=tight.logpdf(method,theta,ys,t,z0,survivor=survivor)
                            reference=tight.logpdf(method,theta,ys,t,z0,survivor=survivor,reference=True)
                            changes["survivor" if survivor else "full"]=dict(production=original,tight=refined,reference=reference,
                                maximum_change=float(np.max(abs(original-refined))),maximum_reference_change=float(np.max(abs(refined-reference))))
                        maximum=max(max(v["maximum_change"],v["maximum_reference_change"]) for v in changes.values())
                        row.update(values=changes,maximum_change=maximum,passed=maximum<cfg["refinement"]["log_density_target"])
                    except (ArithmeticError,ValueError,Warning) as error:
                        row.update(passed=False,failure=f"{type(error).__name__}: {error}")
                    rows.append(row)
            print(f"smoke integration N={n} combination={group}: checked",flush=True)
            store.save_result("smoke-integration-partial.json",dict(rows=rows,complete=False))
    return dict(rows=rows,passed=all(r["passed"] for r in rows),complete=True)


def _tight_fit(method,data,config):
    observed={key:data[key] for key in ("y","time","z0","cell")}
    known={key:value for key,value in config.items() if key!="truth"}
    return fit(method,observed,known,tight=True)


def smoke_refits(store,workers):
    config=store.config
    from .jobs import run_jobs
    jobs=[]
    rows=[]
    for n in config["founders"]:
        data=store.dataset("smoke",n,0,config["smoke_cultures_per_cell"])
        for method in config["methods"]:
            path=f"fits/smoke-refined/n{n}/rep000-{method}.json"
            jobs.append((path,(method,data,config)))
    run_jobs(store,jobs,_tight_fit,workers)
    for n in config["founders"]:
        data=store.dataset("smoke",n,0,config["smoke_cultures_per_cell"])
        for method in config["methods"]:
            original=store.result(f"fits/smoke/n{n}/rep000-{method}.json")
            refined=store.result(f"fits/smoke-refined/n{n}/rep000-{method}.json")
            row=dict(founders=n,method=method,replicate=0,passed=False)
            if original and refined and original["success"] and refined["success"]:
                changes=[]
                for theta in (original["estimate"],refined["estimate"]):
                    changes.append(Likelihood(tight_config(config)).loglik(method,theta,data)-Likelihood(config).loglik(method,theta,data))
                rate_change=float(np.max(abs(np.array(original["estimate"])-refined["estimate"])))
                maximum_change=abs(refined["loglik"]-original["loglik"])
                diagnostic=[]
                c=from_config(refined["estimate"],tight_config(config))
                for group in range(4):
                    first=np.flatnonzero(data["cell"]==group)[0]
                    w=c.survival(float(data["time"][first]),data["z0"][first])
                    diagnostic.append(dict(combination=group,epsilon=w["epsilon"],pi=w["pi"],pi_infinity=w["pi_infinity"]))
                row.update(estimate_change=rate_change,maximum_loglik_change=maximum_change,
                    fixed_parameter_loglik_changes=changes,original=original["estimate"],refined=refined["estimate"],
                    survival_at_refined_fit=diagnostic,
                    passed=rate_change<config["refinement"]["estimate_target"] and max(maximum_change,*map(abs,changes))<config["refinement"]["dataset_loglik_target"])
            else:
                row["failure"]="Missing or unsuccessful production/refined fit"
            rows.append(row)
    return dict(rows=rows,passed=all(r["passed"] for r in rows))


def run_delivery(args,config):
    if args.command in ("screen","inference"):
        raise ValueError("Delivery 1 only: screen, scientific inference, and birth-death profiles are not implemented or authorized in this delivery")
    if args.workers>config["execution"]["max_workers"]:
        raise ValueError("Worker count exceeds the configured execution allowance")
    # Reject an existing no-death destination, even before creating its manifest.
    if Path(args.output).resolve().name != config["experiment"]:
        raise ValueError("Use the explicit birth-death output directory named by the experiment")
    store=Store(args.output,config,retry=args.retry)
    start=args.invocation_started
    store.deadline=start+config["execution"]["stage_wall_hours"]*3600
    passed=False
    finished=False
    stage_path=store.root/"stages"/(args.command+".json")
    if stage_path.exists() and read_json(stage_path)["result"]["status"] in ("running","incomplete"):
        if not args.retry:
            raise RuntimeError(f"Incomplete stage retained at {stage_path}; inspect it and use --retry explicitly")
        stamp=store.provenance["timestamp_utc"].replace(":","-")
        archive=store.root/"attempts"/stamp/"stages"/stage_path.name
        archive.parent.mkdir(parents=True,exist_ok=True)
        stage_path.replace(archive)
    store.save_result(f"stages/{args.command}.json",dict(status="running",workers=args.workers))
    try:
        if args.command=="preflight":
            result=store.result("preflight.json")
            if result is None:
                result=preflight(config)
                store.save_result("preflight.json",result)
            passed=result["passed"]
        else:
            design=store.result("preflight.json")
            if not design or not design["passed"]:
                raise ValueError("Complete the deterministic preflight before other stages")
            if args.command=="checks":
                result=store.result("checks.json")
                if result is None:
                    old=run_unit_checks()
                    implemented=store.result("implementation-checks.json")
                    if implemented is None:
                        implemented=implementation_checks(config)
                        store.save_result("implementation-checks.json",implemented)
                    one=store.result("one-type.json")
                    if one is None:
                        one=comparisons(config)
                        store.save_result("one-type.json",one)
                    result=dict(existing_tests=old,implementation=implemented,one_type_passed=one["passed"],
                                passed=old["passed"] and implemented["passed"] and one["passed"])
                    store.save_result("checks.json",result)
                passed=result["passed"]
            elif args.command=="smoke":
                from .__main__ import run_fits
                checks=store.result("checks.json")
                if not checks or not checks["passed"]:
                    raise ValueError("Complete implementation checks before smoke fitting")
                run_fits(store,"smoke",config["smoke_replicates"],config["smoke_cultures_per_cell"],args.workers)
                results=[store.result(f"fits/smoke/n{n}/rep{r:03d}-{m}.json") for n in config["founders"]
                         for r in range(config["smoke_replicates"]) for m in config["methods"]]
                passed=all(r and r["success"] for r in results)
                store.save_result("smoke.json",dict(passed=passed,fits=len(results),workers=args.workers,seconds=perf_counter()-start))
            elif args.command=="refine":
                for name,action in (("smoke-integration",lambda:smoke_integration_refinement(store)),
                                    ("smoke-refits",lambda:smoke_refits(store,args.workers))):
                    result=store.result(name+".json")
                    if result is None:
                        result=action()
                        store.save_result(name+".json",result)
                passed=all(store.result(name+".json")["passed"] for name in ("smoke-integration","smoke-refits"))
            elif args.command=="report":
                from .birth_death_report import export
                export(store,args.report_dir)
                passed=True
        finished=True
        print(f"Delivery 1 {args.command}: passed={passed}",flush=True)
    finally:
        store.save_result(f"stages/{args.command}.json",dict(status="complete" if finished else "incomplete",passed=passed,workers=args.workers))
        store.save_result(f"timings/{args.command}-{store.provenance['timestamp_utc'].replace(':','-')}.json",
            dict(command=args.command,seconds=perf_counter()-start,workers=args.workers,passed=passed,
                 delivery=1,stage_budget_hours=config["execution"]["stage_wall_hours"]))
    return 0 if passed else 1
