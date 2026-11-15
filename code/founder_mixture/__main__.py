"""Run with PYTHONPATH=code python -m founder_mixture COMMAND."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
INVOCATION_STARTED = perf_counter()
import sys
from . import ROOT
from .storage import Store, read_json
from .checks import run_unit_checks, simulation_checks, moment_solver_refinement
from .diagnostics import METHODS, screen, integration_refinement, fit_refinement
from .inference import fit, profile_sets


def _fit_task(method, data, config):
    observed = {key: data[key] for key in ("y", "time", "z0", "cell")}
    known = {key: value for key, value in config.items() if key != "truth"}
    return fit(method, observed, known)


def _profile_task(method, data, config, fitted):
    observed = {key: data[key] for key in ("y", "time", "z0", "cell")}
    known = {key: value for key, value in config.items() if key != "truth"}
    return profile_sets(method, observed, known, fitted)


def run_fits(store, stage, replicates, per_cell, workers):
    pending = []
    for founders in store.config["founders"]:
        for replicate in range(replicates):
            data = store.dataset(stage, founders, replicate, per_cell)
            for method in store.config.get("methods", METHODS):
                path = f"fits/{stage}/n{founders}/rep{replicate:03d}-{method}.json"
                if store.result(path) is None:
                    pending.append((path, method, data, founders, replicate))
    if any(store.config.get("death", (0., 0.))):
        from .jobs import run_jobs
        run_jobs(store, [(path, (method, data, store.config)) for path, method, data, _, _ in pending], _fit_task, workers)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_task, method, data, store.config): (path, founders, replicate, method)
                   for path, method, data, founders, replicate in pending}
        for future in as_completed(futures):
            path, founders, replicate, method = futures[future]
            result = future.result()
            store.save_result(path, result)
            print(f"{stage} N={founders} replicate={replicate} {method}: success={result['success']} estimate={result['estimate']} seconds={result['seconds']:.2f}", flush=True)


def run_profiles(store, replicates, workers):
    pending = []
    for founders in store.config["founders"]:
        for replicate in range(replicates):
            data = store.dataset("inference", founders, replicate, store.config["cultures_per_cell"])
            for method in store.config.get("methods", METHODS):
                path = f"profiles/n{founders}/rep{replicate:03d}-{method}.json"
                fitted = store.result(f"fits/inference/n{founders}/rep{replicate:03d}-{method}.json")
                if store.result(path) is None:
                    pending.append((path, method, data, fitted, founders, replicate))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_profile_task, method, data, store.config, fitted): (path, founders, replicate, method)
                   for path, method, data, fitted, founders, replicate in pending}
        for future in as_completed(futures):
            path, founders, replicate, method = futures[future]
            result = future.result()
            store.save_result(path, result)
            print(f"profile N={founders} replicate={replicate} {method}: success={result['success']} seconds={result['seconds']:.2f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "checks", "smoke", "screen", "inference", "refine", "report", "cost", "profile-check", "profiles"])
    parser.add_argument("--config", type=str, default=str(ROOT/"configs/founder-mixture-pilot.json"))
    parser.add_argument("--output", type=str, default=str(ROOT/"data/founder-mixture-pilot"))
    parser.add_argument("--report-dir", type=str, default=str(ROOT/"reports/founder-mixture-pilot"),
                        help="destination directory for report figures, tables, and provenance")
    parser.add_argument("--replicates", type=int, choices=[20, 100], default=20)
    parser.add_argument("--delivery", type=int, choices=[1, 2, 3], default=1,
                        help="birth-death delivery to run; Deliveries 2 and 3 require explicit selection")
    parser.add_argument("--retry", action="store_true", help="archive failed or incompatible result attempts before recomputing on unchanged data")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    args.invocation_started = INVOCATION_STARTED
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = read_json(args.config)
    if config.get("death") and any(config["death"]):
        if args.delivery == 3:
            from .birth_death_extended import run_delivery
            status = run_delivery(args, config)
            if args.command == 'report' and status == 0:
                from .birth_death_extended_report import finalize_provenance
                finalize_provenance(args.output, args.report_dir)
            return status
        if args.delivery == 2:
            from .birth_death_scientific import run_delivery
            return run_delivery(args, config)
        from .birth_death_delivery import run_delivery
        return run_delivery(args, config)
    if args.command in ('cost', 'profile-check', 'profiles'):
        parser.error('These commands require the birth-death configuration and --delivery 3')
    if args.command == "preflight":
        parser.error("preflight currently implements the birth-death design")
    store = Store(args.output, config)
    start = perf_counter()
    success = True
    if args.command == "checks":
        result = store.result("checks.json")
        if result is None:
            unit = run_unit_checks()
            simulation = simulation_checks(store) if unit["passed"] else None
            result = dict(unit=unit, simulation=simulation,
                          passed=unit["passed"] and simulation["passed"], seconds=perf_counter()-start)
            store.save_result("checks.json", result)
        success = result["passed"]
        print(f"Checks passed: {success}", flush=True)
    elif args.command == "smoke":
        run_fits(store, "smoke", config["smoke_replicates"], config["smoke_cultures_per_cell"], args.workers)
        smoke_results = [store.result(f"fits/smoke/n{n}/rep{r:03d}-{m}.json")
                         for n in config["founders"] for r in range(config["smoke_replicates"]) for m in config.get("methods", METHODS)]
        success = all(r["success"] for r in smoke_results)
        store.save_result("smoke.json", dict(passed=success, fits=len(smoke_results), seconds=perf_counter()-start))
    elif args.command == "screen":
        if store.result("screen.json") is None:
            store.save_result("screen.json", screen(store))
    elif args.command == "inference":
        run_fits(store, "inference", args.replicates, config["cultures_per_cell"], args.workers)
        if args.replicates == 100:
            run_profiles(store, args.replicates, args.workers)
        store.save_result(f"inference-{args.replicates}.json", dict(replicates=args.replicates,
                          profiles=args.replicates == 100, seconds_this_invocation=perf_counter()-start, workers=args.workers))
    elif args.command == "refine":
        moments = store.result("moment-refinement.json")
        if moments is None:
            moments = moment_solver_refinement(config)
            store.save_result("moment-refinement.json", moments)
        result = integration_refinement(store)
        store.save_result("integration-refinement.json", result)
        fitted = fit_refinement(store, args.workers)
        store.save_result("fit-refinement.json", fitted)
        success = moments["passed"] and result["passed"] and fitted["passed"]
        print(f"Numerical refinement passed: {success}", flush=True)
    if args.command == "report":
        from .report import export_report_assets
        export_report_assets(store, args.report_dir)
    else:
        store.save_result(f"timings/{args.command}-{store.provenance['timestamp_utc'].replace(':', '-')}.json",
                          dict(command=args.command, seconds=perf_counter()-start, workers=args.workers, passed=success))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
