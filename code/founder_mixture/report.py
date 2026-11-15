"""Export figures, tables, and provenance for the maintained LaTeX report."""

import csv
import json
import shutil
from pathlib import Path
import numpy as np
from .diagnostics import METHODS, paired_summary
from .simulation import stream
from .storage import read_json, write_json


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(val) if isinstance(val, (list, dict)) else val for key, val in row.items()})


def inference_summary(store):
    config, rows, paired, records = store.config, [], [], {}
    truth = np.array(config["truth"])
    for founders in config["founders"]:
        for method in METHODS:
            paths = sorted((store.root/f"fits/inference/n{founders}").glob(f"rep*-{method}.json"))
            fits = {int(path.name[3:6]): store.result(str(path.relative_to(store.root))) for path in paths}
            records[founders, method] = fits
            valid = {rep: f for rep, f in fits.items() if f["success"]}
            for rate in range(2):
                row = dict(founders=founders, method=method, rate=f"b{rate+1}", attempted=len(fits),
                           successful=len(valid), failures=len(fits)-len(valid))
                if valid:
                    estimates = np.array([r["estimate"][rate] for r in valid.values()])
                    errors = estimates-truth[rate]
                    row.update(bias=float(errors.mean()), bias_mcse=float(errors.std(ddof=1)/np.sqrt(len(errors))) if len(errors)>1 else None,
                               rmse=float(np.sqrt(np.mean(errors**2))), boundary=sum(f["boundary"][rate] for f in valid.values()),
                               mean_fit_seconds=float(np.mean([f["seconds"] for f in fits.values()])),
                               mean_warm_loglik_seconds=float(np.mean([f["warm_loglik_seconds"] for f in valid.values()])),
                               mean_construction_seconds=float(np.mean([f["timings"]["construction_seconds"] for f in fits.values()])),
                               mean_integration_seconds=float(np.mean([f["timings"]["integration_seconds"] for f in fits.values()])),
                               mean_grid_seconds=float(np.mean([f.get("grid_seconds", 0) for f in fits.values()])),
                               failed_starts=sum(f.get("failed_starts", 0) for f in fits.values()))
                profiles = {rep: store.result(f"profiles/n{founders}/rep{rep:03d}-{method}.json") for rep in fits}
                available = {r: p for r, p in profiles.items() if p is not None}
                if available:
                    resolved = [p for p in available.values() if p["success"]]
                    sets = [p["rates"][rate] for p in resolved]
                    covered = [any(a <= truth[rate] <= b for a, b in p["intervals"]) for p in sets]
                    failed = len(fits)-len(resolved)
                    row.update(profile_attempted=len(available), profile_successful=len(resolved),
                               profile_unresolved=failed,
                               coverage=float(np.mean(covered)) if covered else None,
                               coverage_mcse=float(np.sqrt(np.mean(covered)*(1-np.mean(covered))/len(covered))) if covered else None,
                               coverage_lower=sum(covered)/len(fits), coverage_upper=(sum(covered)+failed)/len(fits),
                               average_set_length=float(np.mean([p["length"] for p in sets])) if sets else None,
                               disconnected=sum(p["disconnected"] for p in sets), flat=sum(p["flat"] for p in sets),
                               bound_truncated=sum(any(p["bound_truncated"]) for p in sets),
                               mean_profile_seconds=float(np.mean([p["seconds"] for p in available.values()])))
                rows.append(row)
        for comparator_index, comparator in enumerate(("G", "M0")):
            left, right = records[founders, "M"], records[founders, comparator]
            ids = sorted(r for r in left.keys() & right.keys() if left[r]["success"] and right[r]["success"])
            for rate in range(2):
                if len(ids)<2:
                    continue
                e1 = np.array([left[r]["estimate"][rate]-truth[rate] for r in ids])
                e2 = np.array([right[r]["estimate"][rate]-truth[rate] for r in ids])
                difference = paired_summary(e1**2-e2**2)
                rng = stream(config["master_seed"], "diagnostics", founders, rate, 0, comparator_index, "bootstrap")
                draws = rng.integers(0, len(ids), (5000, len(ids)))
                rmse1, rmse2 = np.sqrt(np.mean(e1[draws]**2, axis=1)), np.sqrt(np.mean(e2[draws]**2, axis=1))
                reduction = 1-np.sqrt(np.mean(e1**2)/np.mean(e2**2))
                entry = dict(founders=founders, comparison=f"M-{comparator}", rate=f"b{rate+1}",
                             paired_successful=len(ids), paired_mse_difference=difference["mean"],
                             paired_mse_difference_mcse=difference["se"], rmse_reduction=float(reduction),
                             rmse_reduction_ci95=np.quantile(1-rmse1/rmse2, [.025, .975]).tolist(),
                             rmse_difference=float(np.sqrt(np.mean(e1**2))-np.sqrt(np.mean(e2**2))),
                             rmse_difference_ci95=np.quantile(rmse1-rmse2, [.025, .975]).tolist())
                coverage_differences, lengths, costs = [], [], []
                for rep in ids:
                    costs.append(left[rep]["seconds"]-right[rep]["seconds"])
                    p = store.result(f"profiles/n{founders}/rep{rep:03d}-M.json")
                    q = store.result(f"profiles/n{founders}/rep{rep:03d}-{comparator}.json")
                    if p and q and p["success"] and q["success"]:
                        p, q = p["rates"][rate], q["rates"][rate]
                        hit1 = any(a <= truth[rate] <= b for a, b in p["intervals"])
                        hit2 = any(a <= truth[rate] <= b for a, b in q["intervals"])
                        coverage_differences.append(int(hit1)-int(hit2))
                        lengths.append(p["length"]-q["length"])
                entry["paired_fit_seconds"] = paired_summary(costs)
                if coverage_differences:
                    entry["paired_coverage_difference"] = paired_summary(coverage_differences)
                    entry["paired_set_length_difference"] = paired_summary(lengths)
                paired.append(entry)
    return rows, paired, records


def figures(store, directory, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    path = store.root/"screen-predictions.npz"
    colors = dict(G="#3977b8", M="#cd573c", M0="#458254")
    if path.exists():
        with np.load(path) as archive:
            fig, axes = plt.subplots(4, 4, figsize=(14, 11), constrained_layout=True)
            for ni, founders in enumerate(store.config["founders"]):
                for cell in range(4):
                    prefix = f"n{founders}_cell{cell}"
                    observed = archive[prefix+"_observed"]
                    for pi, vector in enumerate((np.array([1, 1]), np.array([1, -1]))):
                        ax = axes[cell, ni*2+pi]
                        pred = {m: archive[prefix+"_"+m] @ vector for m in METHODS}
                        all_values = np.concatenate([observed @ vector]+list(pred.values()))
                        lo, hi = np.quantile(all_values, [.002, .998])
                        bins = np.linspace(lo, hi, 50)
                        width = np.diff(bins)
                        # Counts divided by the original sample size preserve
                        # omitted plotting tails instead of renormalizing them.
                        hist, _ = np.histogram(observed @ vector, bins=bins)
                        ax.stairs(hist/(len(observed)*width), bins, color="black", fill=True, alpha=.17, label="CTMC + noise")
                        for m in METHODS:
                            hist, _ = np.histogram(pred[m], bins=bins)
                            ax.stairs(hist/(len(pred[m])*width), bins, color=colors[m], label=m, linewidth=1.3)
                        ax.set_title(f"N={founders}, type {cell//2+1}, day {store.config['times'][cell%2]:g}")
                        ax.set_xlabel("Total readout $Y_1+Y_2$" if pi == 0 else "Type contrast $Y_1-Y_2$")
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
                        if ni == 0 and pi == 0:
                            ax.set_ylabel("Density")
                        if cell == 0 and ni == 0 and pi == 0:
                            ax.legend(fontsize=8)
            fig.savefig(directory/"distribution.png", dpi=180)
            plt.close(fig)
            names.append("distribution.png")
    if any(records.values()):
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        for ni, founders in enumerate(store.config["founders"]):
            for rate in range(2):
                ax = axes[ni, rate]
                values = [[f["estimate"][rate] for f in records[founders, method].values() if f["success"]] for method in METHODS]
                for j, (method, vals) in enumerate(zip(METHODS, values)):
                    if vals:
                        ax.boxplot([vals], positions=[j], widths=.45, showfliers=False)
                        offsets = (np.arange(len(vals))%11-5)*.022
                        ax.scatter(j+offsets, vals, color=colors[method], s=12, alpha=.5)
                ax.axhline(store.config["truth"][rate], color="black", linestyle="--", linewidth=1)
                ax.set_xticks(range(3), METHODS)
                ax.set_title(f"N={founders}, birth rate $b_{rate+1}$")
                ax.set_ylabel("Estimate (per day)")
        fig.savefig(directory/"inference.png", dpi=180)
        plt.close(fig)
        names.append("inference.png")
    # Representative profiles retain all sampled points and disconnected sets.
    profile_paths = list((store.root/"profiles").glob("n*/rep000-*.json"))
    if profile_paths:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ni, founders in enumerate(store.config["founders"]):
            visible_x = [[], []]
            for method in METHODS:
                profile = store.result(f"profiles/n{founders}/rep000-{method}.json")
                fitted = records[founders, method].get(0)
                refined = store.result(f"profiles/refined/n{founders}/rep000-{method}.json")
                if refined and refined["success"]:
                    profile = refined
                    fitted = store.result(f"fits/refined/n{founders}/rep000-{method}.json")
                if not profile or not profile["success"]:
                    continue
                for rate in range(2):
                    curve = profile["rates"][rate]
                    deviance = 2*(fitted["loglik"]-np.array(curve["loglik"]))
                    axes[ni, rate].plot(curve["x"], deviance, ".-", markersize=3, label=method, color=colors[method])
                    visible_x[rate].extend(np.array(curve["x"])[deviance <= 12].tolist())
                    axes[ni, rate].set_ylim(0, 10)
                    axes[ni, rate].set_title(f"N={founders}, profile of $b_{rate+1}$")
            for rate in range(2):
                if visible_x[rate]:
                    lo, hi = min(visible_x[rate]), max(visible_x[rate])
                    pad = max((hi-lo)*.15, .001)
                    axes[ni, rate].set_xlim(max(store.config["bounds"][rate][0], lo-pad),
                                           min(store.config["bounds"][rate][1], hi+pad))
                axes[ni, rate].axhline(store.config["profiles"]["cutoff"], color="gray", linestyle="--")
                axes[ni, rate].set_xlabel("Birth rate (per day)")
                axes[ni, rate].set_ylabel("Twice log-likelihood decrease")
                axes[ni, rate].legend()
        fig.savefig(directory/"profiles.png", dpi=180)
        plt.close(fig)
        names.append("profiles.png")
    return names


def export_report_assets(store, directory):
    """Refresh supporting artifacts without writing or modifying the report narrative."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table_dir, figure_dir = directory/"tables", directory/"figures"
    provenance_dir = directory/"provenance"
    for name in ("manifest.json", "extension-decision.json", "decision.json"):
        source = store.root/name
        if source.exists():
            provenance_dir.mkdir(parents=True, exist_ok=True)
            if source.resolve() != (provenance_dir/name).resolve():
                shutil.copyfile(source, provenance_dir/name)
    config = store.config
    checks, smoke, screen = [store.result(name+".json") for name in ("checks", "smoke", "screen")]
    integration, refinement = [store.result(name+".json") for name in ("integration-refinement", "fit-refinement")]
    moments = store.result("moment-refinement.json")
    rows, paired, records = inference_summary(store)
    write_csv(table_dir/"inference.csv", rows)
    write_csv(table_dir/"paired-inference.csv", paired)
    figures(store, figure_dir, records)
    if checks and checks.get("simulation"):
        write_json(table_dir/"simulation-checks.json", checks["simulation"])
    if screen:
        score_rows = []
        for row in screen["cells"]:
            a, b = row["M_minus_G"], row["M_minus_M0"]
            errors = row["covariance_relative_frobenius"]
            score_rows.append(dict(founders=row["founders"], founder_type=row["cell"]//2+1, time=row["time"],
                                   M_minus_G=a["mean"], M_minus_G_se=a["se"], M_minus_M0=b["mean"], M_minus_M0_se=b["se"],
                                   M_covariance_relative_error=errors["M"], M0_covariance_relative_error=errors["M0"], mean=row["mean"]))
        write_csv(table_dir/"distribution.csv", score_rows)
        write_json(table_dir/"distribution-details.json", screen)
    for name, result in (("moment-refinement", moments), ("integration-refinement", integration), ("fit-refinement", refinement)):
        if result:
            write_json(table_dir/(name+".json"), result)
    complete = all(len(records[n, m]) >= 20 for n in config["founders"] for m in METHODS)
    numeric_pass = checks and checks["passed"] and smoke and smoke["passed"] and integration and integration["passed"] and refinement and refinement["passed"] and moments and moments["passed"]
    if all(len(records[n, m]) >= 100 for n in config["founders"] for m in METHODS):
        profiles_attempted = all(r.get("profile_attempted", 0) == r["attempted"] for r in rows)
        profiles_refined = refinement and all("profile_refinement_passed" in r for r in refinement["fits"])
        numeric_pass = numeric_pass and profiles_attempted and profiles_refined
    few = [p for p in paired if p["founders"] == 1 and p["comparison"] == "M-G"]
    many = [p for p in paired if p["founders"] == 50 and p["comparison"] == "M-G"]
    resolved_gain = any(p["rmse_reduction_ci95"][0] > 0 for p in few)
    resolved_harm = any(p["rmse_reduction_ci95"][1] < 0 and p["rmse_reduction"] < -.2 for p in many)
    if not complete or not numeric_pass:
        conclusion = "**Inconclusive.** The required stages or numerical checks are incomplete or unresolved. Interpret the affected comparisons only after the listed failures are diagnosed; the report does not treat a failed check as evidence against amplitude mixing."
    elif resolved_harm:
        conclusion = "**Revise a component.** The many-founder control shows a resolved, material deterioration in at least one birth rate. Inspect amplitude shape, conditional mean, and covariance mismatch before expanding this prototype."
    elif resolved_gain:
        conclusion = "**Proceed.** The prototype shows a resolved few-founder RMSE reduction for at least one rate, without a resolved material deterioration in the many-founder control. Judge practical value using the effect sizes and runtime in the tables; roughly 20% is a discussion flag, not a decision threshold. The tables report both rates and the incremental contribution of residual covariance. Prioritize validation of the amplitude shape and conditional mean; this screen alone does not establish interval calibration."
    elif all(p["rmse_reduction_ci95"][1] < .2 for p in few):
        conclusion = "**No useful advantage detected in this design.** Neither birth rate has a resolved material few-founder RMSE advantage in this screen. Smaller effects and their uncertainty remain visible in the tables; a better predictive log score alone does not establish better inference."
    else:
        conclusion = "**Inconclusive.** Paired uncertainty still includes a useful few-founder RMSE benefit. Review the completed sample size and interval assessment before proposing additional computation. If only 20 datasets have run, extend to the prespecified 100 while retaining every existing replicate."
    decision = store.result("decision.json")
    if decision and numeric_pass and all(len(records[n, m]) == decision["replicates"] for n in config["founders"] for m in METHODS):
        conclusion = decision["conclusion"]
    write_json(table_dir/"report-summary.json", dict(inference=rows, paired=paired, numerical_checks_passed=bool(numeric_pass), conclusion=conclusion))
    print(f"Exported report assets to {directory}; revise the LaTeX narrative separately when results change.", flush=True)
