"""Diagnose the retained Delivery 3 estimates without simulating or refitting.

Run from the worktree root with:
    PYTHONPATH=code python -m founder_mixture.birth_death_rmse_diagnostic

Bootstrap resamples reproduce the published Delivery 3 RMSE intervals. New
component and correlation intervals are exploratory, pointwise 95% intervals.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("G", "EG", "M", "M0")
COMPARATORS = ("G", "EG", "M0")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def correlation(values):
    centered = values - values.mean(axis=-2, keepdims=True)
    return ((centered[..., 0] * centered[..., 1]).sum(axis=-1) /
            np.sqrt((centered[..., 0] ** 2).sum(axis=-1) *
                    (centered[..., 1] ** 2).sum(axis=-1)))


def bootstrap(master_seed, founders, index):
    # Same versioned stream keys and PCG64 draws as estimate_summary().
    rng = np.random.default_rng(np.random.SeedSequence(
        [master_seed, 51, founders, 0, 0, index, 4]))
    return rng.integers(0, 100, (2000, 100))


def profile_rows(record, founders, replicate, method):
    assert record["success"] and record["method"] == method
    rows = []
    for rate in record["rates"]:
        # Report a finite change across the whole interval, not a Hessian.
        assert len(rate["intervals"]) == 1 and not any(rate["bound_truncated"])
        lo, hi = rate["intervals"][0]
        x, nuisance = np.array(rate["x"]), np.array(rate["nuisance_estimates"])
        left, right = np.argmin(abs(x - lo)), np.argmin(abs(x - hi))
        assert abs(x[left] - lo) < 1e-12 and abs(x[right] - hi) < 1e-12
        rows.append(dict(founders=founders, replicate=replicate, method=method,
                         interest_rate=f"b{rate['rate'] + 1}", lower=lo, upper=hi,
                         nuisance_at_lower=float(nuisance[left]),
                         nuisance_at_upper=float(nuisance[right]),
                         nuisance_change_per_interest_change=float(
                             (nuisance[right] - nuisance[left]) / (hi - lo))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / "reports/founder-mixture-birth-death"
    tables = report / "tables"
    prefix = "delivery3-rmse-diagnostic"
    input_hashes = {}

    def track(path):
        input_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    config = json.loads(track(report / "configuration.json").read_text())
    truth = np.array(config["truth"])
    fit_rows = read_csv(track(tables / "delivery3-fits.csv"))
    published = read_csv(track(tables / "delivery3-paired.csv"))
    published_metrics = read_csv(track(tables / "delivery3-metrics.csv"))
    sets = read_csv(track(tables / "delivery3-profile-sets.csv"))
    assert len(sets) == 1600 and all(row["resolved"] == "True" for row in sets)
    fits = {}
    for row in fit_rows:
        key = (int(row["founders"]), row["method"], int(row["replicate"]))
        assert key not in fits
        assert row["fit_resolved"] == row["success"] == "True"
        assert not any(json.loads(row["boundary"]))
        fits[key] = np.array(json.loads(row["estimate"]))
    expected = {(n, m, r) for n in config["founders"] for m in METHODS for r in range(100)}
    assert set(fits) == expected and len(fit_rows) == 800

    metrics, paired, joint, changes, individual = [], [], [], [], []
    profiles, profile_summary, refined_changes = [], [], []
    errors = {}
    for n in config["founders"]:
        for index, method in enumerate(METHODS):
            error = np.array([fits[n, method, r] for r in range(100)]) - truth
            errors[n, method] = error
            draws = bootstrap(config["master_seed"], n, 100 + index)
            sample = error[draws]
            joint.append(dict(founders=n, method=method,
                              error_correlation=float(correlation(error)),
                              error_correlation_ci95=ci(correlation(sample))))
            for j in range(2):
                e, b = error[:, j], sample[:, :, j]
                mse, bias, variance = np.mean(e ** 2), e.mean(), e.var(ddof=0)
                assert np.isclose(mse, bias ** 2 + variance, rtol=1e-13, atol=1e-18)
                official = next(row for row in published_metrics
                                if int(row["founders"]) == n and row["method"] == method
                                and row["rate"] == f"b{j + 1}")
                assert abs(np.sqrt(mse) - float(official["rmse"])) < 1e-14
                metrics.append(dict(founders=n, method=method, rate=f"b{j + 1}",
                                    datasets=100, bias=float(bias), bias_ci95=ci(b.mean(axis=1)),
                                    centered_sd=float(np.sqrt(variance)), variance=float(variance),
                                    variance_ci95=ci(b.var(axis=1, ddof=0)), bias_squared=float(bias ** 2),
                                    bias_squared_fraction_of_mse=float(bias ** 2 / mse),
                                    mse=float(mse), rmse=float(np.sqrt(mse))))
            for rep in range(100):
                directory = root / "data/founder-mixture-birth-death/delivery3"
                name = f"n{n}/rep{rep:03d}-{method}.json"
                p = json.loads(track(directory / "profiles" / name).read_text())["result"]
                np.testing.assert_allclose(p["fitted_estimate"], fits[n, method, rep], atol=1e-12, rtol=0)
                production = profile_rows(p, n, rep, method)
                profiles.extend(production)
                if rep < 5:
                    tighter = json.loads(track(directory / "profiles-refined" / name).read_text())["result"]
                    refined = profile_rows(tighter, n, rep, method)
                    for old, new in zip(production, refined):
                        field = "nuisance_change_per_interest_change"
                        refined_changes.append(dict(founders=n, replicate=rep, method=method,
                                                    interest_rate=old["interest_rate"],
                                                    production_slope=old[field], refined_slope=new[field],
                                                    absolute_change=abs(old[field] - new[field]),
                                                    sign_agrees=bool(np.sign(old[field]) == np.sign(new[field]))))
            for j in range(2):
                slopes = np.array([row["nuisance_change_per_interest_change"] for row in profiles
                                   if row["founders"] == n and row["method"] == method
                                   and row["interest_rate"] == f"b{j + 1}"])
                profile_summary.append(dict(founders=n, method=method, interest_rate=f"b{j + 1}",
                                            datasets=len(slopes), positive=int(sum(slopes > 0)),
                                            minimum=float(slopes.min()), lower_quartile=float(np.quantile(slopes, .25)),
                                            median=float(np.median(slopes)), upper_quartile=float(np.quantile(slopes, .75)),
                                            maximum=float(slopes.max())))

        for index, comparator in enumerate(COMPARATORS):
            m, other = errors[n, "M"], errors[n, comparator]
            d = other ** 2 - m ** 2
            delta = m - other
            draws = bootstrap(config["master_seed"], n, 200 + index)
            mb, ob = m[draws], other[draws]
            changes.append(dict(founders=n, comparator=comparator,
                                estimate_change_correlation=float(correlation(delta)),
                                estimate_change_correlation_ci95=ci(correlation(delta[draws])),
                                opposite_direction_changes=int(sum(delta[:, 0] * delta[:, 1] < 0)),
                                both_rates_improve=int(sum((d[:, 0] > 0) & (d[:, 1] > 0))),
                                only_b1_improves=int(sum((d[:, 0] > 0) & (d[:, 1] <= 0))),
                                only_b2_improves=int(sum((d[:, 0] <= 0) & (d[:, 1] > 0))),
                                neither_improves=int(sum((d[:, 0] <= 0) & (d[:, 1] <= 0)))))
            for j in range(2):
                x, a, b = d[:, j], m[:, j], other[:, j]
                ab, bb = mb[:, :, j], ob[:, :, j]
                rmse_reduction = 1 - np.sqrt(np.mean(a ** 2) / np.mean(b ** 2))
                reduction_ci = ci(1 - np.sqrt(np.mean(ab ** 2, axis=1) / np.mean(bb ** 2, axis=1)))
                official = next(row for row in published if int(row["founders"]) == n
                                and row["comparison"] == "M-" + comparator and row["rate"] == f"b{j + 1}")
                np.testing.assert_allclose(reduction_ci, json.loads(official["rmse_reduction_ci95"]), atol=1e-13, rtol=0)
                loo = 1 - np.sqrt((sum(a ** 2) - a ** 2) / (sum(b ** 2) - b ** 2))
                bias_gain = b.mean() ** 2 - a.mean() ** 2
                variance_gain = b.var(ddof=0) - a.var(ddof=0)
                assert np.isclose(x.mean(), bias_gain + variance_gain, atol=1e-18, rtol=1e-12)
                magnitude = np.sort(abs(x))
                paired.append(dict(founders=n, comparator=comparator, rate=f"b{j + 1}", datasets=100,
                                   mse_gain=float(x.mean()), mse_gain_mcse=float(x.std(ddof=1) / 10),
                                   mse_gain_ci95=ci((bb ** 2 - ab ** 2).mean(axis=1)),
                                   rmse_reduction=float(rmse_reduction), rmse_reduction_ci95=reduction_ci,
                                   m_wins=int(sum(x > 0)), ties=int(sum(x == 0)),
                                   squared_bias_gain=float(bias_gain),
                                   squared_bias_gain_ci95=ci(bb.mean(axis=1) ** 2 - ab.mean(axis=1) ** 2),
                                   variance_gain=float(variance_gain),
                                   variance_gain_ci95=ci(bb.var(axis=1, ddof=0) - ab.var(axis=1, ddof=0)),
                                   positive_gain_sum=float(x[x > 0].sum()), negative_gain_sum=float(x[x < 0].sum()),
                                   top5_absolute_contribution_fraction=float(magnitude[-5:].sum() / magnitude.sum()),
                                   top10_absolute_contribution_fraction=float(magnitude[-10:].sum() / magnitude.sum()),
                                   largest_absolute_contribution_fraction=float(magnitude[-1] / magnitude.sum()),
                                   leave_one_out_reduction_min=float(loo.min()), leave_one_out_reduction_max=float(loo.max()),
                                   largest_losses_replicates=np.argsort(x)[:5].tolist(),
                                   largest_gains_replicates=np.argsort(-x)[:5].tolist()))
                for rep in range(100):
                    individual.append(dict(founders=n, comparator=comparator, rate=f"b{j + 1}", replicate=rep,
                                           m_estimate=float(a[rep] + truth[j]), comparator_estimate=float(b[rep] + truth[j]),
                                           m_error=float(a[rep]), comparator_error=float(b[rep]),
                                           squared_error_gain=float(x[rep]), leave_one_out_rmse_reduction=float(loo[rep])))

    for name, rows in [("components", metrics), ("paired", paired), ("joint-errors", joint),
                       ("estimate-changes", changes), ("individual", individual),
                       ("profile-summary", profile_summary), ("profile-slopes", profiles),
                       ("profile-refinement", refined_changes)]:
        write_csv(tables / f"{prefix}-{name}.csv", rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("Squared-error difference = comparator squared error minus M squared error\n"
                 "Positive values favor M", fontsize=12)
    for row, n in enumerate(config["founders"]):
        for j in range(2):
            ax = axes[row, j]
            for method, color in zip(COMPARATORS, ("#0072B2", "#D55E00", "#009E73")):
                value = errors[n, method][:, j] ** 2 - errors[n, "M"][:, j] ** 2
                ax.step(np.sort(value), (np.arange(100) + 1) / 100, where="post", label=method, color=color)
            ax.axvline(0, color="0.4", lw=.8)
            ax.set(title=rf"$N_0={n}$, $b_{j + 1}$", ylabel="Fraction of datasets at or below value",
                   xlabel=r"Squared-error difference (day$^{-2}$)", ylim=(0, 1))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(alpha=.15)
    axes[0, 0].legend(title="Comparator", fontsize=9)
    fig.savefig(report / "figures" / f"{prefix}-paired-errors.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, n in enumerate(config["founders"]):
        for col, comparator in enumerate(("G", "M0")):
            ax = axes[row, col]
            delta = errors[n, "M"] - errors[n, comparator]
            ax.scatter(delta[:, 0], delta[:, 1], s=20, alpha=.7, color="#0072B2")
            ax.axhline(0, color="0.5", lw=.7)
            ax.axvline(0, color="0.5", lw=.7)
            limit = abs(delta).max() * 1.08
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                   xlabel=rf"Change in $\widehat b_1$ from {comparator} to $M$ (day$^{{-1}}$)",
                   ylabel=rf"Change in $\widehat b_2$ from {comparator} to $M$ (day$^{{-1}}$)",
                   title=rf"$N_0={n}$, $M$ versus {comparator}; correlation {correlation(delta):.3f}")
            ax.grid(alpha=.15)
    fig.savefig(report / "figures" / f"{prefix}-estimate-changes.png", dpi=180)
    plt.close(fig)

    for name, digest in input_hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    output_paths = sorted(tables.glob(f"{prefix}-*.csv")) + sorted((report / "figures").glob(f"{prefix}-*.png"))
    evidence = dict(
        description="Exploratory diagnostics of the unchanged Delivery 3 fits and profiles; no new inference runs.",
        reproduction_command="PYTHONPATH=code python -m founder_mixture.birth_death_rmse_diagnostic",
        truth=truth.tolist(), bootstrap_resamples=2000, bootstrap_master_seed=config["master_seed"],
        bootstrap_keys="[master,51,founders,0,0,100+method_index,4] for individual summaries; 200+comparator_index for paired summaries. Methods G,EG,M,M0; comparators G,EG,M0. PCG64.",
        definitions=dict(mse_gain="Comparator MSE minus M MSE; positive favors M.",
                         variance="Centered empirical variance with divisor 100, so MSE = bias squared + variance exactly.",
                         contribution_fraction="Share of sum of absolute paired squared-error differences, not share of net gain.",
                         leave_one_out="Sensitivity diagnostic only; all 100 datasets remain in the reported comparisons.",
                         profile_slope="Change in the optimized nuisance birth rate divided by the change in the interest birth rate, between the two saved confidence-set boundaries. A secant, not a local Hessian or sampling correlation.",
                         uncertainty="Pointwise percentile bootstrap intervals; exploratory components and correlations have no multiplicity adjustment."),
        checks=dict(fits=800, resolved_rate_sets=1600, production_profiles=800,
                    tighter_profiles=40, no_datasets_excluded=True,
                    original_rmse_values_and_intervals_reproduced=True, input_hashes_unchanged=True,
                    mse_decomposition_identity_verified=True,
                    maximum_profile_slope_refinement_change=max(r["absolute_change"] for r in refined_changes),
                    all_refined_profile_slope_signs_agree=all(r["sign_agrees"] for r in refined_changes)),
        metrics=metrics, paired=paired, joint_errors=joint, estimate_changes=changes,
        profile_summary=profile_summary,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
        inputs_sha256=input_hashes,
        outputs_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_paths})
    (tables / f"{prefix}.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence["checks"], indent=2))


if __name__ == "__main__":
    main()
