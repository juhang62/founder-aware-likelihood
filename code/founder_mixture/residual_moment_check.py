"""Deterministic amplitude-averaged residual check; see specification section 11.

Run as ``PYTHONPATH=code python -m founder_mixture.residual_moment_check``.
The high-precision calculation rebuilds the spectral and moment quantities
from decimal input, rather than promoting already rounded NumPy results.
It evaluates exact moment identities, not conditional moments at W=w.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import mpmath as mp
import numpy as np

from . import ROOT
from .model import candidate


def residual_matrix(V, cross, h, variance):
    """Cov[Z-W h], valid for NumPy arrays and mpmath matrices."""
    return V - cross @ h.T - h @ cross.T + variance * (h @ h.T)


def high_precision_model(birth, switching, times):
    """Exact linear moment equations evaluated in the active mp precision."""
    b1, b2 = map(lambda x: mp.mpf(str(x)), birth)
    s12, s21 = map(lambda x: mp.mpf(str(x)), switching)
    A = mp.matrix([[b1-s12, s21], [s12, b2-s21]])
    gap = mp.sqrt((A[0, 0]-A[1, 1])**2 + 4*s12*s21)
    L = (A[0, 0]+A[1, 1]+gap)/2
    lam = (A[0, 0]+A[1, 1]-gap)/2
    if not (min(b1, b2, s12, s21) > 0 and L > 0 and lam < L/2):
        raise ValueError("Positive rates and L > 0, lambda2 < L/2 required")
    r = mp.matrix([(L-A[1, 1])/s12, 1])
    ell = mp.matrix([s12, L-A[0, 0]])
    ell /= (ell.T*r)[0]

    def noise(z):
        switch = s12*z[0] + s21*z[1]
        return mp.matrix([[b1*z[0]+switch, -switch],
                          [-switch, b2*z[1]+switch]])

    D = [noise(mp.eye(2)[:, i]) for i in range(2)]
    c = mp.matrix([(ell.T*d*ell)[0] for d in D])
    v = mp.lu_solve(2*L*mp.eye(2)-A.T, c)
    P = r*ell.T
    S = mp.eye(2)-P
    B = P*noise(r)*P.T/L + S*noise(r)*S.T/(L-2*lam)
    K = mp.zeros(5)
    K[:2, :2] = A
    for j, d in enumerate(D):
        K[2, j], K[3, j], K[4, j] = d[0, 0], d[0, 1], d[1, 1]
    K[2:, 2:] = mp.matrix([[2*A[0, 0], 2*A[0, 1], 0],
                            [A[1, 0], A[0, 0]+A[1, 1], A[0, 1]],
                            [0, 2*A[1, 0], 2*A[1, 1]]])
    moments = {t: mp.expm(K*mp.mpf(str(t)))[:, :2] for t in times}
    return dict(L=L, lambda2=lam, r=r, ell=ell, v=v, B=B, moments=moments)


def high_precision_case(model, t, founders, founder_type):
    out = model["moments"][t][:, founder_type] * founders
    m = out[:2, :]
    V = mp.matrix([[out[2], out[3]], [out[3], out[4]]])
    a = model["ell"][founder_type] * founders
    v0 = model["v"][founder_type] * founders
    h = m/a
    cross = mp.exp(-model["L"]*t)*V*model["ell"]
    exact = residual_matrix(V, cross, h, v0)
    predicted = a*mp.exp(model["L"]*t)*model["B"]
    discrepancy = mp.norm(exact-predicted)/mp.norm(exact)
    cancellation = (mp.norm(V)+2*mp.norm(cross*h.T)+mp.norm(v0*h*h.T))/mp.norm(exact)
    return dict(exact=exact, predicted=predicted, discrepancy=discrepancy,
                mean=m, population_covariance=V, amplitude_mean=a,
                amplitude_variance=v0, cross_covariance=cross,
                cancellation_ratio=cancellation)


def double_case(model, t, founders, founder_type):
    z0 = np.eye(2)[founder_type]*founders
    m, V = model.at(t, z0)
    a, v0, _, _ = model.amplitude(z0)
    h = (m/a)[:, None]
    cross = (np.exp(-model.L*t)*V @ model.ell)[:, None]
    exact = residual_matrix(V, cross, h, v0)
    predicted = a*np.exp(model.L*t)*model.B
    return exact, predicted, np.linalg.norm(exact-predicted)/np.linalg.norm(exact)


def matrix_as_float(matrix):
    return np.array(matrix.tolist(), dtype=float)


def archive_value(value, digits):
    if isinstance(value, mp.matrix):
        return [[mp.nstr(x, digits) for x in row] for row in value.tolist()]
    return mp.nstr(value, digits)


def yule_checks(config):
    """Apply the same residual expansion to exact scalar Yule moments."""
    rows = []
    for birth, t in product((0.3, 0.8, 1.1), (0, *config["times"])):
        E = mp.exp(mp.mpf(str(birth))*t)
        V = mp.matrix([[E*(E-1)]])
        cross, h = mp.matrix([[E-1]]), mp.matrix([[E]])
        exact = residual_matrix(V, cross, h, mp.mpf(1))[0, 0]
        error = abs(exact-E)/E
        rows.append(dict(birth=birth, time=t, relative_error=float(error)))
    return rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures_and_table(rows, report_dir, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    labels = list(dict.fromkeys(row["parameter"] for row in rows))
    for founder_type, ax in enumerate(axes, start=1):
        for label in labels:
            selected = [r for r in rows if r["parameter"] == label
                        and r["founder_type"] == founder_type and r["founders"] == 1]
            theta = selected[0]
            name = f'({theta["b1"]:g}, {theta["b2"]:g})'
            if label == "truth":
                name += " (pilot truth)"
            ax.semilogy([r["time"] for r in selected],
                        [r["relative_discrepancy"] for r in selected],
                        marker="o", markersize=3, label=name)
        ax.set(title=f"Pure founder type {founder_type}", xlabel="Time (days)",
               xticks=config["times"])
        ax.grid(True, which="both", alpha=.2)
    axes[0].set_ylabel("Relative Frobenius discrepancy")
    axes[1].legend(title="Birth rates per day", fontsize=8)
    fig.tight_layout()
    path = report_dir/"figures"/"residual-moment-sweep.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    lines = []
    for label, founder_type in product(labels, (1, 2)):
        selected = [r for r in rows if r["parameter"] == label and r["founders"] == 1
                    and r["founder_type"] == founder_type and r["time"] in (5, 7, 11, 23)]
        if len(selected) != 4:
            raise ValueError("Report table requires times 5, 7, 11, 23")
        row = selected[0]
        values = [f'{100*r["relative_discrepancy"]:.3g}' for r in selected]
        # Scientific notation is rendered as LaTeX rather than unexplained e notation.
        def latex_number(value):
            if "e" in value:
                mantissa, exponent = value.split("e")
                return rf"${mantissa}\times10^{{{int(exponent)}}}$"
            return value
        lines.append(f'({row["b1"]:g}, {row["b2"]:g}) & {founder_type} & '
                     + " & ".join(map(latex_number, values)) + r" \\")
    header = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
              r"$(b_1,b_2)$ & Founder type & Day 5 & Day 7 & Day 11 & Day 23 \\",
              r"\midrule"]
    (report_dir/"tables"/"selected-times.tex").write_text(
        "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}"])+"\n")


def run(config_path, output, report_dir):
    tick = perf_counter()
    config = json.loads(config_path.read_text())
    precision, refined = config["decimal_precision"], config["refined_decimal_precision"]
    if not 30 <= precision < refined:
        raise ValueError("Require increasing decimal precisions, starting at >=30")
    if config["founders"] != [1, 50] or min(config["times"]) <= 0:
        raise ValueError("This check retains the specified founder counts and positive times")
    parameters = [("truth", config["truth"])] + [
        (f"corner_{i+1}", theta) for i, theta in enumerate(product(*config["bounds"]))]
    rows, validation, archive, spectra, total_yule = [], [], [], [], []
    scaling_error = mp.mpf(0)
    initial_error = mp.mpf(0)
    for name, theta in parameters:
        print(f"Checking {name}: birth={tuple(theta)}", flush=True)
        with mp.workdps(precision):
            ordinary = high_precision_model(theta, config["switching"], config["times"])
            cases = {(t, n, i): high_precision_case(ordinary, t, n, i)
                     for t, n, i in product(config["times"], config["founders"], range(2))}
        double = candidate(tuple(theta), tuple(config["switching"]), tuple(config["times"]))
        with mp.workdps(refined):
            reference = high_precision_model(theta, config["switching"], [0, *config["times"]])
            spectra.append(dict(parameter=name, b1=theta[0], b2=theta[1], L=float(reference["L"]),
                                lambda2=float(reference["lambda2"]),
                                spectral_margin=float(reference["L"]/2-reference["lambda2"])))
            for i in range(2):
                zero = high_precision_case(reference, 0, 1, i)
                h0 = mp.eye(2)[:, i]/reference["ell"][i]
                target = reference["v"][i]*h0*h0.T
                initial_error = max(initial_error, mp.norm(zero["exact"]-target)/mp.norm(target))
            for t, n, i in product(config["times"], config["founders"], range(2)):
                high = high_precision_case(reference, t, n, i)
                low = cases[t, n, i]
                d_exact, d_predicted, d_error = double_case(double, t, n, i)
                key = dict(parameter=name, b1=theta[0], b2=theta[1], founders=n, founder_type=i+1, time=t)
                row = dict(**key, relative_discrepancy=float(high["discrepancy"]),
                           cancellation_ratio=float(high["cancellation_ratio"]))
                for label in ("exact", "predicted"):
                    for j, k in ((0, 0), (0, 1), (1, 1)):
                        row[f"{label}_{j+1}{k+1}"] = float(high[label][j, k])
                    matrix = matrix_as_float(high[label])
                    if np.linalg.eigvalsh(matrix)[0] <= 0:
                        raise ArithmeticError(f"Nonpositive residual matrix: {key}, {label}")
                rows.append(row)
                validation.append(dict(**key,
                    precision_matrix_relative_change=float(mp.norm(low["exact"]-high["exact"])/mp.norm(high["exact"])),
                    precision_prediction_relative_change=float(mp.norm(low["predicted"]-high["predicted"])/mp.norm(high["predicted"])),
                    precision_discrepancy_absolute_change=float(abs(low["discrepancy"]-high["discrepancy"])),
                    double_matrix_relative_error=float(mp.norm(mp.matrix(d_exact.tolist())-high["exact"])/mp.norm(high["exact"])),
                    double_prediction_relative_error=float(mp.norm(mp.matrix(d_predicted.tolist())-high["predicted"])/mp.norm(high["predicted"])),
                    double_discrepancy=float(d_error),
                    double_discrepancy_absolute_error=float(abs(mp.mpf(d_error)-high["discrepancy"]))))
                archive.append(dict(**key, **{k: archive_value(v, refined) for k, v in high.items()}))
                if theta[0] == theta[1]:
                    # With equal birth rates the total population is exactly Yule,
                    # even though types keep switching. This independently checks
                    # the two-type calculation and amplitude normalization.
                    ones = mp.matrix([1, 1])
                    target = n*mp.exp(mp.mpf(str(theta[0]))*t)
                    total_yule.append(dict(**key,
                        exact_relative_error=float(abs((ones.T*high["exact"]*ones)[0]-target)/target),
                        predicted_relative_error=float(abs((ones.T*high["predicted"]*ones)[0]-target)/target)))
                one = high_precision_case(reference, t, 1, i)
                scaling_error = max(scaling_error, mp.norm(high["exact"]-n*one["exact"])/mp.norm(high["exact"]))
    with mp.workdps(refined):
        yule = yule_checks(config)
    summary = dict(cases=len(rows), decimal_precision=precision, refined_decimal_precision=refined,
                   max_precision_matrix_relative_change=max(r["precision_matrix_relative_change"] for r in validation),
                   max_precision_prediction_relative_change=max(r["precision_prediction_relative_change"] for r in validation),
                   max_precision_discrepancy_absolute_change=max(r["precision_discrepancy_absolute_change"] for r in validation),
                   max_double_matrix_relative_error=max(r["double_matrix_relative_error"] for r in validation),
                   max_double_discrepancy_absolute_error=max(r["double_discrepancy_absolute_error"] for r in validation),
                   max_cancellation_ratio=max(r["cancellation_ratio"] for r in rows),
                   max_yule_relative_error=max(r["relative_error"] for r in yule),
                   max_equal_birth_total_yule_relative_error=max(max(r["exact_relative_error"], r["predicted_relative_error"]) for r in total_yule),
                   max_founder_scaling_relative_error=float(scaling_error),
                   max_initial_time_relative_error=float(initial_error),
                   all_matrices_positive_definite=True, spectra=spectra)
    accuracy = float(config["precision_relative_tolerance"])
    identity = float(config["identity_relative_tolerance"])
    summary["passed"] = (max(summary[k] for k in (
        "max_precision_matrix_relative_change", "max_precision_prediction_relative_change",
        "max_precision_discrepancy_absolute_change")) < accuracy
        and max(summary[k] for k in ("max_yule_relative_error", "max_equal_birth_total_yule_relative_error", "max_founder_scaling_relative_error",
                                    "max_initial_time_relative_error")) < identity)
    if not summary["passed"]:
        raise ArithmeticError(f"Numerical checks failed: {summary}")
    write_json(output/"high-precision-moments.json", archive)
    write_csv(report_dir/"tables"/"residual-moments.csv", rows)
    write_csv(report_dir/"tables"/"numerical-refinement.csv", validation)
    write_json(report_dir/"tables"/"validation.json", dict(summary=summary, yule=yule, equal_birth_total_yule=total_yule))
    figures_and_table(rows, report_dir, config)

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()

    sources = [Path(__file__).resolve(), Path(__file__).with_name("model.py").resolve(),
               Path(__file__).with_name("__init__.py").resolve(), config_path.resolve()]
    manifest = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), config=config,
                    method="Exact moment identities, mpmath matrix exponential; no random sampling",
                    random_seed=None, seconds=perf_counter()-tick, python=platform.python_version(),
                    versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mpmath", "matplotlib")},
                    git_revision=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
                    git_status=git("status", "--short"),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    artifact_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [output/"high-precision-moments.json", *sorted((report_dir/"tables").glob("*")),
                                               report_dir/"figures"/"residual-moment-sweep.png"]})
    write_json(output/"manifest.json", manifest)
    write_json(report_dir/"provenance"/"manifest.json", manifest)
    print(json.dumps(dict(summary=summary, seconds=manifest["seconds"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/residual-moment-check.json")
    parser.add_argument("--output", type=Path, default=ROOT/"data/residual-moment-check")
    parser.add_argument("--report-dir", type=Path, default=ROOT/"reports/residual-moment-check")
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
