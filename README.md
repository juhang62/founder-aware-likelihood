# Founder-aware likelihoods for branching populations

Code and summary results for the manuscript *Large populations from few founders: founder-aware likelihoods for inferring cell proliferation rates* by Lifeng Han and Marley M. Assefa (manuscript in preparation).

The study uses only synthetic data. Every simulated dataset is generated from the configurations and random seeds in this repository, so the repository holds the code, the configurations, and the summary tables from which the manuscript's figures and tables are built. Raw simulated observations, fits, and profiles are not included; they can be regenerated with the commands below.

## Contents

| Path | Contents |
|---|---|
| `code/founder_mixture/` | Exact event-driven simulation of the two-type branching process with proliferation, death, and reversible phenotypic switching; the five likelihoods; maximum-likelihood fitting and profile-likelihood confidence sets; numerical checks; and report export. |
| `configs/` | Experiment configurations, including the parameter values, observation times, founder counts, numerical settings, and master random seeds. |
| `reports/*/tables/*.csv` | The nine saved summary tables used by the manuscript's figures and tables. |
| `paper/build_assets.py` | Builds the manuscript's three figures and four supplementary tables from those summary tables. |

The likelihoods are labeled as in the manuscript. $G$ is the population Gaussian likelihood of Gunnarsson, Foo, and Leder (2023, *J. Theor. Biol.* 573:111497); $EG$ is our adaptation of it with an explicit empty-culture component; $M$ is the founder-aware likelihood; $M_0$ is $M$ with the approximate conditional covariance $\widehat Q$ set to zero; and $M_{\mathrm{const}}$ (`Mconst` in the code) replaces $\widehat Q$ by a fitted constant conditional variance for living cultures.

## Installation

The calculations used Python 3.10 with the package versions pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the manuscript figures and tables

This step uses only the saved summary tables and takes a few seconds. It performs no simulation, fitting, or bootstrap resampling.

```bash
python paper/build_assets.py
```

It writes `paper/figures/figure-1-yule.pdf`, `figure-2-prediction.pdf`, and `figure-3-inference.pdf`; the supplementary tables `paper/tables/prediction.tex`, `paired.tex`, `bias.tex`, and `no-death.tex`; and `paper/asset-manifest.json`, which lists the SHA-256 hash of each input table and the inputs of each product.

## Rerun the experiments

Run all commands from the repository root. Generated datasets, fits, profiles, and caches are written under `data/`, which is ignored by Git. Each stage resumes from completed results and records the configuration, software versions, source hashes, and runtime. Fitting and profiling are the expensive stages; the commands below use four worker processes, or two for the $M_{\mathrm{const}}$ extension.

```bash
export PYTHONPATH=code
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/founder-mixture-matplotlib
```

**Automated checks.** The formula checks take under a minute:

```bash
python -m unittest founder_mixture.checks.FormulaChecks -v
```

**Experiments without death** (`configs/founder-mixture-pilot.json` and `configs/founder-mixture-pilot-days-9-11.json`; supplementary table `no-death.tex`). For each configuration `C` and output directory `D`, run

```bash
for stage in checks smoke screen; do
  python -m founder_mixture $stage --config C --output D --workers 4
done
python -m founder_mixture inference --replicates 100 --config C --output D --workers 4
python -m founder_mixture refine --config C --output D --workers 4
python -m founder_mixture report --config C --output D --report-dir reports/NAME
```

with `D=data/founder-mixture-pilot` and `NAME=founder-mixture-pilot` for the first configuration, and `D=data/founder-mixture-pilot-days-9-11` and `NAME=founder-mixture-pilot-days-9-11` for the second.

**Birth–death experiment with $G$, $EG$, $M$, and $M_0$** (`configs/founder-mixture-birth-death.json`; Figures 1–3). The stages run in three deliveries; each later delivery reuses the datasets and fits of the earlier ones.

```bash
C=configs/founder-mixture-birth-death.json
D=data/founder-mixture-birth-death
R=reports/founder-mixture-birth-death
for stage in preflight checks smoke refine; do
  python -m founder_mixture $stage --config $C --output $D --workers 4
done
python -m founder_mixture report    --config $C --output $D --report-dir $R
python -m founder_mixture checks    --delivery 2 --config $C --output $D --workers 4
python -m founder_mixture screen    --delivery 2 --config $C --output $D --workers 4
python -m founder_mixture inference --delivery 2 --config $C --output $D --replicates 20 --workers 4
python -m founder_mixture refine    --delivery 2 --config $C --output $D --workers 4
python -m founder_mixture report    --delivery 2 --config $C --output $D --report-dir $R
python -m founder_mixture checks    --delivery 3 --config $C --output $D --workers 4
python -m founder_mixture profile-check --delivery 3 --config $C --output $D --workers 4
python -m founder_mixture cost      --delivery 3 --config $C --output $D --workers 4
python -m founder_mixture inference --delivery 3 --config $C --output $D --replicates 100 --workers 4
python -m founder_mixture profiles  --delivery 3 --config $C --output $D --workers 4
python -m founder_mixture refine    --delivery 3 --config $C --output $D --workers 4
python -m founder_mixture.birth_death_extended_audit
python -m founder_mixture.birth_death_coverage_refinement --workers 4
python -m founder_mixture report    --delivery 3 --config $C --output $D --report-dir $R
```

**Addition of $M_{\mathrm{const}}$** (`configs/founder-mixture-birth-death-mconst.json`; Figure 3 and supplementary tables `paired.tex` and `bias.tex`). This extension fits $M_{\mathrm{const}}$ to the same 100 datasets per founder count without changing the earlier results. Its commands require one numerical thread per worker and reject other settings.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
C=configs/founder-mixture-birth-death-mconst.json
D=data/founder-mixture-birth-death/mconst
python -m founder_mixture.mconst_snapshot --config $C --output $D
python -m founder_mixture.mconst checks --config $C --output $D --workers 2
python -m founder_mixture.mconst pilot  --config $C --output $D --workers 2
for stage in inference profiles refine; do
  python -m founder_mixture.mconst_progress $stage --config $C --output $D --workers 2
done
python -m founder_mixture.mconst_coverage --config $C --output $D --workers 2
python -m founder_mixture.mconst cost   --config $C --output $D --workers 1
python -m founder_mixture.mconst report --config $C --output $D
python -m founder_mixture.mconst audit  --config $C --output $D
```

The report stages write their summary tables into `reports/`, after which `python paper/build_assets.py` rebuilds the figures and tables from the regenerated summaries.

## Reproducibility notes

- `python paper/build_assets.py` has been checked in this repository: it reproduces the manuscript's four supplementary tables byte for byte and its three figures identically when rendered. The complete simulation and fitting sequence has not been rerun from this repository.
- Random streams are derived from `numpy.random.SeedSequence` keys built from the master seed in each configuration and the stage, founder count, founder type, observation time, replicate, and purpose; the generator is PCG64. Increasing the number of replicates does not change earlier datasets. NumPy does not guarantee identical streams across versions or platforms, so use the pinned versions to regenerate the same datasets.
- The published-formula check in `birth_death_validation.py` records the hash of a local copy of Gunnarsson et al. (2023) when the environment variable `GUNNARSSON2023_PDF` points to one. The article is not distributed here, and the check runs without it. This optional provenance record is the only change from the code used for the manuscript.

## License

MIT; see `LICENSE`.
