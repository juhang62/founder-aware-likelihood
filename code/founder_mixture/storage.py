"""Atomic resumable artifacts with configuration, code, and environment records."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import importlib.metadata
import numpy as np
from . import ROOT, __version__
from .simulation import generate, STAGES, PURPOSES


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, default=serializable, allow_nan=False)+"\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serializable).encode()).hexdigest()


def validate_config(config):
    """Reject unsupported or invalid designs before writing any artifacts."""
    bounds = np.asarray(config["bounds"], float)
    if bounds.shape != (2, 2) or not np.all(bounds[:, 0] < bounds[:, 1]) or np.any(bounds <= 0):
        raise ValueError("Two positive, increasing birth-rate bounds are required")
    if len(config["times"]) != 2 or len(set(config["times"])) != 2 or min(config["times"]) <= 0:
        raise ValueError("This pilot requires two distinct positive endpoints")
    if len(config["founders"]) != 2 or len(set(config["founders"])) != 2:
        raise ValueError("This pilot requires two distinct founder counts")
    integers = [*config["founders"], config["cultures_per_cell"], config["smoke_cultures_per_cell"],
                config["screen_cultures_per_cell"], config["moment_check_cultures"], config["population_cap"]]
    if any(not isinstance(x, int) or x < 1 for x in integers):
        raise ValueError("Founder counts, culture counts, and population cap must be positive integers")
    if len(config["switching"]) != 2 or min(config["switching"]) <= 0 or config["measurement_variance"] <= 0:
        raise ValueError("Positive switching rates and measurement variance are required")
    death = np.asarray(config.get("death", (0., 0.)), float)
    if death.shape != (2,) or not np.isfinite(death).all() or np.any(death < 0):
        raise ValueError("Two finite nonnegative death rates are required")
    methods = config.get("methods", ["G", "M", "M0"])
    if not methods or len(set(methods)) != len(methods) or set(methods)-{"G", "EG", "M", "M0"}:
        raise ValueError("Invalid method selection")
    if np.any(death) and methods != ["G", "EG", "M", "M0"]:
        raise ValueError("The birth-death experiment explicitly requires all four methods")
    truth = np.asarray(config["truth"])
    if truth.shape != (2,) or np.any(truth < bounds[:, 0]) or np.any(truth > bounds[:, 1]):
        raise ValueError("Generating rates must lie in the fitted box")
    from .model import from_config
    from itertools import product
    for point in product(*bounds):
        from_config(point, config)
    for key in ("integration", "tight_integration"):
        settings = config[key]
        if not 0 < settings["tail_probability"] < .5 or min(settings["atol"], settings["rtol"]) <= 0:
            raise ValueError(f"Invalid {key} settings")


def provenance(config):
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/"code/founder_mixture").glob("*.py"))}
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), config=config,
                config_digest=digest(config), package_version=__version__, python=platform.python_version(),
                versions={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "numba", "matplotlib"]},
                git_revision=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
                git_status=git("status", "--short"), source_sha256=files,
                seed_scheme="SeedSequence([master, stage, founders, founder_type, time_index, replicate, purpose]); PCG64",
                stage_ids=STAGES, purpose_ids=PURPOSES,
                scientific_config={k: v for k, v in config.items() if k not in
                    ("execution", "integration", "tight_integration", "optimization", "tight_optimization", "extinction", "tight_extinction", "profiles", "refinement")},
                execution_config={k: v for k, v in config.items() if k in
                    ("execution", "integration", "tight_integration", "optimization", "tight_optimization", "extinction", "tight_extinction", "profiles", "refinement")},
                array_order="founder type, then time index, then culture; components type 1, type 2",
                thread_environment={k: __import__("os").environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_CACHE_DIR")})


class Store:
    def __init__(self, root, config, retry=False):
        validate_config(config)
        self.root, self.config = Path(root), config
        self.retry = retry
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance(config)
        manifest = self.root/"manifest.json"
        if manifest.exists():
            if read_json(manifest)["config_digest"] != digest(config):
                raise ValueError("Configuration changed: choose a separate --output directory; original results are preserved")
        else:
            write_json(manifest, self.provenance)
        write_json(self.root/"runs"/(self.provenance["timestamp_utc"].replace(":", "-")+".json"), self.provenance)

    def dataset(self, stage, founders, replicate, per_cell):
        path = self.root/"observations"/stage/f"n{founders}"/f"rep{replicate:03d}.npz"
        meta = path.with_suffix(".json")
        if path.exists():
            record = read_json(meta)
            if record["per_cell"] != per_cell or record["config_digest"] != digest(self.config):
                raise ValueError(f"Incompatible saved data: {path}")
            key = "code/founder_mixture/simulation.py"
            if record["source_sha256"][key] != self.provenance["source_sha256"][key]:
                raise ValueError(f"Simulator changed: preserve {path} and choose a separate output directory")
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        else:
            tick = perf_counter()
            data = generate(self.config, stage, founders, replicate, per_cell)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **data)
            temporary.replace(path)
            write_json(meta, dict(stage=stage, founders=founders, replicate=replicate,
                                 per_cell=per_cell, seconds=perf_counter()-tick,
                                 incomplete=int(np.sum(~data["complete"])), **self.provenance))
        if not np.all(data["complete"]):
            raise RuntimeError(f"Incomplete CTMC paths recorded in {path}; stage stopped without replacing or excluding them")
        return data

    def save_result(self, path, result):
        write_json(self.root/path, dict(result=result, provenance=self.provenance))

    def result(self, path):
        full = self.root/path
        if not full.exists():
            return None
        record = read_json(full)
        if record["provenance"]["config_digest"] != digest(self.config):
            raise ValueError(f"Incompatible saved result: {full}")
        # Explicit retries archive the entire earlier attempt and its provenance.
        # Data have separate compatibility checks and are never regenerated here.
        names = ["model.py", "likelihood.py", "inference.py", "extinction.py", "simulation.py"]
        if str(path).startswith(("preflight", "checks", "implementation-checks")):
            names += ["checks.py", "birth_death_checks.py"]
        if str(path).startswith(("one-type", "checks")):
            names += ["one_type.py"]
        if str(path).startswith(("smoke-integration", "smoke-refits")):
            names += ["birth_death_delivery.py"]
        if str(path).startswith("delivery2/"):
            names += ["birth_death_scientific.py", "birth_death_validation.py", "birth_death_diagnostics.py", "checks.py"]
        changed = [name for name in names if record["provenance"]["source_sha256"].get(f"code/founder_mixture/{name}")
                   != self.provenance["source_sha256"].get(f"code/founder_mixture/{name}")]
        result = record["result"]
        failed = result.get("passed", result.get("success", True)) is False
        if self.retry and (changed or failed):
            stamp = self.provenance["timestamp_utc"].replace(":", "-")
            archived = self.root/"attempts"/stamp/path
            archived.parent.mkdir(parents=True, exist_ok=True)
            full.replace(archived)
            write_json(archived.with_suffix(".retry.json"), dict(changed_sources=changed, prior_failed=failed,
                original_path=str(path), reason="Explicit --retry preserves the old attempt and reuses unchanged observations"))
            return None
        if changed:
            raise ValueError(f"Computational source changed ({', '.join(changed)}); use --retry to archive and recompute {full} on the same observations")
        return result
