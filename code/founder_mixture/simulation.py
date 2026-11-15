"""Exact exponential-waiting-time simulation with stable independent streams."""

import numpy as np
from numba import njit

# Integer identifiers are a versioned part of the experiment, not Python hashes.
STAGES = {"checks": 11, "smoke": 21, "screen": 31, "inference": 41, "diagnostics": 51}
PURPOSES = {"ctmc": 1, "measurement": 2, "predictions": 3, "bootstrap": 4, "yule": 5}


def stream(seed, stage, founders, founder_type, time_index, replicate, purpose):
    return np.random.default_rng(np.random.SeedSequence([
        seed, STAGES[stage], founders, founder_type, time_index, replicate, PURPOSES[purpose]
    ]))


@njit(cache=True)
def simulate_counts(rng, count, z0, t, birth, switching, cap):
    """Return counts and completion flags; a capped path is never replaced."""
    states = np.empty((count, 2), dtype=np.int64)
    complete = np.ones(count, dtype=np.bool_)
    for j in range(count):
        z1, z2 = z0[0], z0[1]
        clock = 0.
        while True:
            rate1, rate2 = birth[0]*z1, birth[1]*z2
            rate3, rate4 = switching[0]*z1, switching[1]*z2
            total = rate1+rate2+rate3+rate4
            if total == 0:
                break
            clock += rng.exponential(1./total)
            if clock > t:
                break
            if z1+z2 >= cap:
                complete[j] = False
                break
            draw = rng.random()*total
            if draw < rate1:
                z1 += 1
            elif draw < rate1+rate2:
                z2 += 1
            elif draw < rate1+rate2+rate3:
                z1 -= 1
                z2 += 1
            else:
                z1 += 1
                z2 -= 1
        states[j, 0], states[j, 1] = z1, z2
    return states, complete


@njit(cache=True)
def simulate_birth_death_counts(rng, count, z0, t, birth, death, switching, cap):
    states = np.empty((count, 2), dtype=np.int64)
    complete = np.ones(count, dtype=np.bool_)
    for j in range(count):
        z1, z2 = z0[0], z0[1]
        clock = 0.
        while z1+z2:
            rates = np.array([birth[0]*z1, birth[1]*z2, death[0]*z1, death[1]*z2,
                              switching[0]*z1, switching[1]*z2])
            total = rates.sum()
            if total == 0:
                break
            clock += rng.exponential(1./total)
            if clock > t:
                break
            if z1+z2 >= cap:
                complete[j] = False
                break
            draw = rng.random()*total
            event = 0
            cumulative = rates[0]
            while draw >= cumulative and event < 5:
                event += 1
                cumulative += rates[event]
            if event == 0:
                z1 += 1
            elif event == 1:
                z2 += 1
            elif event == 2:
                z1 -= 1
            elif event == 3:
                z2 -= 1
            elif event == 4:
                z1 -= 1
                z2 += 1
            else:
                z1 += 1
                z2 -= 1
        states[j, 0], states[j, 1] = z1, z2
    return states, complete


def generate(config, stage, founders, replicate, per_cell):
    arrays = {key: [] for key in ["y", "counts", "z0", "time", "cell", "complete"]}
    for founder_type in range(2):
        z0 = np.zeros(2, dtype=np.int64)
        z0[founder_type] = founders
        for time_index, t in enumerate(config["times"]):
            args = (config["master_seed"], stage, founders, founder_type, time_index, replicate)
            death = np.asarray(config.get("death", (0., 0.)), float)
            common = (stream(*args, "ctmc"), per_cell, z0, t, np.array(config["truth"]))
            if np.any(death):
                counts, complete = simulate_birth_death_counts(*common, death, np.array(config["switching"]), config["population_cap"])
            else:
                # Preserve the original event ordering and random draws exactly.
                counts, complete = simulate_counts(*common, np.array(config["switching"]), config["population_cap"])
            error = stream(*args, "measurement").normal(
                0., np.sqrt(config["measurement_variance"]), size=(per_cell, 2))
            arrays["y"].append(counts+error)
            arrays["counts"].append(counts)
            arrays["z0"].append(np.tile(z0, (per_cell, 1)))
            arrays["time"].append(np.full(per_cell, t))
            arrays["cell"].append(np.full(per_cell, 2*founder_type+time_index))
            arrays["complete"].append(complete)
    return {key: np.concatenate(parts) for key, parts in arrays.items()}
