"""Run the unchanged Mconst calculations with periodic progress records.

Only wraps completed searches for operational reporting. Numerical functions,
arguments, return values, caches, and checkpoint policy are unchanged. Matched
cost measurements use mconst directly, without this logging wrapper.
"""

import hashlib
import os
import resource
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from . import mconst
from .mconst_inference import Problem
from .storage import write_json

IDENTITIES={}
OUTPUT=None
STAGE=None
ORIGINAL_SOLVE=Problem.solve
ORIGINAL_PROVENANCE=mconst.provenance


def fingerprint(y):
    return hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()


def execution_provenance(config):
    record=ORIGINAL_PROVENANCE(config)
    record['execution_command']=[sys.executable,'-m','founder_mixture.mconst_progress',*sys.argv[1:]]
    record['progress_reporting']='Completed-search counts only; original numerical functions and arguments unchanged.'
    return record


def solve_with_progress(self,fixed=None,extra=()):
    if not hasattr(self,'_progress_started'):
        self._progress_started=perf_counter()
        self._progress_last=self._progress_started
        self._progress_searches=0
        self._progress_identity=IDENTITIES.get(fingerprint(self.data['y']),fingerprint(self.data['y'])[:12])
    result=ORIGINAL_SOLVE(self,fixed=fixed,extra=extra)
    self._progress_searches+=1
    now=perf_counter()
    if now-self._progress_last>=60 or not fixed:
        accuracy='tight' if self.tight else 'production'
        record=dict(dataset=self._progress_identity,stage=STAGE,accuracy=accuracy,pid=os.getpid(),
            completed_searches=self._progress_searches,unique_objectives=len(self.objectives),
            seconds=now-self._progress_started,last_fixed=fixed,last_loglik=result['loglik'],
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            checkpoint_cost=self.checkpoint.cost() if self.checkpoint else None)
        write_json(OUTPUT/'progress'/f'{self._progress_identity}-{STAGE}-{accuracy}-{os.getpid()}.json',record)
        print(f"{STAGE} {self._progress_identity} {accuracy}: {self._progress_searches} searches, "
              f"{len(self.objectives)} objective values, {now-self._progress_started:.0f}s; "
              f"peak RSS {record['peak_rss_mib']:.0f} MiB; fixed={fixed}",flush=True)
        self._progress_last=now
    return result


def main():
    global OUTPUT,STAGE
    args=sys.argv[1:]
    if not args:return mconst.main()
    STAGE=args[0]
    OUTPUT=Path(args[args.index('--output')+1] if '--output' in args else 'data/founder-mixture-birth-death/mconst').resolve()
    for n in (1,50):
        for path in (OUTPUT.parent/f'observations/inference/n{n}').glob('rep*.npz'):
            with np.load(path,allow_pickle=False) as data:
                IDENTITIES[fingerprint(data['y'])]=f'n{n}-{path.stem}'
    Problem.solve=solve_with_progress
    mconst.provenance=execution_provenance
    return mconst.main()


if __name__=='__main__':raise SystemExit(main())
