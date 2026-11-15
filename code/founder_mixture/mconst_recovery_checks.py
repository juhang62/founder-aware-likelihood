"""Kill and resume deterministic fit/profile processes to check recovery."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
from unittest.mock import patch
import numpy as np
from . import ROOT, mconst_inference as inference
from .mconst_checkpoint import Checkpoint, durable_json, envelope, recover
from .storage import read_json, digest


def example(root, kind, scenario, interrupt=None):
    config = read_json(ROOT/'configs/founder-mixture-birth-death.json')
    config['mconst'] = read_json(ROOT/'configs/founder-mixture-birth-death-mconst.json')
    config['optimization'] = dict(config['optimization'], grid_size=3, starts=2)
    config['mconst'] = dict(config['mconst'], variance_grid_size=3)
    config['profiles'] = dict(config['profiles'], grid_size=9, nuisance_grid_size=3)
    data = dict(y=np.array([[10., 10.]]), time=np.array([9.]), z0=np.array([[1,0]]), cell=np.array([0]))
    state = dict(step='', calls=0, writes=0)
    def kill(phase):
        durable_json(Path(root)/'interruption.json', dict(phase=phase, state=state.copy()))
        os.kill(os.getpid(), signal.SIGKILL)
    def hook(event):
        if event.startswith('before_step:'):
            state['step'] = event
            state['calls'] = 0
        if event == 'before_replace':
            state['writes'] += 1
            if interrupt == 'write' and state['writes'] == 6:
                kill('checkpoint replacement')
    def objective(t):
        state['calls'] += 1
        if interrupt == 'optimize' and state['writes'] >= 4 and ':start:' in state['step'] and state['calls'] == 5:
            kill('inside local optimization')
        x,y,u=t
        if scenario == 'disconnected':
            return -1000*min((x-.5)**2,(x-.9)**2)-1000*(y-.7)**2-(u-1)**2
        if scenario == 'truncated':
            return -100*((x-.3)**2+(y-.7)**2+u**2)
        if scenario == 'flat':
            return -100*(y-.7)**2-(u-1)**2
        return -.5*((x-.8)**2+(y-.6-.4*(x-.8))**2+(u-1-.3*(x-.8)-.2*(y-.6))**2)/.02**2
    class Analytic:
        def __init__(self, *args): self.timing = {}
        def loglik(self, method, theta, data): return objective(theta)
    metadata = dict(founders=1, replicate=0, method='Mconst', calculation=kind,
                    observed_inputs_sha256=digest(data), config_digest=digest(config),
                    numerical_sources={'test':digest(Path(__file__).read_text())}, solver='analytic', scenario=scenario)
    checkpoint = Checkpoint(Path(root)/'checkpoints', metadata, hook)
    with patch.object(inference, 'Likelihood', Analytic), patch.object(inference, 'tail_bound', return_value=-1e6):
        if kind == 'fit':
            result = inference.fit(data, config, checkpoint=checkpoint)
        else:
            theta = dict(disconnected=[.5,.7,1.], truncated=[.3,.7,0.], flat=[.7,.7,1.]).get(scenario,[.8,.6,1.])
            fitted = dict(success=True, theta=theta, estimate=theta[:2], loglik=-.1 if scenario=='failure' else 0., variance_cap=100.)
            result = inference.profile_sets(data, config, fitted, checkpoint=checkpoint)
    result['checkpoint_cost'] = checkpoint.cost()
    durable_json(Path(root)/'result.json', result)


def scientific_result(result, kind):
    keys = ('success','estimate','theta','loglik','u','boundary','variance_boundary','reason') if kind=='fit' else ('success','rates','reason')
    return {key:result[key] for key in keys if key in result}


def run_checks(output):
    root = Path(output)/uuid.uuid4().hex
    root.mkdir(parents=True)
    rows = []
    def run(path, kind, scenario, interrupt=None):
        command = [sys.executable,'-m','founder_mixture.mconst_recovery_checks','--example',str(path),'--kind',kind,'--scenario',scenario]
        if interrupt: command += ['--interrupt',interrupt]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode != (-signal.SIGKILL if interrupt else 0):
            raise AssertionError(f'{command}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return command
    for kind, scenario in (('fit','quadratic'),('profile','quadratic'),('profile','disconnected'),
                            ('profile','truncated'),('profile','flat'),('profile','failure')):
        reference = root/f'{kind}-{scenario}-reference'
        command = run(reference,kind,scenario)
        expected = read_json(reference/'result.json')
        phases = ('optimize','write') if scenario=='quadratic' else ('write',)
        for phase in phases:
            path = root/f'{kind}-{scenario}-{phase}'
            interrupted = run(path,kind,scenario,phase)
            resumed = run(path,kind,scenario)
            actual = read_json(path/'result.json')
            # Exact equality is stronger than the specified numerical targets;
            # timings and cache counters are deliberately excluded.
            same = scientific_result(actual,kind)==scientific_result(expected,kind)
            reused = actual['checkpoint_cost']['reused_steps']>0
            rows.append(dict(kind=kind,scenario=scenario,phase=phase,passed=bool(same and reused),
                             same_scientific_result=same,reused_steps=actual['checkpoint_cost']['reused_steps'],
                             reference_command=command,interrupt_command=interrupted,resume_command=resumed))
    path = root/'compatibility'
    metadata = dict(data='a', settings='a', source='a', solver='a')
    Checkpoint(path, metadata)
    rejected = []
    for key in metadata:
        try:
            Checkpoint(path, dict(metadata, **{key:'changed'}))
        except ValueError:
            rejected.append(key)
    path = root/'corrupt.json'
    durable_json(path,envelope({'version':1}))
    durable_json(path,envelope({'version':2}))
    path.write_text('{incomplete')
    fallback = recover(path)=={'version':1}
    # Confirm the recovered primary remains usable on the next replacement.
    durable_json(path,envelope({'version':3}))
    fallback = fallback and recover(path)=={'version':3}
    result = dict(passed=all(r['passed'] for r in rows) and len(rejected)==4 and fallback,
                  rows=rows,rejected_metadata_changes=rejected,corrupt_primary_recovered=fallback,
                  output=str(root),scientific_datasets_generated=0)
    durable_json(root/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example')
    parser.add_argument('--kind',choices=['fit','profile'])
    parser.add_argument('--scenario',default='quadratic')
    parser.add_argument('--interrupt',choices=['optimize','write'])
    parser.add_argument('--output',default='data/founder-mixture-birth-death/mconst/recovery-tests')
    args=parser.parse_args()
    if args.example:
        example(args.example,args.kind,args.scenario,args.interrupt)
    else:
        result=run_checks(args.output)
        print(json.dumps(result,indent=2))
        return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
