"""Checkpointed addendum stages with a 24-hour total invocation limit."""

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from .storage import read_json,write_json

STAGES=['checks','pilot','inference','profiles','refine','coverage','cost','report','audit']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-stage',choices=STAGES,default='checks')
    parser.add_argument('--wait-for-pilot',action='store_true')
    parser.add_argument('--config',default='configs/founder-mixture-birth-death-mconst.json')
    parser.add_argument('--output',default='data/founder-mixture-birth-death/mconst')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--retry',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error('Use one or two workers')
    from .mconst import execution_limits
    execution_limits()
    os.nice(10)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();wall_start=time.time();deadline=started+86400-30
    stamp=datetime.now(timezone.utc).isoformat().replace(':','-')
    path=root/'workflow.json'
    if path.exists():
        previous=root/'workflow-history'/f'{stamp}.json';previous.parent.mkdir(exist_ok=True);path.replace(previous)
    state=dict(status='starting',completed=[],started_utc=stamp,allowance_hours=24,shutdown_reserve_seconds=30,
               workers=args.workers,numerical_threads=1,
               command=getattr(sys,'orig_argv',[sys.executable,*sys.argv]))
    def checkpoint():
        state['seconds']=time.monotonic()-started;write_json(path,state)
    checkpoint()
    process=None
    try:
        if args.wait_for_pilot:
            state['status']='waiting_for_pilot';checkpoint()
            print('Waiting for the active numerical pilot.',flush=True)
            while not (root/'pilot.json').exists():
                if time.monotonic()>=deadline:raise TimeoutError('24-hour workflow allowance exhausted while waiting for the pilot')
                failed=[p for p in (root/'timings').glob('pilot-*.json') if p.stat().st_mtime>=wall_start and not read_json(p)['result']['success']]
                if failed:raise RuntimeError('The active pilot stopped; inspect its preserved results')
                time.sleep(5)
            if not read_json(root/'pilot.json')['result']['passed']:raise RuntimeError('Pilot did not pass; expansion remains blocked')
        with (root/'workflow.log').open('a',buffering=1) as log:
            for stage in STAGES[STAGES.index(args.from_stage):]:
                module='founder_mixture.mconst_progress' if stage in ('pilot','inference','profiles','refine') else 'founder_mixture.mconst'
                if stage=='coverage':module='founder_mixture.mconst_coverage'
                command=[sys.executable,'-m',module]
                if stage!='coverage':command.append(stage)
                command += ['--config',args.config,'--output',args.output,'--workers',str(1 if stage=='cost' else args.workers)]
                if args.retry:command.append('--retry')
                state.update(status='running',stage=stage,stage_command=command);checkpoint()
                print('Starting '+stage,flush=True);log.write('Starting '+stage+'\n')
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
                lines=queue.Queue()
                def forward(stream):
                    for line in stream:lines.put(line)
                reader=threading.Thread(target=forward,args=(process.stdout,),daemon=True);reader.start()
                while process.poll() is None or reader.is_alive() or not lines.empty():
                    if time.monotonic()>=deadline:raise TimeoutError('24-hour workflow allowance exhausted; completed work is retained')
                    try:line=lines.get(timeout=1)
                    except queue.Empty:continue
                    log.write(line);print(line,end='',flush=True)
                code=process.wait();process=None
                if code:raise RuntimeError(f'{stage} exited with code {code}; inspect retained numerical outcomes')
                state['completed'].append(stage);checkpoint()
        state['status']='computations_finished';checkpoint()
        print('Computations finished; narrative interpretation and final LaTeX build remain.',flush=True)
        return 0
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        state.update(status='stopped',reason=f'{type(error).__name__}: {error}');checkpoint()
        raise


if __name__=='__main__':raise SystemExit(main())
