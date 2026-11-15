"""Checkpoint process jobs and retain incomplete attempts on interruption."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from .storage import read_json


def run_jobs(store, jobs, worker, workers):
    """Jobs are (result path, positional arguments); retries must be explicit."""
    pending=[]
    for path,args in jobs:
        if store.result(path) is not None:
            continue
        status_path=store.root/"jobs"/path
        if status_path.exists():
            status=read_json(status_path)["result"]
            if status["status"] in ("running","incomplete") and not store.retry:
                raise RuntimeError(f"Incomplete job retained at {status_path}; use --retry on the same observations")
            stamp=store.provenance["timestamp_utc"].replace(":","-")
            archive=store.root/"attempts"/stamp/"jobs"/path
            archive.parent.mkdir(parents=True,exist_ok=True)
            status_path.replace(archive)
        store.save_result(f"jobs/{path}",dict(status="running",result_path=path,workers=workers))
        pending.append((path,args))
    if not pending:
        return
    deadline=getattr(store,"deadline",perf_counter()+store.config.get("execution",{}).get("stage_wall_hours",24)*3600)
    pool=ProcessPoolExecutor(max_workers=workers)
    futures={}
    finished=set()
    interrupted=False
    try:
        futures={pool.submit(worker,*args):path for path,args in pending}
        for future in as_completed(futures,timeout=max(.001,deadline-perf_counter())):
            path=futures[future]
            # A worker exception is an incomplete job; never replace it by a fit.
            result=future.result()
            store.save_result(path,result)
            store.save_result(f"jobs/{path}",dict(status="complete",result_path=path,success=result.get("success"),workers=workers))
            finished.add(path)
            print(f"{path}: success={result.get('success')} seconds={result.get('seconds',0):.2f}",flush=True)
    except BaseException as error:
        interrupted=True
        for path,_ in pending:
            if path not in finished:
                store.save_result(f"jobs/{path}",dict(status="incomplete",result_path=path,
                    reason=f"{type(error).__name__}: {error}",workers=workers))
        # Stop active processes before shutdown, rather than waiting beyond the
        # stage allowance. Completed results have already been saved atomically.
        for process in list(pool._processes.values()):
            process.terminate()
        raise
    finally:
        pool.shutdown(wait=not interrupted,cancel_futures=interrupted)
