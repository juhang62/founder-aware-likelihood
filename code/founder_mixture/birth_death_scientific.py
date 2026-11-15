"""Explicit Delivery 2 stages; never extends to 100 replicates or profiles."""

from pathlib import Path
from time import perf_counter
from .storage import Store, read_json
from .birth_death_validation import published_comparator, simulation_validation
from .birth_death_diagnostics import screen, inference_summary, integration_refinement, fit_refinement


def run_delivery(args, config):
    if args.command not in ('checks', 'screen', 'inference', 'refine', 'report'):
        raise ValueError('Use Delivery 1 for preflight and smoke stages')
    if args.replicates != 20:
        raise ValueError('Delivery 2 is limited to 20 datasets per founder count; Delivery 3 requires review')
    if args.workers > config['execution']['max_workers']:
        raise ValueError('Worker count exceeds the configured allowance')
    if Path(args.output).resolve().name != config['experiment']:
        raise ValueError('Use the explicit birth-death output directory')
    store = Store(args.output, config, retry=args.retry)
    start = args.invocation_started
    store.deadline = start + config['execution']['stage_wall_hours']*3600
    # The frozen configuration records the original Delivery 1 default; this
    # explicit CLI selection authorizes Delivery 2 without changing data keys.
    for path in ('preflight.json', 'checks.json', 'smoke.json', 'smoke-integration.json', 'smoke-refits.json'):
        result = store.result(path)
        if not result or not result['passed']:
            raise ValueError(f'Complete Delivery 1 prerequisite {path} before Delivery 2')
    stage = f'delivery2/stages/{args.command}.json'
    path = store.root/stage
    if path.exists() and read_json(path)['result']['status'] in ('running', 'incomplete'):
        if not args.retry:
            raise RuntimeError(f'Incomplete stage retained at {path}; inspect and use --retry explicitly')
        archive = store.root/'attempts'/store.provenance['timestamp_utc'].replace(':', '-')/stage
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.replace(archive)
    store.save_result(stage, dict(status='running', workers=args.workers, delivery=2))
    finished, passed = False, False
    def cached(name, action):
        result = store.result(f'delivery2/{name}.json')
        if result is None:
            result = action()
            store.save_result(f'delivery2/{name}.json', result)
        return result
    try:
        if args.command == 'checks':
            comparator = cached('comparator', lambda: published_comparator(config))
            simulation = cached('simulation', lambda: simulation_validation(store))
            passed = comparator['passed'] and simulation['passed']
        else:
            # Failed validation may be reported, but blocks scientific stages.
            if args.command != 'report':
                for name in ('comparator', 'simulation'):
                    result = store.result(f'delivery2/{name}.json')
                    if not result or not result['passed']:
                        raise ValueError(f'Delivery 2 {name} validation must pass before {args.command}')
            if args.command == 'screen':
                passed = cached('screen', lambda: screen(store))['passed']
            elif args.command == 'inference':
                screened = store.result('delivery2/screen.json')
                if not screened or not screened['passed']:
                    raise ValueError('Complete the independent screen before inference')
                from .__main__ import run_fits
                run_fits(store, 'inference', 20, config['cultures_per_cell'], args.workers)
                result = cached('inference', lambda: inference_summary(store))
                passed = result['passed']
            elif args.command == 'refine':
                result = store.result('delivery2/inference.json')
                if not result or not result['complete']:
                    raise ValueError('Record all 160 inference outcomes before refinement')
                integration = cached('integration', lambda: integration_refinement(store))
                refits = cached('refits', lambda: fit_refinement(store, args.workers))
                passed = integration['passed'] and refits['passed']
            elif args.command == 'report':
                from .birth_death_scientific_report import export
                export(store, args.report_dir)
                passed = True
        finished = True
        print(f'Delivery 2 {args.command}: passed={passed}', flush=True)
    finally:
        store.save_result(stage, dict(status='complete' if finished else 'incomplete', passed=passed, workers=args.workers, delivery=2))
        store.save_result(f"timings/delivery2-{args.command}-{store.provenance['timestamp_utc'].replace(':', '-')}.json",
                          dict(command=args.command, delivery=2, seconds=perf_counter()-start, workers=args.workers,
                               passed=passed, complete=finished, stage_budget_hours=config['execution']['stage_wall_hours']))
    return 0 if passed else 1
