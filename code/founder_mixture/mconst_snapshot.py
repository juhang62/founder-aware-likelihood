"""Freeze prior artifacts and Git-verified numerical sources before Mconst."""

import argparse
import hashlib
from pathlib import Path
import subprocess
from . import ROOT
from .storage import read_json,write_json


def snapshot(config_path,output):
    cfg=read_json(config_path);root=Path(output).resolve();base=root.parent
    dest=root/'baseline';manifest=dest/'manifest.json'
    if manifest.exists():
        existing=read_json(manifest)
        print(f"Existing baseline retained: {len(existing['files'])} artifact hashes",flush=True)
        return
    for n in (1,50):
        for rep in range(100):
            paths=[base/f'observations/inference/n{n}/rep{rep:03d}.npz']
            paths += [base/f'{folder}/n{n}/rep{rep:03d}-{method}.json' for folder in ('fits/inference','delivery3/profiles') for method in ('G','EG','M','M0')]
            for path in paths:
                if not path.exists():raise ValueError(f'Complete or reproduce Delivery 3 first: missing {path}')
    dest.mkdir(parents=True,exist_ok=True)
    paths=[p for p in base.rglob('*') if p.is_file() and 'mconst' not in p.relative_to(base).parts]
    paths += [p for p in (ROOT/'reports/founder-mixture-birth-death').rglob('*') if p.is_file() and not p.name.startswith('mconst-')]
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    revision=cfg['baseline_revision'];sources={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--','code/founder_mixture'],cwd=ROOT,text=True).splitlines()
    for name in names:
        if not name.endswith('.py'):continue
        content=subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=ROOT)
        sources[name]=hashlib.sha256(content).hexdigest();(dest/Path(name).name).write_bytes(content)
    write_json(manifest,dict(git_revision=revision,files=files,source_sha256=sources))
    print(f'Frozen {len(files)} artifact hashes and {len(sources)} original sources',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/founder-mixture-birth-death-mconst.json')
    parser.add_argument('--output',default='data/founder-mixture-birth-death/mconst')
    args=parser.parse_args();snapshot(args.config,args.output)
