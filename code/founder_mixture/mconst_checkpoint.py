"""Durable, compatible checkpoints for deterministic optimization replay.

One file holds each constrained search. Replaying the same control flow loads
completed steps; it never changes the ordering or introduces later maximizers
as earlier starting points. A checksum and retained previous generation allow
recovery after an incomplete write. Disposable likelihood caches are rebuilt.
"""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import uuid
from .storage import digest, serializable

FORMAT = 1


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value, hook=None):
    """Install a complete JSON file, retaining the previous durable generation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    previous = path.with_name(path.name+'.previous')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, default=serializable, allow_nan=False, separators=(',', ':'))+'\n')
        stream.flush()
        if hook:
            hook('before_fsync')
        os.fsync(stream.fileno())
    if path.exists():
        # Hard-linking retains the old inode until the new directory entry is
        # durable. Never move away the only valid primary before replacement.
        backup = path.with_name(path.name+'.backup-tmp')
        if backup.exists():
            backup.unlink()
        os.link(path, backup)
        os.replace(backup, previous)
        sync_directory(path.parent)
    if hook:
        hook('before_replace')
    os.replace(temp, path)
    sync_directory(path.parent)
    if hook:
        hook('after_replace')


def envelope(value):
    return dict(sha256=digest(value), payload=value)


def recover(path):
    """Return the last valid generation; incomplete temporary files are ignored."""
    path = Path(path)
    failures = []
    for source in (path, path.with_name(path.name+'.previous')):
        if not source.exists():
            continue
        try:
            record = json.loads(source.read_text())
            if record['sha256'] != digest(record['payload']):
                raise ValueError('Checksum mismatch')
            if source != path:
                # Preserve corrupt primary and restore without promoting it to
                # the backup generation on the next checkpoint write.
                if path.exists():
                    os.replace(path, path.with_name(path.name+'.invalid-'+uuid.uuid4().hex))
                durable_json(path, record)
            return record['payload']
        except (ValueError, KeyError, TypeError) as error:
            failures.append(f'{source}: {error}')
    if failures:
        raise ValueError('No valid checkpoint generation: '+'; '.join(failures))
    return None


class Checkpoint:
    def __init__(self, root, metadata, hook=None):
        self.root = Path(root)
        self.metadata = dict(metadata, checkpoint_format=FORMAT)
        self.hook = hook
        self.started = perf_counter()
        self.io_seconds = 0.
        self.recovery_seconds = 0.
        self.reused_steps = 0
        self.new_steps = 0
        self.sequence = 0
        self.previous_cost = dict(seconds=0., checkpoint_seconds=0., recovery_seconds=0.)
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root/'identity.json'
        saved = recover(identity)
        if saved is not None and saved != self.metadata:
            raise ValueError('Incompatible checkpoint data, configuration, sources, solver, or fitted maximum')
        if saved is None:
            durable_json(identity, envelope(self.metadata))
        # Each saved step carries the job's accumulated cost. Taking the latest
        # sequence avoids counting the same invocation once per search file.
        for path in self.root.glob('state-*.json'):
            record = recover(path)
            if record['metadata_sha256'] != digest(self.metadata):
                raise ValueError('Incompatible checkpoint record')
            if record['sequence'] > self.sequence:
                self.sequence = record['sequence']
                self.previous_cost = record['cost']
        self.recovery_seconds = perf_counter()-self.started
        self.resumed = self.sequence > 0

    def cost(self):
        return dict(seconds=self.previous_cost['seconds']+perf_counter()-self.started,
                    checkpoint_seconds=self.previous_cost['checkpoint_seconds']+self.io_seconds,
                    recovery_seconds=self.previous_cost['recovery_seconds']+self.recovery_seconds,
                    resumed=self.resumed, reused_steps=self.reused_steps, new_steps=self.new_steps,
                    unsaved_interruption_time='Not recoverable after abrupt termination; reported totals are lower bounds for interrupted attempts.')

    def scope(self, key, inputs):
        return Scope(self, key, inputs)

    def write(self, path, state):
        self.sequence += 1
        value = dict(metadata_sha256=digest(self.metadata), sequence=self.sequence, cost=self.cost(), state=state)
        tick = perf_counter()
        durable_json(path, envelope(value), self.hook)
        self.io_seconds += perf_counter()-tick


class Scope:
    def __init__(self, checkpoint, key, inputs):
        self.checkpoint = checkpoint
        self.path = checkpoint.root/('state-'+hashlib.sha256(key.encode()).hexdigest()+'.json')
        tick = perf_counter()
        saved = recover(self.path)
        checkpoint.recovery_seconds += perf_counter()-tick
        self.state = saved['state'] if saved else dict(key=key, inputs=inputs, steps={}, objectives=[], result=None)
        if digest(self.state['inputs']) != digest(inputs) or self.state['key'] != key:
            raise ValueError('Checkpoint search order or starting points changed')
        self.capture = None

    def save(self):
        if self.capture:
            self.capture(self.state)
        self.checkpoint.write(self.path, self.state)

    def step(self, key, function):
        if key in self.state['steps']:
            self.checkpoint.reused_steps += 1
            return self.state['steps'][key]
        self.state['pending_step'] = key
        # Persist the pending operation before entering its optimizer. Its
        # completed predecessors remain available if this operation is killed.
        self.save()
        if self.checkpoint.hook:
            self.checkpoint.hook('before_step:'+key)
        result = function()
        self.state['steps'][key] = result
        self.state['pending_step'] = None
        self.checkpoint.new_steps += 1
        self.save()
        return result

    def finish(self, result):
        self.state['result'] = result
        self.state['pending_step'] = None
        self.save()
        return result


def operation(scope, key, function):
    return scope.step(key, function) if scope is not None else function()
