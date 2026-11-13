#!/usr/bin/env python3
"""Run sequential argv manifests and always restore llama-swap.

The operator must authorize the interruption. A positional manifest contains
ninfer_tune.py arguments; --program-manifest contains complete argv lists, without
a shell. Optional cumulative accounting reserves the last two hours for validation
and restoration, and deadlines also stop a running job.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ninfer_tune as tune

CHILD_STOP_TIMEOUT = 210


def read_jobs(path: Path) -> list[list[str]]:
    jobs = json.loads(path.read_text())
    if not isinstance(jobs, list) or any(
            not isinstance(job, list) or not job or not job[0] or
            not all(isinstance(value, str) for value in job) for job in jobs):
        raise ValueError('manifest must be a list of nonempty argument lists')
    return jobs


class BudgetLedger:
    """A separate lock survives atomic ledger replacement; unfinished runs block reuse."""

    def __init__(self, path: Path, budget: float, reserve: float):
        if not math.isfinite(budget) or not math.isfinite(reserve) or not 0 <= reserve < budget:
            raise ValueError('budget must exceed the nonnegative reserve')
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.with_name(path.name + '.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.data = json.loads(path.read_text()) if path.exists() else {
                'version': 1, 'budget_seconds': budget, 'reserve_seconds': reserve,
                'used_seconds': 0, 'sessions': []}
            if (self.data.get('version') != 1 or
                    self.data.get('budget_seconds') != budget or
                    self.data.get('reserve_seconds') != reserve):
                raise ValueError('ledger budget/reserve differ; use the recorded limits')
            used = self.data.get('used_seconds')
            if not isinstance(used, (int, float)) or not math.isfinite(used) or used < 0:
                raise ValueError('invalid ledger used_seconds')
            if not isinstance(self.data.get('sessions'), list):
                raise ValueError('invalid ledger sessions')
            if any(entry.get('finished') is None for entry in self.data['sessions']):
                raise ValueError('ledger has an unfinished interruption; reconcile its restoration first')
            if any(entry.get('restored') is False for entry in self.data['sessions']):
                raise ValueError('ledger records a failed restoration; reconcile service readiness first')
        except BaseException:
            self.lock.close()
            raise
        self.entry: dict | None = None

    def remaining(self, validation: bool) -> float:
        ceiling = self.data['budget_seconds'] - (0 if validation else self.data['reserve_seconds'])
        return ceiling - self.data['used_seconds']

    def write(self) -> None:
        temporary = self.path.with_name(self.path.name + '.next')
        temporary.write_text(json.dumps(self.data, indent=2) + '\n')
        os.replace(temporary, self.path)

    def start(self, manifest: Path, validation: bool) -> None:
        self.started = time.monotonic()
        self.entry = {'started': time.time(), 'finished': None,
                      'phase': 'validation' if validation else 'exploration',
                      'manifest': str(manifest.resolve())}
        self.data['sessions'].append(self.entry)
        self.write()

    def finish(self, error: str | None) -> None:
        if self.entry is not None:
            elapsed = time.monotonic() - self.started
            self.entry.update({'finished': time.time(), 'seconds': elapsed,
                               'restored': error is None, 'restoration_error': error})
            self.data['used_seconds'] += elapsed
            self.write()

    def close(self) -> None:
        self.lock.close()


def server_groups(registry: Path | None) -> dict[int, str]:
    return tune.registered_server_groups(registry)


def stop_child(child: subprocess.Popen | None, registry: Path | None = None) -> None:
    if child is None:
        return
    # The harness owns its group; each server owns another group and registers
    # it. Give the harness a chance to stop those servers before cleaning orphans.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + CHILD_STOP_TIMEOUT
    try:
        child.wait(timeout=CHILD_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass
    signalled: set[int] = set()
    while time.monotonic() < deadline:
        groups = server_groups(registry)
        for pid, state in groups.items():
            if state == 'stopping':
                signalled.add(pid)  # The harness has already sent graceful TERM.
            elif pid not in signalled:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                signalled.add(pid)
        if not any(tune.process_group_alive(pid) for pid in [child.pid, *groups]):
            return
        time.sleep(.2)
    for pid in [child.pid, *server_groups(registry)]:
        if pid == os.getpgrp():
            raise RuntimeError('refusing to signal the session process group')
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=10)


def wait_service_ready(model: str, timeout: float) -> None:
    tune.wait_model_ready(model, timeout)


def run(args: argparse.Namespace) -> None:
    manifest = args.program_manifest or args.manifest
    jobs = read_jobs(manifest)
    root = Path(__file__).resolve().parent.parent
    ledger = BudgetLedger(args.budget_ledger, args.budget_seconds, args.reserve_seconds) if args.budget_ledger else None
    registry = None
    try:
        registry_file = tempfile.NamedTemporaryFile(prefix='ninfer-session-groups-', suffix='.jsonl', delete=False)
        registry = Path(registry_file.name)
        registry_file.close()
        remaining = ledger.remaining(args.validation) if ledger else None
        # Exploration stops at the four-hour ceiling and restores inside the final
        # reserve. Validation also leaves time for graceful child exit + preload.
        if remaining is not None and args.validation:
            remaining -= CHILD_STOP_TIMEOUT + args.restore_timeout
        if remaining is not None and remaining <= 0:
            raise RuntimeError('interruption budget exhausted for this phase; service left running')
        cutoff = time.monotonic() + remaining if remaining is not None else None
        if args.deadline is not None:
            requested = time.monotonic() + args.deadline - time.time()
            cutoff = min(cutoff, requested) if cutoff is not None else requested
        if cutoff is not None and cutoff <= time.monotonic():
            raise RuntimeError('session deadline has passed; service left running')
        tune.gpu_state()  # Must work BEFORE stopping the service.
        results = []
        child = None

        def interrupt(*_: object) -> None:
            raise KeyboardInterrupt

        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupt)
        if ledger:
            ledger.start(manifest, args.validation)
        try:
            subprocess.run(['docker', 'compose', 'stop', 'llama-swap'], cwd=root, check=True)
            tune.wait_gpu_idle(0)
            tune.preflight(False)
            index = 0
            waiting = False
            while True:
                if cutoff is not None and time.monotonic() >= cutoff:
                    print('Session deadline reached; restoring service.', flush=True)
                    break
                if args.watch:
                    updated = read_jobs(manifest)
                    if updated[:index] != jobs[:index]:
                        raise ValueError('watch manifest may only append jobs')
                    jobs = updated
                if index == len(jobs):
                    if not args.watch or (args.finish_file and args.finish_file.exists()):
                        break
                    if not waiting:
                        print('Queue drained; waiting for appended jobs.', flush=True)
                        waiting = True
                    time.sleep(min(2, max(0, cutoff - time.monotonic())) if cutoff is not None else 2)
                    continue
                waiting = False
                job = jobs[index]
                started = time.time()
                command = job if args.program_manifest else [sys.executable, '-u', str(root / 'bench/ninfer_tune.py'), *job]
                print('JOB', json.dumps(job), flush=True)
                registry.write_text('')
                env = dict(os.environ, NINFER_SESSION_SERVER_GROUPS=str(registry))
                child = subprocess.Popen(command, cwd=root, start_new_session=True, env=env)
                timed_out = False
                try:
                    timeout = max(0, cutoff - time.monotonic()) if cutoff is not None else None
                    rc = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop_child(child, registry)
                    rc = child.returncode
                finally:
                    results.append({'args': job, 'returncode': child.poll(), 'started': started,
                                    'seconds': time.time() - started, 'timed_out': timed_out})
                    manifest.with_suffix('.results.json').write_text(json.dumps(results, indent=2))
                if timed_out:
                    child = None
                    raise TimeoutError('running job reached the session deadline; restoring service')
                if rc:
                    raise RuntimeError(f'job failed with exit {rc}')
                stop_child(child, registry)  # Also clean a normally exiting driver's survivors.
                child = None
                index += 1
        finally:
            # Repeated signals must not interrupt child cleanup or restoration.
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            restoration_error = None
            try:
                try:
                    stop_child(child, registry)
                finally:
                    try:
                        # swap-check can bring the container up inside a job.
                        # Stop it explicitly even if that child was forcibly killed.
                        subprocess.run(['docker', 'compose', 'stop', 'llama-swap'], cwd=root, check=True)
                    finally:
                        try:
                            tune.wait_gpu_idle(0)
                        finally:
                            subprocess.run(['docker', 'compose', 'up', '-d', 'llama-swap'], cwd=root, check=True)
                            if ledger:
                                wait_service_ready(args.ready_model, args.restore_timeout)
            except BaseException as error:
                restoration_error = f'{type(error).__name__}: {error}'
                raise
            finally:
                if ledger:
                    ledger.finish(restoration_error)
    finally:
        if ledger:
            ledger.close()
        if registry:
            registry.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?')
    parser.add_argument('--program-manifest', type=Path, help='complete argv jobs, for sequential GPU operator tests')
    parser.add_argument('--deadline', type=float, help='Unix time; stops a running job and restores service')
    parser.add_argument('--watch', action='store_true', help='accept appended jobs until deadline or finish-file')
    parser.add_argument('--finish-file', type=Path, help='stop after the queue drains when this file exists')
    parser.add_argument('--budget-ledger', type=Path, help='cumulative interruption ledger shared across campaign runs')
    parser.add_argument('--budget-seconds', type=float, default=21600)
    parser.add_argument('--reserve-seconds', type=float, default=7200)
    parser.add_argument('--validation', action='store_true', help='allow work inside the final validation/restoration reserve')
    parser.add_argument('--ready-model', default='ninfer-v100', help='preload whose readiness ends ledger accounting')
    parser.add_argument('--restore-timeout', type=float, default=900, help='seconds reserved for preload readiness')
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.program_manifest):
        parser.error('provide either a tuning manifest or --program-manifest')
    if args.validation and not args.budget_ledger:
        parser.error('--validation requires --budget-ledger')
    if not math.isfinite(args.restore_timeout) or args.restore_timeout <= 0:
        parser.error('--restore-timeout must be positive')
    if args.deadline is not None and not math.isfinite(args.deadline):
        parser.error('--deadline must be finite')
    run(args)


if __name__ == '__main__':
    main()
