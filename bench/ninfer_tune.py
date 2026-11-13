#!/usr/bin/env python3
"""Capacity probing and performance sweeps for ninfer-serve on the local V100.

Runs ninfer-serve straight from the dist bundle, so llama-swap must be stopped first.
Arguments after a bare `--` are ninfer-serve flags that override the chosen profile.

  ninfer_tune.py probe  --profile c1 -- --max-context 240000
  ninfer_tune.py search --profile c1 --lo 225000 --hi 262144
  ninfer_tune.py bench  --profile c1 --tests smoke,ladder -- --max-context 240000
  ninfer_tune.py bench  --profile c1 --quick-host --tests decode -- --max-context 98304 --draft-tokens 3
  ninfer_tune.py bench  --profile c2 --tests conc,interf,agent -- --max-context 190000
  ninfer_tune.py bench  --profile c1-final --tests smoke,ladder      # shipped C=1 config, 253952 and 3 checkpoint slots
  ninfer_tune.py bench  --profile c2-final --tests smoke,ladder,interf,conc,lineages --depths 8192,32768,65536
  ninfer_tune.py bench  --profile c2-shared --tests smoke,interf,conc,ladder --depths 65536   # one shared 256000 pool
  ninfer_tune.py report
  ninfer_tune.py bench --profile all-target --tests smoke,speed,cache,history -- --kv-capacity 253952
  ninfer_tune.py bench --profile all-final --tests smoke,speed --repeats 3

NInfer-all profiles select the pinned v3 runtime/artifact and explicit hybrid caching.
For a controlled cache A/B, --cache-mode original translates the profile to the
legacy catalog and omits hybrid persistence; the runtime and artifact stay the same.
Use --runtime/--launcher/--artifact/--runs-dir to select another bundle or output root.
Benchmark caches are private, saved on the models volume, then removed unless
--keep-cache is given. Production caches are rejected. ninfer_session.py runs a
sequential manifest during an authorized interruption and always restores llama-swap.
swap-check validates the configured production model through unload/reload and stop/start.

Tests: smoke, ladder (prefill/decode/vision/needles by depth), decode (multi-task decode speed and MTP
acceptance), conc (parallel lanes), interf (decode stall while the other lane prefills), agent (multi-turn
tool replay: TTFT and prefix-cache reuse), lineages (several long conversations plus aux requests: cache
retention). Runs land in $NINFER_TUNE_RUNS (rows.jsonl is authoritative).
Additional tests: speed (matched cold payloads and cached followups), cache (prompt
and tool-history mutations), history (edited archived tool sessions), restart
(orderly save, restoration/reuse, and incompatible binary-stamp rejection).
Running ladder,restart together restores the ladder's last text-only branch;
the incompatible-cache check then uses a separate short cold prompt.
"""
from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import hashlib
import http.client
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import struct
import sys
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable

ORIGINAL_ROOT = Path('/models/engine/geoffwatts-ninfer-v100')
ALL_ROOT = Path('/models/engine/iamwavecut-ninfer-all')
RUNS = Path(os.environ.get('NINFER_TUNE_RUNS', ORIGINAL_ROOT / 'bench-runs'))
DIST = ORIGINAL_ROOT / 'runtime/standard'
SRC = ORIGINAL_ROOT / 'source/standard'
ARTIFACT = "/models/ninfer/qwen3_8_27b_nvfp4.ninfer"
ALL_DIST = ALL_ROOT / 'runtime/active'
ALL_ARTIFACT = '/models/ninfer/qwen3_8_27b_nvfp4.v3.ninfer'
PRODUCTION_CACHE_ROOTS = (
    ALL_ROOT / 'cache', Path('/models/ninfer-all/cache'), ALL_ROOT / 'archive',
    ALL_ROOT / 'archive/legacy-models-ninfer-all/cache',
    Path('/models/.ninfer-all-rollback-20261029/cache'),
)
NVIDIA_LIB_DIRS = ("/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64")
CORPUS_ROOTS = (
    "/home/simon/comfyui/comfy",
    "/home/simon/comfyui/comfy_extras",
    "/home/simon/comfyui/comfy_api_nodes",
    "/home/simon/hermes-webui/api",
    "/home/simon/hermes-webui/static",
)
GIB = 1024**3
KIB_PER_TOKEN = 40.3  # int8 main KV + MTP pool, from serve.log capacity deltas
HEADROOM_GIB = 1.0  # --kv-capacity auto sizing headroom

PROFILES = {
    'all-final': '--max-context 253952 --kv-capacity auto --kv-headroom-mib 256 '
    '--max-concurrency 1 --kv-dtype int8 --use-alt-prefix-caching --host-cache-mib 16384 '
    '--device-snapshot-slots 2 --device-profile auto '
    f'--device-profile-path {ALL_ROOT}/profiles/ninfer-all-pg500-33163f0-20261029.json '
    '--spec mtp --draft-tokens 3 --ngram-draft-tokens 15 --lm-head-draft '
    '--preserve-thinking --vision --prefill-chunk 4096',
    'all-initial': '--max-context 131072 --kv-capacity auto --max-concurrency 1 --kv-dtype int8 '
    '--use-alt-prefix-caching --host-cache-mib 16384 --device-snapshot-slots 4 '
    '--device-profile off --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision --prefill-chunk 2048',
    'all-target': '--max-context 253952 --kv-capacity auto --max-concurrency 1 --kv-dtype int8 '
    '--use-alt-prefix-caching --host-cache-mib 16384 --device-snapshot-slots 4 '
    '--device-profile off --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision --prefill-chunk 2048',
    "c1": "--max-context 225000 --kv-capacity auto --max-concurrency 1 --kv-dtype int8 "
    "--device-state-slots 1 --host-state-slots 8 --host-kv-mib 16384 "
    "--spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision",
    "c2": "--max-context 190000 --kv-capacity auto --max-concurrency 2 --kv-dtype int8 "
    "--device-state-slots 2 --host-state-slots 8 --host-kv-mib 16384 "
    "--spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision",
    # shipped configs (llama-swap.yaml): explicit KV capacity, no auto headroom
    "c1-final": "--max-context 253952 --kv-capacity 253952 --max-concurrency 1 --kv-dtype int8 "
    "--device-state-slots 3 --host-state-slots 8 --host-kv-mib 16384 --max-private-continuations 4 "
    "--spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision --prefill-chunk 2048",
    "c2-final": "--max-context 139264 --kv-capacity 278528 --max-concurrency 2 --kv-dtype int8 "
    "--device-state-slots 2 --host-state-slots 8 --host-kv-mib 16384 "
    "--spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision",
    "c2-shared": "--max-context 256000 --kv-capacity 256000 --max-concurrency 2 --kv-dtype int8 "
    "--device-state-slots 2 --host-state-slots 8 --host-kv-mib 16384 "
    "--spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking --vision --prefill-chunk 2048",
}
BOOL_FLAGS = {
    '--use-alt-prefix-caching', '--use-original-prefix-caching', '--adaptive-mtp',
    "--vision", "--no-cuda-graph", "--no-prefix-reuse", "--lm-head-draft",
    "--no-thinking", "--preserve-thinking", "--cors", "--greedy",
}
QUICK_HOST = {"--host-kv-mib": "1024", "--host-state-slots": "2"}

LADDER = (4096, 16384, 32768, 65536, 98304, 131072, 163840, 196608, 229376)
Q_STEP = "Write a detailed technical analysis of the last file shown above, at least 400 words. Do not quote the code."
Q_VISION = (
    "Images follow in order. Read the number in the title banner of image 1, the number on the blue sign in "
    "image 2, and the title text of image 3 (any later image is only colored blocks). "
    "Answer on one line as: <number> | <number> | <title>."
)
Q_FINAL = (
    "Reply with only two values mentioned earlier in this conversation: the deployment "
    "passphrase for project KESTREL, then the maintenance code for BEACON."
)
Q_AGENT = "Investigate the code above with the tools and report briefly what you find."
DECODE_TASKS = (
    "Rewrite the last file shown above in full, with better docstrings and type hints.",
    "Write a detailed code review of the files above listing at least twelve concrete issues.",
    "List every function defined in the last file above as a JSON array with name, parameters and a one-line summary.",
    "Explain step by step the control flow of the last file above, then derive its time complexity.",
    "Translate the last file above into idiomatic Rust, keeping the structure.",
    "Write a README.md documenting how to install and use the code above, with examples.",
)
TOOL_NAMES = ("read_file", "write_file", "list_dir", "grep", "run_tests", "git_diff", "git_log", "shell", "fetch_url", "edit_file")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- server flags

def parse_flags(tokens: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        name = tokens[i]
        if not name.startswith("--"):
            raise SystemExit(f"unexpected token in server args: {name}")
        if name in BOOL_FLAGS:
            flags[name] = ""
            i += 1
        else:
            if i + 1 >= len(tokens):
                raise SystemExit(f"missing value for {name}")
            flags[name] = tokens[i + 1]
            i += 2
    return flags


def render_flags(flags: dict[str, str]) -> list[str]:
    out: list[str] = []
    for name, value in flags.items():
        out.append(name)
        if value != "":
            out.append(value)
    return out


# Plain decode keeps the finalist's cache, precision, context and vision policy;
# --spec none must not retain any explicit draft width or proposal-head flag.
PROFILES['all-plain'] = shlex.join(render_flags({
    name: ('none' if name == '--spec' else value)
    for name, value in parse_flags(PROFILES['all-final'].split()).items()
    if name not in ('--draft-tokens', '--ngram-draft-tokens', '--lm-head-draft')
}))


def build_flags(args: argparse.Namespace, server_args: list[str], quick: bool) -> dict[str, str]:
    flags = parse_flags(PROFILES[args.profile].split())
    cache_mode = getattr(args, 'cache_mode', None) or 'hybrid'
    if args.runtime == 'all' and cache_mode == 'original':
        # Explicit diagnostic A/B: translate the profile, never mix cache APIs.
        state_slots = flags.get('--device-snapshot-slots', '2')
        for name in ('--use-alt-prefix-caching', '--device-snapshot-slots',
                     '--cache-taps-per-request', '--cache-tap-ladder', '--cache-tap-min-gap',
                     '--prefix-cache-file'):
            flags.pop(name, None)
        flags.update({'--use-original-prefix-caching': '', '--device-state-slots': state_slots})
    flags["--pending-timeout-ms"] = "3600000"
    flags["--log-stats-interval-ms"] = "1000"
    if quick:
        flags.update({'--host-cache-mib': '1024'} if args.runtime == 'all' else QUICK_HOST)
    flags.update(parse_flags(server_args))
    for name in args.drop:
        flags.pop("--" + name.lstrip("-"), None)
    if args.runtime == 'all':
        legacy = set(flags) & {'--device-state-slots', '--host-state-slots', '--host-kv-mib',
                              '--max-private-continuations', '--no-prefix-reuse', '--use-original-prefix-caching',
                              '--slot-save-path', '--context-cache-disk-mib'}
        if cache_mode == 'original':
            hybrid = set(flags) & {'--use-alt-prefix-caching', '--device-snapshot-slots',
                                   '--prefix-cache-file', '--cache-taps-per-request',
                                   '--cache-tap-ladder', '--cache-tap-min-gap'}
            if hybrid or '--use-original-prefix-caching' not in flags:
                raise SystemExit(f'Original-cache diagnostic has hybrid-only flags: {sorted(hybrid)}')
        elif legacy or '--use-alt-prefix-caching' not in flags:
            raise SystemExit(f'NInfer-all benchmarks require explicit hybrid caching; incompatible flags: {sorted(legacy)}')
    return flags


# ------------------------------------------------------------------------- GPU

def gpu_state() -> dict[str, float]:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,power.limit', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise SystemExit('GPU monitoring unavailable: ' + (result.stderr or result.stdout).strip())
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise SystemExit('This harness requires exactly one visible GPU')
    used, total, free, limit = (float(x) for x in rows[0].split(','))
    return {'mem_used_mib': used, 'mem_total_mib': total, 'mem_free_mib': free, 'power_limit_w': limit}


def gpu_apps() -> list[dict[str, Any]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
        check=True,
    ).stdout
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                apps.append({"pid": int(parts[0]), "name": parts[1], "mib": float(parts[2])})
            except ValueError:
                pass
    return apps


def preflight(force: bool) -> None:
    state = gpu_state()  # --force never bypasses failed monitoring
    busy = gpu_apps()
    if busy and not force:
        desc = ", ".join(f"pid {a['pid']} {a['name']} {a['mib']:.0f} MiB" for a in busy)
        raise SystemExit(f"GPU is busy ({desc}). Stop llama-swap first, or pass --force.")
    if state['mem_used_mib'] > 64 and not force:
        raise SystemExit(f"GPU is not idle ({state['mem_used_mib']:.0f} MiB used). Stop other GPU users first.")


def wait_gpu_idle(baseline_mib: float, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gpu_state()["mem_used_mib"] <= baseline_mib + 500:
            return
        time.sleep(1.0)
    raise RuntimeError('GPU memory did not return to baseline after server stop')


class GpuMonitor:
    QUERY = "memory.used,power.draw,clocks.sm,temperature.gpu,clocks_event_reasons.active,memory.free"

    def __init__(self) -> None:
        self.samples: list[tuple[Any, ...]] = []
        self.total_mib = gpu_state()["mem_total_mib"]
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                reasons = int(parts[4], 16)
            except ValueError:
                reasons = 0
            self.samples.append((time.monotonic(), *(self._num(p) for p in parts[:4]), reasons,
                                 self._num(parts[5]) if len(parts) > 5 else None))

    @staticmethod
    def _num(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)

    def stats(self, t0: float, t1: float) -> dict[str, Any]:
        samples = list(self.samples)
        # A dead monitor must not make later requests pass using an old sample.
        rows = [s for s in samples if t0 - 1.0 <= s[0] <= t1 + 1.0]
        if not rows:
            return {}

        def col(i: int) -> list[float]:
            return [s[i] for s in rows if s[i] is not None]

        mem, power, sm, temp = col(1), col(2), col(3), col(4)
        return {
            "vram_max_mib": max(mem) if mem else None,
            'vram_free_min_mib': min(col(6)) if col(6) else None,
            "power_w": sum(power) / len(power) if power else None,
            "sm_mhz": sum(sm) / len(sm) if sm else None,
            "cap_pct": 100.0 * sum(1 for s in rows if s[5] & 0x4) / len(rows),
            "temp_c": max(temp) if temp else None,
        }


# ---------------------------------------------------------------------- server

CAP_RE = re.compile(
    r"capacity \| KV ([\d,]+) tokens, (\w+), (\w+) \| pages ([\d,]+)/([\d,]+) \| "
    r"runtime ([\d.]+) (GiB|MiB) \| free ([\d.]+) (GiB|MiB)"
)
WEIGHTS_RE = re.compile(r"weights ready \| ([\d.]+) GiB \| ([\d.]+)s")
READY_RE = re.compile(r"listening on http://[\d.]+:(\d+)")
CACHE_RE = re.compile(r"context cache \| (.*)")


def _gib(value: str, unit: str) -> float:
    return float(value) / (1024.0 if unit == "MiB" else 1.0)


def parse_capacity(text: str) -> dict[str, Any] | None:
    m = CAP_RE.search(text)
    if not m:
        return None
    kv, dtype, mode, pages, pages_max, runtime, runtime_unit, free, free_unit = m.groups()
    cap: dict[str, Any] = {
        "kv_tokens": int(kv.replace(",", "")), "kv_dtype": dtype, "kv_mode": mode,
        "pages": int(pages.replace(",", "")), "pages_max": int(pages_max.replace(",", "")),
        "runtime_gib": _gib(runtime, runtime_unit), "free_gib": _gib(free, free_unit),
    }
    if m := WEIGHTS_RE.search(text):
        cap["weights_gib"], cap["weights_s"] = float(m.group(1)), float(m.group(2))
    if m := CACHE_RE.search(text):
        cap["context_cache"] = m.group(1).strip()
    return cap


def pick_port() -> int:
    for port in range(18100, 18200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port in 18100-18199")


def server_group_event(pid: int, event: str, registry: Path | None = None) -> None:
    registry = registry or os.environ.get('NINFER_SESSION_SERVER_GROUPS')
    if registry:
        # One append per event, shared by this harness and its restoring session.
        with open(registry, 'a') as output:
            output.write(json.dumps({'pid': pid, 'event': event}) + '\n')


def registered_server_groups(registry: Path | None, offset: int = 0) -> dict[int, str]:
    groups: dict[int, str] = {}
    if registry is None or not registry.exists():
        return groups
    with registry.open('rb') as source:
        source.seek(offset)
        lines = source.read().decode('utf-8', 'replace').splitlines()
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(item, dict) and isinstance(item.get('pid'), int) and item['pid'] > 1 and
                item.get('event') in ('start', 'stopping', 'done')):
            if item['event'] == 'done':
                groups.pop(item['pid'], None)
            else:
                groups[item['pid']] = item['event']
    return groups


def process_group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Server:
    def __init__(self, run_dir: Path, flags: dict[str, str], args: argparse.Namespace | None = None) -> None:
        self.run_dir = run_dir
        self.port = pick_port()
        self.flags = dict(flags)
        self.runtime = getattr(args, 'runtime', 'original')
        self.launcher = getattr(args, 'launcher', None)
        self.artifact = getattr(args, 'artifact', None) or (ALL_ARTIFACT if self.runtime == 'all' else ARTIFACT)
        self.shutdown_timeout = getattr(args, 'shutdown_timeout', 150)
        self.shutdown: dict[str, Any] = {}
        self.owned_cache = False
        if (self.runtime == 'all' and '--use-original-prefix-caching' not in self.flags
                and '--no-prefix-reuse' not in self.flags):
            # Each configuration owns its cache. Never touch the production cache.
            if '--prefix-cache-file' not in self.flags:
                cache_dir = getattr(args, 'cache_dir', None)
                cache_root = Path(cache_dir) if cache_dir else RUNS
                self.flags['--prefix-cache-file'] = str(cache_root / run_dir.name / 'prefix.cache')
                self.owned_cache = True
        if self.runtime == 'all' and '--prefix-cache-file' in self.flags:
            cache = Path(self.flags['--prefix-cache-file']).resolve()
            cache_root = Path(getattr(args, 'cache_dir', None) or RUNS).resolve()
            if any(cache.is_relative_to(root.resolve()) for root in PRODUCTION_CACHE_ROOTS) or not (
                    cache.is_relative_to(RUNS.resolve()) or cache.is_relative_to(cache_root)):
                raise ValueError(f'benchmark cache must be under {RUNS} or {cache_root}: {cache}')
        if self.runtime == 'all':
            self.flags.setdefault('--device-profile-path', str(run_dir / 'device-profile.json'))
        self.flags.update({"--host": "127.0.0.1", "--port": str(self.port),
                           "--request-log-jsonl": str(run_dir / "server.jsonl")})
        self.log_path = run_dir / "server.log"
        self.proc: subprocess.Popen | None = None
        self.group_registry: Path | None = None
        self.group_registry_offset = 0
        self.text = ""
        self._pos = 0

    def command(self) -> list[str]:
        if self.launcher:
            return [self.launcher, self.artifact, *render_flags(self.flags)]
        nvidia = next((d for d in NVIDIA_LIB_DIRS if os.path.exists(f"{d}/libcuda.so.1")), "")
        if self.runtime == 'all':
            # The production wrapper redirects to serve.log; invoke the same loader here
            # with identical signal handling and a private log instead.
            return ['env', '--default-signal=INT,TERM', '--block-signal=INT,TERM',
                    str(ALL_DIST / 'ld-linux-x86-64.so.2'), '--library-path',
                    str(ALL_DIST / 'system-lib') + (f':{nvidia}' if nvidia else ''),
                    str(ALL_DIST / 'ninfer-serve'), self.artifact, *render_flags(self.flags)]
        libpath = f"{DIST}/lib" + (f":{nvidia}" if nvidia else "")
        return [str(DIST / "lib/ld-linux-x86-64.so.2"), "--library-path", libpath,
                str(DIST / "ninfer-serve"), self.artifact, *render_flags(self.flags)]

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if '--prefix-cache-file' in self.flags:
            Path(self.flags['--prefix-cache-file']).parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command()
        (self.run_dir / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env['NINFER_LOG_FILE'] = str(self.log_path)
        env['NINFER_DEVICE_PROFILES'] = self.flags.get('--device-profile-path', str(self.run_dir / 'device-profile.json'))
        self.group_registry = Path(env.get('NINFER_SESSION_SERVER_GROUPS', str(self.run_dir / 'process-groups.jsonl')))
        self.group_registry.touch(exist_ok=True)
        self.group_registry_offset = self.group_registry.stat().st_size
        env['NINFER_SESSION_SERVER_GROUPS'] = str(self.group_registry)
        (self.run_dir / 'runtime.json').write_text(json.dumps({'runtime': self.runtime,
            'artifact': self.artifact, 'command': cmd, 'shutdown_timeout': self.shutdown_timeout}, indent=2))
        with open(self.log_path, "wb") as out:
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        try:
            server_group_event(self.proc.pid, 'start', self.group_registry)
        except BaseException:
            # A registration failure must not orphan an untracked inference group.
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise

    def _pump(self) -> None:
        with open(self.log_path, "rb") as f:
            f.seek(self._pos)
            data = f.read()
        self._pos += len(data)
        self.text += data.decode("utf-8", "replace")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tail(self, lines: int = 25) -> str:
        self._pump()
        return "\n".join(self.text.splitlines()[-lines:])

    def wait(self, stage: str, timeout: float) -> dict[str, Any]:
        """stage 'capacity' returns once the ledger is printed, 'ready' once the server is healthy."""
        assert self.proc
        deadline = time.monotonic() + timeout
        cap_at: float | None = None
        while time.monotonic() < deadline:
            self._pump()
            cap = parse_capacity(self.text)
            if cap and cap_at is None:
                cap_at = time.monotonic()
            listening = READY_RE.search(self.text)
            if stage == "capacity" and cap and (listening or time.monotonic() - cap_at > 10):
                return {"ok": True, "capacity": cap}
            if listening and cap and stage == "ready":
                try:
                    status, body = http_json(self.port, "GET", "/health", timeout=10)
                except OSError:
                    status, body = 0, None
                if status == 200:
                    return {"ok": True, "capacity": cap}
            rc = self.proc.poll()
            if rc is not None:
                return {"ok": False, "capacity": cap, "returncode": rc, "error": self.tail()}
            time.sleep(0.5)
        return {"ok": False, "capacity": parse_capacity(self.text), "error": "startup timeout\n" + self.tail()}

    def stop(self) -> None:
        t0 = time.monotonic()
        forced = False
        if self.proc:
            pid = self.proc.pid
            owned = {pid}
            signalled: set[int] = set()
            deadline = time.monotonic() + self.shutdown_timeout
            while True:
                registered = registered_server_groups(self.group_registry, self.group_registry_offset)
                owned.update(registered)
                if os.getpgrp() in owned:
                    raise RuntimeError('refusing to signal the harness process group')
                # A profiler may give its application a different group. Stop
                # the application first and let nsys finish exporting naturally.
                targets = owned - {pid} or {pid}
                for target in targets - signalled:
                    if registered.get(target) != 'stopping':
                        server_group_event(target, 'stopping', self.group_registry)
                        try:
                            os.killpg(target, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    signalled.add(target)
                self.proc.poll()  # Reap the leader so its zombie does not hold the group.
                alive = [target for target in owned if process_group_alive(target)]
                if not alive:
                    break
                if time.monotonic() >= deadline:
                    forced = True
                    for target in alive:
                        try:
                            os.killpg(target, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(.1)
            self.proc.wait(timeout=10)
            for target in owned:
                if not process_group_alive(target):
                    server_group_event(target, 'done', self.group_registry)
            self._pump()
            self.shutdown = {'returncode': self.proc.returncode, 'forced': forced,
                             'seconds': time.monotonic() - t0,
                             'cache_saved': 'prefix cache saved:' in self.text}
            (self.run_dir / 'shutdown.json').write_text(json.dumps(self.shutdown, indent=2))

    def cleanup_cache(self, keep: bool = False) -> None:
        if not self.owned_cache:
            return
        cache = Path(self.flags['--prefix-cache-file'])
        if cache.exists():
            (self.run_dir / 'cache-file.json').write_text(json.dumps({
                'path': str(cache), 'bytes': cache.stat().st_size, 'retained': keep}, indent=2))
            if not keep:
                cache.unlink()


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pos = 0
        self.events: list[dict[str, Any]] = []
        self.cursor = 0

    def poll(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self.pos += end + 1
        new = []
        for line in data[: end + 1].splitlines():
            try:
                new.append(json.loads(line))
            except ValueError:
                pass
        self.events.extend(new)
        return new

    def skip_to_end(self) -> None:
        self.poll()
        self.cursor = len(self.events)

    def next_done(self, timeout: float = 20.0) -> dict[str, Any] | None:
        """Consume the next request_start and return its matching request_done/request_error."""
        deadline = time.monotonic() + timeout
        rid: Any = None
        started = False
        while True:
            self.poll()
            while self.cursor < len(self.events):
                ev = self.events[self.cursor]
                name = ev.get("event")
                if not started and name == "request_start":
                    rid, started = find_key(ev, "request_id"), True
                elif started and name in ("request_done", "request_error"):
                    if rid is None or find_key(ev, "request_id") == rid:
                        self.cursor += 1
                        return ev
                self.cursor += 1
            if time.monotonic() > deadline:
                return None
            time.sleep(0.2)


def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


# ------------------------------------------------------------------ HTTP client

def http_json(port: int, method: str, path: str, body: Any = None, timeout: float = 60.0) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers = {"Connection": "close"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, data, headers)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw.decode("utf-8", "replace")
    finally:
        conn.close()


def wait_model_ready(model: str, timeout: float) -> None:
    """Model-list 'loaded' means spawned in llama-swap; require real inference."""
    deadline = time.monotonic() + timeout
    payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
               'max_tokens': 1, 'temperature': 0, 'reasoning_effort': 'none', 'stream': False}
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            status, body = http_json(8082, 'POST', '/v1/chat/completions', payload,
                                     timeout=min(30.0, remaining))
            choices = body.get('choices') if isinstance(body, dict) else None
            if (status == 200 and isinstance(choices, list) and choices and
                    isinstance(choices[0], dict) and isinstance(choices[0].get('message'), dict)):
                return
        except (OSError, RuntimeError):
            pass
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f'restored service preload {model} did not complete a readiness request')


@dataclasses.dataclass
class ChatResult:
    t0: float = 0.0
    status: int = 0
    error: str = ""
    request_id: str = ""
    t_first: float | None = None
    t_end: float = 0.0
    content: list[str] = dataclasses.field(default_factory=list)
    reasoning: list[str] = dataclasses.field(default_factory=list)
    tool_calls: dict[int, dict[str, str]] = dataclasses.field(default_factory=dict)
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    timings: dict[str, Any] = dataclasses.field(default_factory=dict)
    finish: str = ""
    chunks: list[tuple[float, int | None]] = dataclasses.field(default_factory=list)
    aborted: bool = False

    @property
    def ok(self) -> bool:
        return (self.status == 200 and not self.error) or self.aborted

    @property
    def text(self) -> str:
        return "".join(self.content)

    @property
    def thinking(self) -> str:
        return "".join(self.reasoning)

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t0

    def calls(self) -> list[dict[str, Any]]:
        return [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for _, c in sorted(self.tool_calls.items())]


def chat_payload(model: str, messages: list[dict[str, Any]], max_tokens: int, *, effort: str = "none",
                 tools: list[dict[str, Any]] | None = None, per_token: bool = False, seed: int = 1234) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True}, "reasoning_effort": effort, "seed": seed,
        "return_progress": True,
    }
    if per_token:
        payload["timings_per_token"] = True
    if tools:
        payload["tools"] = tools
    return payload


def logged_metrics(done: dict[str, Any]) -> dict[str, Any]:
    """Both log schemas expose precise engine timings, even without SSE timings."""
    result, timings = done.get('result', {}), done.get('timings_seconds', {})
    computed, generated = result.get('computed_prefill_tokens'), result.get('completion_tokens')
    spec = done.get('speculative', {})
    return {
        'prompt_tokens': result.get('prompt_tokens'), 'completion_tokens': generated,
        'cache_n': result.get('prefix_cache_hit_tokens'), 'prompt_n': computed,
        'prefill_tps': computed / timings['prefill'] if computed and timings.get('prefill') else None,
        'decode_tps': generated / timings['decode'] if generated and timings.get('decode') else None,
        'engine_ttft_s': timings.get('ttft'), 'prefill_s': timings.get('prefill'),
        'decode_s': timings.get('decode'),
        'draft_acc': spec.get('accepted_tokens', 0) / spec['drafted_tokens'] if spec.get('drafted_tokens') else None,
        'ngram_drafted': spec.get('ngram_drafted_tokens'), 'ngram_accepted': spec.get('ngram_accepted_tokens'),
    }


def chat_stream(port: int, payload: dict[str, Any], *, abort_after_chunks: int | None = None,
                stop: threading.Event | None = None, on_first: Callable[[], None] | None = None,
                on_progress: Callable[[dict[str, Any]], None] | None = None) -> ChatResult:
    res = ChatResult(t0=time.monotonic())
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(payload).encode(),
                     {"Content-Type": "application/json", "Connection": "close"})
        resp = conn.getresponse()
        res.status = resp.status
        res.request_id = resp.getheader("x-request-id", "") or ""
        if resp.status != 200:
            res.error = resp.read(8192).decode("utf-8", "replace")
            return res
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            now = time.monotonic()
            if "error" in ev:
                res.error = json.dumps(ev["error"])
                break
            if ev.get("prompt_progress") and on_progress:
                on_progress(ev["prompt_progress"])
            visible = False
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    res.content.append(delta["content"])
                    visible = True
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    res.reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
                    visible = True
                for tc in delta.get("tool_calls") or []:
                    slot = res.tool_calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["name"] = fn.get("name") or slot["name"]
                    slot["arguments"] += fn.get("arguments") or ""
                    visible = True
                if choice.get("finish_reason"):
                    res.finish = choice["finish_reason"]
            if ev.get("usage"):
                res.usage = ev["usage"]
            if ev.get("timings"):
                res.timings = ev["timings"]
            if visible:
                if res.t_first is None:
                    res.t_first = now
                    if on_first:
                        on_first()
                res.chunks.append((now - res.t0, (ev.get("timings") or {}).get("predicted_n")))
                if abort_after_chunks and len(res.chunks) >= abort_after_chunks:
                    res.aborted = True
                    break
            if stop is not None and stop.is_set():
                res.aborted = True
                break
    finally:
        res.t_end = time.monotonic()
        conn.close()
    return res


def metrics(res: ChatResult) -> dict[str, Any]:
    t, u = res.timings or {}, res.usage or {}
    prompt_n, prompt_ms = t.get("prompt_n"), t.get("prompt_ms")
    draft_n, accepted = t.get("draft_n"), t.get("draft_n_accepted")
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "cache_n": t.get("cache_n"),
        "prompt_n": prompt_n,
        "ttft_s": res.ttft,
        "prefill_tps": prompt_n / (prompt_ms / 1000.0) if prompt_n and prompt_ms else None,
        "decode_tps": t.get("predicted_per_second") or None,
        "draft_acc": accepted / draft_n if draft_n else None,
        "finish": res.finish,
    }


def to_anthropic(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(p["text"] for p in content if p.get("type") == "text")
        if m["role"] == "system":
            system.append(content or "")
        elif m["role"] in ("user", "assistant") and content:
            out.append({"role": m["role"], "content": content})
    body: dict[str, Any] = {"messages": out}
    if system:
        body["system"] = "\n".join(system)
    return body


# ----------------------------------------------------------------------- corpus

class Corpus:
    EXTS = {".py", ".js", ".ts", ".css", ".html", ".md", ".yaml", ".yml"}

    def __init__(self, roots: tuple[str, ...], per_file: int = 20000, cap: int = 8_000_000) -> None:
        parts: list[str] = []
        total = 0
        for root in roots:
            for path in sorted(Path(root).rglob("*")):
                if total >= cap:
                    break
                if path.suffix not in self.EXTS or ".min." in path.name or not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if not text.strip() or max(map(len, text.splitlines())) > 400:
                    continue
                block = f"\n### FILE: {path}\n{text[:per_file]}\n"
                parts.append(block)
                total += len(block)
        self.text = "".join(parts)
        if len(self.text) < 2_000_000:
            raise SystemExit(f"corpus too small ({len(self.text)} chars); check CORPUS_ROOTS")

    def cursor(self, idx: int) -> "Cursor":
        pos = (idx * 1_000_003) % len(self.text)
        nl = self.text.find("\n", pos)
        return Cursor(self.text, nl + 1 if nl >= 0 else 0)


class Cursor:
    def __init__(self, text: str, pos: int) -> None:
        self.text, self.pos = text, pos

    def take(self, n: int) -> str:
        n = min(n, len(self.text) - 1)
        end = self.pos + n
        s = self.text[self.pos:end] if end <= len(self.text) else self.text[self.pos:] + self.text[: end - len(self.text)]
        self.pos = end % len(self.text)
        return s


def make_tools(n: int = 32) -> list[dict[str, Any]]:
    tools = []
    for i in range(n):
        name = TOOL_NAMES[i % len(TOOL_NAMES)]
        tools.append({"type": "function", "function": {
            "name": f"tool_{i:02d}_{name}",
            "description": f"Utility {i}: performs the {name} operation on the workspace and returns the result as text.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Target file or directory"},
                "query": {"type": "string", "description": "Search pattern or argument"}}, "required": ["path"]}}})
    return tools


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def synth_png(width: int, height: int, block: int = 64) -> str:
    rng = random.Random(5)
    per_row = width // block + 1
    colors = [bytes(rng.randrange(256) for _ in range(3)) for _ in range(per_row * (height // block + 1))]
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        base = (y // block) * per_row
        for x in range(0, width, block):
            raw += colors[base + x // block] * min(block, width - x)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def vision_ok(text: str) -> bool:
    return "731" in text and "24" in text and "left panel" in text.lower()


# ---------------------------------------------------------------------- context

class Ctx:
    def __init__(self, server: Server, mon: GpuMonitor, run_dir: Path, args: argparse.Namespace) -> None:
        self.server, self.mon, self.run_dir, self.args = server, mon, run_dir, args
        self.port = server.port
        self.jsonl = JsonlTail(run_dir / "server.jsonl")
        self.rows_path = run_dir / "rows.jsonl"
        self.nonce = uuid.uuid4().hex[:6]
        self.rng = random.Random(7)
        self.convs = 0
        self.failures: list[dict[str, Any]] = []
        self.lanes = int(server.flags.get("--max-concurrency", "1"))
        self.max_context = int(server.flags["--max-context"])
        self.vision = "--vision" in server.flags
        self.model = http_json(self.port, "GET", "/v1/models")[1]["data"][0]["id"]
        self.corpus = Corpus(CORPUS_ROOTS)
        self.cpt = 3.2
        media = SRC / "examples/cli/media"
        self.small = [png_uri(media / n) for n in ("visual_chart.png", "natural_scene.png", "compare_left.png")]
        self.small_tokens = 384 + 384 + 240 + 6  # one token per 32x32 px plus start/end markers
        self.big: str | None = None
        self.big_tokens = 0
        spec = getattr(args, "big_image", "none")
        if self.vision and spec != "none":
            w, h = (int(x) for x in spec.split("x"))
            self.big = synth_png(w, h)
            self.big_tokens = -(-w // 32) * -(-h // 32) + 2
        self._last_progress = 0.0
        self.calibrate()

    def calibrate(self) -> None:
        sample = self.corpus.cursor(999).take(40000)
        n = self.count([{"role": "user", "content": sample}])
        self.cpt = len(sample) / max(1, n - 20)
        (self.run_dir / 'corpus.json').write_text(json.dumps({'roots': CORPUS_ROOTS,
            'chars': len(self.corpus.text), 'sha256': hashlib.sha256(self.corpus.text.encode()).hexdigest(),
            'chars_per_token': self.cpt}, indent=2))
        log(f"calibrated {self.cpt:.2f} chars/token on corpus ({len(self.corpus.text):,} chars)")

    def count(self, messages: list[dict[str, Any]]) -> int:
        body = to_anthropic(messages)
        body["model"] = self.model
        status, js = http_json(self.port, "POST", "/v1/messages/count_tokens", body, timeout=600)
        if status != 200:
            raise RuntimeError(f"count_tokens failed: {status} {js}")
        return int(js["input_tokens"])

    def next_conv(self) -> int:
        self.convs += 1
        return self.convs - 1

    def emit(self, row: dict[str, Any]) -> None:
        if row.get('ok') is False or any(row.get(k) is False for k in ('vision_ok', 'needle_a', 'needle_b')):
            self.failures.append(row)
        with open(self.rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(one_line(row))

    def progress(self, pp: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last_progress > 30 and pp.get("total"):
            self._last_progress = now
            log(f"  prefill {pp.get('processed', 0):,}/{pp['total']:,} ({pp.get('time_ms', 0) / 1000:.0f}s)")

    def chat(self, payload: dict[str, Any], test: str, *, abort_after_chunks: int | None = None,
             **extra: Any) -> tuple[ChatResult, dict[str, Any]]:
        self.jsonl.skip_to_end()
        t0 = time.monotonic()
        res = chat_stream(self.port, payload, abort_after_chunks=abort_after_chunks, on_progress=self.progress)
        t1 = time.monotonic()
        row: dict[str, Any] = {"test": test, **extra, **metrics(res), **self.mon.stats(t0, t1),
                               "wall_s": t1 - t0, "ok": res.ok}
        row['payload_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        row['tool_choice'] = payload.get('tool_choice', 'auto')
        row['tool_count'] = len(payload.get('tools', []))
        if res.status != 200:
            row["error"] = res.error[:300]
        else:
            done = self.jsonl.next_done()
            if done:
                row.update({k: v for k, v in logged_metrics(done).items() if v is not None})
                row["path"] = find_key(done, "prefix_reuse_path")
                row["queue_s"] = find_key(done, "queue_wait_seconds")
                host = find_key(done, "host_exposed_seconds")
                row["host_s"] = host.get("total") if isinstance(host, dict) else None
                row['log_schema'] = done.get('schema_version')
        if row.get('prompt_tokens'):
            row['cache_fraction'] = (row.get('cache_n') or 0) / row['prompt_tokens']
        if self.args.min_free_mib:
            free = row.get('vram_free_min_mib')
            row['memory_margin_ok'] = free is not None and free >= self.args.min_free_mib
            if not row['memory_margin_ok']:
                row['ok'] = False
        self.emit(row)
        return res, row


def one_line(row: dict[str, Any]) -> str:
    parts = [f"[{row.get('test')}]"]
    parts += [f"{k}={row[k]}" for k in ("variant", "kind", "lineage", "step", "turn") if k in row]

    def add(key: str, label: str, spec: str, unit: str = "") -> None:
        if row.get(key) is not None:
            parts.append(f"{label}={format(row[key], spec)}{unit}")

    add("prompt_tokens", "prompt", ",d")
    add("cache_n", "cache", ",d")
    add("ttft_s", "ttft", ".1f", "s")
    add("prefill_tps", "prefill", ".0f", "t/s")
    add("decode_tps", "decode", ".1f", "t/s")
    add("completion_tokens", "out", ",d")
    add("draft_acc", "acc", ".0%")
    add("agg_decode_tps", "agg", ".1f", "t/s")
    add("batch", "batch", ".2f")
    add("vram_max_mib", "vram", ",.0f", "MiB")
    add("vram_free_min_mib", "free", ",.0f", "MiB")
    add("sm_mhz", "sm", ".0f", "MHz")
    add("cap_pct", "cap", ".0f", "%")
    if row.get("path"):
        parts.append(f"path={row['path']}")
    if row.get("finish"):
        parts.append(f"finish={row['finish']}")
    for key in ("needle_a", "needle_b", "vision_ok", "turns", "misses", "error"):
        if key in row:
            parts.append(f"{key}={row[key]}")
    return " ".join(parts)


class Conv:
    """Chat history that grows by corpus slabs, so each request extends the previous one."""

    def __init__(self, ctx: Ctx, system: str | None = None, tools: list[dict[str, Any]] | None = None,
                 fixed_idx: int | None = None) -> None:
        self.ctx = ctx
        self.idx = ctx.next_conv() if fixed_idx is None else fixed_idx
        self.cursor = ctx.corpus.cursor(self.idx)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        self.tools = tools
        self.seen = 0
        self.offset = 0
        self.text_count = 0
        self.slab_chars = 0
        self.header = f"[session {ctx.nonce}-{self.idx}]\n"
        self.last_calls: list[dict[str, Any]] = []

    def _message(self, chars: int, question: str, images: tuple[str, ...], inject: str | None,
                 role: str, call_id: str | None) -> dict[str, Any]:
        slab = self.cursor.take(chars)
        if inject:
            mid = slab.find("\n", len(slab) // 2)
            slab = slab[: mid + 1] + inject + "\n" + slab[mid + 1:]
        self.slab_chars = len(slab)
        head = self.header if not any(m["role"] == "user" for m in self.messages) else ""
        text = f"{head}{slab}\n\n{question}"
        if role == "tool":
            return {"role": "tool", "tool_call_id": call_id, "content": text}
        if images:
            return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images]
                    + [{"type": "text", "text": text}]}
        return {"role": "user", "content": text}

    def add(self, *, question: str, tokens: int | None = None, target: int | None = None,
            images: tuple[str, ...] = (), inject: str | None = None, role: str = "user",
            call_id: str | None = None, precise: bool = False) -> None:
        """Append a turn of `tokens` new tokens, or sized so the whole prompt is ~`target` tokens."""
        ctx = self.ctx
        start = self.cursor.pos
        need = (target - self.seen) if target is not None else (tokens or 0)
        chars = max(400, int(need * ctx.cpt))
        if target is None:
            self.messages.append(self._message(chars, question, images, inject, role, call_id))
            return
        prev_chars, prev_n = None, None
        for _ in range(6):
            self.cursor.pos = start
            msg = self._message(chars, question, images, inject, role, call_id)
            text_n = ctx.count(self.messages + [msg])
            n = text_n + self.offset
            err = target - n
            if (0 <= err <= 96) if precise else (abs(err) <= max(64, 0.012 * target)):
                break
            if prev_chars is not None and n != prev_n:
                ctx.cpt = max(1.0, min(8.0, (chars - prev_chars) / (n - prev_n)))
            prev_chars, prev_n = chars, n
            chars = max(400, chars + int((err - (48 if precise else 0)) * ctx.cpt))
        self.messages.append(msg)
        self.text_count = text_n

    def commit(self, res: ChatResult) -> None:
        pt, ct = res.usage.get("prompt_tokens"), res.usage.get("completion_tokens", 0)
        if pt:
            if self.text_count:
                self.offset = pt - self.text_count
            self.seen = pt + ct
        msg: dict[str, Any] = {"role": "assistant", "content": res.text or None}
        if res.thinking:
            msg["reasoning_content"] = res.thinking
        self.last_calls = res.calls() if res.finish == "tool_calls" else []
        if self.last_calls:
            msg["tool_calls"] = self.last_calls
        elif not res.text:
            msg["content"] = ""
        self.messages.append(msg)

    def followup(self, tokens: int, question: str) -> None:
        if self.last_calls:
            first, rest = self.last_calls[0], self.last_calls[1:]
            self.add(question=question, tokens=tokens, role="tool", call_id=first["id"])
            for call in rest:
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": "ok"})
        else:
            self.add(question=question, tokens=tokens)


# ------------------------------------------------------------------------ tests

def resolve_depths(text: str, ctx: Ctx, tokens: int) -> list[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(ctx.max_context - tokens - 160 if item == "max" else int(item))
    return out


def test_smoke(ctx: Ctx, args: argparse.Namespace) -> None:
    ctx.chat(chat_payload(ctx.model, [{"role": "user", "content": "Say hi."}], 16), "smoke", kind="warmup")
    conv = Conv(ctx)
    conv.add(target=2000, question="Summarize the purpose of the last file in two sentences.")
    ctx.chat(chat_payload(ctx.model, conv.messages, 128), "smoke", kind="text")
    if ctx.vision:
        messages = [{"role": "user", "content": [
            *({"type": "image_url", "image_url": {"url": u}} for u in ctx.small), {"type": "text", "text": Q_VISION}]}]
        res, _ = ctx.chat(chat_payload(ctx.model, messages, 200), "smoke", kind="vision")
        ctx.emit({"test": "smoke", "kind": "vision-check", "vision_ok": vision_ok(res.text), 'answer': res.text})


def test_ladder(ctx: Ctx, args: argparse.Namespace) -> None:
    final_tokens = 128
    steps = resolve_depths(args.depths, ctx, 0) if args.depths else [d for d in LADDER if d < ctx.max_context - 16384]
    plan = [("step", d) for d in steps]
    if not args.ladder_light:
        if ctx.vision:
            plan.append(("vision", ctx.max_context - 8192))
        plan.append(("final", ctx.max_context - final_tokens - 160))
    needle_a = f"ember-quartz-{ctx.rng.randint(1000, 9999)}"
    needle_b = f"{ctx.rng.randint(1000, 9999)}-delta"
    line_a = f"# NOTE: the deployment passphrase for project KESTREL is {needle_a}."
    line_b = f"# NOTE: the maintenance code for BEACON is {needle_b}."
    inner = range(1, len(plan) - 1)
    mid = min(inner, key=lambda i: abs(plan[i][1] - plan[-1][1] / 2)) if len(plan) > 2 else 0
    conv = Conv(ctx)
    for i, (kind, target) in enumerate(plan):
        inject = None if args.ladder_light else "\n".join(line for line, at in ((line_a, 0), (line_b, mid)) if at == i) or None
        question = {"step": Q_STEP, "vision": Q_VISION, "final": Q_FINAL}[kind]
        images = (*ctx.small, *([ctx.big] if ctx.big else [])) if kind == "vision" else ()
        image_tokens = ctx.small_tokens + ctx.big_tokens if kind == "vision" else 0
        log(f"ladder {kind} target {target:,}: fitting prompt")
        conv.add(target=target - image_tokens, question=question, images=images,
                 inject=inject, precise=(kind == "final"))
        out = final_tokens if kind == "final" else args.new_tokens
        res, row = ctx.chat(chat_payload(ctx.model, conv.messages, out), "ladder", kind=kind, step=i, target=target)
        if not res.ok or res.status != 200:
            log("ladder aborted: request failed")
            return
        conv.commit(res)
        answer = res.text + res.thinking
        if kind == 'step' and not args.ladder_light:
            # Keep the latest text-only branch: disk restoration can be checked
            # at depth without relying on process-local prepared media identities.
            ctx.restart_payload = chat_payload(ctx.model, copy.deepcopy(conv.messages) +
                [{'role': 'user', 'content': Q_FINAL}], 128)
            ctx.restart_expected = tuple(value for value, at in ((needle_a, 0), (needle_b, mid)) if i >= at)
        if kind == "vision":
            ctx.emit({"test": "ladder", "kind": "vision-check", "vision_ok": vision_ok(answer), 'answer': answer})
        if kind == "final":
            ctx.emit({"test": "ladder", "kind": "needle-check", "needle_a": needle_a in answer,
                      "needle_b": needle_b in answer, 'answer': answer, 'expected': [needle_a, needle_b]})


def test_decode(ctx: Ctx, args: argparse.Namespace) -> None:
    conv = Conv(ctx, fixed_idx=700)
    for depth in (int(x) for x in args.decode_depths.split(",")):
        tokens, seconds, drafted, accepted, last_prompt = 0, 0.0, 0, 0, 0
        for i, task in enumerate(DECODE_TASKS[: args.decode_tasks]):
            if i == 0:
                conv.add(target=depth, question=task)
            else:
                conv.add(tokens=120, question=task)
            res, row = ctx.chat(chat_payload(ctx.model, conv.messages, args.decode_tokens), "decode",
                                kind="task", step=i, target=depth)
            if not res.ok or res.status != 200:
                return
            conv.commit(res)
            timing = res.timings or {}
            n, tps = res.usage.get("completion_tokens"), timing.get("predicted_per_second")
            if n and tps:
                tokens += n
                seconds += n / tps
            drafted += timing.get("draft_n") or 0
            accepted += timing.get("draft_n_accepted") or 0
            last_prompt = row.get("prompt_tokens") or last_prompt
        ctx.emit({"test": "decode", "kind": "summary", "target": depth, "prompt_tokens": last_prompt,
                  "completion_tokens": tokens, "decode_tps": tokens / seconds if seconds else None,
                  "draft_acc": accepted / drafted if drafted else None})


def steady_decode(events: list[dict[str, Any]], lanes: int) -> dict[str, Any]:
    full, any_batch = [], []
    for ev in events:
        if ev.get("event") != "throughput":
            continue
        try:
            rounds = int(ev["decode_batch"]["rounds"])
            row_rounds = int(ev["decode_batch"]["row_rounds"])
            tokens = int(ev["tokens"]["committed_decode"])
            seconds = float(ev["interval_seconds"])
            sched = ev["scheduler"]
        except (KeyError, TypeError, ValueError):
            continue
        if rounds <= 0:
            continue
        any_batch.append((tokens, seconds, rounds, row_rounds))
        if (int(ev["tokens"]["computed_prefill"]) == 0 and row_rounds == lanes * rounds
                and int(sched["running"]) == lanes and int(sched["prefilling"]) == 0):
            full.append((tokens, seconds, rounds, row_rounds))
    use = full or any_batch
    seconds = sum(s for _, s, _, _ in use)
    rounds = sum(r for _, _, r, _ in use)
    return {
        "agg_decode_tps": sum(t for t, _, _, _ in use) / seconds if seconds else None,
        "batch": sum(rr for _, _, _, rr in use) / rounds if rounds else None,
        "steady_intervals": len(full),
    }


def test_conc(ctx: Ctx, args: argparse.Namespace) -> None:
    if ctx.lanes < 2:
        log("conc skipped: --max-concurrency is 1")
        return
    for depth in resolve_depths(args.conc_depths, ctx, args.conc_tokens):
        convs = [Conv(ctx) for _ in range(ctx.lanes)]
        for c in convs:
            c.add(target=depth, question=Q_STEP)
        payloads = [chat_payload(ctx.model, c.messages, args.conc_tokens, seed=1000 + i) for i, c in enumerate(convs)]
        results: list[ChatResult | None] = [None] * len(convs)
        ctx.jsonl.skip_to_end()
        barrier = threading.Barrier(len(convs))

        def run(i: int) -> None:
            barrier.wait()
            results[i] = chat_stream(ctx.port, payloads[i], on_progress=ctx.progress if i == 0 else None)

        log(f"conc: {len(convs)} simultaneous requests at depth {depth:,}")
        t0 = time.monotonic()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(convs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t1 = time.monotonic()
        time.sleep(1.5)
        ctx.jsonl.poll()
        stats = ctx.mon.stats(t0, t1)
        for i, res in enumerate(results):
            assert res is not None
            ctx.emit({"test": "conc", "kind": "stream", "step": i, "target": depth, **metrics(res), "ok": res.ok,
                      **({"error": res.error[:300]} if res.status != 200 else {})})
        ctx.emit({"test": "conc", "kind": "summary", "target": depth, "wall_s": t1 - t0, **stats,
                  **steady_decode(ctx.jsonl.events[ctx.jsonl.cursor:], ctx.lanes)})
        if any(r is None or not r.ok for r in results):
            return


def rate(chunks: list[tuple[float, int | None]], t0: float, t1: float) -> float | None:
    def cum(t: float) -> int:
        total = 0
        for at, n in chunks:
            if at > t:
                break
            total = n if n is not None else total
        return total
    return (cum(t1) - cum(t0)) / (t1 - t0) if t1 - t0 >= 2.0 else None


def test_interf(ctx: Ctx, args: argparse.Namespace) -> None:
    if ctx.lanes < 2:
        log("interf skipped: --max-concurrency is 1")
        return
    a, b = Conv(ctx), Conv(ctx)
    a.add(target=args.interf_a, question=Q_STEP)
    b.add(target=args.interf_b, question=Q_STEP)
    pa = chat_payload(ctx.model, a.messages, 4096, per_token=True)
    pb = chat_payload(ctx.model, b.messages, 64, seed=99)
    first, stop = threading.Event(), threading.Event()
    box: dict[str, ChatResult] = {}
    ctx.jsonl.skip_to_end()
    t0 = time.monotonic()
    ta = threading.Thread(target=lambda: box.__setitem__("a", chat_stream(ctx.port, pa, stop=stop, on_first=first.set)))
    ta.start()
    first.wait(timeout=3600)
    time.sleep(6.0)
    t_b = time.monotonic()
    log(f"interf: A decoding at depth {args.interf_a:,}; sending B (cold {args.interf_b:,})")
    res_b = chat_stream(ctx.port, pb, on_progress=ctx.progress)
    time.sleep(8.0)
    stop.set()
    ta.join()
    res_a = box["a"]
    base = res_a.t0
    tb0, tb1 = t_b - base, t_b - base + (res_b.ttft or 0.0)
    ta0, te = (res_a.t_first or 0.0) + 1.0, res_a.chunks[-1][0] if res_a.chunks else 0.0
    gaps = [y[0] - x[0] for x, y in zip(res_a.chunks, res_a.chunks[1:]) if tb0 <= y[0] <= tb1 + 5]
    ctx.emit({"test": "interf", "kind": "a-alone", "target": args.interf_a, "decode_tps": rate(res_a.chunks, ta0, tb0)})
    ctx.emit({"test": "interf", "kind": "a-during-b-prefill", "target": args.interf_b,
              "decode_tps": rate(res_a.chunks, tb0, min(tb1, te)), "max_gap_s": max(gaps) if gaps else None})
    ctx.emit({"test": "interf", "kind": "a-both-decoding", "decode_tps": rate(res_a.chunks, tb1, te)})
    ctx.emit({"test": "interf", "kind": "b", "target": args.interf_b, **metrics(res_b), **ctx.mon.stats(t0, time.monotonic())})


def agent_system(ctx: Ctx) -> str:
    guide = Conv(ctx, fixed_idx=500).cursor.take(int(6000 * ctx.cpt))
    return "You are a coding agent working in a repository. Use the tools to investigate. Keep each step brief.\n\n" \
           "Repository guidelines and reference material:\n" + guide


def test_agent(ctx: Ctx, args: argparse.Namespace) -> None:
    tools = make_tools()
    system = agent_system(ctx)
    for variant in args.agent_variants.split(","):
        log(f"agent variant {variant}")
        conv = Conv(ctx, system=system, tools=tools)
        conv.add(tokens=max(2000, args.agent_start - 9500), question=Q_AGENT)  # system + 32 tools ~9.5K
        for turn in range(args.agent_turns):
            if turn > 0 and "interleave" in variant:
                title = [{"role": "user", "content": "Write a 3-5 word title for this chat: " + Cursor(ctx.corpus.text, 12345).take(900)}]
                ctx.chat(chat_payload(ctx.model, title, 1024, effort="low"), "agent", variant=variant, kind="distract-title", turn=turn)
            if turn > 0 and "vision" in variant and ctx.vision:
                vision = [{"role": "user", "content": [
                    *({"type": "image_url", "image_url": {"url": u}} for u in ctx.small), {"type": "text", "text": Q_VISION}]}]
                ctx.chat(chat_payload(ctx.model, vision, 400), "agent", variant=variant, kind="distract-vision", turn=turn)
            payload = chat_payload(ctx.model, conv.messages, 1024, effort="low", tools=tools)
            if turn == args.agent_turns // 2 and "cancel" in variant:
                ctx.chat(payload, "agent", abort_after_chunks=1, variant=variant, kind="aborted", turn=turn)
            res, _ = ctx.chat(payload, "agent", variant=variant, kind="turn", turn=turn)
            if res.status != 200:
                break
            conv.commit(res)
            conv.followup(args.agent_step, Q_AGENT)


def test_lineages(ctx: Ctx, args: argparse.Namespace) -> None:
    tools = make_tools()
    system = agent_system(ctx)
    sizes = [int(x) for x in args.lineage_sizes.split(",")]
    convs = [Conv(ctx, system=system, tools=tools) for _ in sizes]
    for i, conv in enumerate(convs):
        conv.header = f'[lineage sample {i}]\n'
    turns: list[dict[str, Any]] = []

    def step(i: int, rnd: int) -> bool:
        conv = convs[i]
        res, row = ctx.chat(chat_payload(ctx.model, conv.messages, 512, effort="low", tools=tools), "lineages",
                            kind="turn", lineage=i, turn=rnd)
        if res.status != 200:
            return False
        turns.append(row)
        conv.commit(res)
        conv.followup(args.agent_step, Q_AGENT)
        return True

    for i, (conv, size) in enumerate(zip(convs, sizes)):
        conv.add(tokens=max(2000, size - 9500), question=Q_AGENT)
        if not step(i, 0):
            return
    for rnd in range(1, args.lineage_rounds + 1):
        for i in range(len(convs)):
            text = Cursor(ctx.corpus.text, 40000 + 997 * (rnd * 16 + i)).take(900)
            title = [{"role": "user", "content": f"Write a 3-5 word title for this chat [lineage-{rnd}-{i}]: {text}"}]
            ctx.chat(chat_payload(ctx.model, title, 256, effort="low"), "lineages", kind="aux", lineage=i, turn=rnd)
            if not step(i, rnd):
                return
    later = [r for r in turns if r["turn"] > 0]
    misses = [r for r in later if (r.get("cache_n") or 0) < 0.5 * (r.get("prompt_tokens") or 1)]
    ctx.emit({"test": "lineages", "kind": "summary", "turns": len(later), "misses": len(misses),
              'fixture_version': 2,
              "ttft_s": sum(r.get("ttft_s") or 0 for r in later) / max(1, len(later)),
              "prompt_tokens": max(r.get("prompt_tokens") or 0 for r in later)})


def test_speed(ctx: Ctx, args: argparse.Namespace) -> None:
    """Identical saved payloads across runtimes; new leading nonce per cold sample."""
    fixture_dir = Path(args.fixtures)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for depth in map(int, args.speed_depths.split(',')):
            fixture = fixture_dir / f'speed-{depth}-r{repeat}.json'
            if not fixture.exists():
                conv = Conv(ctx, fixed_idx=800 + repeat * 10 + depth // 8192)
                conv.header = f'[speed sample {repeat} depth {depth}]\n'
                conv.add(target=depth, question=DECODE_TASKS[0], precise=True)
                payload = chat_payload('qwen3.8-27b', conv.messages, args.speed_tokens)
                payload.update(temperature=0, presence_penalty=0, frequency_penalty=0)
                fixture.write_text(json.dumps(payload))
            payload = json.loads(fixture.read_text())
            payload['max_tokens'] = args.speed_tokens
            payload['model'] = ctx.model
            res, row = ctx.chat(payload, 'speed', kind='cold', target=depth, repeat=repeat)
            enough = (row.get('completion_tokens') or 0) >= 256
            cold = (row.get('cache_n') or 0) < 128
            ctx.emit({'test': 'speed-check', 'target': depth, 'repeat': repeat,
                      'ok': res.ok and enough and cold, 'enough_output': enough, 'cold': cold})
            if not res.ok:
                raise RuntimeError('speed request failed')
            follow = copy.deepcopy(payload)
            follow['messages'].extend([{'role': 'assistant', 'content': res.text},
                                      {'role': 'user', 'content': 'Continue the rewrite in full.'}])
            follow['max_tokens'] = 64
            ctx.chat(follow, 'speed', kind='followup', target=depth, repeat=repeat)


def test_cache(ctx: Ctx, args: argparse.Namespace) -> None:
    """Stable recurrence, changed tool results/definitions, and earlier history edits."""
    conv = Conv(ctx, system='Use the reference files below. Answer briefly; do not call tools.', tools=make_tools())
    conv.header = '[cache interface test]\n'
    conv.add(target=args.cache_depth, question='Describe the purpose of the last file in one sentence.')
    base = chat_payload(ctx.model, conv.messages, 64, tools=conv.tools)
    base.update(temperature=0, tool_choice='auto')
    res, _ = ctx.chat(base, 'cache', kind='initial')
    if not res.ok:
        raise RuntimeError('cache initial request failed')
    ctx.chat(base, 'cache', kind='identical')
    conv.commit(res)
    for call in conv.last_calls:
        conv.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'No additional result required.'})
    conv.messages.extend([
        {'role': 'user', 'content': 'Read the file and summarize it.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_cache_1', 'type': 'function',
            'function': {'name': conv.tools[0]['function']['name'], 'arguments': '{"path":"app.py"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_cache_1', 'content': 'Version A: def main(): return 17'},
    ])
    baseline = chat_payload(ctx.model, conv.messages, 64, tools=conv.tools)
    baseline.update(temperature=0, tool_choice='auto')
    ctx.chat(baseline, 'cache', kind='tool-session')
    for kind in ('changed-result', 'reordered-tools', 'added-tool', 'removed-tool', 'edited-history'):
        # Re-prime the same ancestor so each mutation has an equal starting point.
        ctx.chat(baseline, 'cache', kind='reprime', variant=kind)
        changed = copy.deepcopy(baseline)
        if kind == 'changed-result':
            changed['messages'][-1]['content'] = 'Version B: def main(): return 29'
        elif kind == 'reordered-tools':
            changed['tools'].reverse()
        elif kind == 'added-tool':
            changed['tools'].append(make_tools(33)[-1])
        elif kind == 'removed-tool':
            changed['tools'].pop()
        else:
            text = changed['messages'][1]['content']
            midpoint = len(text) // 2
            changed['messages'][1]['content'] = text[:midpoint] + '\nHistory edited here.\n' + text[midpoint:]
        ctx.chat(changed, 'cache', kind=kind)
    growing = copy.deepcopy(baseline)
    for turn in range(3):
        result, _ = ctx.chat(growing, 'cache', kind='growing-session', turn=turn)
        reply = {'role': 'assistant', 'content': result.text or None}
        calls = result.calls()
        if calls:
            reply['tool_calls'] = calls
        growing['messages'].append(reply)
        for call in calls:
            growing['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Result: no further action needed.'})
        growing['messages'].append({'role': 'user',
            'content': ctx.corpus.cursor(1500 + turn).take(6000) + '\nSummarize briefly.'})
    changed = copy.deepcopy(growing)
    # Editing a later turn preserves a full earlier turn, unlike an edit inside
    # the first large message. This exposes the value of history snapshots.
    later_users = [m for m in changed['messages'][2:] if m['role'] == 'user' and len(m.get('content', '')) > 1000]
    if later_users:
        later_users[0]['content'] = 'Edited earlier turn.\n' + later_users[0]['content']
        ctx.chat(changed, 'cache', kind='edited-earlier-turn')


def test_history(ctx: Ctx, args: argparse.Namespace) -> None:
    """Edit an archived tool session without pre-existing intermediate endpoints."""
    tools = make_tools()
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'Summarize the recorded tool history briefly; do not call tools.\n' +
         ctx.corpus.cursor(1600).take(6000)},
        {'role': 'user', 'content': '[archived history snapshot test] Assess these recorded tool results.'},
    ]
    for turn in range(4):
        call_id = f'archive_{turn}'
        messages.extend([
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call_id, 'type': 'function',
                'function': {'name': tools[0]['function']['name'], 'arguments': '{"path":"app.py"}'}}]},
            {'role': 'tool', 'tool_call_id': call_id, 'content': ctx.corpus.cursor(1700 + turn).take(4000)},
        ])
    payload = chat_payload(ctx.model, messages, 32, tools=tools)
    payload['temperature'] = 0
    ctx.chat(payload, 'history', kind='cold-archive')
    changed = copy.deepcopy(payload)
    results = [m for m in changed['messages'] if m['role'] == 'tool']
    results[2]['content'] = 'Edited earlier result.\n' + results[2]['content']
    ctx.chat(changed, 'history', kind='edited-archive')


def incompatible_cache(source: Path, target: Path) -> None:
    """Preserve a valid v3 cache, changing only its binary compatibility stamp."""
    with source.open('rb') as src, target.open('wb') as dst:
        prefix = src.read(16)
        if len(prefix) != 16 or prefix[:8] != b'NINFHPC1':
            raise ValueError('unrecognized hybrid cache header')
        version, length = struct.unpack('<II', prefix[8:])
        if version != 3 or not 0 < length <= 65536:
            raise ValueError('unsupported hybrid cache version')
        header = bytearray(src.read(length))
        start = header.find(b';build=')
        if start < 0 or len(header) != length:
            raise ValueError('hybrid cache has no build compatibility stamp')
        start += len(b';build=')
        header[start] = ord('0') if header[start] != ord('0') else ord('1')
        dst.write(prefix)
        dst.write(header)
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def cache_rejection_logged(text: str) -> bool:
    return ('prefix cache not restored | saved prefix cache belongs to a different model, KV format or build' in text
            and 'prefix cache restored |' not in text and 'prefix cache partly restored |' not in text)


def start_rejected_cache(ctx: Ctx, args: argparse.Namespace, flags: dict[str, str],
                         cache: Path, name: str, reason: str) -> Server:
    trial = dict(flags, **{'--prefix-cache-file': str(cache)})
    server = Server(ctx.run_dir / name, trial, args)
    ctx.server = server  # The outer finally owns cleanup, including a failed startup.
    server.start()
    info = server.wait('ready', args.startup_timeout)
    (server.run_dir / 'startup.json').write_text(json.dumps(info, indent=2))
    if not info['ok']:
        raise RuntimeError(f'incompatible cache prevented safe startup: {info}')
    ctx.port = server.port
    ctx.jsonl = JsonlTail(server.run_dir / 'server.jsonl')
    rejected = cache_rejection_logged(server.text)
    ctx.emit({'test': 'restart', 'kind': name, 'incompatibility': reason,
              'ok': rejected, 'cache': str(cache)})
    if not rejected:
        raise RuntimeError(f'no explicit compatibility rejection for {reason}; a cold unrelated prompt is insufficient')
    return server


def test_kernel_policy_cache_rejection(ctx: Ctx, args: argparse.Namespace,
                                       flags: dict[str, str], cache: Path) -> None:
    mode = flags.get('--v100-kernels')
    if mode not in ('native', 'ported'):
        return
    other = 'ported' if mode == 'native' else 'native'
    source = cache
    for index, (source_mode, target_mode) in enumerate(((mode, other), (other, mode))):
        isolated = cache.parent / f'policy-{source_mode}-to-{target_mode}.cache'
        shutil.copy2(source, isolated)  # Preserve the primary valid saved cache byte-for-byte.
        ctx.restart_cache_files.append(isolated)
        trial = dict(flags, **{'--v100-kernels': target_mode})
        server = start_rejected_cache(ctx, args, trial, isolated,
                                      f'policy-{source_mode}-to-{target_mode}',
                                      f'kernel policy {source_mode} -> {target_mode}')
        if index == 0:
            # Produce a real small opposite-policy cache for the reverse check.
            conv = Conv(ctx)
            conv.add(target=2000, question='Reply only: OK')
            _, row = ctx.chat(chat_payload(ctx.model, conv.messages, 64), 'restart',
                              kind=f'policy-{target_mode}-prime')
            if not row['ok']:
                raise RuntimeError('opposite kernel policy could not prime a valid cache')
            server.stop()
            if (server.shutdown.get('forced') or server.shutdown.get('returncode') != 0 or
                    not server.shutdown.get('cache_saved') or not isolated.is_file() or
                    not isolated.stat().st_size):
                raise RuntimeError(f'opposite kernel policy failed to save: {server.shutdown}')
            source = isolated


def test_restart(ctx: Ctx, args: argparse.Namespace) -> None:
    if ctx.server.runtime != 'all':
        raise ValueError('restart test requires NInfer-all hybrid caching')
    continuation = getattr(ctx, 'restart_payload', None)
    expected = getattr(ctx, 'restart_expected', ())
    if continuation:
        continuation = copy.deepcopy(continuation)
        result, _ = ctx.chat(continuation, 'restart', kind='prime-from-ladder')
        if not result.ok or any(value not in result.text + result.thinking for value in expected):
            raise RuntimeError('validated ladder branch could not be continued before saving')
        reply: dict[str, Any] = {'role': 'assistant', 'content': result.text}
        if result.thinking:
            reply['reasoning_content'] = result.thinking
        continuation['messages'].extend([reply, {'role': 'user', 'content': Q_FINAL}])
    else:
        conv = Conv(ctx)
        conv.add(target=args.cache_depth, question='Summarize the last file in one sentence.')
        result, _ = ctx.chat(chat_payload(ctx.model, conv.messages, 64), 'restart', kind='prime')
        conv.commit(result)
        conv.followup(64, 'Briefly explain one more detail.')
        continuation = chat_payload(ctx.model, conv.messages, 64)
    (ctx.run_dir / 'restart-continuation.json').write_text(json.dumps(continuation))
    old = ctx.server
    old.stop()
    if old.shutdown.get('forced') or old.shutdown.get('returncode') != 0 or not old.shutdown.get('cache_saved'):
        raise RuntimeError(f'graceful cache save failed: {old.shutdown}')
    cache = Path(old.flags['--prefix-cache-file'])
    if not cache.exists() or not cache.stat().st_size:
        raise RuntimeError('cache save produced no file')
    ctx.emit({'test': 'restart', 'kind': 'saved', 'ok': True, 'bytes': cache.stat().st_size, **old.shutdown})
    restart_flags = dict(old.flags)
    if restart_flags.get('--device-profile') == 'calibrate':
        restart_flags['--device-profile'] = 'auto'
    new = Server(ctx.run_dir / 'restart', restart_flags, args)
    ctx.server = new  # outer finally must stop the latest process even after a failed wait
    base = gpu_state()
    new.start()
    info = new.wait('ready', args.startup_timeout)
    save_capacity(new.run_dir, 'restart', new.flags, info, base)
    if not info['ok']:
        raise RuntimeError(info)
    ctx.port = new.port
    ctx.jsonl = JsonlTail(new.run_dir / 'server.jsonl')
    result, row = ctx.chat(continuation, 'restart', kind='restored-continuation')
    restored = 'prefix cache restored |' in new.text or 'prefix cache partly restored |' in new.text
    ctx.emit({'test': 'restart', 'kind': 'restore-check', 'ok': restored and row['ok'] and row.get('cache_fraction', 0) > .95})
    if not restored or row.get('cache_fraction', 0) <= .95:
        raise RuntimeError('cache restoration did not reuse the saved prefix')
    if any(value not in result.text + result.thinking for value in expected):
        raise RuntimeError('restored ladder continuation returned incorrect values')
    new.stop()
    ctx.restart_cache_files = []
    isolated = cache.parent / 'incompatible.cache'
    incompatible_cache(cache, isolated)
    ctx.restart_cache_files.append(isolated)
    incompatible = start_rejected_cache(ctx, args, restart_flags, isolated,
                                        'incompatible', 'binary build stamp')
    cold_conv = Conv(ctx)
    cold_conv.add(target=2000, question='Summarize briefly.')
    _, cold = ctx.chat(chat_payload(ctx.model, cold_conv.messages, 64), 'restart', kind='incompatible-cold')
    ctx.emit({'test': 'restart', 'kind': 'incompatible-check', 'incompatibility': 'binary build stamp',
              'ok': cold['ok'] and (cold.get('cache_n') or 0) < 128 and cold.get('prompt_tokens', 0) > 1000})
    if '--v100-kernels' in restart_flags:
        incompatible.stop()
        if not args.keep_cache:
            isolated.unlink(missing_ok=True)  # Avoid holding three full ladder caches at once.
        test_kernel_policy_cache_rejection(ctx, args, restart_flags, cache)


TESTS: dict[str, Callable[[Ctx, argparse.Namespace], None]] = {
    "smoke": test_smoke, "ladder": test_ladder, "decode": test_decode, "conc": test_conc, "interf": test_interf,
    "agent": test_agent, "lineages": test_lineages,
    'speed': test_speed, 'cache': test_cache, 'restart': test_restart,
    'history': test_history,
}


# --------------------------------------------------------------------- commands

def new_run_dir(label: str) -> Path:
    path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_capacity(run_dir: Path, label: str, flags: dict[str, str], info: dict[str, Any], base: dict[str, float]) -> None:
    record = {"label": label, "flags": flags, "baseline": base, **info}
    (run_dir / "capacity.json").write_text(json.dumps(record, indent=2))


def probe_once(args: argparse.Namespace, flags: dict[str, str], run_dir: Path, label: str) -> dict[str, Any]:
    base = gpu_state()
    server = Server(run_dir, flags, args)
    try:
        server.start()
        info = server.wait("capacity", args.startup_timeout)
    finally:
        server.stop()
        server.cleanup_cache(getattr(args, 'keep_cache', False))
    wait_gpu_idle(base["mem_used_mib"])
    save_capacity(run_dir, label, server.flags, info, base)
    return info


def describe(info: dict[str, Any]) -> str:
    cap = info.get("capacity")
    if not info["ok"] or not cap:
        last = (info.get("error") or "").strip().splitlines()
        return "FAILED: " + (last[-1] if last else "unknown")
    return (f"KV {cap['kv_tokens']:,} tokens, pages {cap['pages']:,}/{cap['pages_max']:,}, "
            f"runtime {cap['runtime_gib']:.2f} GiB, free {cap['free_gib']:.2f} GiB")


def cmd_probe(args: argparse.Namespace, server_args: list[str]) -> None:
    preflight(args.force)
    flags = build_flags(args, server_args, quick=not args.full_host)
    label = args.label or f"probe-{args.profile}-{flags['--max-context']}"
    info = probe_once(args, flags, new_run_dir(label), label)
    print(describe(info))
    if not info["ok"]:
        print(info.get("error", ""))
        raise SystemExit(1)


def cmd_search(args: argparse.Namespace, server_args: list[str]) -> None:
    preflight(args.force)
    flags = build_flags(args, server_args, quick=not args.full_host)
    lanes = int(flags.get("--max-concurrency", "1"))
    label = args.label or f"search-{args.profile}-{args.pred}"
    parent = new_run_dir(label)
    step = args.step
    mult = 1.0 if args.kv_eq_context else args.kv_mult
    reserve = args.reserve_gib if mult else HEADROOM_GIB
    per_token_gib = KIB_PER_TOKEN * 1024 / GIB * (lanes if args.pred == "pool" else (mult or 1.0))

    last_free = [0.0]

    def probe(s: int) -> bool:
        trial = dict(flags)
        trial["--max-context"] = str(s)
        if mult:
            trial["--kv-capacity"] = str(int(mult * s))
        log(f"probing --max-context {s:,}")
        info = probe_once(args, trial, parent / f"S{s}", f"{label}-S{s}")
        log("  " + describe(info))
        cap = info.get("capacity")
        last_free[0] = cap["free_gib"] if cap else 0.0
        if not info["ok"] or not cap:
            return False
        if args.pred == 'pool':
            return cap['kv_tokens'] >= lanes * s if args.runtime == 'all' else cap['pages'] == cap['pages_max']
        return True

    if not probe(args.lo):
        raise SystemExit(f"--lo {args.lo:,} is already infeasible")
    best, free, fail = args.lo, last_free[0], None
    while fail is None and best < args.hi:
        gain = max((free - reserve) / per_token_gib * 0.97, 2 * step)
        nxt = min(int((best + gain) // step * step), args.hi)
        if nxt <= best:
            break
        if probe(nxt):
            best, free = nxt, last_free[0]
        else:
            fail = nxt
    lo, hi = best, fail
    while hi is not None and hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if mid <= lo:
            break
        if probe(mid):
            lo = mid
        else:
            hi = mid
    print(f"\nlargest feasible --max-context ({args.pred}): {lo:,}"
          + ("" if fail is not None else "  (reached --hi without a failure)"))
    (parent / "result.json").write_text(json.dumps({"best": lo, "pred": args.pred, "flags": flags}, indent=2))


def cmd_bench(args: argparse.Namespace, server_args: list[str]) -> None:
    preflight(args.force)
    flags = build_flags(args, server_args, quick=args.quick_host)
    label = args.label or f"{args.profile}-{flags['--max-context']}"
    run_dir = new_run_dir(label)
    (run_dir / 'workload.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items() if k != 'fn'}, indent=2))
    base = gpu_state()
    mon = GpuMonitor()
    server = Server(run_dir, flags, args)
    ctx = None
    try:
        mon.start()
        server.start()
        info = server.wait("ready", args.startup_timeout)
        save_capacity(run_dir, label, server.flags, info, base)
        log(describe(info))
        if not info["ok"]:
            raise RuntimeError(info.get('error', 'startup failed'))
        ctx = Ctx(server, mon, run_dir, args)
        for name in args.tests.split(","):
            if not ctx.server.alive():
                log("server died; stopping")
                print(ctx.server.tail())
                raise RuntimeError('benchmark server died')
            log(f"== test {name}")
            TESTS[name.strip()](ctx, args)
            if ctx.failures:
                raise RuntimeError(f'{len(ctx.failures)} benchmark checks failed; inspect rows.jsonl')
        if ctx.failures:
            raise RuntimeError(f'{len(ctx.failures)} benchmark checks failed; inspect rows.jsonl')
    finally:
        (ctx.server if ctx else server).stop()
        server.cleanup_cache(args.keep_cache)
        if ctx and ctx.server is not server and not args.keep_cache:
            # All restart-test files are isolated under this run.
            for isolated in getattr(ctx, 'restart_cache_files', []):
                if isolated.exists():
                    isolated.unlink()
        mon.stop()
        wait_gpu_idle(base["mem_used_mib"])
    stopped = ctx.server if ctx else server
    if stopped.shutdown.get('forced') or stopped.shutdown.get('returncode') != 0:
        raise RuntimeError(f'server shutdown failed: {stopped.shutdown}')
    print(f"\nresults: {run_dir}")


def cmd_report(args: argparse.Namespace, server_args: list[str]) -> None:
    caps = sorted(RUNS.rglob("capacity.json"))
    print(f"{'run':44} {'ok':3} {'max-ctx':>8} {'kv':>8} {'pages':>11} {'runtime':>8} {'free':>6} {'base MiB':>8}  flags")
    for path in caps:
        d = json.loads(path.read_text())
        cap, flags = d.get("capacity") or {}, d.get("flags", {})
        short = " ".join(f"{k[2:]}={flags[k]}" for k in ("--max-concurrency", "--kv-dtype", "--device-state-slots",
                         "--prefill-chunk", "--draft-tokens", "--kv-capacity") if k in flags)
        short += " vision" if "--vision" in flags else ""
        print(f"{str(path.parent.relative_to(RUNS))[:44]:44} {'y' if d['ok'] else 'n':3} {flags.get('--max-context', ''):>8} "
              f"{cap.get('kv_tokens', 0):>8,} {str(cap.get('pages', 0)) + '/' + str(cap.get('pages_max', 0)):>11} "
              f"{cap.get('runtime_gib', 0):>8.2f} {cap.get('free_gib', 0):>6.2f} {d.get('baseline', {}).get('mem_used_mib', 0):>8.0f}  {short}")
    for path in sorted(RUNS.glob("*/rows.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        print(f"\n{path.parent.name}")
        for row in rows:
            if row.get("test") != "smoke" or row.get("kind") != "warmup":
                print("  " + one_line(row))
        for depth in sorted({r['target'] for r in rows if r.get('test') == 'speed' and r.get('kind') == 'cold'}):
            samples = [r for r in rows if r.get('test') == 'speed' and r.get('kind') == 'cold'
                       and r.get('target') == depth and r.get('ok') and (r.get('completion_tokens') or 0) >= 256
                       and (r.get('cache_n') or 0) < 128]
            if samples:
                print(f"  speed median at {depth:,}: n={len(samples)}, prefill="
                      f"{statistics.median(r['prefill_tps'] for r in samples):.1f}, decode="
                      f"{statistics.median(r['decode_tps'] for r in samples):.1f} t/s")


def cmd_swap_check(args: argparse.Namespace, server_args: list[str]) -> None:
    if server_args:
        raise SystemExit('swap-check does not accept direct server flags')
    from ninfer_swap_check import run
    run(args, sys.modules[__name__])


def interrupt(*_: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    global RUNS
    argv = sys.argv[1:]
    server_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, server_args = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", choices=sorted(PROFILES), default="c1")
    common.add_argument("--label", default="")
    common.add_argument("--drop", action="append", default=[], help="server flag to remove, e.g. vision")
    common.add_argument("--force", action="store_true", help="skip the busy-GPU check")
    common.add_argument("--startup-timeout", type=int, default=900)
    common.add_argument('--shutdown-timeout', type=int, default=150)
    common.add_argument('--runtime', choices=('original', 'all'))
    common.add_argument('--cache-mode', choices=('hybrid', 'original'),
                        help='NInfer-all cache implementation; default hybrid, original is an explicit diagnostic A/B')
    common.add_argument('--launcher', help='custom relocatable runtime launcher')
    common.add_argument('--artifact', help='artifact path (runtime-specific default)')
    common.add_argument('--runs-dir', type=Path,
                        help='records directory; defaults to the selected engine bench-runs or NINFER_TUNE_RUNS')
    common.add_argument('--cache-dir', type=Path,
                        help='isolated benchmark caches; defaults to hybrid-caches under the records directory')
    common.add_argument('--keep-cache', action='store_true', help='retain benchmark cache after recording shutdown')

    p = sub.add_parser("probe", parents=[common], help="start once and print the capacity ledger")
    p.add_argument("--full-host", action="store_true", help="keep the profile's host KV/state sizes")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("search", parents=[common], help="find the largest --max-context that starts")
    p.add_argument("--lo", type=int, required=True)
    p.add_argument("--hi", type=int, default=262144)
    p.add_argument("--step", type=int, default=512)
    p.add_argument("--pred", choices=("start", "pool"), default="start",
                   help="start: server starts; pool: auto pool reaches max-concurrency x max-context")
    p.add_argument("--kv-eq-context", action="store_true", help="set --kv-capacity equal to --max-context")
    p.add_argument("--kv-mult", type=float, default=0.0, help="set --kv-capacity to this multiple of --max-context")
    p.add_argument("--reserve-gib", type=float, default=0.3, help="free VRAM the explicit-capacity search aims to keep")
    p.add_argument("--full-host", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("bench", parents=[common], help="start a server and run workloads")
    p.add_argument("--tests", default="smoke,ladder")
    p.add_argument("--quick-host", action="store_true", help="use small host KV/state for faster starts")
    p.add_argument("--depths", default="", help="ladder step depths, comma separated")
    p.add_argument("--ladder-light", action="store_true", help="ladder without the vision and final needle steps")
    p.add_argument("--decode-depths", default="8192,32768,65536")
    p.add_argument("--decode-tasks", type=int, default=4)
    p.add_argument("--decode-tokens", type=int, default=400)
    p.add_argument("--lineage-sizes", default="45000,28000,18000")
    p.add_argument("--lineage-rounds", type=int, default=3)
    p.add_argument("--big-image", default="1920x1080", help="WxH of the extra synthetic image in the ladder vision step, or none")
    p.add_argument("--new-tokens", type=int, default=384)
    p.add_argument("--conc-depths", default="8192,32768")
    p.add_argument("--conc-tokens", type=int, default=512)
    p.add_argument("--interf-a", type=int, default=8192)
    p.add_argument("--interf-b", type=int, default=32768)
    p.add_argument("--agent-variants", default="plain,interleave,cancel")
    p.add_argument("--agent-start", type=int, default=40000)
    p.add_argument("--agent-turns", type=int, default=8)
    p.add_argument("--agent-step", type=int, default=3000)
    p.add_argument('--speed-depths', default='8192,32768,65536')
    p.add_argument('--speed-tokens', type=int, default=384)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--fixtures', default=str(Path(__file__).resolve().parent.parent / '.local/ninfer-fixtures'))
    p.add_argument('--cache-depth', type=int, default=32768)
    p.add_argument('--min-free-mib', type=float, default=0,
                   help='fail if any request lacks this measured free-VRAM margin')
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser('swap-check', parents=[common], help='verify production save/restore/reuse through llama-swap')
    p.add_argument('--expected-context', type=int, required=True)
    p.add_argument('--swap-model', default='ninfer-all-v100')
    p.add_argument('--reference-model', default='ninfer-v100')
    p.add_argument('--production-cache-file', required=True)
    p.add_argument('--serve-log', default=str(ALL_ROOT / 'logs/serve.log'))
    p.set_defaults(fn=cmd_swap_check)

    p = sub.add_parser("report", help="print capacity and result tables from earlier runs")
    p.add_argument('--runs-dir', type=Path)
    p.add_argument('--runtime', choices=('original', 'all'), default='original')
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    if args.cmd != 'report':
        args.runtime = args.runtime or ('all' if args.profile.startswith('all-') else 'original')
    default_root = ALL_ROOT if args.runtime == 'all' else ORIGINAL_ROOT
    args.runs_dir = args.runs_dir or Path(os.environ.get('NINFER_TUNE_RUNS', default_root / 'bench-runs'))
    RUNS = args.runs_dir.resolve()
    if args.cmd != 'report':
        args.cache_dir = args.cache_dir or RUNS / 'hybrid-caches'
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    args.fn(args, server_args)


if __name__ == "__main__":
    main()
