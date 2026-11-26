#!/usr/bin/env python3.11
"""Run paired Serve trials and evaluate the release gates for the V100 port."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ninfer_precision import paired_bootstrap  # noqa: E402


REPEATS = 8
SEED = 1729
PHASES = ("controls", "head", "drafter", "gdn", "flash", "cold", "w4a8", "capacity",
          "cache", "deployment", "quality")
MODEL = "v100-qualification"
CANDIDATES = ("head", "drafter", "gdn", "flash")
ROUTES = ",".join(CANDIDATES)
W4A8_COUNTERS = ("w4a8_narrow", "w4a8_wide")
CACHE_FIRST_PAIR_CHECKS = ("incompatible_cache", "incompatible_native_cache", "incompatible_w4a8_cache")
PROSE = {
    "prose": "Write an account of how a city rebuilt its public transit network.",
    "code": "Write a Python function that parses a web server access log, including tests.",
    "list": "List 25 practical tips for maintaining a home network, one sentence each.",
}
SPEED_CASES = ("prose", "code", "list", "prefill-11k", "prefill-33k", "ttft-11k", "ttft-33k")
WORDS = (
    "proxmox volta tensor cache kernel module router vlan bridge storage thin pool snapshot "
    "backup restore latency throughput packet socket thread mutex scheduler interrupt firmware "
    "bios uefi partition journal inode"
).split()


@dataclass(frozen=True)
class Arm:
    name: str
    baseline: bool = False
    candidates: str = ROUTES
    w4a8: bool = False
    ported: bool = True


OLD = Arm("baseline", baseline=True)
OLD_NATIVE = Arm("baseline-native", baseline=True, ported=False)
NATIVE = Arm("candidate-native", candidates="none", ported=False)
CONTROL = Arm("candidate-ported-control", candidates="none")
NEW = Arm("candidate")
W4A8 = Arm("candidate-w4a8", w4a8=True)


@dataclass(frozen=True)
class DeploymentProfile:
    context: int
    requested_capacity: int | str
    minimum_capacity: int
    prefill_chunk: int
    drafts: int
    vision: bool
    host_cache_mib: int
    device_snapshot_slots: int
    requests: dict[str, list[dict]]


@dataclass(frozen=True)
class Run:
    phase: str
    kv: str
    drafts: int
    number: int
    arm: Arm
    deployment: DeploymentProfile | None = None


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    complete_rows = path.read_bytes().split(b"\n")[:-1]
    return [json.loads(line) for line in complete_rows]


def endpoint(base: str, path: str, timeout: float, body: dict | None = None):
    headers = {} if body is None else {"Content-Type": "application/json"}
    data = None if body is None else json.dumps(body).encode()
    return urlopen(Request(base + path, data=data, headers=headers), timeout=timeout)


def host_record(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"argv": command, "code": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except (OSError, subprocess.SubprocessError) as error:
        return {"argv": command, "error": str(error)}


def route_selections(path: Path) -> dict[str, int]:
    records = re.findall(r"v100 kernel route selections: ((?:[a-z0-9_]+=\d+ )+)"
                         r"\(graph replays not counted\)",
                         path.read_text(encoding="utf-8", errors="replace"))
    if len(records) != 1:
        raise ValueError(f"expected one final Engine route counter record, found {len(records)}")
    fields = [field.split("=") for field in records[0].split()]
    counts = {key: int(value) for key, value in fields}
    if len(fields) != len(counts) or not {"ported", "fallback"} <= counts.keys():
        raise ValueError("incomplete or duplicate final Engine route counters")
    return counts


def check_route_selections(run: Run, counts: dict[str, int], *, smoke: bool = False) -> None:
    named = set(CANDIDATES + W4A8_COUNTERS)
    if not run.arm.baseline and not named <= counts.keys():
        raise RuntimeError("candidate binary did not emit all named route counters")
    if not run.arm.ported:
        if any(value for key, value in counts.items() if key != "fallback"):
            raise RuntimeError("native control selected a Volta ported route")
        return
    if counts["ported"] == 0:
        raise RuntimeError("ported control or candidate did not exercise a ported route")
    if run.arm.baseline:
        return
    selected = set(run.arm.candidates.split(",")) & set(CANDIDATES)
    allowed = set(selected)
    if run.kv != "bf16":
        allowed.discard("flash")
    if run.arm.w4a8:
        allowed.update(W4A8_COUNTERS)
    required = allowed.copy()
    if run.drafts == 0:
        required.discard("drafter")
    if smoke:
        required -= {"drafter", "gdn", "flash", "w4a8_wide"}
    rejected = sorted(name for name in named - allowed if counts[name] > 0)
    missing = sorted(name for name in required if counts[name] == 0)
    if rejected or missing:
        raise RuntimeError(f"unexpected Volta route selections: disabled={rejected}, missing={missing}")


@contextmanager
def sample_process_gpu_memory(pid: int, folder: Path):
    """Sample this server PID's device memory, independent of CUDA index remapping."""
    samples = []
    readiness = []
    sampler = None
    reader = None
    error = None
    argv = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits", "-lms", "100"]
    try:
        sampler = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True)
    except OSError as exception:
        error = str(exception)

    def read_samples():
        for line in sampler.stdout:
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 3 or fields[0] != str(pid):
                continue
            try:
                samples.append((time.monotonic(), fields[1], int(fields[2])))
            except ValueError:
                continue

    if sampler is not None:
        reader = threading.Thread(target=read_samples, daemon=True)
        reader.start()
    try:
        yield readiness
    finally:
        if sampler is not None:
            if sampler.poll() is None:
                sampler.terminate()
                try:
                    sampler.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sampler.kill()
                    sampler.wait()
            reader.join()
        ready = readiness[0] if readiness else math.inf
        finish = readiness[1] if len(readiness) > 1 else math.inf
        loading = [mib for timestamp, _, mib in samples if timestamp < ready]
        running = [mib for timestamp, _, mib in samples if ready <= timestamp < finish]
        dump(folder / "gpu-memory.json", {
            "argv": argv, "pid": pid, "sampling_error": error,
            "sampling_status": "sampled" if samples else "unavailable",
            "device_uuids": sorted({uuid for _, uuid, _ in samples}),
            "load_samples": len(loading), "runtime_samples": len(running),
            "load_peak_mib": max(loading, default=None),
            "runtime_peak_mib": max(running, default=None),
            "samples": samples,
        })


def wait_until(probe, timeout: float, explanation: str):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = probe()
        if result is not None:
            return result
        time.sleep(0.1)
    raise TimeoutError(explanation)


def prompt(text: str, tokens: int) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def speed_workloads(base: str, timeout: float) -> dict[str, dict]:
    workloads = {name: prompt(text, 300) for name, text in PROSE.items()}
    for name, target in (("prefill-11k", 11600), ("prefill-33k", 33000)):
        prng = random.Random(7)
        words = [prng.choice(WORDS) for _ in range(2 * target)]
        suffix = "\nSummarise the above in one sentence."
        first, last = 1, len(words)
        while first < last:
            count = (first + last) // 2
            text = " ".join(words[:count]) + suffix
            with endpoint(base, "/tokenize", timeout, {"content": text}) as response:
                tokens = len(json.load(response)["tokens"])
            if tokens < target:
                first = count + 1
            else:
                last = count
        workloads[name] = prompt(" ".join(words[:first]) + suffix, 8)
    return workloads


def validate_cache_item(name: str, item: dict) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("body"), dict):
        raise ValueError(f"cache {name} requests must wrap an HTTP body")
    required = {"prompt_tokens", "minimum_cached_tokens", "expected_text"}
    if not required <= item.keys() or type(item["prompt_tokens"]) is not int or \
            type(item["minimum_cached_tokens"]) is not int or \
            not isinstance(item["expected_text"], str):
        raise ValueError(f"cache {name} requests need token and exact-text expectations")
    maximum = item.get("maximum_cached_tokens", item["prompt_tokens"])
    if (type(maximum) is not int or not 0 <= item["minimum_cached_tokens"] <= maximum
            <= item["prompt_tokens"] or item["prompt_tokens"] <= 0):
        raise ValueError(f"invalid token counts in cache {name} fixture")
    for field in ("expected_native_text", "expected_w4a8_text"):
        if field in item and not isinstance(item[field], str):
            raise ValueError(f"expected text for {field} must be a string")


def cache_fixture(path: Path) -> dict:
    fixture = read_json(path)
    expected_counts = {"warm": 2, "pressure": 47}
    for name, count in expected_counts.items():
        if name not in fixture or not isinstance(fixture[name], list) or len(fixture[name]) != count:
            raise ValueError(f"cache fixture needs {count} requests in {name!r}")
    for name in ("media", "near_limit"):
        if name in fixture and (not isinstance(fixture[name], list) or not fixture[name]):
            raise ValueError(f"cache {name} needs a nonempty request list")
    if "media" in fixture and len(fixture["media"]) != 2:
        raise ValueError("cache media fixture needs a priming request and a changed-media request")
    for name in (*expected_counts, "media", "near_limit"):
        for item in fixture.get(name, ()):
            validate_cache_item(name, item)
    if "media" in fixture:
        old, changed = fixture["media"]
        if ("maximum_cached_tokens" not in changed or old["body"] == changed["body"] or
                old["expected_text"] == changed["expected_text"]):
            raise ValueError("changed media must have a new body, expected answer and reuse ceiling")
    if any(item["prompt_tokens"] != 90787 for item in fixture["warm"]):
        raise ValueError("both warm requests must have 90,787 expanded prompt tokens")
    if fixture["warm"][0]["body"] != fixture["warm"][1]["body"]:
        raise ValueError("the warm fixture must replay the exact same request")
    if not fixture["warm"][1]["minimum_cached_tokens"]:
        raise ValueError("the warmed request must prove prefix reuse")
    return fixture


def deployment_profile(value: dict) -> DeploymentProfile:
    if not isinstance(value, dict):
        raise ValueError("deployment profile must be an object")

    def integer(name, default=None, minimum=1):
        number = value.get(name, default)
        if type(number) is not int or number < minimum:
            raise ValueError(f"deployment {name} must be an integer of at least {minimum}")
        return number

    context = integer("context")
    minimum_capacity = integer("minimum_capacity")
    chunk = integer("prefill_chunk")
    drafts = integer("drafts", 3, 0)
    vision = value.get("vision", False)
    host_cache_mib = integer("host_cache_mib", 20480)
    snapshot_slots = integer("device_snapshot_slots", 4, 0)
    capacity = value.get("kv_capacity", "auto")
    if capacity != "auto" and (type(capacity) is not int or capacity < minimum_capacity):
        raise ValueError("deployment kv_capacity must be auto or hold at least minimum_capacity")
    if chunk % 128 or drafts > 15 or type(vision) is not bool:
        raise ValueError("deployment requires a 128-token chunk multiple, 0–15 drafts and boolean vision")

    restart = value.get("restart")
    if not isinstance(restart, dict):
        raise ValueError("deployment restart must contain prime and after request lists")
    requests = {"near_limit": value.get("near_limit"), "frontier": value.get("frontier"),
                "restart_prime": restart.get("prime"), "restart": restart.get("after")}
    requests.update({key: value[key] for key in ("warm", "pressure") if key in value})
    for case, items in requests.items():
        if not isinstance(items, list) or not items:
            raise ValueError(f"deployment {case} needs a nonempty request list")
        names = set()
        for item in items:
            validate_cache_item(case, item)
            name = item.get("name")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError(f"deployment {case} needs unique request names")
            names.add(name)
            if item["prompt_tokens"] > context:
                raise ValueError(f"deployment {case} request {name!r} exceeds its context")
    near = requests["near_limit"]
    if near[0]["prompt_tokens"] < context - chunk:
        raise ValueError("first deployment near-limit prompt must be within one chunk of context")
    if near[0]["minimum_cached_tokens"] or near[0].get("maximum_cached_tokens", 0) != 0:
        raise ValueError("first deployment near_limit request must declare zero cached tokens")
    if not any(item["minimum_cached_tokens"] == item.get("maximum_cached_tokens") > 0
               for item in requests["frontier"][1:]):
        raise ValueError("deployment frontier must prime then verify an exact positive reuse frontier")
    if requests["restart"][0]["minimum_cached_tokens"] == 0:
        raise ValueError("first deployment restart request must prove reuse without priming")
    if "warm" in requests:
        warm = requests["warm"]
        if (len(warm) != 2 or any(item["prompt_tokens"] != 90787 for item in warm) or
                warm[0]["body"] != warm[1]["body"] or not warm[1]["minimum_cached_tokens"]):
            raise ValueError("deployment warm requires an exact 90,787-token replay with reuse")
    if "pressure" in requests and len(requests["pressure"]) != 47:
        raise ValueError("deployment pressure needs 47 requests")
    return DeploymentProfile(context, capacity, minimum_capacity, chunk, drafts, vision,
                             host_cache_mib, snapshot_slots, requests)


def server_command(args: argparse.Namespace, run: Run, port: int,
                   request_log: Path, cache_path: Path | None = None) -> list[str]:
    binary = args.baseline if run.arm.baseline else args.candidate
    is_cache = run.phase == "cache"
    long_context = is_cache or (run.phase == "capacity" and run.kv == "int8")
    context = args.cache_context if long_context else args.cold_context
    capacity = "auto" if run.phase == "capacity" else str(context)
    if is_cache:
        capacity = getattr(args, "matched_cache_kv_capacity", args.cache_kv_capacity)
    chunk = args.cache_prefill_chunk if long_context else 2048
    vision = args.cache_vision if long_context else args.vision
    host_cache_mib, snapshot_slots = 20480, 4
    if run.deployment is not None:
        profile = run.deployment
        context, capacity = profile.context, str(profile.requested_capacity)
        chunk, vision = profile.prefill_chunk, profile.vision
        host_cache_mib, snapshot_slots = profile.host_cache_mib, profile.device_snapshot_slots
    command = [
        str(binary), str(args.artifact), "--host", "127.0.0.1", "--port", str(port),
        "--device", str(args.device), "--model-id", MODEL, "--request-log-jsonl", str(request_log),
        "--max-context", str(context), "--kv-capacity", capacity, "--max-concurrency", "1",
        "--prefill-chunk", str(chunk), "--kv-dtype", run.kv,
        "--v100-kernels", "ported" if run.arm.ported else "native",
        "--spec", "mtp" if run.drafts else "none", "--ngram-draft-tokens", "0",
    ]
    if run.drafts:
        command += ["--draft-tokens", str(run.drafts), "--lm-head-draft"]
    if run.phase in ("cache", "capacity", "deployment"):
        command += [
            "--use-alt-prefix-caching", "--host-cache-mib", str(host_cache_mib),
            "--device-snapshot-slots", str(snapshot_slots),
        ]
    else:
        command.append("--no-prefix-reuse")
    if cache_path is not None:
        command += ["--prefix-cache-file", str(cache_path)]
    if vision:
        command.append("--vision")
    if run.arm.w4a8:
        command.append("--nvfp4-w4a8-int")
    return command


def loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def verify_scheduling(engine: dict, kv: str, drafts: int, chunk: int) -> None:
    # The public --kv-dtype shorthand differs from the canonical request-log name.
    codec = {"bf16":"bf16", "int8":"int8-group64"}[kv]
    if (engine["kv_cache"] != codec or engine["prefill_chunk"] != chunk or
            engine["max_concurrency"] != 1 or not engine["cuda_graph"] or
            engine["speculative_backend"] != ("mtp" if drafts else "none") or
            (drafts and (engine["speculative_draft_window"] != drafts or
                         engine["proposal_head"] != "optimized"))):
        raise RuntimeError(f"effective scheduling differs from the requested profile: {engine}")


def verify_deployment_start(engine: dict, profile: DeploymentProfile) -> None:
    cache = engine["context_cache"]
    capacity_mode = "auto" if profile.requested_capacity == "auto" else "explicit"
    if (engine["max_context"] != profile.context or engine["kv_capacity_mode"] != capacity_mode
            or engine["vision"] != profile.vision or not engine["prefix_reuse"]
            or not cache["enabled"] or cache["mode"] != "hybrid"
            or cache["host_cache_budget_bytes"] != profile.host_cache_mib * 2**20
            or cache["hybrid"]["device_snapshot_slots"] != profile.device_snapshot_slots
            or not cache["hybrid"]["persistent"]
            or (profile.drafts and engine["speculative_draft_window"] != profile.drafts)
            or engine["speculative_backend"] != ("mtp" if profile.drafts else "none")):
        raise RuntimeError("effective Serve deployment settings differ from the frozen profile")
    capacity = engine["kv_capacity"]
    if type(capacity) is not int or capacity < profile.minimum_capacity:
        raise RuntimeError(f"physical KV capacity {capacity} is below {profile.minimum_capacity}")


@contextmanager
def serving(args: argparse.Namespace, run: Run, folder: Path, cache_path: Path | None = None):
    folder.mkdir(parents=True)
    request_log = folder / "requests.jsonl"
    port = loopback_port()
    base = f"http://127.0.0.1:{port}"
    command = server_command(args, run, port, request_log, cache_path)
    env = os.environ.copy()
    env.pop("NINFER_V100_CANDIDATES", None)
    env.pop("NINFER_GDN_TWO_STAGE", None)
    env["NINFER_PREFILL_ALIGN"] = "0"
    if not run.arm.baseline:
        env["NINFER_V100_CANDIDATES"] = run.arm.candidates
    dump(folder / "launch.json", {
        "argv": command, "NINFER_PREFILL_ALIGN": "0",
        "execution_environment": {key: env.get(key) for key in (
            "CUDA_VISIBLE_DEVICES","CUDA_LAUNCH_BLOCKING","CUDA_DEVICE_MAX_CONNECTIONS",
            "NINFER_DEVICE_PROFILES","NINFER_SM70_CUBLAS_MIN","NINFER_GDN_TWO_STAGE",
            "NINFER_PROMPT_FAST","NINFER_PROMPT_PACK_GQA","NINFER_PROMPT_PV_F16",
            "NINFER_PROMPT_SPLITS","NINFER_SMALLT_PV_F16","NINFER_ATTN_PARALLEL_TILES")},
        "NINFER_V100_CANDIDATES": env.get("NINFER_V100_CANDIDATES"),
    })
    with (folder / "server.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=env)

        def started():
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode} during startup")
            try:
                with endpoint(base, "/v1/models", 2) as response:
                    models = json.load(response)["data"]
                if not any(model["id"] == MODEL for model in models):
                    return None
            except OSError:
                return None
            starts = [item for item in read_jsonl(request_log)
                      if item.get("event") == "server_start"]
            if not starts:
                return None
            startup = starts[-1]
            engine = startup["engine"]
            if not run.arm.baseline and (
                    engine["v100_kernels"] != ("ported" if run.arm.ported else "native") or
                    engine["nvfp4_w4a8_int"] != run.arm.w4a8):
                raise RuntimeError(f"effective Volta arithmetic differs from trial arm: {engine}")
            long_context = run.phase == "cache" or (run.phase == "capacity" and run.kv == "int8")
            expected_chunk = (run.deployment.prefill_chunk if run.deployment else
                              args.cache_prefill_chunk if long_context else 2048)
            verify_scheduling(engine, run.kv, run.drafts, expected_chunk)
            return startup

        with sample_process_gpu_memory(process.pid, folder) as readiness:
            try:
                startup = wait_until(started, args.startup_timeout, "server readiness timeout")
                readiness.append(time.monotonic())
                dump(folder / "startup.json", startup)
                yield base, request_log, startup
            finally:
                if readiness:
                    readiness.append(time.monotonic())
                if process.poll() is None:
                    proc_status = Path(f"/proc/{process.pid}/status")
                    try:
                        high_water = next((line for line in proc_status.read_text().splitlines()
                                           if line.startswith("VmHWM:")), None)
                        dump(folder / "process-memory.json", {"VmHWM": high_water})
                    except OSError:
                        pass
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=args.stop_timeout)
                    except subprocess.TimeoutExpired as error:
                        process.kill()
                        process.wait()
                        raise RuntimeError("server did not stop within its clean-stop timeout") from error
                if process.returncode != 0:
                    raise RuntimeError(f"server exited with status {process.returncode}; see {folder}")


def request_result(base: str, request_log: Path, body: dict, timeout: float) -> dict:
    before = len(read_jsonl(request_log))
    data = dict(body, model=MODEL, stream=True, stream_options={"include_usage": True})
    started = time.monotonic()
    first_event = None
    events = []
    content = []
    done = False
    usage = None
    with endpoint(base, "/v1/chat/completions", timeout, data) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line.removeprefix(b"data:").strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            events.append(event)
            if event.get("error"):
                raise RuntimeError(f"stream emitted an error: {event['error']}")
            for choice in event.get("choices", ()):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                content.append(text)
                if first_event is None and (text or delta.get("reasoning_content")
                                            or delta.get("tool_calls")):
                    first_event = time.monotonic() - started
            if event.get("usage") is not None:
                usage = event["usage"]
    elapsed = time.monotonic() - started
    if not done or first_event is None or usage is None:
        raise RuntimeError("stream missing visible output, usage or [DONE]")

    def completed():
        records = read_jsonl(request_log)[before:]
        errors = [item for item in records if item.get("event") == "request_error"]
        if errors:
            raise RuntimeError(f"request log contains an error: {errors[-1]}")
        results = [item for item in records if item.get("event") == "request_done"]
        if len(results) > 1:
            raise RuntimeError("more than one request completed on the isolated server")
        return results[0] if results else None

    record = wait_until(completed, 5, "no matching request_done record")
    counts = record["result"]
    for key in ("prompt_tokens", "completion_tokens"):
        if counts[key] != usage[key]:
            raise RuntimeError(f"stream and server log disagree on {key}")
    times = record["timings_seconds"]
    rounds = record["engine_timing"]["decode"]["rounds"]
    drafts = record["speculative"]
    return {
        "prompt_tokens": counts["prompt_tokens"], "output_tokens": counts["completion_tokens"],
        "computed_tokens": counts["computed_prefill_tokens"],
        "cached_tokens": counts["prefix_cache_hit_tokens"],
        "cache_path": counts["prefix_reuse_path"],
        "ttft": first_event, "elapsed": elapsed, "timings": times,
        "decode_rounds": rounds, "speculative": drafts,
        "decode": (counts["completion_tokens"] - 1) / times["decode"] if times["decode"] > 0 else 0,
        "prefill": counts["computed_prefill_tokens"] / times["prefill"] if times["prefill"] > 0 else 0,
        "wall_prefill": counts["prompt_tokens"] / elapsed if elapsed > 0 else 0,
        "ms_per_round": times["decode"] * 1000 / rounds if rounds else None,
        "ms_per_mtp_round": times["decode"] * 1000 / drafts["rounds"] if drafts["rounds"] else None,
        "draft_acceptance": drafts["accepted_tokens"] / drafts["drafted_tokens"] if drafts["drafted_tokens"] else None,
        "text": "".join(content), "events": events, "request_log": record,
    }


def valid_speed_case(name: str, body: dict, result: dict, drafts: int) -> None:
    if result["cached_tokens"] or result["computed_tokens"] != result["prompt_tokens"]:
        raise RuntimeError(f"{name} requires a fully cold prefill")
    if result["output_tokens"] != body["max_tokens"]:
        raise RuntimeError(f"{name} stopped early ({result['output_tokens']} output tokens)")
    if drafts and not result["speculative"]["drafted_tokens"]:
        raise RuntimeError(f"{name} did not exercise MTP")
    if name == "prefill-11k" and not 11000 <= result["prompt_tokens"] <= 12500:
        raise RuntimeError("11.6K prompt token count is outside its qualification band")
    if name == "prefill-33k" and not 31500 <= result["prompt_tokens"] <= 34500:
        raise RuntimeError("33K prompt token count is outside its qualification band")


def measured_request(base: str, log: Path, body: dict, args: argparse.Namespace,
                     folder: Path, name: str) -> dict:
    result = request_result(base, log, body, args.request_timeout)
    dump(folder / f"{name}.json", result)
    return result


def trials_evidence(folder: Path, run: Run, *, smoke: bool = False) -> dict:
    counters = route_selections(folder / "server.log")
    check_route_selections(run, counters, smoke=smoke)
    gpu_memory = read_json(folder / "gpu-memory.json")
    del gpu_memory["samples"]
    return {"route_selections": counters, "gpu_memory": gpu_memory,
            "engine": read_json(folder / "startup.json")["engine"]}


def verify_cache_request(item: dict, result: dict, label: str, *, cold: bool = False,
                         text_key: str = "expected_text") -> None:
    minimum = 0 if cold else item["minimum_cached_tokens"]
    maximum = 0 if cold else item.get("maximum_cached_tokens", item["prompt_tokens"])
    expected = item.get(text_key)
    if (result["prompt_tokens"] != item["prompt_tokens"] or
            not minimum <= result["cached_tokens"] <= maximum or
            (cold and result["computed_tokens"] != result["prompt_tokens"]) or
            (expected is not None and result["text"] != expected)):
        raise RuntimeError(f"{label} failed its frozen output or prefix reuse expectations")


def run_cache_cases(args: argparse.Namespace, run: Run, fixture: dict,
                    folder: Path) -> dict:
    folder.mkdir(parents=True)
    timings = {}
    counts = {}
    evidence = {}
    checks = dict.fromkeys(("restart", "media", "near_limit", *CACHE_FIRST_PAIR_CHECKS), "open")
    check_errors = {}

    def replay(kind, requests, cache_path=None, *, arm=run.arm, cold=False, smoke=False,
               text_key="expected_text"):
        trial = Run(run.phase, run.kv, run.drafts, run.number, arm)
        trial_folder = folder / kind
        results = []
        with serving(args, trial, trial_folder, cache_path) as (base, log, _):
            for index, item in enumerate(requests):
                result = measured_request(base, log, item["body"], args, trial_folder,
                                          f"request-{index:02d}")
                verify_cache_request(item, result, f"{kind} request {index}", cold=cold,
                                     text_key=text_key)
                results.append(result)
        if cold:
            log_text = (trial_folder / "server.log").read_text(encoding="utf-8", errors="replace")
            if ("prefix cache not restored | saved prefix cache belongs to a different model, "
                    "KV format or build" not in log_text):
                raise RuntimeError(f"{kind} did not report rejection of the copied cache fingerprint")
        evidence[kind] = dict(trials_evidence(trial_folder, trial, smoke=smoke),
                              expected_text_validated=all(text_key in item for item in requests))
        return results

    with TemporaryDirectory(prefix="private-cache-", dir=folder) as temporary:
        private = Path(temporary)
        warm_file = private / "warm.cache"
        warm = replay("warm", fixture["warm"], warm_file, smoke=True)
        if not warm_file.is_file() or warm_file.stat().st_size == 0:
            raise RuntimeError("warm clean stop did not produce a private persistent cache")
        checks["warm_file_bytes"] = warm_file.stat().st_size
        replay("warm-restart", fixture["warm"][1:], warm_file, smoke=True)
        checks["restart"] = "pass"
        if not run.arm.baseline and run.number == 1:
            probes = [("incompatible_cache", OLD, "expected_text"),
                      ("incompatible_native_cache", NATIVE, "expected_native_text")]
            if getattr(args, "evaluate_w4a8", False):
                probes.append(("incompatible_w4a8_cache", W4A8, "expected_w4a8_text"))
            for kind, arm, text_key in probes:
                incompatible = private / f"{kind}.cache"
                try:
                    try:
                        shutil.copyfile(warm_file, incompatible)
                        replay(kind, fixture["warm"][1:], incompatible, arm=arm, cold=True,
                               smoke=True, text_key=text_key)
                    finally:
                        incompatible.unlink(missing_ok=True)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    if arm != W4A8:
                        raise
                    checks[kind] = "fail"
                    check_errors[kind] = str(error)
                else:
                    checks[kind] = "pass"
        warm_file.unlink()
        pressure_path = private / "pressure.cache"
        pressure = replay("pressure", fixture["pressure"], pressure_path, smoke=True)
        pressure_path.unlink(missing_ok=True)
        for kind in ("media", "near_limit"):
            if kind in fixture:
                replay(kind, fixture[kind], private / f"{kind}.cache", smoke=True)
                checks[kind] = "pass"

    for kind, results in (("warm", warm), ("pressure", pressure)):
        counts[kind] = [[result["prompt_tokens"], result["output_tokens"]] for result in results]
    timings["warm"] = warm[-1]["ttft"]
    timings["pressure"] = math.fsum(result["ttft"] for result in pressure)
    capacity = {item["engine"]["kv_capacity"] for item in evidence.values()}
    if len(capacity) != 1:
        raise RuntimeError("cache check servers did not have matched physical KV capacity")
    return {"cases": timings, "counts": counts, "checks": checks, "cache_evidence": evidence,
            "check_errors": check_errors, "engine": evidence["warm"]["engine"]}


def run_deployment_cases(args: argparse.Namespace, run: Run, folder: Path, record: dict) -> None:
    profile = run.deployment
    folder.mkdir(parents=True)

    def replay(case, path, *, cold_first=False, restored=False):
        trial_folder = folder / case
        results = {"status": "fail", "requests": []}
        record["checks"][case] = results
        with serving(args, run, trial_folder, path) as (base, log, startup):
            engine = startup["engine"]
            results["observed_capacity"] = engine["kv_capacity"]
            verify_deployment_start(engine, profile)
            for index, item in enumerate(profile.requests[case]):
                result = measured_request(base, log, item["body"], args, trial_folder,
                                          f"request-{index:02d}")
                observation = {key: result[key] for key in (
                    "prompt_tokens", "computed_tokens", "cached_tokens", "cache_path",
                    "output_tokens", "ttft", "elapsed")}
                observation.update(status="fail", name=item["name"],
                                   expected_prompt_tokens=item["prompt_tokens"],
                                   minimum_cached_tokens=item["minimum_cached_tokens"],
                                   maximum_cached_tokens=0 if cold_first and index == 0 else
                                   item.get("maximum_cached_tokens", item["prompt_tokens"]))
                results["requests"].append(observation)
                verify_cache_request(item, result, f"{case} {item['name']}",
                                     cold=cold_first and index == 0)
                observation["status"] = "pass"
        if restored:
            startup_log = (trial_folder / "server.log").read_text(encoding="utf-8", errors="replace")
            if not re.search(r"prefix cache (?:partly )?restored \|", startup_log):
                raise RuntimeError("restart did not acknowledge the private persistent cache")
            results["restored"] = True
        results.update(trials_evidence(trial_folder, run, smoke=True), status="pass")

    with TemporaryDirectory(prefix="private-cache-", dir=folder) as temporary:
        private = Path(temporary)
        for case in ("near_limit", "frontier", "warm", "pressure"):
            if case not in profile.requests:
                continue
            cache_path = private / f"{case}.cache"
            try:
                replay(case, cache_path, cold_first=case == "near_limit")
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                record["checks"][case]["error"] = str(error)
            cache_path.unlink(missing_ok=True)
        restart_path = private / "restart.cache"
        try:
            replay("restart_prime", restart_path)
            if not restart_path.is_file() or restart_path.stat().st_size == 0:
                raise RuntimeError("restart primer did not save a private persistent cache")
            record["file_bytes"] = restart_path.stat().st_size
            replay("restart", restart_path, restored=True)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            check = record["checks"].setdefault("restart", {"status": "fail", "requests": []})
            check["error"] = str(error)
    if any(check["status"] != "pass" for check in record["checks"].values()):
        raise RuntimeError("one or more deployment checks failed")


def run_trial(args: argparse.Namespace, run: Run, fixture: dict | None) -> dict:
    folder = args.output / run.phase / f"{run.kv}-mtp{run.drafts}-{run.number:02d}" / run.arm.name
    row = {"phase": run.phase, "kv": run.kv, "drafts": run.drafts, "repeat": run.number,
           "arm": run.arm.name, "folder": str(folder), "status": "fail", "cases": {}}
    print(f"{run.phase} {run.kv} MTP{run.drafts} pair {run.number}/{REPEATS} {run.arm.name}",
          flush=True)
    try:
        if run.phase == "cache":
            row.update(run_cache_cases(args, run, fixture, folder))
        elif run.deployment is not None:
            profile = run.deployment
            row["deployment"] = {
                "context": profile.context, "requested_capacity": profile.requested_capacity,
                "minimum_capacity": profile.minimum_capacity, "prefill_chunk": profile.prefill_chunk,
                "vision": profile.vision, "host_cache_mib": profile.host_cache_mib,
                "device_snapshot_slots": profile.device_snapshot_slots, "checks": {},
            }
            run_deployment_cases(args, run, folder, row["deployment"])
        else:
            with serving(args, run, folder) as (base, log, startup):
                row["engine"] = startup["engine"]
                row["memory"] = startup.get("memory")
                measured_request(base, log, prompt("Write a few words.", 24), args, folder, "warmup")
                if run.phase != "capacity":
                    row["cases"], row["counts"] = run_speed_cases(
                        args, run, base, log, folder)
            row.update(trials_evidence(folder, run, smoke=run.phase == "capacity"))
        row["status"] = "ok"
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        row["error"] = str(error)
        print(f"{folder}: {error}", file=sys.stderr, flush=True)
    return row


def run_speed_cases(args: argparse.Namespace, run: Run, base: str, log: Path,
                    folder: Path) -> tuple[dict, dict]:
    metrics = {}
    counts = {}
    frozen = args.output / "speed-workloads.json"
    if not frozen.exists():
        dump(frozen, speed_workloads(base, args.request_timeout))
    for name, body in read_json(frozen).items():
        result = measured_request(base, log, body, args, folder, name)
        valid_speed_case(name, body, result, run.drafts)
        metric = "prefill" if name.startswith("prefill") else "decode"
        metrics[name] = result[metric]
        if metric == "prefill":
            metrics[name.replace("prefill", "ttft")] = result["ttft"]
        counts[name] = [result["prompt_tokens"], result["output_tokens"]]
    return metrics, counts


def rate_change(log_ratio: float) -> float:
    return math.expm1(log_ratio)


def latency_reduction(log_speedup: float) -> float:
    return -math.expm1(-log_speedup)


def comparison(phase: str, rows: list[dict], kv: str, drafts: int,
               old: str, new: str) -> list[dict]:
    matched = [row for row in rows if row["phase"] == phase and row["kv"] == kv
               and row["drafts"] == drafts and row["arm"] in (old, new)]
    index = {(row["repeat"], row["arm"]): row for row in matched}
    pairs = [(index.get((number, old)), index.get((number, new)))
             for number in range(1, REPEATS + 1)]
    gate = {"phase": phase, "kv": kv, "drafts": drafts, "old": old, "new": new}
    if len(index) != len(matched):
        return [dict(gate, status="fail", reason="duplicate observations")]
    if any(row["status"] != "ok" for row in matched):
        return [dict(gate, status="fail", reason="one or more trials failed")]
    if any(a is None or b is None for a, b in pairs):
        return [dict(gate, status="open", reason="one or more pairs are missing")]
    expected = {"warm", "pressure"} if phase == "cache" else set(SPEED_CASES)
    for a, b in pairs:
        if set(a["cases"]) != expected or set(b["cases"]) != expected:
            return [dict(gate, status="fail", reason="case coverage differs")]
        if a.get("counts") != b.get("counts"):
            return [dict(gate, status="fail", reason="prompt or completion counts differ within a pair")]
        fields = ("max_context", "kv_capacity", "prefill_chunk", "max_concurrency",
                  "kv_cache", "cuda_graph", "vision", "gdn_state_fp16")
        if any(a.get("engine", {}).get(name) != b.get("engine", {}).get(name) for name in fields):
            return [dict(gate, status="fail", reason="effective timing profiles differ within a pair")]
        if any(not math.isfinite(value) or value <= 0
               for row in (a, b) for value in row["cases"].values()):
            return [dict(gate, status="fail", reason="missing or invalid timing metric")]
    def case_is_latency(case: str) -> bool:
        return phase == "cache" or case.startswith("ttft")

    samples = {}
    for case in sorted(expected):
        logs = [(math.log(a["cases"][case]), math.log(b["cases"][case]), 1) for a, b in pairs]
        samples[case] = logs if case_is_latency(case) else [(b, a, n) for a, b, n in logs]
    results = []
    for case, observations in samples.items():
        transform = latency_reduction if case_is_latency(case) else rate_change
        stats = paired_bootstrap(observations, transform, alpha=.05)
        lower, upper = stats["ci"]
        require_gain = phase == "flash" and case == "prefill-33k"
        if phase == "controls":
            status = "pass"
        elif require_gain:
            status = "pass" if lower > 0 else "fail" if upper <= 0 else "open"
        else:
            status = "fail" if upper < -.02 else "open" if stats["estimate"] < -.02 else "pass"
        changes = [transform(a - b) for a, b, _ in observations]
        results.append(dict(gate, case=case, status=status, pair_range=[min(changes), max(changes)],
                            **stats))
    if (phase == "cold" and drafts == 3) or phase == "w4a8":
        grouped = [
            (math.fsum(samples[case][i][0] for case in PROSE) / len(PROSE),
             math.fsum(samples[case][i][1] for case in PROSE) / len(PROSE), 1)
            for i in range(REPEATS)
        ]
        stats = paired_bootstrap(grouped, math.expm1, alpha=.05)
        minimum = .03 if phase == "cold" else 0
        estimate = stats["estimate"]
        lower, upper = stats["ci"]
        status = ("pass" if estimate >= minimum - 1e-12 and lower > 0
                  else "fail" if estimate < minimum - 1e-12 or upper <= 0 else "open")
        results.append(dict(gate, case="MTP3 decode geometric mean", status=status, **stats))
    return results


def quality(path: Path | None) -> list[dict]:
    gate = {"phase": "quality"}
    if path is None or not path.is_file():
        return [dict(gate, comparison=mode, kv=kv, status="open", reason="frozen paired quality evidence unavailable")
                for mode in ("a16","w4a8") for kv in ("int8","bf16")]
    evidence = read_json(path)
    if not isinstance(evidence.get("source"), str) or not evidence["source"].strip():
        raise ValueError("quality evidence must name the raw scoring and grading records")
    pairs = {}
    for pair in evidence["comparisons"]:
        key = (pair["comparison"], pair["kv_dtype"])
        if key in pairs or key[0] not in ("a16", "w4a8") or key[1] not in ("bf16", "int8"):
            raise ValueError(f"unexpected or repeated quality comparison: {key}")
        pairs[key] = pair
    gates = []
    for mode in ("a16", "w4a8"):
        for kv in ("int8", "bf16"):
            case = dict(gate, comparison=mode, kv=kv)
            pair = pairs.get((mode, kv))
            if pair is None:
                gates.append(dict(case, status="open", reason="missing paired quality evidence"))
            else:
                gates.extend(quality_pair(pair, case))
    return gates


def quality_pair(evidence: dict, gate: dict) -> list[dict]:
    documents = evidence["documents"]
    tasks = evidence["tasks"]
    if len({row["id"] for row in documents}) != len(documents) or \
            len({row["id"] for row in tasks}) != len(tasks):
        raise ValueError("duplicate streams or tasks in quality evidence")
    if gate["comparison"] == "w4a8":
        if documents:
            raise ValueError(
                "w4a8 is narrow decode/verify-only (T<=4); chunked causal scoring is vacuous"
            )
    elif len(documents) != 16 or any(row["tokens"] != 8192 for row in documents):
        return [dict(gate, status="open", reason="quality needs 16 matched frozen streams of 8,192 targets")]
    if len(tasks) != 384:
        return [dict(gate, status="open", reason="quality needs 384 matched graded task pairs")]
    if any(type(row[arm]) is not bool for row in tasks for arm in ("candidate", "control")):
        raise ValueError("paired task grades must be booleans")
    clusters = {}
    for task in tasks:
        clusters.setdefault(task["cluster"], []).append(task)
    if len(clusters) < 2:
        raise ValueError("quality task grades must retain the original fixture clusters")
    measures = {}
    if gate["comparison"] != "w4a8":
        measures["perplexity"] = (
            [(row["candidate_mean_nll"], row["control_mean_nll"], row["tokens"])
             for row in documents], math.expm1, .005)
    measures["accuracy"] = (
        [(sum(row["candidate"] for row in group) / len(group),
          sum(row["control"] for row in group) / len(group), len(group))
         for group in clusters.values()], float, .02)
    out = []
    for case, (samples, transform, margin) in measures.items():
        stats = paired_bootstrap(samples, transform=transform)
        lower, upper = stats["ci"]
        status = ("pass" if -margin <= lower <= upper <= margin else "fail"
                  if lower > margin or upper < -margin else "open")
        out.append(dict(gate, case=case, status=status, margin=margin, **stats))
    return out


def matrix():
    yield "controls", "int8", 3, OLD_NATIVE, OLD
    yield "controls", "int8", 3, NATIVE, CONTROL
    for name in CANDIDATES:
        kv = "bf16" if name == "flash" else "int8"
        yield name, kv, 3, CONTROL, Arm(f"candidate-{name}-only", candidates=name)
    for kv in ("int8", "bf16"):
        for drafts in (0, 3):
            yield "cold", kv, drafts, OLD, NEW
        yield "w4a8", kv, 3, NEW, W4A8
        yield "capacity", kv, 3, OLD, NEW
    yield "cache", "int8", 3, OLD, NEW


def matched_cache_capacity(rows: list[dict]) -> int:
    probes = [row for row in rows if row["phase"] == "capacity" and row["kv"] == "int8"]
    arms = {row["arm"]: row for row in probes}
    if (len(probes) != 2 or set(arms) != {OLD.name, NEW.name} or
            any(row["status"] != "ok" for row in arms.values())):
        raise RuntimeError("INT8 capacity probes did not complete for both cache arms")
    capacities = [row["engine"]["kv_capacity"] for row in arms.values()]
    if any(type(value) is not int or value <= 0 for value in capacities):
        raise RuntimeError("invalid reported physical capacity for the cache comparison")
    return min(capacities)


def cache_check_results(rows: list[dict], evaluate_w4a8: bool = False) -> list[dict]:
    matching = [row for row in rows if row["phase"] == "cache"]
    results = []
    cases = ("restart", "media", "near_limit", *CACHE_FIRST_PAIR_CHECKS)
    for case in cases:
        if case == "incompatible_w4a8_cache" and not evaluate_w4a8:
            continue
        first_pair_only = case in CACHE_FIRST_PAIR_CHECKS
        expected = [row for row in matching if not first_pair_only or
                    (row["arm"] == NEW.name and row["repeat"] == 1)]
        required = 1 if first_pair_only else 2 * REPEATS
        statuses = [row.get("checks", {}).get(case, "open") for row in expected]
        if "fail" in statuses or any(row["status"] != "ok" for row in expected):
            status = "fail"
        elif len(statuses) != required or any(status != "pass" for status in statuses):
            status = "open"
        else:
            status = "pass"
        results.append({"phase": "cache", "case": case, "status": status,
                        "observations": statuses, "errors": [
                            row["check_errors"][case] for row in expected
                            if case in row.get("check_errors", {})]})
    return results


def capacity_results(rows: list[dict], kv: str) -> list[dict]:
    gate = {"phase": "capacity", "kv": kv}
    matching = [row for row in rows if row["phase"] == "capacity" and row["kv"] == kv]
    if any(row["status"] != "ok" for row in matching):
        return [dict(gate, status="fail", reason="one or both capacity starts failed")]
    arms = {row["arm"]: row for row in matching}
    if len(arms) != len(matching):
        return [dict(gate, status="fail", reason="duplicate capacity observations")]
    if set(arms) != {"baseline", "candidate"}:
        return [dict(gate, status="open", reason="one or both capacity starts are missing")]
    old = arms["baseline"]["engine"]["kv_capacity"]
    new = arms["candidate"]["engine"]["kv_capacity"]
    return [dict(gate, status="pass", baseline=old, candidate=new, delta_tokens=new-old,
                 baseline_memory=arms["baseline"]["memory"],
                 candidate_memory=arms["candidate"]["memory"],
                 baseline_gpu_memory=arms["baseline"]["gpu_memory"],
                 candidate_gpu_memory=arms["candidate"]["gpu_memory"])]


def run_deployments(args: argparse.Namespace, rows: list[dict], gates: list[dict]) -> None:
    path = args.deployment_fixture
    fixture = {}
    fixture_error = None
    if path is not None and path.is_file():
        try:
            fixture = read_json(path)
            if not isinstance(fixture, dict) or fixture.keys() - {"int8", "bf16"}:
                raise ValueError("deployment fixture must be an object keyed by int8 and/or bf16")
        except (OSError, ValueError) as error:
            fixture_error = str(error)
    for kv in ("int8", "bf16"):
        gate = {"phase": "deployment", "kv": kv}
        if fixture_error is not None:
            gates.append(dict(gate, status="fail", reason=fixture_error))
            continue
        if kv not in fixture:
            gates.append(dict(gate, status="open", reason="frozen deployment profile unavailable"))
            continue
        try:
            profile = deployment_profile(fixture[kv])
        except ValueError as error:
            gates.append(dict(gate, status="fail", reason=str(error)))
            continue
        arms = {}
        for arm in (OLD, NEW):
            row = run_trial(args, Run("deployment", kv, profile.drafts, 1, arm, profile), None)
            rows.append(row)
            append(args.output / "trials.jsonl", row)
            arms[arm.name] = row
        checks = {}
        for case in profile.requests:
            observed = {name: row["deployment"]["checks"].get(case, {})
                        for name, row in arms.items()}
            checks[case] = {name: {key: item[key] for key in ("status", "observed_capacity", "error")
                                   if key in item} for name, item in observed.items()}
        passed = all(row["status"] == "ok" for row in arms.values())
        gates.append(dict(gate, status="pass" if passed else "fail", context=profile.context,
                          minimum_capacity=profile.minimum_capacity,
                          requested_capacity=profile.requested_capacity,
                          checks=checks,
                          errors={name: row["error"] for name, row in arms.items() if "error" in row}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("baseline", "candidate", "artifact", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, action="append")
    parser.add_argument("--cache-fixture", type=Path)
    parser.add_argument("--deployment-fixture", type=Path)
    parser.add_argument("--quality-evidence", type=Path)
    parser.add_argument("--cold-context", type=int, default=65536)
    parser.add_argument("--cache-context", type=int, default=262144)
    parser.add_argument("--cache-kv-capacity", default="auto")
    parser.add_argument("--cache-prefill-chunk", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--stop-timeout", type=float, default=180)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--cache-vision", action="store_true")
    return parser.parse_args()


def gate_status(gates: list[dict]) -> str:
    if any(gate["status"] == "fail" for gate in gates):
        return "fail"
    if not gates or any(gate["status"] == "open" for gate in gates):
        return "open"
    return "pass"


def main() -> int:
    args = parse_args()
    if sys.platform != "linux":
        raise SystemExit("Serve qualification needs a Linux V100 test host")
    for name in ("baseline", "candidate", "artifact", "output"):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.baseline, args.candidate, args.artifact):
        if not path.is_file():
            raise SystemExit(f"missing explicit input: {path}")
    if min(args.startup_timeout, args.request_timeout, args.stop_timeout) <= 0:
        raise SystemExit("timeouts must be positive")
    if args.cache_kv_capacity != "auto" and (
        not args.cache_kv_capacity.isdecimal() or int(args.cache_kv_capacity) <= 0
    ):
        raise SystemExit("--cache-kv-capacity needs a positive token count or auto")
    if args.cache_prefill_chunk <= 0 or args.cache_prefill_chunk % 128:
        raise SystemExit("--cache-prefill-chunk needs a positive multiple of 128 tokens")
    args.output.mkdir(parents=True, exist_ok=False)
    phases = set(args.phase or PHASES)
    args.evaluate_w4a8 = "w4a8" in phases
    dump(args.output / "manifest.json", {
        "tool_sha256": digest(Path(__file__)), "inputs": {
            name: {"path": str(getattr(args, name)), "sha256": digest(getattr(args, name))}
            for name in ("baseline", "candidate", "artifact", "cache_fixture",
                         "deployment_fixture", "quality_evidence")
            if getattr(args, name) is not None and getattr(args, name).is_file()
        },
        "phases": sorted(phases), "repeats": REPEATS, "bootstrap_seed": SEED,
        "device": args.device, "cold_context": args.cold_context, "cache_context": args.cache_context,
        "cache_kv_capacity": args.cache_kv_capacity, "cache_prefill_chunk": args.cache_prefill_chunk,
        "vision": args.vision, "cache_vision": args.cache_vision,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "matrix": [{"phase": phase, "kv": kv, "drafts": drafts,
                    "control": first.name, "candidate": second.name}
                   for phase, kv, drafts, first, second in matrix()],
        "gpu": host_record([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,power.limit,"
            "clocks.max.sm,clocks.max.mem,ecc.mode.current,driver_version",
            "--format=csv",
        ]),
        "toolkit": host_record(["nvcc", "--version"]),
    })
    fixture = None
    gates = []
    if "cache" in phases:
        if args.cache_fixture and args.cache_fixture.is_file():
            try:
                fixture = cache_fixture(args.cache_fixture)
            except (ValueError, KeyError, TypeError) as error:
                gates.append({"phase": "cache", "status": "fail", "reason": str(error)})
        else:
            gates.append({"phase": "cache", "status": "open",
                          "reason": "frozen warm/pressure fixture unavailable"})
    rows = []
    try:
        for phase, kv, drafts, first, second in matrix():
            capacity_prerequisite = (phase == "capacity" and kv == "int8" and fixture is not None
                                     and args.cache_kv_capacity == "auto")
            if (phase not in phases and not capacity_prerequisite) or (
                    phase == "cache" and fixture is None):
                continue
            if phase == "cache":
                try:
                    shared = (matched_cache_capacity(rows) if args.cache_kv_capacity == "auto"
                              else int(args.cache_kv_capacity))
                    args.matched_cache_kv_capacity = str(shared)
                    dump(args.output / "matched-cache-capacity.json", {
                        "requested": args.cache_kv_capacity, "shared_physical_capacity": shared,
                    })
                except (RuntimeError, KeyError) as error:
                    gates.append({"phase": "cache", "status": "fail", "reason": str(error)})
                    continue
            count = 1 if phase == "capacity" else REPEATS
            for number in range(1, count + 1):
                order = (first, second) if number % 2 else (second, first)
                for arm in order:
                    row = run_trial(args, Run(phase, kv, drafts, number, arm), fixture)
                    rows.append(row)
                    append(args.output / "trials.jsonl", row)
        if "deployment" in phases:
            run_deployments(args, rows, gates)
    except KeyboardInterrupt:
        gates.append({"phase": "session", "status": "open", "reason": "interrupted"})
    for phase, kv, drafts, first, second in matrix():
        if phase not in phases or (phase == "cache" and fixture is None):
            continue
        if phase == "capacity":
            gates.extend(capacity_results(rows, kv))
        else:
            gates.extend(comparison(phase, rows, kv, drafts, first.name, second.name))
    if "cache" in phases and fixture is not None:
        gates.extend(cache_check_results(rows, args.evaluate_w4a8))
    if "quality" in phases:
        try:
            gates.extend(quality(args.quality_evidence))
        except (ValueError, KeyError, TypeError) as error:
            gates.append({"phase": "quality", "status": "fail", "reason": str(error)})
    for phase in sorted(set(PHASES) - phases):
        gates.append({"phase": phase, "status": "open", "reason": "phase not run"})
    status = gate_status(gates)
    policies = {}
    for policy in ("a16", "w4a8"):
        relevant = []
        for gate in gates:
            if gate["phase"] == "quality" and "comparison" not in gate:
                relevant.append(gate)  # Missing or unreadable evidence affects both comparisons.
                continue
            w4a8 = (gate["phase"] == "w4a8" or gate.get("comparison") == "w4a8" or
                    gate.get("case") == "incompatible_w4a8_cache")
            if w4a8 == (policy == "w4a8"):
                relevant.append(gate)
        policies[policy] = gate_status(relevant)
    dump(args.output / "summary.json", {"status": status, "performance_and_quality": policies, "gates": gates})
    print(f"{status}: {args.output / 'summary.json'}")
    return {"pass": 0, "fail": 1, "open": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
