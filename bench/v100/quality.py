#!/usr/bin/env python3.11
"""Score the original frozen V100 quality inputs through the public Engine."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ninfer_precision_grade import grade  # noqa: E402


KV_TYPES = ("int8", "bf16")
ARM_PHASES = {
    "baseline": ("score", "generate"),
    "a16": ("score", "score-narrow", "generate"),
    "w4a8": ("score-narrow", "generate"),
}
TASK_COUNTS = {
    "tool_json": 128, "code": 96, "arithmetic": 64,
    "retrieval_en": 32, "retrieval_zh": 32, "history": 32,
}
WINDOWS = {(0, 1, 4096), (2048, 4096, 6144), (4096, 6144, 8192), (4097, 8192, 8193)}
CANDIDATES = "head,drafter,gdn,flash"


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_jobs(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(data: bytes):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def index(rows, label: str) -> dict:
    result = {}
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"invalid or duplicate ID in {label}: {key!r}")
        result[key] = row
    return result


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_identity(path: Path) -> dict:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"expected a file: {path}")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def token_list(tokens: list) -> bool:
    return isinstance(tokens, list) and bool(tokens) and all(
        type(token) is int and 0 <= token < 2**31 for token in tokens)


def load_fixture(root: Path):
    fixture = root / "fixtures"
    original = json.loads((fixture / "freeze.json").read_bytes())["files"]
    token_freeze = json.loads((fixture / "tokens-freeze.json").read_bytes())["files"]
    hashes = {}

    def locked(name: str, manifest: dict):
        path = fixture / name
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if manifest.get(f"fixtures/{name}") != digest:
            raise ValueError(f"frozen fixture hash mismatch: {path}")
        hashes[name] = digest
        return read_jsonl(data) if path.suffix == ".jsonl" else json.loads(data)

    streams = index(locked("corpus.json", original), "frozen corpus")
    cases = index(locked("quality.json", original), "frozen tasks")
    case_order = locked("quality-order.json", original)
    token_rows = index(locked("tokens.jsonl", token_freeze), "frozen tokens")
    windows = locked("score-windows.json", token_freeze)
    index(windows, "frozen score windows")
    if (len(streams) != 16 or
            sum(key.startswith("heldout-") for key in streams) != 12 or
            sum(key.startswith("legacy-") for key in streams) != 4 or
            any(row.get("target_tokens") != 8192 for row in streams.values())):
        raise ValueError("expected the frozen 12 held-out and four legacy score streams")
    if (Counter(case["domain"] for case in cases.values()) != TASK_COUNTS or
            len(case_order) != len(cases) or set(case_order) != cases.keys()):
        raise ValueError("expected all 384 frozen tasks in the original order")
    if len(windows) != len(streams) * len(WINDOWS):
        raise ValueError("expected four original score windows per document")

    for key in streams.keys() | cases.keys():
        tokens = token_rows.get(key, {}).get("tokens")
        if not token_list(tokens) or (key in streams and len(tokens) != 8193):
            raise ValueError(f"invalid frozen token vector for {key}")
    observed = defaultdict(set)
    for row in windows:
        doc = row["document"]
        begin, target, end = row["input_begin"], row["target_begin"], row["target_end"]
        interval = (begin, target, end)
        if (doc not in streams or row["domain"] != streams[doc]["domain"] or
                interval not in WINDOWS or interval in observed[doc] or
                row["first_target"] != target - begin or
                row["tokens"] != token_rows[doc]["tokens"][begin:end]):
            raise ValueError(f"score window differs from the original token protocol: {row['id']}")
        observed[doc].add(interval)
    if observed.keys() != streams.keys() or any(seen != WINDOWS for seen in observed.values()):
        raise ValueError("frozen score coverage is incomplete")

    generation = [{"id": key, "tokens": token_rows[key]["tokens"], "reuse": False,
                   "max_tokens": 512 if cases[key]["domain"] == "code" else 128}
                  for key in case_order]
    if any(len(job["tokens"]) + job["max_tokens"] > 8192 for job in generation):
        raise ValueError("a frozen generation input does not fit its original context")
    return streams, cases, windows, generation, hashes


def cluster(case: dict) -> str:
    domain = case["domain"]
    family = {
        "code": ("code-algorithm", 12),
        "tool_json": ("tool-schema", 8),
        "arithmetic": ("arithmetic-operation", 4),
    }.get(domain)
    if family is None:
        return case["id"]
    name, count = family
    return f"{name}-{int(case['id'].rsplit('-', 1)[1]) % count}"


def run_probe(command: list[str], output: Path, env: dict, timeout: int) -> dict:
    with output.open("x", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                       timeout=timeout, check=True)
    path = Path(command[4])
    return index(read_jsonl(path.read_bytes()), str(path))


def check_w4a8_route(log: Path, enabled: bool) -> int:
    reports = re.findall(r"\bw4a8_narrow=(\d+)\s+w4a8_wide=(\d+)",
                         log.read_text(encoding="utf-8", errors="replace"))
    if len(reports) != 1:
        raise ValueError(f"missing or ambiguous candidate route counts: {log}")
    narrow, wide = map(int, reports[0])
    if (enabled and narrow == 0) or (not enabled and (narrow or wide)):
        raise ValueError(f"candidate did not demonstrate the expected W4A8 route: {log}")
    return narrow


def score_documents(records: dict, windows: list[dict], mode: str) -> dict:
    if records.keys() != {job["id"] for job in windows}:
        raise ValueError("causal scoring did not return every frozen window")
    totals = defaultdict(list)
    for job in windows:
        row = records[job["id"]]
        logprobs = row.get("logprobs", [])
        if ("error" in row or row.get("mode") != mode or row.get("k") != 0 or
                row.get("first_target") != job["first_target"] or
                row.get("input_tokens") != len(job["tokens"]) or
                len(logprobs) != job["target_end"] - job["target_begin"] or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in logprobs)):
            raise ValueError(f"invalid score record for {job['id']}")
        totals[job["document"]].extend(logprobs)
    return {doc: -math.fsum(logprobs) / len(logprobs) for doc, logprobs in totals.items()}


def grade_tasks(records: dict, cases: dict, generation: list[dict]) -> dict:
    if records.keys() != cases.keys():
        raise ValueError("generation did not return every frozen task")
    grades = {}
    for job in generation:
        key = job["id"]
        row = records[key]
        if ("error" in row or row.get("mode") != "generate" or row.get("k") != 0 or
                row.get("reused_tokens") != 0 or row.get("drafted_tokens") != 0 or
                row.get("accepted_tokens") != 0 or not isinstance(row.get("tokens"), list) or
                len(row["tokens"]) > job["max_tokens"] or not isinstance(row.get("content"), str)):
            raise ValueError(f"invalid generation record for {key}")
        result = grade(cases[key], row["content"])
        if type(result.get("correct")) is not bool:
            raise ValueError(f"no deterministic grade for {key}")
        grades[key] = result
    return grades


def compare(kv: str, name: str, score_mode: str, streams: dict, cases: dict,
            control: tuple[dict, dict], candidate: tuple[dict, dict]) -> dict:
    control_nll, control_grade = control
    candidate_nll, candidate_grade = candidate
    return {
        "comparison": name, "kv_dtype": kv,
        "score_mode": score_mode,
        "documents": [{"id": key, "domain": row["domain"], "tokens": 8192,
                       "control_mean_nll": control_nll[key],
                       "candidate_mean_nll": candidate_nll[key]} for key, row in streams.items()],
        "tasks": [{"id": key, "domain": row["domain"], "cluster": cluster(row),
                   "control": control_grade[key]["correct"],
                   "candidate": candidate_grade[key]["correct"]} for key, row in cases.items()],
    }


def run(args: argparse.Namespace) -> None:
    streams, cases, windows, generation, hashes = load_fixture(args.fixtures)
    probes = {"baseline": args.baseline_probe.resolve(), "candidate": args.candidate_probe.resolve()}
    artifact = args.artifact.resolve()
    artifact_identity = file_identity(artifact)
    probe_identities = {name: {**file_identity(path), "sha256": hash_file(path)}
                        for name, path in probes.items()}
    score_jobs, generation_jobs = args.output / "score-jobs.jsonl", args.output / "generation-jobs.jsonl"
    write_jobs(score_jobs, windows)
    write_jobs(generation_jobs, generation)
    env = dict(os.environ, NINFER_V100_CANDIDATES=CANDIDATES)
    metadata = {"artifact": artifact_identity, "probes": probe_identities,
                "fixtures": {"root": str(args.fixtures.resolve()), "sha256": hashes},
                "environment": {key: env.get(key) for key in (
                    "NINFER_V100_CANDIDATES", "NINFER_PREFILL_ALIGN", "CUDA_VISIBLE_DEVICES")},
                "jobs": {path.name: hash_file(path) for path in (score_jobs, generation_jobs)},
                "commands": []}
    manifest = args.output / "manifest.json"
    comparisons = []
    for kv in KV_TYPES:
        results = {}
        for arm, phases in ARM_PHASES.items():
            probe = probes["baseline" if arm == "baseline" else "candidate"]
            arithmetic = "w4a8" if arm == "w4a8" else "a16"
            for mode in phases:
                jobs, context = (generation_jobs, "8192") if mode == "generate" else (score_jobs, "4096")
                stem = args.output / f"{kv}-{arm}-{mode}"
                command = [str(probe), mode, str(artifact), str(jobs), str(stem.with_suffix(".jsonl")),
                           "0", context, kv, arithmetic]
                metadata["commands"].append(command)
                dump(manifest, metadata)
                log = stem.with_suffix(".log")
                records = run_probe(command, log, env, args.timeout)
                if arm != "baseline":
                    selections = metadata.setdefault("w4a8_narrow_selections", {})
                    selections[f"{kv}-{arm}-{mode}"] = check_w4a8_route(log, arm == "w4a8")
                if mode == "generate":
                    grades = grade_tasks(records, cases, generation)
                    dump(args.output / f"{kv}-{arm}-grades.json", grades)
                    results[arm, mode] = grades
                else:
                    results[arm, mode] = score_documents(records, windows, mode)
        for mode, control, candidate in (("score", "baseline", "a16"), ("score-narrow", "a16", "w4a8")):
            comparisons.append(compare(kv, candidate, mode, streams, cases,
                (results[control, mode], results[control, "generate"]),
                (results[candidate, mode], results[candidate, "generate"])))
    dump(manifest, metadata)
    if file_identity(artifact) != artifact_identity or any(
            hash_file(path) != probe_identities[name]["sha256"] for name, path in probes.items()):
        raise ValueError("artifact or quality probe changed during the run")
    dump(args.output / "quality-evidence.json",
         {"source": str(manifest.resolve()), "comparisons": comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-probe", type=Path, required=True)
    parser.add_argument("--candidate-probe", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True, help="original campaign root")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--timeout", type=int, default=7200, help="per probe phase in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except FileNotFoundError as error:
        dump(args.output / "status.json", {"status": "open", "reason": str(error)})
        print(f"Quality gate open: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        dump(args.output / "status.json", {"status": "open", "reason": str(error)})
        print(f"Quality run incomplete: {error}", file=sys.stderr)
        return 1
    dump(args.output / "status.json", {"status": "complete"})
    print(args.output / "quality-evidence.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
