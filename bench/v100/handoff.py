#!/usr/bin/env python3.11
"""Build pinned source trees and collect the V100 release evidence, one rerunnable stage at a time."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from xml.etree import ElementTree

BASE = "42d8d03ffe7ac9e9c5cd1cb227a195c099a03f88"
STAGES = ("preflight", "build", "cpu", "correctness", "gdn", "quality", "performance", "report")
CPU = r"^ninfer_(v100_arch_guard|v100_kernel_policy.*|v100_fp8_loading(_native)?|serve_options|request_log|cli_options|prefix_cache_index|context_cache_defaults)_test$"
GPU = r"^ninfer_(v100_(port|native_packed|w4a8|fp8.*|gdn.*)|gated_delta_net(_two_stage)?|softmax_attention(_volta_flash|_int8_prompt)?|linear_(q4|q8)_(a16|a8int)|linear_swiglu_(q4_a16|q8_a16|nvfp4))_test$"
VOLTA_LEAVES = ("v100_cutlass.cu", "v100_drafter_gemv.cu", "v100_fp8.cu", "v100_fp8_cutlass.cu",
                "v100_qpn.cu", "v100_w4a8.cu", "volta_flash.cu", "volta_flash_bridge.cu")
MIXED_DISPATCHER = "sm70_project.cu"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def attempt(root, stage):
    for i in range(1, 10000):
        path = root / f"{stage}-{i:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"too many attempts for {stage}")


class Session:
    def __init__(self, args, folder):
        self.args = args
        self.folder = folder
        self.rows = []

    def add(self, name, status, **extra):
        self.rows.append(dict(name=name, status=status, **extra))
        write(self.folder / "checks.json", self.rows)
        print(f"{status:>5}  {name}", flush=True)

    def run(self, name, argv, *, env=None, classify=None, cwd=None):
        path = self.folder / f"{name}.log"
        argv = list(map(str, argv))
        try:
            with path.open("wb") as output:
                rc = subprocess.run(argv, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, check=False).returncode
        except OSError as exc:
            self.add(name, "open", command=argv, reason=str(exc))
            return False
        status = classify(rc) if classify else ("pass" if rc == 0 else "fail")
        self.add(name, status, command=argv, log=path, returncode=rc)
        return status == "pass"

    def ctest(self, name, build, pattern, env=None):
        junit = self.folder / f"{name}.xml"
        self.run(name, ["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error",
                        "--output-junit", junit, "-R", pattern], env=env)
        if junit.is_file():
            for case in ElementTree.parse(junit).iter("testcase"):
                status = ("open" if case.find("skipped") is not None else
                          "fail" if any(case.find(t) is not None for t in ("error","failure")) else "pass")
                self.add(f"{name}/{case.get('name')}", status, evidence=junit)


def inspect(args):
    if args.stage == "preflight" and not args.artifact.is_file():
        raise ValueError(f"missing artifact {args.artifact}")
    for tree in (args.baseline_source, args.candidate_source):
        if not (tree / "CMakeLists.txt").is_file():
            raise ValueError(f"missing source tree {tree}")
    if args.baseline_source == args.candidate_source:
        raise ValueError("use separate baseline and candidate checkouts")
    rev = subprocess.check_output(["git", "-C", str(args.baseline_source), "rev-parse", "HEAD"],
                                  text=True).strip()
    if rev != BASE:
        raise ValueError(f"baseline is {rev}; expected {BASE}")
    if subprocess.check_output(["git", "-C", str(args.baseline_source), "status", "--porcelain"]):
        raise ValueError("baseline has local changes; use a clean checkout at the pinned revision")
    header = (args.cutlass / "include/cutlass/version.h").read_text(encoding="utf-8")
    parts = dict(re.findall(r"#define\s+CUTLASS_(MAJOR|MINOR|PATCH)\s+(\d+)",header))
    version = tuple(parts.get(key) for key in ("MAJOR","MINOR","PATCH"))
    if version != ("4","4","2"):
        raise ValueError(f"CUTLASS version {version}; expected 4.4.2")
    version = subprocess.check_output([str(args.cuda / "bin/nvcc"), "--version"],text=True)
    if not re.search(r"release 12\.9\b", version):
        raise ValueError("the paired build requires CUDA 12.9")


def mixed_codegen_check(s, build):
    database = json.loads((build / "compile_commands.json").read_text())
    expected = {name: ["sm_70"] for name in VOLTA_LEAVES}
    expected[MIXED_DISPATCHER] = ["sm_70", "sm_75"]
    entries = {name: [] for name in expected}
    for row in database:
        name = Path(row["file"]).name
        if name not in entries:
            continue
        argv = iter(row.get("arguments") or shlex.split(row["command"]))
        specs = []
        for arg in argv:
            option, equals, value = arg.partition("=")
            if option in ("--generate-code", "-gencode"):
                specs.append(value if equals else next(argv, ""))
        targets = set()
        for spec in specs:
            _, _, code = spec.partition(",code=")
            targets.update(code.strip("[]").split(","))
        entries[name].append({"gencode": specs,
                              "sass": sorted(item for item in targets if item.startswith("sm_")),
                              "ptx": sorted(item for item in targets if item.startswith("compute_"))})

    errors = []
    for name, commands in entries.items():
        if len(commands) != 1:
            errors.append(f"{name} has {len(commands)} compile entries, expected one")
            continue
        command = commands[0]
        if command["sass"] != expected[name] or (name in VOLTA_LEAVES and command["ptx"]):
            errors.append(f"{name} has SASS {command['sass']} and PTX {command['ptx']}")
    s.add("mixed-codegen", "fail" if errors else "pass", codegen=entries,
          **({"reason": "; ".join(errors)} if errors else {}))


def nonvolta_check(s, build):
    database = json.loads((build/"compile_commands.json").read_text())
    leaves = set(VOLTA_LEAVES)
    selected = leaves.intersection(Path(row["file"]).name for row in database)
    if selected:
        raise ValueError(f"non-Volta build selected these Volta kernels: {sorted(selected)}")
    stub, = [row for row in database if Path(row["file"]).name == "v100_native_stubs.cpp"]
    argv = stub.get("arguments") or shlex.split(stub["command"])
    compile_args = list(argv)
    compile_args[compile_args.index("-o")+1] = str(s.folder/"nonvolta-stubs.o")
    s.run("nonvolta-stubs",compile_args,cwd=stub["directory"])
    s.add("nonvolta-source-selection","pass", excluded=sorted(leaves))


def current_quality(results, artifact, probes):
    attempts = sorted(results.glob("quality-*/inputs.json"))
    if not attempts:
        return None
    latest = attempts[-1].parent / "observations"
    evidence = latest / "quality-evidence.json"
    if not evidence.is_file():
        return None
    manifest = json.loads(Path(json.loads(evidence.read_text())["source"]).read_text())
    stat = artifact.stat()
    file = manifest["artifact"]
    if (str(artifact),stat.st_size,stat.st_mtime_ns) != (file["path"],file["bytes"],file["mtime_ns"]):
        return None
    for arm, path in probes.items():
        if not path.is_file():
            return None
        with path.open("rb") as binary:
            if hashlib.file_digest(binary,"sha256").hexdigest() != manifest["probes"][arm]["sha256"]:
                return None
    return evidence


def run_stage(s):
    a = s.args
    build_root = a.results / "build"
    old, new = build_root/"baseline-70", build_root/"candidate-70"
    script = a.candidate_source / "bench/v100"
    if a.stage in ("preflight","build"):
        inspect(a)
    if a.stage == "preflight":
        for label, tree in (("baseline",a.baseline_source),("candidate",a.candidate_source)):
            s.run(f"{label}-revision", ["git","-C",tree,"rev-parse","HEAD"])
            s.run(f"{label}-status", ["git","-C",tree,"status","--short"])
        s.run("nvcc", [a.cuda/"bin/nvcc","--version"])
        s.add("cutlass","pass", version="4.4.2", header=a.cutlass/"include/cutlass/version.h")
        s.run("gpu", ["nvidia-smi","-q"])
        s.add("private-fixtures", "pass" if a.fixture_root.is_dir() else "open", path=a.fixture_root)
    elif a.stage == "build":
        for name, tree, arch in (("baseline-70", a.baseline_source, "70"),
                                 ("candidate-70", a.candidate_source, "70"),
                                 ("candidate-mixed", a.candidate_source, "70;75"),
                                 ("candidate-nonvolta", a.candidate_source, "75")):
            dest = build_root/name
            opts = ["cmake", "-S", tree, "-B", dest, "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release","-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                    f"-DCMAKE_CUDA_COMPILER={a.cuda}/bin/nvcc", f"-DCUDAToolkit_ROOT={a.cuda}",
                    f"-DNINFER_V100_CUTLASS_SOURCE={a.cutlass}", "-DNINFER_BUILD_APPS=ON",
                    "-DNINFER_BUILD_BENCHMARKS=ON", "-DBUILD_TESTING=ON",
                    f"-DPython3_EXECUTABLE={sys.executable}"]
            if name == "candidate-mixed":
                opts.append("-DNINFER_CUDA_COMPRESSION_MODE=speed")
            if arch == "70":
                opts += [f"-DCMAKE_PROJECT_ninfer_INCLUDE={script}/quality_probe.cmake",
                         "-DNINFER_V100_QUALITY_W4A8="+("ON" if tree == a.candidate_source else "OFF")]
            if s.run("configure-"+name, opts):
                if name == "candidate-nonvolta":
                    nonvolta_check(s,dest)
                else:
                    targets = []
                    if name == "candidate-mixed":
                        mixed_codegen_check(s, dest)
                        targets = ["--target","ninfer-serve"]
                    elif name == "baseline-70":
                        # The unrelated stand-alone multi-GPU probe in the pinned tree lacks
                        # its peer-preflight object. Build every product, test and benchmark.
                        targets = ["--target","apps/all","tests/all","bench/all","ninfer-v100-quality-probe"]
                    s.run("compile-"+name, ["cmake", "--build", dest, "-j"] +
                          ([] if a.jobs is None else [str(a.jobs)]) + targets)
    elif a.stage == "cpu":
        s.ctest("candidate-cpu",new,CPU)
        s.run("runner-tests",[sys.executable,"-m","unittest","discover","-s",script,"-v"])
    elif a.stage == "correctness":
        for name, dest in (("baseline",old),("candidate",new)):
            s.ctest(name+"-ops",dest,GPU,dict(os.environ,NINFER_OP_REPORT_STATS="1"))
        s.ctest("flash-suspend",new,"^ninfer_qwen3_5_volta_flash_suspend_real_test$",
                dict(os.environ,NINFER_TEST_ARTIFACT=str(a.artifact)))
        dflash = os.environ.get("NINFER_TEST_V100_DFLASH2_ARTIFACT")
        if dflash:
            s.ctest("dflash2-fused-heads",new,r"^ninfer_qwen3_5_v100_dflash2_(full|optimized)_real_test$")
        else:
            s.add("dflash2-fused-heads","skipped-optional", reason="NINFER_TEST_V100_DFLASH2_ARTIFACT absent")
        sanitizer = a.cuda/"bin/compute-sanitizer"
        s.run("flash-memcheck",[sanitizer,"--tool","memcheck","--error-exitcode","85",
                                new/"tests/ninfer_tests","ninfer_softmax_attention_test","--volta-flash-only"],
              env=dict(os.environ,NINFER_V100_CANDIDATES="flash"),
              classify=lambda rc: "open" if rc==77 else "pass" if rc==0 else "fail")
        for kv in ("int8","bf16"):
            for scenario in ("restore-exact-mtp","interleaved","full-pool","persist","vision"):
                s.ctest(f"cache-{kv}-{scenario}",new,"^ninfer_qwen3_5_hybrid_prefix_real_test$",
                        dict(os.environ,NINFER_HYBRID_V100_POLICY="ported",NINFER_TEST_ARTIFACT=str(a.artifact),
                             NINFER_HYBRID_KV_DTYPE=kv,
                             NINFER_HYBRID_REAL_SCENARIO=scenario))
    elif a.stage == "gdn":
        for heads in (32,48):
            for width in (15,16,32,63,64,65,128,512,1024,2048):
                modes = ["profile","recurrent"]
                if width >= 16:
                    modes.append("two-stage")
                if width >= 64:
                    modes.append("chunked")
                for mode in modes:
                    env = dict(os.environ,NINFER_V100_CANDIDATES="gdn")
                    env.pop("NINFER_GDN_TWO_STAGE",None)
                    if mode in ("two-stage","chunked"):
                        env["NINFER_GDN_TWO_STAGE"]="1" if mode=="two-stage" else "0"
                    s.run(f"{heads}-{width}-{mode}",[new/"bench/ninfer_benches",
                          "ninfer_gated_delta_net_bench","--running","--qk-heads","16",
                          "--value-heads",heads,"--tokens",width,"--warmup","20","--repeat","100","--csv",
                          *(["--v100-ported"] if mode=="recurrent" else [])],env=env)
    elif a.stage in ("quality","performance"):
        if a.stage == "quality":
            argv = [script/"quality.py","--baseline-probe",old/"ninfer-v100-quality-probe",
                    "--candidate-probe",new/"ninfer-v100-quality-probe","--fixtures",a.fixture_root]
        else:
            argv = [script/"qualify.py","--baseline",old/"apps/ninfer-serve","--candidate",
                    new/"apps/ninfer-serve","--cache-fixture",a.fixture_root/"cache.json",
                    "--deployment-fixture",a.fixture_root/"deployment.json"]
            try:
                evidence = current_quality(a.results,a.artifact,{
                    "baseline":old/"ninfer-v100-quality-probe","candidate":new/"ninfer-v100-quality-probe"})
            except (ValueError,OSError,KeyError) as exc:
                evidence = None
                s.add("quality-evidence","open",reason=str(exc))
            if evidence:
                argv += ["--quality-evidence",evidence]
        s.run(a.stage,[sys.executable,*argv,"--artifact",a.artifact,"--output",s.folder/"observations"],
              classify=lambda rc: {0:"pass",2:"open"}.get(rc,"fail"))


def workload_rows(result_root):
    for attempt_dir in sorted(result_root.glob("performance-*")):
        root = attempt_dir / "observations"
        trials = root / "trials.jsonl"
        if trials.is_file():
            with trials.open(encoding="utf-8") as stream:
                for line in stream:
                    trial = json.loads(line)
                    base = {key:trial.get(key) for key in ("phase","kv","drafts","repeat","arm","status","error")}
                    base["attempt"] = attempt_dir.name
                    folder = root / trial["phase"] / (
                        f"{trial['kv']}-mtp{trial['drafts']}-{trial['repeat']:02d}") / trial["arm"]
                    requests = (sorted(folder.glob("*/request-*.json")) if trial["phase"] in ("cache","deployment")
                                else [folder / f"{name}.json" for name in
                                      ("prose","code","list","prefill-11k","prefill-33k")])
                    recorded = False
                    for path in requests:
                        if not path.is_file():
                            continue
                        record = json.loads(path.read_text())
                        metrics = {key:record.get(key) for key in (
                            "prompt_tokens","output_tokens","computed_tokens","cached_tokens",
                            "decode","prefill","ttft","ms_per_round","ms_per_mtp_round","draft_acceptance","decode_rounds","elapsed")}
                        metrics["engine_prefill_s"] = record["timings"]["prefill"]
                        metrics["speculative"] = record.get("speculative")
                        audit = trial.get("cache_evidence",{}).get(path.parent.name,trial)
                        metrics["cache_path"] = record.get("cache_path")
                        profile = trial.get("deployment", {})
                        metrics["requested_capacity"] = profile.get("requested_capacity")
                        metrics["minimum_capacity"] = profile.get("minimum_capacity")
                        spec = profile.get("checks", {}).get(path.parent.name, {})
                        if spec:
                            audit = spec
                            index = int(path.stem.removeprefix("request-"))
                            observations = spec.get("requests", [])
                            if index < len(observations):
                                for key in ("expected_prompt_tokens","minimum_cached_tokens","maximum_cached_tokens"):
                                    metrics[key] = observations[index].get(key)
                                metrics["fixture_request"] = observations[index].get("name")
                            metrics["context"] = profile["context"]
                            metrics["check_status"] = spec.get("status")
                        yield dict(base, kind="request", workload=str(path.relative_to(folder)),
                                   route_selections=audit.get("route_selections"), memory=audit.get("gpu_memory"),
                                   capacity=spec.get("observed_capacity",audit.get("engine",{}).get("kv_capacity")), **metrics)
                        recorded = True
                    if not recorded or base["status"] != "ok":
                        yield dict(base, kind="trial", memory=trial.get("gpu_memory"),
                                   capacity=trial.get("engine",{}).get("kv_capacity"))
        summary = root / "summary.json"
        if summary.is_file():
            for gate in json.loads(summary.read_text())["gates"]:
                yield dict(attempt=attempt_dir.name, kind="gate", **gate)


def write_workloads(root):
    # Keep the CSV rectangular while preserving per-workload confidence, diagnostics and failures.
    rows = list(workload_rows(root))
    for checks in sorted(root.glob("*-*/checks.json")):
        for check in json.loads(checks.read_text()):
            rows.append(dict(kind="check", attempt=checks.parent.name, phase=checks.parent.name.rsplit("-",1)[0],
                             case=check["name"], status=check["status"],
                             reason=check.get("reason"), evidence=check.get("evidence",check.get("log"))))
    keys = dict.fromkeys(("attempt","kind","phase","kv","drafts","repeat","arm","workload","case","status"))
    for row in rows:
        keys.update(dict.fromkeys(row))
    with (root/"workloads.csv").open("w",newline="",encoding="utf-8") as out:
        table = csv.DictWriter(out, fieldnames=keys)
        table.writeheader()
        for row in rows:
            table.writerow({k: json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def report(args):
    checks = {}
    for stage in STAGES[:-1]:
        reports = sorted(args.results.glob(stage+"-*/checks.json"))
        checks[stage] = [dict(path=str(p), rows=json.loads(p.read_text())) for p in reports]
    # Some release gates require the test agent's interpretation even after timings exist.
    manual = ["choose measured per-geometry GDN thresholds", "qualified context and cache frontiers for both profiles",
              "restart, changed media, near-full admission, concurrency and incompatible-policy persistence",
              "automatic selection on accepted profile followed by explicit native rollback"]
    write(args.results/"report.json",dict(baseline=BASE, candidate_source=args.candidate_source,
          attempts=checks, unrun=[k for k,v in checks.items() if not v], human_review=manual,
          promotion="pending V100 evidence and review"))
    write_workloads(args.results)
    print(args.results/"report.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=STAGES)
    for name in ("baseline-source","candidate-source","artifact","fixture-root","results","cutlass","cuda"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--jobs",type=int,help="override only when parallel compilation exceeds host resources")
    a = p.parse_args()
    if sys.version_info[:2] != (3,11):
        p.error("use Python 3.11")
    for k, value in vars(a).items():
        if isinstance(value,Path): setattr(a,k,value.resolve())
    if a.jobs is not None and a.jobs < 1: p.error("--jobs must be positive")
    a.results.mkdir(parents=True,exist_ok=True)
    if a.stage == "report":
        report(a)
        return 0
    folder = attempt(a.results,a.stage)
    write(folder/"inputs.json",vars(a))
    s = Session(a,folder)
    try:
        run_stage(s)
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        s.add(a.stage,"fail",reason=str(exc))
    return 1 if any(i["status"]=="fail" for i in s.rows) else 2 if any(i["status"]=="open" for i in s.rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
