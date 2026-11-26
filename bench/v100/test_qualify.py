import argparse
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify


CASES = qualify.SPEED_CASES


def rows(phase, change, kv="int8", drafts=3, old="baseline", new="candidate"):
    return [
        {"phase": phase, "kv": kv, "drafts": drafts, "repeat": repeat,
         "arm": arm, "status": "ok", "cases": {case: changes.get(case, 100) for case in CASES},
         "counts": {case: [325, 300] for case in CASES}}
        for repeat in range(1, qualify.REPEATS + 1)
        for arm, changes in ((old, {}), (new, change))
    ]


class PerformanceGates(unittest.TestCase):
    def test_geometric_mean_accepts_three_percent(self):
        evidence = rows("cold", dict.fromkeys(qualify.PROSE, 103))
        gates = qualify.comparison("cold", evidence, "int8", 3, "baseline", "candidate")
        self.assertTrue(all(gate["status"] == "pass" for gate in gates))
        self.assertAlmostEqual(gates[-1]["estimate"], .03)

    def test_faster_average_does_not_hide_slow_prompt(self):
        evidence = rows("cold", {"prose": 96, "code": 119, "list": 119})
        gates = qualify.comparison("cold", evidence, "int8", 3, "baseline", "candidate")
        indexed = {gate["case"]: gate["status"] for gate in gates}
        self.assertEqual(indexed["prose"], "fail")
        self.assertEqual(indexed["MTP3 decode geometric mean"], "pass")

    def test_missing_failed_and_mismatched_trials_are_classified(self):
        evidence = rows("cold", {})
        self.assertEqual(qualify.comparison(
            "cold", evidence[:-1], "int8", 3, "baseline", "candidate")[0]["status"], "open")
        evidence[0]["status"] = "fail"
        self.assertEqual(qualify.comparison(
            "cold", evidence, "int8", 3, "baseline", "candidate")[0]["status"], "fail")
        evidence[0]["status"] = "ok"
        evidence[0]["counts"]["code"] = [324, 300]
        self.assertEqual(qualify.comparison(
            "cold", evidence, "int8", 3, "baseline", "candidate")[0]["status"], "fail")

    def test_flash_requires_positive_long_prefill(self):
        evidence = rows("flash", {"prefill-11k": 105, "prefill-33k": 100})
        gates = qualify.comparison("flash", evidence, "int8", 3, "baseline", "candidate")
        indexed = {gate["case"]: gate["status"] for gate in gates}
        self.assertEqual(indexed["prefill-33k"], "fail")

    def test_cache_latency_threshold_uses_new_over_old(self):
        evidence = rows("cache", {})
        for row in evidence:
            row["cases"] = {"warm": 1.0201 if row["arm"] == "candidate" else 1.0,
                            "pressure": 1.0}
        gates = qualify.comparison("cache", evidence, "int8", 3, "baseline", "candidate")
        warm = next(gate for gate in gates if gate["case"] == "warm")
        self.assertAlmostEqual(warm["estimate"], -.0201)
        self.assertEqual(warm["status"], "fail")


class Deployment(unittest.TestCase):
    @staticmethod
    def profile():
        def request(name, tokens, cached=0):
            return {"name": name, "body": qualify.prompt(name, 5), "prompt_tokens": tokens,
                    "minimum_cached_tokens": cached, "maximum_cached_tokens": cached,
                    "expected_text": "answer"}

        return {"context": 65536, "minimum_capacity": 64000, "kv_capacity": 66560,
                "prefill_chunk": 2048, "drafts": 0, "vision": True,
                "host_cache_mib": 4096, "device_snapshot_slots": 2,
                "near_limit": [request("long prompt", 65000)],
                "frontier": [request("prime", 1500), request("known shared prefix", 1600, 1536)],
                "restart": {"prime": [request("persist", 1800)],
                            "after": [request("resume", 2000, 1792)]}}

    def test_locks_long_admission_and_positive_exact_frontier(self):
        value = self.profile()
        self.assertEqual(qualify.deployment_profile(value).requested_capacity, 66560)
        value["near_limit"][0]["prompt_tokens"] = 62000
        with self.assertRaisesRegex(ValueError, "near-limit"):
            qualify.deployment_profile(value)
        value = self.profile()
        value["frontier"][1]["maximum_cached_tokens"] = 1600
        with self.assertRaisesRegex(ValueError, "exact positive reuse frontier"):
            qualify.deployment_profile(value)
        value = self.profile()
        value["restart"]["after"][0]["minimum_cached_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "without priming"):
            qualify.deployment_profile(value)

    def test_profile_specific_serve_arguments(self):
        args = argparse.Namespace(baseline=Path("old"), candidate=Path("new"), artifact=Path("model"),
                                  device=1, cold_context=4096, vision=False)
        profile = qualify.deployment_profile(self.profile())
        run = qualify.Run("deployment", "bf16", profile.drafts, 1, qualify.OLD, profile)
        command = qualify.server_command(args, run, 10001, Path("log"), Path("private.cache"))
        for option, expected in {"--max-context": "65536", "--kv-capacity": "66560",
                                 "--kv-dtype": "bf16", "--prefill-chunk": "2048",
                                 "--host-cache-mib": "4096", "--device-snapshot-slots": "2",
                                 "--prefix-cache-file": "private.cache", "--spec": "none"}.items():
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(command[0], "old")
        self.assertIn("--vision", command)

    def test_records_long_prompt_restart_and_capacity_failure_independently(self):
        value = self.profile()
        engine = {"max_context": value["context"], "kv_capacity_mode": "explicit",
                  "kv_capacity": 66560, "vision": True, "prefix_reuse": True,
                  "speculative_backend": "none",
                  "context_cache": {"enabled": True, "mode": "hybrid",
                                    "host_cache_budget_bytes": 4096 * 2**20,
                                    "hybrid": {"device_snapshot_slots": 2, "persistent": True}}}
        batches = (value["near_limit"], value["frontier"], *value["restart"].values())
        requests = {item["name"]: item for batch in batches for item in batch}
        starts = []

        @contextmanager
        def serve(_args, run, folder, cache):
            starts.append((run.arm.name, folder.name))
            folder.mkdir(parents=True)
            if folder.name == "restart":
                self.assertEqual(cache.read_bytes(), b"saved")
                (folder / "server.log").write_text("prefix cache restored | stored snapshots")
            observed = dict(engine, kv_capacity=63999) if (
                run.arm == qualify.NEW and folder.name == "near_limit") else engine
            yield "base", Path("log"), {"engine": observed}
            cache.write_bytes(b"saved")

        def measure(_base, _log, body, _args, _folder, _name):
            item = requests[body["messages"][0]["content"]]
            prompt = item["prompt_tokens"]
            cached = item["minimum_cached_tokens"]
            return {"prompt_tokens": prompt, "computed_tokens": prompt - cached,
                    "cached_tokens": cached, "cache_path": "host" if cached else "miss",
                    "text": "answer", "output_tokens": 5, "ttft": .5, "elapsed": 1.5}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "deployment.json"
            qualify.dump(path, {"int8": value})
            args = argparse.Namespace(output=output, deployment_fixture=path)
            rows, gates = [], []
            with patch.object(qualify, "serving", serve), \
                    patch.object(qualify, "measured_request", measure), \
                    patch.object(qualify, "trials_evidence", return_value={}):
                qualify.run_deployments(args, rows, gates)
        self.assertEqual([(gate["kv"], gate["status"]) for gate in gates],
                         [("int8", "fail"), ("bf16", "open")])
        self.assertEqual(starts, [(arm, case) for arm in ("baseline", "candidate") for case in
                                  ("near_limit", "frontier", "restart_prime", "restart")])
        old, new = [row["deployment"]["checks"] for row in rows]
        self.assertEqual(old["near_limit"]["requests"][0]["computed_tokens"], 65000)
        self.assertEqual(new["near_limit"]["observed_capacity"], 63999)
        self.assertEqual(new["frontier"]["requests"][1]["cached_tokens"], 1536)
        self.assertTrue(new["restart"]["restored"])

    def test_optional_w4a8_failure_preserves_a16_cache_trial(self):
        item = {"body": qualify.prompt("saved", 5), "prompt_tokens": 100,
                "minimum_cached_tokens": 0, "expected_text": "base"}

        @contextmanager
        def serve(_args, run, folder, cache):
            if run.arm == qualify.W4A8:
                raise RuntimeError("W4A8 launch failed")
            folder.mkdir(parents=True)
            (folder / "server.log").write_text(
                "prefix cache not restored | saved prefix cache belongs to a different model, "
                "KV format or build")
            yield "base", Path("log"), {}
            cache.write_bytes(b"saved")

        result = {"cached_tokens": 0, "computed_tokens": 100, "prompt_tokens": 100,
                  "output_tokens": 5, "text": "base", "ttft": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), evaluate_w4a8=True)
            fixture = {"warm": [item, item], "pressure": [item]}
            run = qualify.Run("cache", "int8", 3, 1, qualify.NEW)
            with patch.object(qualify, "serving", serve), \
                    patch.object(qualify, "measured_request", return_value=result), \
                    patch.object(qualify, "trials_evidence", return_value={"engine": {"kv_capacity": 100}}):
                row = qualify.run_trial(args, run, fixture)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["cases"], {"warm": 1.0, "pressure": 1.0})
        gates = {gate["case"]: gate for gate in qualify.cache_check_results([row], evaluate_w4a8=True)}
        self.assertEqual(gates["incompatible_native_cache"]["status"], "pass")
        self.assertEqual(gates["incompatible_w4a8_cache"]["status"], "fail")
        self.assertEqual(gates["incompatible_w4a8_cache"]["errors"], ["W4A8 launch failed"])


class InputsAndCommands(unittest.TestCase):
    def test_speed_prompt_lengths_use_the_artifact_tokenizer(self):
        def tokenize(_base, path, _timeout, body):
            self.assertEqual(path, "/tokenize")
            # Model a tokenizer that uses two tokens per word so a word count is insufficient.
            return io.StringIO(json.dumps({"tokens": [1, 2] * len(body["content"].split())}))

        with patch.object(qualify, "endpoint", side_effect=tokenize):
            workloads = qualify.speed_workloads("http://localhost", 3)
        for name, target in (("prefill-11k", 11600), ("prefill-33k", 33000)):
            self.assertEqual(2 * len(workloads[name]["messages"][0]["content"].split()), target)

    def test_order_and_ablation_scope(self):
        plan = list(qualify.matrix())
        self.assertEqual(plan[0], ("controls", "int8", 3, qualify.OLD_NATIVE, qualify.OLD))
        self.assertEqual(plan[1], ("controls", "int8", 3, qualify.NATIVE, qualify.CONTROL))
        for name, row in zip(qualify.CANDIDATES, plan[2:6], strict=True):
            phase, kv, drafts, control, candidate = row
            self.assertEqual((phase, kv, drafts), (name, "bf16" if name == "flash" else "int8", 3))
            self.assertEqual(control.candidates, "none")
            self.assertEqual(candidate.candidates, name)
        self.assertEqual(plan[6][0], "cold")

    def test_route_counts_are_real_and_arm_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("[info] v100 kernel route selections: ported=32 fallback=5 "
                            "head=7 drafter=0 gdn=0 flash=0 w4a8_narrow=0 w4a8_wide=0 "
                            "(graph replays not counted)\n", encoding="utf-8")
            counts = qualify.route_selections(path)
            treatment = qualify.Run("head", "int8", 3, 1, qualify.Arm("head", candidates="head"))
            qualify.check_route_selections(treatment, counts)
            with self.assertRaisesRegex(RuntimeError, "missing=.*head"):
                qualify.check_route_selections(treatment, dict(counts, head=0))
            with self.assertRaisesRegex(RuntimeError, "disabled=.*flash"):
                qualify.check_route_selections(treatment, dict(counts, flash=1))
            with self.assertRaisesRegex(RuntimeError, "all named route counters"):
                qualify.check_route_selections(treatment, {"ported": 1, "fallback": 0})
            legacy = qualify.Run("controls", "int8", 3, 1, qualify.OLD_NATIVE)
            qualify.check_route_selections(legacy, {"ported": 0, "fallback": 0})
            qualify.check_route_selections(legacy, {"ported": 0, "fallback": 42})
            native = qualify.Run("controls", "int8", 3, 1, qualify.NATIVE)
            native_counts = dict.fromkeys(qualify.CANDIDATES + qualify.W4A8_COUNTERS, 0)
            native_counts.update(ported=0, fallback=128)
            qualify.check_route_selections(native, native_counts)
            with self.assertRaisesRegex(RuntimeError, "native control"):
                qualify.check_route_selections(legacy, {"ported": 1, "fallback": 42})
            with self.assertRaisesRegex(RuntimeError, "native control"):
                qualify.check_route_selections(native, dict(native_counts, head=1))
            path.write_text("[info] v100 kernel route selections: ported=32 fallback=5\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected one final"):
                qualify.route_selections(path)

    def test_incompatible_native_cache_probe_allows_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            log = folder / "server.log"
            qualify.dump(folder / "startup.json", {"engine": {"v100_kernels": "native"}})
            qualify.dump(folder / "gpu-memory.json", {"samples": []})
            run = qualify.Run("cache", "int8", 3, 1, qualify.NATIVE)
            record = ("[info] v100 kernel route selections: ported={ported} fallback=128 "
                      "head=0 drafter=0 gdn=0 flash=0 w4a8_narrow=0 w4a8_wide=0 "
                      "(graph replays not counted)\n")
            log.write_text(record.format(ported=0), encoding="utf-8")
            observed = qualify.trials_evidence(folder, run, smoke=True)
            self.assertEqual(observed["route_selections"]["fallback"], 128)
            log.write_text(record.format(ported=1), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "native control selected"):
                qualify.trials_evidence(folder, run, smoke=True)

    def test_cache_capacity_is_shared_and_changed_media_has_reuse_ceiling(self):
        records = [{"phase": "capacity", "kv": "int8", "arm": arm.name, "status": "ok",
                    "engine": {"kv_capacity": size}} for arm, size in
                   ((qualify.OLD, 270976), (qualify.NEW, 271040))]
        self.assertEqual(qualify.matched_cache_capacity(records), 270976)
        with self.assertRaisesRegex(RuntimeError, "both cache arms"):
            qualify.matched_cache_capacity(records[:1])
        fixture = {"prompt_tokens": 100, "minimum_cached_tokens": 80,
                   "maximum_cached_tokens": 93, "expected_text": "blue"}
        result = {"prompt_tokens": 100, "cached_tokens": 93, "text": "blue"}
        qualify.verify_cache_request(fixture, result, "media")
        with self.assertRaisesRegex(RuntimeError, "prefix reuse expectations"):
            qualify.verify_cache_request(fixture, dict(result, cached_tokens=94), "media")

    def test_incompatible_policy_checks_cold_frontier_and_its_own_text(self):
        fixture = {"prompt_tokens": 100, "minimum_cached_tokens": 90,
                   "expected_text": "base", "expected_w4a8_text": "alternate"}
        cold = {"prompt_tokens": 100, "computed_tokens": 100, "cached_tokens": 0,
                "text": "alternate"}
        qualify.verify_cache_request(fixture, cold, "policy", cold=True, text_key="expected_w4a8_text")
        qualify.verify_cache_request(fixture, cold, "policy", cold=True, text_key="expected_native_text")
        with self.assertRaisesRegex(RuntimeError, "prefix reuse expectations"):
            qualify.verify_cache_request(fixture, dict(cold, computed_tokens=99), "policy",
                                         cold=True, text_key="expected_w4a8_text")
        with self.assertRaisesRegex(RuntimeError, "output"):
            qualify.verify_cache_request(fixture, dict(cold, text="base"), "policy", cold=True,
                                         text_key="expected_w4a8_text")

    def test_cache_shape_safeguards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            item = {"body": qualify.prompt("example", 5), "prompt_tokens": 90787,
                    "minimum_cached_tokens": 90000, "expected_text": "example answer"}
            qualify.dump(path, {"warm": [item, item], "pressure": [item] * 47})
            self.assertEqual(len(qualify.cache_fixture(path)["pressure"]), 47)
            qualify.dump(path, {"warm": [dict(item, prompt_tokens=90786), item],
                                "pressure": [item] * 47})
            with self.assertRaisesRegex(ValueError, "90,787"):
                qualify.cache_fixture(path)

    def test_coverage_and_equivalence_for_quality(self):
        self.assertEqual(qualify.quality(None)[0]["status"], "open")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            documents = [{"id": f"d{i:02d}", "tokens": 8192, "control_mean_nll": 1.5,
                          "candidate_mean_nll": 1.501} for i in range(16)]
            tasks = [{"id": f"t{i:03d}", "cluster": f"case-{i // 4}", "candidate": True,
                      "control": True} for i in range(384)]
            comparisons = [
                dict(comparison=mode, kv_dtype=kv,
                     score_mode="score-narrow" if mode == "w4a8" else "score",
                     documents=documents, tasks=tasks)
                for mode in ("a16", "w4a8") for kv in ("int8", "bf16")
            ]
            qualify.dump(path, {"source": "paired raw records", "comparisons": comparisons})
            gates = qualify.quality(path)
            self.assertEqual(len(gates), 8)
            self.assertTrue(all(item["status"] == "pass" for item in gates))
            narrow = comparisons[2]
            qualify.dump(path, {"source": "paired raw records", "comparisons": [dict(narrow, score_mode="score")]})
            gates = qualify.quality(path)
            self.assertIn("quality needs matched score-narrow evidence", [item.get("reason") for item in gates])
            regressed = [dict(item, candidate_mean_nll=1.51) for item in documents]
            qualify.dump(path, {"source": "paired raw records", "comparisons": [dict(narrow, documents=regressed)]})
            gate = next(item for item in qualify.quality(path) if item.get("case") == "perplexity")
            self.assertEqual((gate["comparison"], gate["kv"], gate["status"]), ("w4a8", "int8", "fail"))
            documents[0]["tokens"] -= 1
            qualify.dump(path, {"source": "paired raw records", "comparisons": comparisons})
            self.assertEqual(qualify.quality(path)[0]["status"], "open")

    def test_baseline_flags_and_w4a8_comparison_arms(self):
        args = argparse.Namespace(baseline=Path("old"), candidate=Path("new"), artifact=Path("weights"),
                                  device=0, cache_context=262144, cold_context=65536,
                                  cache_kv_capacity="auto", cache_prefill_chunk=2048,
                                  cache_vision=True, vision=False)
        old = qualify.Run("cold", "int8", 3, 1, qualify.OLD)
        new = qualify.Run("w4a8", "int8", 3, 1, qualify.W4A8)
        cache = qualify.Run("cache", "int8", 3, 1, qualify.NEW)
        base = qualify.server_command(args, old, 10000, Path("old.log"))
        treatment = qualify.server_command(args, new, 10000, Path("new.log"))
        cache_run = qualify.server_command(args, cache, 10000, Path("cache.log"))
        self.assertEqual(base[0], "old")
        self.assertIn("--no-prefix-reuse", base)
        native = qualify.Run("controls", "int8", 3, 1, qualify.OLD_NATIVE)
        native_command = qualify.server_command(args, native, 10000, Path("old.log"))
        self.assertEqual(native_command[native_command.index("--v100-kernels") + 1], "native")
        self.assertNotIn("--nvfp4-w4a8-int", base)
        self.assertIn("--nvfp4-w4a8-int", treatment)
        self.assertIn("--vision", cache_run)
        self.assertNotIn("--no-prefix-reuse", cache_run)
        self.assertEqual(cache_run[cache_run.index("--host-cache-mib") + 1], "20480")

    def test_cold_measurements_do_not_accept_cache_or_missing_mtp(self):
        body = qualify.prompt("example", 300)
        result = {"cached_tokens": 1, "computed_tokens": 3, "prompt_tokens": 4,
                  "output_tokens": 300, "speculative": {"drafted_tokens": 0}}
        with self.assertRaisesRegex(RuntimeError, "fully cold prefill"):
            qualify.valid_speed_case("code", body, result, 3)
        result.update(cached_tokens=0, computed_tokens=4)
        with self.assertRaisesRegex(RuntimeError, "did not exercise MTP"):
            qualify.valid_speed_case("code", body, result, 3)

    def test_stream_timing_and_log_counts_join(self):
        class Response(list):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        response = Response([
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"word"}}]}\n',
            b'data: {"usage":{"prompt_tokens":21,"completion_tokens":4},"choices":[]}\n',
            b'data: [DONE]\n',
        ])
        done = {"event": "request_done", "result": {
            "prompt_tokens": 21, "completion_tokens": 4, "computed_prefill_tokens": 11,
            "prefix_cache_hit_tokens": 10, "prefix_reuse_path": "device",
        }, "timings_seconds": {"decode": .04, "prefill": .02},
            "engine_timing": {"decode": {"rounds": 2}},
            "speculative": {"drafted_tokens": 2, "accepted_tokens": 1, "rounds": 1}}
        with patch.object(qualify, "endpoint", return_value=response), \
                patch.object(qualify, "read_jsonl", side_effect=[[], [done]]):
            result = qualify.request_result("http://localhost", Path("requests.jsonl"), {}, 1)
        self.assertEqual(result["cached_tokens"], 10)
        self.assertEqual(result["text"], "word")
        self.assertEqual(result["decode"], 75)
        self.assertEqual(result["prefill"], 550)
        self.assertEqual(result["ms_per_round"], 20)
        self.assertEqual(result["ms_per_mtp_round"], 40)
        self.assertEqual(result["draft_acceptance"], .5)

    def test_canonical_int8_log_and_effective_draft_configuration(self):
        engine = dict(kv_cache="int8-group64", prefill_chunk=2048, max_concurrency=1,
                      cuda_graph=True, speculative_backend="mtp", speculative_draft_window=3,
                      proposal_head="optimized")
        qualify.verify_scheduling(engine,"int8",3,2048)
        for changed in (dict(kv_cache="bf16"), dict(proposal_head="full"), dict(speculative_draft_window=1)):
            with self.assertRaises(RuntimeError):
                qualify.verify_scheduling(engine | changed,"int8",3,2048)


if __name__ == "__main__":
    unittest.main()
