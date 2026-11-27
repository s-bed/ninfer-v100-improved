"""CPU checks for preserving and pairing the frozen quality protocol."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import quality


class QualityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.streams = [
            {"id": f"{prefix}-{index}", "domain": "text", "target_tokens": 8192}
            for prefix, count in (("heldout", 12), ("legacy", 4)) for index in range(count)
        ]
        self.cases = [
            {"id": f"{domain}-{index:03}", "domain": domain}
            for domain, count in quality.TASK_COUNTS.items() for index in range(count)
        ]
        self.order = [case["id"] for case in self.cases]
        tokens = list(range(8193))
        windows = [
            {"id": f"{row['id']}/{target}", "document": row["id"], "domain": row["domain"],
             "input_begin": begin, "target_begin": target, "target_end": end,
             "first_target": target - begin, "tokens": tokens[begin:end]}
            for row in self.streams for begin, target, end in sorted(quality.WINDOWS)
        ]
        for name, data in (("corpus.json", self.streams), ("quality.json", self.cases),
                           ("quality-order.json", self.order), ("score-windows.json", windows)):
            quality.dump(self.fixtures / name, data)
        quality.write_jobs(self.fixtures / "tokens.jsonl", [
            {"id": row["id"], "tokens": tokens} for row in self.streams
        ] + [{"id": row["id"], "tokens": [1, 2, 3]} for row in self.cases])
        for manifest, names in (
            ("freeze.json", ("corpus.json", "quality.json", "quality-order.json")),
            ("tokens-freeze.json", ("score-windows.json", "tokens.jsonl")),
        ):
            quality.dump(self.fixtures / manifest, {"files": {
                f"fixtures/{name}": hashlib.sha256((self.fixtures / name).read_bytes()).hexdigest()
                for name in names}})

    def test_hash_validation_precedes_execution(self):
        quality.load_fixture(self.root)
        (self.fixtures / "quality-order.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            quality.load_fixture(self.root)

    def test_complete_arms_pair_same_frozen_inputs(self):
        for name in ("baseline-probe", "candidate-probe", "model.ninfer"):
            (self.root / name).write_text(name)
        output = self.root / "quality-run"
        output.mkdir()
        calls = []

        def simulated_probe(command, log, env, _timeout):
            binary, mode, artifact, jobs, _, drafts, context, kv, arithmetic = command
            calls.append((Path(binary).name, mode, kv, arithmetic))
            self.assertEqual(drafts, "0")
            self.assertEqual(Path(artifact), self.root / "model.ninfer")
            self.assertEqual(context, "8192" if mode == "generate" else "4096")
            self.assertEqual(env["NINFER_V100_CANDIDATES"], quality.CANDIDATES)
            if Path(binary).name == "candidate-probe":
                count = 1000 if arithmetic == "w4a8" else 0
                log.write_text(f"v100 route selections: w4a8_narrow={count} w4a8_wide=0\n")
            records = {}
            for job in quality.read_jsonl(Path(jobs).read_bytes()):
                row = {"id": job["id"], "mode": mode, "k": 0}
                if mode != "generate":
                    first = job["first_target"]
                    nll = {("score", "a16"): 2.0, ("score-narrow", "a16"): 2.5,
                           ("score-narrow", "w4a8"): 2.502}[mode, arithmetic]
                    row.update(first_target=first, input_tokens=len(job["tokens"]),
                               logprobs=[-nll] * (len(job["tokens"]) - first))
                else:
                    row.update(content="reply", tokens=[3], reused_tokens=0, drafted_tokens=0,
                               accepted_tokens=0)
                records[job["id"]] = row
            return records

        args = argparse.Namespace(baseline_probe=self.root / "baseline-probe",
                                  candidate_probe=self.root / "candidate-probe",
                                  artifact=self.root / "model.ninfer", fixtures=self.root,
                                  output=output, timeout=10)
        with patch.object(quality, "run_probe", side_effect=simulated_probe), \
                patch.object(quality, "grade", return_value={"correct": True}):
            quality.run(args)

        self.assertEqual(len(calls), 14)
        self.assertEqual(Counter(arm for arm, _, _, _ in calls),
                         {"baseline-probe": 4, "candidate-probe": 10})
        for kv in quality.KV_TYPES:
            self.assertIn(("candidate-probe", "score-narrow", kv, "a16"), calls)
            self.assertIn(("candidate-probe", "score-narrow", kv, "w4a8"), calls)
        report = json.loads((output / "quality-evidence.json").read_bytes())
        self.assertEqual(len(report["comparisons"]), 4)
        for row in report["comparisons"]:
            self.assertEqual(len(row["tasks"]), 384)
            self.assertEqual(len(row["documents"]), 16)
            if row["comparison"] == "w4a8":
                self.assertEqual(row["score_mode"], "score-narrow")
                self.assertEqual(row["documents"][0]["control_mean_nll"], 2.5)
                self.assertAlmostEqual(row["documents"][0]["candidate_mean_nll"], 2.502)
            else:
                self.assertEqual(row["score_mode"], "score")
                self.assertEqual(row["documents"][0]["control_mean_nll"], 2.0)
                self.assertAlmostEqual(row["documents"][0]["candidate_mean_nll"], 2.0)
        self.assertEqual(len({entry["cluster"] for entry in report["comparisons"][0]["tasks"]}),
                         120)
        routes = json.loads((output / "manifest.json").read_bytes())["w4a8_narrow_selections"]
        for kv in quality.KV_TYPES:
            self.assertGreater(routes[f"{kv}-w4a8-score-narrow"], 0)
            self.assertEqual(routes[f"{kv}-a16-score-narrow"], 0)

    def test_missing_and_incomplete_scores_remain_ineligible(self):
        (self.fixtures / "tokens.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            quality.load_fixture(self.root)
        with self.assertRaisesRegex(ValueError, "every frozen window"):
            quality.score_documents({}, [{"id": "missing"}], "score")
        log = self.root / "probe.log"
        log.write_text("v100 route selections: w4a8_narrow=0 w4a8_wide=0\n")
        with self.assertRaisesRegex(ValueError, "expected W4A8 route"):
            quality.check_w4a8_route(log, True)


if __name__ == "__main__":
    unittest.main()
