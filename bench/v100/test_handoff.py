import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import handoff


class MixedCodegen(unittest.TestCase):
    def test_manifest_keeps_volta_leaves_native_and_dispatcher_mixed(self):
        volta = "--generate-code=arch=compute_70,code=[sm_70]"
        turing = "arch=compute_75,code=[compute_75,sm_75]"
        rows = [{"file": f"/src/{name}", "command": f"nvcc '{volta}' -c /src/{name}"}
                for name in handoff.VOLTA_LEAVES]
        rows.append({"file": "/src/sm70_project.cu",
                     "arguments": ["nvcc", volta, "-gencode", turing, "-c", "/src/sm70_project.cu"]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "compile_commands.json"
            session = Mock()
            handoff.write(path, rows)
            handoff.mixed_codegen_check(session, root)
            self.assertEqual(session.add.call_args.args, ("mixed-codegen", "pass"))
            codegen = session.add.call_args.kwargs["codegen"]
            self.assertEqual(codegen["v100_drafter_gemv.cu"][0]["sass"], ["sm_70"])
            self.assertEqual(codegen["sm70_project.cu"][0]["sass"], ["sm_70", "sm_75"])
            self.assertEqual(codegen["sm70_project.cu"][0]["ptx"], ["compute_75"])

            rows[0]["command"] += f" --generate-code='{turing}'"
            rows[-1]["arguments"] = ["nvcc", volta]
            rows[1]["command"] += " -gencode arch=compute_70,code=compute_70"
            handoff.write(path, rows)
            handoff.mixed_codegen_check(session, root)
            self.assertEqual(session.add.call_args.args, ("mixed-codegen", "fail"))
            reason = session.add.call_args.kwargs["reason"]
            self.assertIn("v100_cutlass.cu has SASS ['sm_70', 'sm_75']", reason)
            self.assertIn("v100_drafter_gemv.cu has SASS ['sm_70'] and PTX ['compute_70']", reason)
            self.assertIn("sm70_project.cu has SASS ['sm_70']", reason)


class Report(unittest.TestCase):
    def test_partial_cache_attempt_and_missing_gate_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = handoff.attempt(root,"performance") / "observations"
            requests = stage / "cache/int8-mtp3-01/candidate/warm"
            requests.mkdir(parents=True)
            trial = dict(phase="cache",kv="int8",drafts=3,repeat=1,arm="candidate",
                         status="fail",error="restart failed",
                         cache_evidence={"warm":{"engine":{"kv_capacity":99100},
                                                 "route_selections":{"head":23}}})
            (stage/"trials.jsonl").write_text(json.dumps(trial)+"\n")
            handoff.write(requests/"request-00.json",dict(timings={"prefill":0.8},ttft=0.9))
            handoff.write(stage/"summary.json",dict(gates=[dict(phase="quality",status="open")]))
            handoff.write_workloads(root)
            with (root/"workloads.csv").open() as source:
                observed, failed, missing = csv.DictReader(source)
            self.assertEqual((observed["capacity"], observed["ttft"]), ("99100","0.9"))
            self.assertEqual(json.loads(observed["route_selections"]), {"head":23})
            self.assertEqual((failed["kind"],failed["error"]), ("trial","restart failed"))
            self.assertEqual((missing["phase"],missing["status"]), ("quality","open"))
