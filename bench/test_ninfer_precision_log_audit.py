"""Error markers from flat suites and bounded wide replay attempts."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ninfer_precision_analysis import run_error_markers


class PrecisionLogAuditTests(unittest.TestCase):
    def test_attempt_failures_and_archived_pilot_markers_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)/'runs/r5'
            expected={
                'native-k3-before-cache-control/server.log':['FATAL archived watchdog'],
                'quality-k0/process.log':['ERROR generated fixture'],
                'operators-wide-k0/attempts/attempt-001/process.log':['ERROR qualification failed','force-kill child'],
                'operators-wide-k0/attempts/attempt-002/server.log':['CRITICAL retry failure'],
                'operators-wide-k0-before-resume-attempt-001/attempts/attempt-001/process.log':['engine recovery'],
            }
            ignored=[
                'operators-wide-k0/source/build/process.log',
                'operators-wide-k0/attempts/attempt-001/source/build/process.log',
                'operators-wide-k0/attempts/not-an-attempt/process.log',
                '../other-engine/quality-k0/process.log',
                '../../probes/r5/build/process.log',
            ]
            for name,lines in expected.items():
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('ordinary progress\n'+'\n'.join(lines)+'\n')
            for name in ignored:
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('ERROR unrelated source/build log\n')
            actual={r['path']:r['lines'] for r in run_error_markers(base)}
            self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
