"""Precision-only interruption accounting and restoration gates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import ninfer_precision_analysis as analysis


class PrecisionAccountingTests(unittest.TestCase):
    def ledger(self,root,name,used,sessions):
        path=root/'control'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'budget_seconds':21600 if name=='budget.json' else 7200,
            'reserve_seconds':1800 if name=='budget.json' else 600,
            'used_seconds':used,'sessions':sessions}))

    def session(self,seconds=12,**changes):
        return {'started':1,'finished':1+seconds,'seconds':seconds,'restored':True,**changes}

    def test_original_only_preserves_target_and_restoration_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.ledger(root,'budget.json',12,[self.session()])
            result=analysis.interruption_accounting(root,require_restored=True)
            self.assertEqual(result['used_seconds'],12)
            self.assertEqual(result['session_count'],1)
            self.assertEqual(result['original_target_seconds'],21600)
            self.assertEqual(result['original_restoration_reserve_seconds'],1800)
            self.assertFalse(result['continuation_started'])

    def test_continuation_is_added_and_agentic_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.ledger(root,'budget.json',20000,[self.session(20000)])
            self.ledger(root,'resume-budget.json',2400,[self.session(2400)])
            unrelated=root/'agentic/control/budget.json';unrelated.parent.mkdir(parents=True)
            unrelated.write_text(json.dumps({'used_seconds':999999,'sessions':[{'started':1}]}))
            result=analysis.interruption_accounting(root,require_restored=True)
            self.assertEqual(result['used_seconds'],22400)
            self.assertEqual(result['session_count'],2)
            self.assertEqual(result['over_original_target_seconds'],800)
            self.assertEqual(result['original_target_seconds'],21600)
            self.assertTrue(result['continuation_started'])
            self.assertEqual({s['ledger'] for s in result['sessions']},
                {'control/budget.json','control/resume-budget.json'})

    def test_empty_unstarted_resume_ledger_contributes_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.ledger(root,'budget.json',12,[self.session()])
            resume=root/'control/resume-budget.json'
            for data in [{},{'used_seconds':0,'sessions':[]}]:
                with self.subTest(data=data):
                    resume.write_text(json.dumps(data))
                    result=analysis.interruption_accounting(root,require_restored=True)
                    self.assertEqual(result['used_seconds'],12)
                    self.assertEqual(result['session_count'],1)
                    self.assertFalse(result['continuation_started'])

    def test_finalizer_refuses_any_unfinished_or_unrestored_precision_session(self):
        for ledger_name in ['budget.json','resume-budget.json']:
            for changes in [{'finished':None},{'restored':False},{'restored':None}]:
                with self.subTest(ledger=ledger_name,changes=changes),tempfile.TemporaryDirectory() as temporary:
                    root=Path(temporary)
                    self.ledger(root,'budget.json',12,[self.session()])
                    self.ledger(root,ledger_name,12,[self.session(**changes)])
                    with patch.object(analysis,'analyze_acceptance') as expensive:
                        with self.assertRaisesRegex(ValueError,'every started precision session'):
                            analysis.finalize(root)
                        expensive.assert_not_called()

    def test_no_started_session_cannot_finalize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.ledger(root,'budget.json',0,[])
            with self.assertRaises(ValueError):analysis.interruption_accounting(root,require_restored=True)

    def test_unattributed_or_invalid_recorded_time_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.ledger(root,'budget.json',12,[self.session()])
            for used,sessions in [(12,[]),(-1,[]),(float('inf'),[]),(True,[])]:
                with self.subTest(used=used):
                    self.ledger(root,'resume-budget.json',used,sessions)
                    with self.assertRaises(ValueError):analysis.interruption_accounting(root)


if __name__=='__main__':unittest.main()
