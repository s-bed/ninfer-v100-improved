"""CPU-only behavioral checks for the native-cache agentic benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ninfer_agentic as a


class AgenticTests(unittest.TestCase):
    def test_native_command_profiles_preserve_distinct_cache_contracts(self):
        config=(a.REPO/'llama-swap.yaml').read_text()
        _,old=a.source_flags(config,'v100');_,vendor=a.source_flags(config,'vendor');_,r5=a.source_flags(config,'r5')
        self.assertEqual(old['--host-kv-mib'],'16384');self.assertEqual(old['--host-state-slots'],'8')
        self.assertEqual(old['--device-state-slots'],'3');self.assertEqual(old['--max-private-continuations'],'4')
        self.assertEqual(old['--prefill-chunk'],'2048');self.assertNotIn('--use-alt-prefix-caching',old)
        self.assertEqual(r5['--host-cache-mib'],'16384');self.assertEqual(r5['--device-snapshot-slots'],'2')
        self.assertEqual(r5['--prefill-chunk'],'4096');self.assertEqual(r5['--kv-capacity'],'auto')
        self.assertEqual(r5['--ngram-draft-tokens'],'15')
        self.assertEqual(vendor,{k:v for k,v in r5.items() if k!='--v100-kernels'})
        self.assertNotIn('--v100-kernels',vendor)

    def test_fixture_mutations_are_frozen_and_tools_really_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'fixtures').mkdir()
            cases=a.fixtures(root);again=a.fixtures(root)
            self.assertEqual(cases,again);self.assertEqual(len(cases),3)
            images=[a.sha(root/case['image']) for case in cases]
            self.assertEqual(len(set(images)),3)
            for case in cases:
                base=case['base'];m=case['mutations']
                self.assertEqual(base['tool_choice'],'auto')
                self.assertEqual(len(base['tools']),8)
                self.assertGreater(len(base['messages'][3]['content']),20000)
                self.assertEqual(m['tools-reordered']['tools'],base['tools'][::-1])
                self.assertEqual(len(m['tool-added']['tools']),9)
                self.assertEqual(m['late-result-edit']['messages'][:-2],base['messages'][:-2])
                self.assertEqual(m['tool-schema-changed']['messages'],base['messages'])
                self.assertIn('include_context',m['tool-schema-changed']['tools'][0]['function']['parameters']['properties'])
                for message in base['messages']:
                    for call in message.get('tool_calls',[]):
                        self.assertTrue(any(t['function']['name']==call['function']['name'] for t in base['tools']))

    def test_live_history_preserves_actual_reasoning_and_tool_identifiers(self):
        result=a.tune.ChatResult(status=200,content=['answer'],reasoning=['analysis'],
            tool_calls={0:{'id':'call_actual','name':'read_file','arguments':'{"path":"x.py"}'}})
        messages=[];a.append_assistant(messages,result)
        self.assertEqual(messages[0]['reasoning_content'],'analysis')
        self.assertEqual(messages[0]['tool_calls'][0]['id'],'call_actual')
        self.assertEqual(messages[1]['tool_call_id'],'call_actual')
        self.assertIn('No external tool was executed',messages[1]['content'])
        for arguments in ['{"path":','[]','{"path": NaN}']:
            malformed=copy.deepcopy(result);malformed.tool_calls[0]['arguments']=arguments;messages=[]
            self.assertFalse(a.append_assistant(messages,malformed,a.make_tools()))
            self.assertEqual(messages,[])

    def test_correctness_is_typed_json_and_requires_no_tool_execution(self):
        want=a.expected(0)
        good=a.tune.ChatResult(status=200,content=[json.dumps(want)])
        self.assertTrue(a.grade(good,want)['correct'])
        good.content=['```json\n'+json.dumps(want)+'\n```']
        self.assertTrue(a.grade(good,want)['correct'])
        wrong=copy.deepcopy(want);wrong['checks'][0]='other';good.content=[json.dumps(wrong)]
        self.assertFalse(a.grade(good,want)['correct'])
        good.content=['{"result":"x","result":"x"}']
        self.assertFalse(a.grade(good,want)['correct'])
        good.content=[json.dumps(want)];good.tool_calls={0:{'id':'x','name':'run_tests','arguments':'{}'}}
        self.assertFalse(a.grade(good,want)['correct'])
        self.assertFalse(a.typed_equal(True,1))

    def test_analysis_keeps_failed_cases_and_reports_small_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'fixtures').mkdir();a.write(root/'fixtures/protocol.json',{})
            (root/'bench').mkdir()
            for key in a.ENGINES:
                directory=root/'runs'/key;directory.mkdir(parents=True)
                for repeat in range(3):
                    correct=(repeat==2) if key=='r5' else (repeat!=1)
                    fast_wrong=key=='r5' and not correct
                    a.append(directory/'records.jsonl',{'repeat':repeat,'stage':'new-lineage','setup':False,
                        'grade':{'correct':correct},'payload_sha256':'same','wall_s':.1 if fast_wrong else 2+repeat,
                        'client_ttft_s':.05 if fast_wrong else 1+repeat,'prompt_tokens':8000,'cache_n':40,'cache_fraction':.005,
                        'completion_tokens':50,'decode_tps':20})
            with patch.object(a,'REPO',root):result=a.analyze(root)
            self.assertEqual(result['engines']['v100']['stages']['new-lineage']['n'],3)
            self.assertEqual(result['engines']['v100']['stages']['new-lineage']['correct'],2)
            self.assertEqual(result['engines']['v100']['stages']['new-lineage']['wall_s']['median'],3)
            paired=result['comparisons']['r5_vs_v100']['new-lineage']
            self.assertEqual(paired['paired_n'],3)
            self.assertLess(paired['wall_ratio']['median'],.1)
            self.assertEqual(paired['both_correct']['paired_n'],1)
            self.assertEqual([p['repeat'] for p in paired['both_correct']['pairs']],[2])
            self.assertEqual(paired['both_correct']['wall_ratio']['n'],1)
            self.assertEqual(paired['both_correct']['wall_ratio']['median'],1)
            text=(root/'NINFER_AGENTIC.txt').read_text()
            self.assertIn('all paired n=3',text)
            self.assertIn('both-correct secondary subset n=1',text)
            self.assertEqual(len(result['engines']['v100']['correctness_failures']),1)

    def test_precision_gate_rejects_unfinished_campaign(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(a,'PRECISION',Path(temporary)):
            with self.assertRaisesRegex(ValueError,'finish precision finalization first'):a.precision_done()


if __name__=='__main__':unittest.main()
