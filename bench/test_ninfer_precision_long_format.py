"""Long-sentinel needle presence does not imply exact answer formatting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ninfer_precision_analysis import integration


class PrecisionLongFormatTests(unittest.TestCase):
    def test_extra_text_preserves_retrieval_but_fails_strict_answer_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);fixtures=root/'fixtures';fixtures.mkdir()
            for name in ['quality','native-subset','mtp-subset']:
                (fixtures/(name+'.json')).write_text('[]')
            (fixtures/'long-jobs.json').write_text(json.dumps([
                {'id':'long-131072','max_tokens':64},{'id':'long-229376','max_tokens':64}]))
            directory=root/'runs/v100/long-k3';directory.mkdir(parents=True)
            rows=[{'id':'long-131072','content':'maple-7c90 519204<|endoftext|>','reused_tokens':0},
                  {'id':'long-229376','content':'maple-7c90 519204\n.\n\nUnrelated text<|endoftext|>','reused_tokens':131108}]
            (directory/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            vendor=root/'runs/vendor/long-k3';vendor.mkdir(parents=True)
            wrong={'id':'long-131072','content':'KESTREL-519204\nThe passphrase is maple-7c90.\n','tokens':[0]*64}
            (vendor/'records.jsonl').write_text(json.dumps(wrong)+'\n')
            result=integration(root)
            first,second=result['v100']['long_context']
            self.assertTrue(first['retrieval_correct'])
            self.assertTrue(first['both_needle_strings_present'])
            self.assertTrue(first['ordered_answer_present'])
            self.assertTrue(first['strict_answer_format_correct'])
            self.assertIsNone(first['format_deviation'])
            self.assertTrue(second['retrieval_correct'])
            self.assertTrue(second['ordered_answer_present'])
            self.assertFalse(second['strict_answer_format_correct'])
            self.assertEqual(second['format_deviation'],'extra trailing text')
            associated=result['vendor']['long_context'][0]
            self.assertTrue(associated['retrieval_correct'])
            self.assertTrue(associated['both_needle_strings_present'])
            self.assertFalse(associated['ordered_answer_present'])
            self.assertFalse(associated['strict_answer_format_correct'])
            self.assertTrue(associated['output_token_limit_reached'])
            missing=result['r5']['long_context'][0]
            self.assertFalse(missing['available'])
            self.assertFalse(missing['strict_answer_format_correct'])
            self.assertIsNone(missing['format_deviation'])


if __name__=='__main__':unittest.main()
