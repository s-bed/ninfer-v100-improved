"""Supplemental historical operator contracts preserve common/raw results."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ninfer_precision_analysis import operator_contract_summary,supplemental_operator_contract


class PrecisionOperatorContractTests(unittest.TestCase):
    def criteria(self):
        return {'gdn_input_proj:q7':{'relative_l2':.003,'gross_absolute':.004,'gross_relative_to_max_reference':.0035},
                'gdn_input_proj:q8':{'relative_l2':1/256,'gross_absolute':1/256,'gross_relative_to_max_reference':2/256}}

    def rows(self):
        return [{'engine':engine,'operator':'gdn','family':'gdn_input_proj','qtype':7,'width':1,
                 'count':128,'nonfinite':0,'relative_l2':.001,'max_abs':.5 if engine=='r5' else .3,
                 'reference_max_abs':100,'represented_weight_differences':0} for engine in ['v100','vendor','r5']]

    def test_historical_gross_bound_can_fail_an_unchanged_common_pass(self):
        rows=self.rows();before=copy.deepcopy(rows)
        result=operator_contract_summary(rows,self.criteria(),['gdn'],[1])
        self.assertEqual(rows,before)
        self.assertTrue(.5<=.004+(2/256)*100) # Existing common-vendor bound.
        self.assertFalse(next(r for r in result['records'] if r['engine']=='r5')['passes_contract'])
        self.assertEqual(result['comparisons']['r5_vs_v100']['verdict'],'lost')
        self.assertEqual(result['comparisons']['r5_vs_vendor']['verdict'],'lost')
        self.assertIn('retained',result['comparisons']['vendor_vs_v100']['verdict'])

    def test_fp8_keeps_its_original_distinct_bound(self):
        rows=self.rows()
        for row in rows:row['qtype']=8
        result=operator_contract_summary(rows,self.criteria(),['gdn'],[1])
        self.assertTrue(all(row['passes_contract'] for row in result['records']))

    def test_missing_or_duplicate_shapes_prevent_a_retention_claim(self):
        result=operator_contract_summary(self.rows(),self.criteria(),['gdn'],[1,2])
        self.assertTrue(all(r['verdict']=='inconclusive' for r in result['comparisons'].values()))
        rows=self.rows();rows.append(dict(rows[-1]))
        result=operator_contract_summary(rows,self.criteria(),['gdn'],[1])
        self.assertEqual(result['comparisons']['r5_vs_v100']['verdict'],'inconclusive')
        self.assertEqual(result['comparisons']['r5_vs_v100']['duplicate_operator_width_records'][0]['records'],2)

    def test_nonfinite_or_different_weights_cannot_pass(self):
        rows=self.rows();rows[-1].update(max_abs=0,relative_l2=0,nonfinite=1)
        result=operator_contract_summary(rows,self.criteria(),['gdn'],[1])
        self.assertFalse(result['records'][-1]['passes_contract'])
        rows[-1].update(nonfinite=0,represented_weight_differences=1)
        result=operator_contract_summary(rows,self.criteria(),['gdn'],[1])
        self.assertEqual(result['comparisons']['r5_vs_v100']['verdict'],'inconclusive')

    def test_sidecar_does_not_replace_raw_or_common_policy_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'analysis').mkdir();(root/'provenance').mkdir()
            raw=root/'analysis/operators.jsonl';common=root/'analysis/operator-summary.json'
            raw.write_text(''.join(json.dumps(r)+'\n' for r in self.rows()))
            common.write_text('{"preserved_common_result":true}\n')
            previous=(raw.read_bytes(),common.read_bytes())
            policy={'original_v100':{'criteria':self.criteria()},'expected_operators':['gdn'],'expected_widths':[1]}
            (root/'provenance/operator-tolerance-provenance.json').write_text(json.dumps(policy))
            result=supplemental_operator_contract(root)
            self.assertTrue(result['primary_common_vendor_policy_unchanged'])
            self.assertTrue((root/'analysis/operator-original-v100-contract.json').exists())
            self.assertEqual((raw.read_bytes(),common.read_bytes()),previous)


if __name__=='__main__':unittest.main()
