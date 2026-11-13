import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ninfer_precision as campaign
from ninfer_precision_grade import grade
from ninfer_precision_analysis import distribution


class PrecisionTests(unittest.TestCase):
    def test_missing_acceptance_coverage_cannot_pass_with_zero_errors(self):
        from ninfer_precision_analysis import acceptance_coverage,complete_acceptance_coverage
        with tempfile.TemporaryDirectory() as temporary:
            coverage,jobs=acceptance_coverage(Path(temporary))
            self.assertEqual(coverage['expected_fixture_count'],54)
            self.assertEqual(jobs,[])
            complete_acceptance_coverage(coverage,[])
            self.assertEqual(coverage['acceptance_errors'],0)
            self.assertFalse(coverage['coverage_complete'])
            self.assertEqual(coverage['overall_verdict'],'inconclusive')
            self.assertTrue(all(not f['control_validation_present'] for f in coverage['fixtures']))

    def test_duplicate_acceptance_round_is_not_chosen_arbitrarily(self):
        from ninfer_precision_analysis import acceptance_coverage
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);suite=root/'runs/r5/decode-k3';directory=suite/'observations/decode-8192-0'
            directory.mkdir(parents=True)
            campaign.write_json(suite/'control-validation.json',{'passed':True})
            p={'cursor':1,'round':0,'width':3,'extent':3,'drafts':[1,2,3]}
            for _ in range(2):campaign.append(directory/'proposal-inputs.jsonl',p)
            campaign.append(directory/'acceptance.jsonl',{'cursor':1,'round':0,'count':1,'accepted':0,'unforced_bonus':7})
            campaign.append(directory/'observations.jsonl',{'phase':'verify','script_cursor':1,'vocab':2,'offset':0})
            (directory/'logits.bf16').write_bytes(bytes(16))
            coverage,jobs=acceptance_coverage(root)
            fixture=next(f for f in coverage['fixtures'] if f['engine']=='r5' and f['k']==3 and f['fixture']=='decode-8192-0')
            self.assertEqual(fixture['duplicate_round_records'],[{'cursor':1,'duplicates':{'proposal':2}}])
            self.assertFalse(fixture['record_coverage_complete'])
            self.assertEqual(jobs,[])

    def test_budget_tail_does_not_establish_full_rejection_coverage(self):
        from ninfer_precision_analysis import complete_acceptance_coverage
        fixture={'engine':'r5','k':3,'fixture':'case','record_coverage_complete':True,'eligible_cursors':[1,128]}
        coverage={'expected_fixture_count':1,'fixtures':[fixture]}
        common={'engine':'r5','k':3,'fixture':'case','extent':3,'acceptance_correct':True,'bonus_correct':True}
        records=[{**common,'cursor':1,'remaining_budget':128,'expected_accepted':3},
                 {**common,'cursor':128,'remaining_budget':1,'expected_accepted':0,'pattern':'reject_all'}]
        complete_acceptance_coverage(coverage,records)
        self.assertEqual(fixture['pattern_counts'],{'full':1,'budget_tail':1})
        self.assertEqual(fixture['missing_patterns'],['partial','reject_all'])
        self.assertEqual(coverage['overall_verdict'],'inconclusive')

    def test_fresh_package_profiles_do_not_add_cache_capacity_or_files(self):
        from ninfer_precision_run import package_flags
        from ninfer_tune import Server
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for key in campaign.ENGINES:
                flags=package_flags(directory,key)
                self.assertIn('--no-prefix-reuse',flags)
                self.assertFalse(set(flags)&{'--host-state-slots','--host-kv-mib','--device-state-slots',
                    '--host-cache-mib','--device-snapshot-slots','--use-alt-prefix-caching','--prefix-cache-file'})
                args=argparse.Namespace(runtime='original' if key=='v100' else 'all',cache_dir=temporary)
                with patch('ninfer_tune.pick_port',return_value=19001):
                    server=Server(directory,flags,args)
                self.assertNotIn('--prefix-cache-file',server.flags)
                image_flags=package_flags(directory,key,images=True)
                self.assertNotIn('--no-prefix-reuse',image_flags)
                if key!='v100':self.assertIn('--prefix-cache-file',image_flags)
            protected=package_flags(directory,'r5')
            protected['--prefix-cache-file']=str(campaign.ENGINE/'cache/protected.cache')
            with patch('ninfer_tune.pick_port',return_value=19001),self.assertRaises(ValueError):
                Server(directory,protected,argparse.Namespace(runtime='all',cache_dir=temporary))
    def test_stream_comparison_reports_difference_and_truncation(self):
        a=io.BytesIO(b'XXabcdefghij');b=io.BytesIO(b'YabcDefghij')
        result=campaign.compare_range(a,b,2,1,10,chunk=3)
        self.assertFalse(result['equal'])
        self.assertEqual(result['first_difference'],3)
        with self.assertRaises(ValueError):
            campaign.compare_range(io.BytesIO(b'a'),io.BytesIO(b'a'),0,0,2)

    def test_artifact_header_versions(self):
        with tempfile.TemporaryDirectory() as d:
            for version,prefix in [(2,16),(3,32)]:
                path=Path(d)/str(version)
                metadata=b'{"objects":[]}'
                path.write_bytes(b'NINFER\0'+bytes([version])+struct.pack('<Q',len(metadata))+
                                 b'0'*(prefix-16)+metadata+b'\0'*4096)
                directory,start=campaign.artifact_directory(path)
                self.assertEqual(directory,{'objects':[]})
                self.assertEqual(start,4096)

    def test_fixtures_fixed_counts_and_known_arithmetic(self):
        cases=campaign.quality_cases()
        self.assertEqual(cases,campaign.quality_cases())
        self.assertEqual(len({x['id'] for x in cases}),384)
        self.assertEqual(sum(x['domain']=='tool_json' for x in cases),128)
        self.assertEqual(sum(x['domain']=='code' for x in cases),96)
        for case in cases:
            if case['grader']=='json':
                self.assertTrue(grade(case,json.dumps(case['expected'],ensure_ascii=False))['correct'])

    def test_strict_json_types_and_duplicate_keys(self):
        case={'grader':'json','expected':{'active':True,'count':1}}
        self.assertFalse(grade(case,'{"active":1,"count":1}')['correct'])
        self.assertFalse(grade(case,'{"active":true,"active":true,"count":1}')['correct'])
        self.assertFalse(grade(case,'{"active":true,"count":1,"extra":null}')['correct'])
        self.assertTrue(grade(case,'{"active":true,"count":1.0}')['correct'])

    def test_code_runs_assertions_and_rejects_escape(self):
        case={'grader':'code','function':'solve','expected':[[[3],6],[[0],0]]}
        self.assertTrue(grade(case,'def solve(x):\n return x*2')['correct'])
        self.assertFalse(grade(case,'def solve(x):\n return 6')['correct'])
        for code in ['import os\ndef solve(x): return os.getcwd()',
                     'def solve(x): return x.__class__',
                     'def solve(x): return eval("2*x")']:
            self.assertFalse(grade(case,code)['correct'])

    def test_paired_documents_weight_tokens_and_correct_direction(self):
        estimate=campaign.paired_bootstrap([(1.01,1,1),(2.02,2,3)],replicates=500)
        self.assertAlmostEqual(estimate['estimate'],.0175)
        self.assertEqual(estimate['units'],2)
        self.assertAlmostEqual(estimate['confidence'],1-.05/6)
        self.assertEqual(campaign.verdict([-.004,.004],.005,True),'retained')
        self.assertEqual(campaign.verdict([-.009,-.006],.005,True),'gained')
        self.assertEqual(campaign.verdict([.006,.009],.005,True),'lost')
        self.assertEqual(campaign.verdict([-.006,.003],.005,True),'inconclusive')
        self.assertEqual(campaign.verdict([.03,.04],.02,False),'gained')

    def test_distribution_centering_and_kl_direction(self):
        same=distribution([1.,2.,3.],[11.,12.,13.])
        self.assertAlmostEqual(same['kl_left_right'],0,places=12)
        self.assertAlmostEqual(same['centered_logit_rmse'],0,places=12)
        self.assertTrue(same['argmax_agreement'])
        different=distribution([0.,0.,0.],[0.,0.,4.])
        self.assertGreater(different['kl_left_right'],0)
        self.assertGreater(different['kl_right_left'],0)

    def test_long_state_comparison_requires_identical_executed_history(self):
        from ninfer_precision_analysis import long_state_comparisons
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for key,generated in [('v100',[3,2]),('r5',[1,2])]:
                directory=root/'runs'/key/'long-k3';observed=directory/'observations/case'
                observed.mkdir(parents=True)
                campaign.append(directory/'jobs.jsonl',{'id':'case','tokens':[10,11]})
                campaign.append(directory/'records.jsonl',{'id':'case','tokens':generated})
                parts=[{'kind':'recurrent','layer':0,'bytes':8,'dtype':1}]
                for offset,frontier in [(0,2),(8,3)]:
                    campaign.append(observed/'states.jsonl',{'execution_frontier':frontier,
                        'ledger_frontier':frontier+1,'offset':offset,'parts':parts})
                (observed/'states.bin').write_bytes(struct.pack('<ffff',1,2,3,4))
            result=long_state_comparisons(root)
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['generated_executed_prefix'],0)
            self.assertTrue(result[0]['exact_bytes'])

    @unittest.skipUnless((Path(__file__).resolve().parent.parent/'.local/precision-oracle').exists(),'compile CPU oracle first')
    def test_independent_fp64_codec_oracle_native_and_qpn(self):
        binary=Path(__file__).resolve().parent.parent/'.local/precision-oracle'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest=[]
            for q,k in [(7,64),(8,128)]:
                n=128;fixture=root/f'x{q}';fixture.mkdir()
                activations=[((col*3+j)%9-4)/16 for col in range(128) for j in range(k)]
                bf=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]>>16
                (fixture/'activation.bf16').write_bytes(struct.pack('<'+'H'*len(activations),*[bf(x) for x in activations]))
                if q==7:
                    codes=[[(row*3+j*5)%16 for j in range(k)] for row in range(n)]
                    scales=bytearray(n*k//16)
                    for row in range(n):
                        for group in range(k//16):
                            offset=((row//128)*(k//64)+group//4)*512+(row%32)*16+((row%128)//32)*4+group%4
                            scales[offset]=[0x30,0x38,0x40,0x48][(row+group)%4]
                    native=bytes(codes[row][j]|(codes[row][j+1]<<4) for row in range(n) for j in range(0,k,2))
                    def weight(row,j):
                        code=codes[row][j];scale=[.5,1.,2.,4.][(row+j//16)%4]
                        return [0,.5,1.,1.5,2.,3.,4.,6.][code&7]*(-1 if code&8 else 1)*scale/(1 if row<n//2 else 2)
                else:
                    enc=[0x30,0x38,0x40,0x48,0xb0,0xb8,0xc0,0xc8]
                    val=[.5,1.,2.,4.,-.5,-1.,-2.,-4.]
                    codes=[[enc[(row*3+j*5)%8] for j in range(k)] for row in range(n)]
                    native=bytes(x for row in codes for x in row)
                    scales=struct.pack('<'+'H'*n,*[bf([.5,1.,2.,4.][r%4]) for r in range(n)])
                    def weight(row,j):return val[(row*3+j*5)%8]*[.5,1.,2.,4.][row%4]
                values=[sum(weight(row,j)*activations[col*k+j] for j in range(k)) for col in range(128) for row in range(n)]
                for layout in [2 if q==7 else 3,4,7]:
                    directory=root/f'q{q}-layout{layout}';directory.mkdir()
                    data=native;row_scales=scales
                    if layout in [4,7]:
                        packed=bytearray(len(native));packed_scales=bytearray(len(scales))
                        for row in range(n):
                            lane=(row%32//8)*4+row%4+(16 if row%8>=4 else 0)
                            if q==7:
                                order=[0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]
                                for group in range(k//16):
                                    offset=((row//32)*(k//16)+group)*32+lane
                                    for byte in range(8):packed[offset*8+byte]=codes[row][group*16+order[byte*2]]|(codes[row][group*16+order[byte*2+1]]<<4)
                                    native_scale=((row//128)*(k//64)+group//4)*512+(row%32)*16+((row%128)//32)*4+group%4
                                    packed_scales[offset]=scales[native_scale]
                            else:
                                for j in range(k):
                                    offset=((((row//32)*(k//128)+j//128)*8+(j%128)//16)*32+lane)*16+j%16
                                    packed[offset]=codes[row][j]
                        data=bytes(packed)
                        row_scales=bytes(packed_scales) if q==7 else scales
                    (directory/'codes.bin').write_bytes(data);(directory/'scales.bin').write_bytes(row_scales)
                    (directory/'weight.json').write_text(json.dumps({'n':n,'k':k,'qtype':q,'layout':layout,
                        'weight_divisor_rows':n//2 if q==7 else n,'weight_divisors':[1.,2.] if q==7 else [1.], 'output_rows':n,'output_element_bytes':4,
                        'family':'linear','fixture':str(fixture)}))
                    for width in campaign.WIDTHS:
                        (directory/f'width-{width}.bin').write_bytes(struct.pack('<'+'f'*(n*width),*values[:n*width]))
                    if q==8 and layout==3:
                        for width in [1024,2048]:
                            repeated=values*(width//128)
                            (directory/f'width-{width}.bin').write_bytes(struct.pack('<'+'f'*len(repeated),*repeated))
                    canonical=root/f'q{q}-layout{2 if q==7 else 3}'
                    manifest.append({'canonical':str(canonical),'candidate':str(directory)})
            manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest));output=root/'out.jsonl'
            subprocess.run([str(binary),str(manifest_path),str(output)],check=True,capture_output=True,timeout=30)
            results=campaign.rows(output)
            self.assertEqual(len(results),74)
            self.assertEqual({r['width'] for r in results if r['width']>128},{1024,2048})
            self.assertTrue(all(r['max_abs']==0 and r['represented_weight_differences']==0 for r in results))


if __name__=='__main__': unittest.main()
