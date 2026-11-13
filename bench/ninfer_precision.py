#!/usr/bin/env python3.11
"""Frozen, paired three-engine precision campaign. No production config writes.

Preparation and analysis are CPU-only. GPU jobs must run under ninfer_session.py.
All JSONL records are append-only and flushed so a deadline preserves completed work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path('/models/engine/iamwavecut-ninfer-all')
DONOR = Path('/models/engine/geoffwatts-ninfer-v100')
DEFAULT_ROOT = ENGINE / 'bench-runs/precision-20261030'
ENGINES = {
    'v100': {'revision': 'b37d0dd3e1163b9d802d8bccfa89918bf68d793e',
             'source': DONOR / 'source/standard', 'bundle': DONOR / 'runtime/standard',
             'artifact': Path('/models/ninfer/qwen3_8_27b_nvfp4.ninfer')},
    'vendor': {'revision': '33163f05fb54459c19e2574607f5d252f4be2ba4',
               'source': ENGINE / 'source/vendor-33163f05fb54',
               'bundle': ENGINE / 'runtime/vendor-33163f05fb54/current',
               'artifact': Path('/models/ninfer/qwen3_8_27b_nvfp4.v3.ninfer')},
    'r5': {'revision': 'ec94ca8d97aebcae2e5429b83925b2c08252143a',
           'source': ENGINE / 'source/ported', 'bundle': ENGINE / 'runtime/ported-33163f0-r5',
           'artifact': Path('/models/ninfer/qwen3_8_27b_nvfp4.v3.ninfer')},
}
WIDTHS = [1, 2, 4, 6, 8, 16, 32, 33, 63, 64, 65, 128]
PAIRS = [('r5', 'v100'), ('r5', 'vendor'), ('vendor', 'v100')]
SETTINGS = {'context': 4096, 'stride': 2048, 'target_tokens_per_stream': 8192,
            'kv_dtype': 'int8', 'gdn_state_fp16': False, 'temperature': 0,
            'seed': 1729, 'power_watts': 150, 'ppl_margin': .005,
            'accuracy_margin': .02, 'family_alpha': .05,
            'primary_claims': 6, 'bootstrap_replicates': 20000,
            'template': '<|im_start|>system\nFollow the request exactly. Give only the requested answer.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'}
READ_ERRORS = []


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.next')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def rows(path: Path) -> list[dict]:
    text=path.read_text();lines=text.splitlines();result=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not text.endswith('\n'):
                READ_ERRORS.append({'path':str(path),'line':i+1,'error':'interrupted trailing JSONL record'})
            else:raise
    return result


def artifact_directory(path: Path) -> tuple[dict, int]:
    with path.open('rb') as f:
        header = f.read(16)
        if header[:7] != b'NINFER\0' or header[7] not in (2, 3):
            raise ValueError('unsupported artifact header')
        length = struct.unpack_from('<Q', header, 8)[0]
        prefix = 16 if header[7] == 2 else 32
        if not 0 < length < 64 * 1024 * 1024:
            raise ValueError('unbounded artifact directory')
        f.seek(prefix)
        return json.loads(f.read(length)), ((prefix + length + 4095) // 4096) * 4096


def compare_range(a, b, aoffset: int, boffset: int, length: int, chunk: int = 4 << 20) -> dict:
    """Bounded memory, exact byte comparison, including all embedded scales/divisors."""
    a.seek(aoffset)
    b.seek(boffset)
    ah, bh = hashlib.sha256(), hashlib.sha256()
    first_difference = None
    for pos in range(0, length, chunk):
        size = min(chunk, length - pos)
        av, bv = a.read(size), b.read(size)
        if len(av) != size or len(bv) != size:
            raise ValueError('truncated artifact payload')
        ah.update(av)
        bh.update(bv)
        if first_difference is None and av != bv:
            first_difference = pos + next(i for i, (x, y) in enumerate(zip(av, bv)) if x != y)
    return {'bytes': length, 'equal': first_difference is None,
            'first_difference': first_difference, 'v2_sha256': ah.hexdigest(), 'v3_sha256': bh.hexdigest()}


def artifact_check(root: Path) -> None:
    old_path, new_path = ENGINES['v100']['artifact'], ENGINES['vendor']['artifact']
    old, old_start = artifact_directory(old_path)
    new, new_start = artifact_directory(new_path)
    if len(new['files']) != 1:
        raise ValueError('this campaign expects the actual single-file serving artifact')
    upgrade_path = ENGINES['vendor']['source'] / 'tools/upgrade_ninfer_v2_to_v3.py'
    spec = importlib.util.spec_from_file_location('precision_upgrade', upgrade_path)
    upgrade = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upgrade)
    normalized = upgrade.make_directory(old['identity'], old['objects'])
    old_by_id = {o['id']: o for o in normalized['objects']}
    new_by_id = {o['id']: o for o in new['objects']}
    mapping = {}
    binding_errors = []
    for name, binding in normalized['bindings'].items():
        other = new['bindings'].get(name)
        if other is None or binding.keys() != other.keys():
            binding_errors.append(name)
            continue
        op = binding.get('parts', [{'object': binding.get('object')}])
        np = other.get('parts', [{'object': other.get('object')}])
        if len(op) != len(np):
            binding_errors.append(name)
            continue
        for left, right in zip(op, np):
            if left.get('range') != right.get('range'):
                binding_errors.append(name)
            oi, ni = left['object'], right['object']
            if oi in mapping and mapping[oi] != ni:
                binding_errors.append(name)
            mapping[oi] = ni
    # A fused gate/up scalar in v2 is represented by two equal auxiliaries in v3.
    # Pair uses by parameter; input-edge names changed during the v3 conversion.
    new_uses = {}
    for use in new['uses']:
        new_uses.setdefault(use['parameter'], []).append(use)
    auxiliary_pairs = set()
    use_errors = []
    for use in normalized['uses']:
        for other in new_uses.get(use['parameter'], []):
            if use['activation_policy'] != other['activation_policy']:
                use_errors.append(use['parameter'])
            for name, value in use.get('auxiliaries', {}).items():
                right = other.get('auxiliaries', {}).get(name)
                if right is None:
                    use_errors.append(use['parameter'])
                else:
                    auxiliary_pairs.add((value['object'], right['object']))
    for oi, ni in sorted(auxiliary_pairs):
        mapping.setdefault(oi, ni)
    differences = []
    matched = 0
    total_bytes = 0
    records = root / 'provenance/artifact-objects.jsonl'
    cached = {r['v2_id']:r for r in rows(records) if r.get('equal') and r.get('metadata_equal')} if records.exists() else {}
    if records.exists():
        initial = root / 'provenance/artifact-comparison.json'
        saved = root / 'provenance/artifact-comparison-initial.json'
        if initial.exists() and not saved.exists():
            saved.write_bytes(initial.read_bytes())
        records = root / 'provenance/artifact-objects-extended.jsonl'
        if records.exists(): raise ValueError('extended artifact evidence already exists')
    with old_path.open('rb') as a, new_path.open('rb') as b:
        for oid, obj in old_by_id.items():
            if obj['kind'] != 'tensor':
                continue
            nid = mapping.get(oid)
            other = new_by_id.get(nid)
            metadata_equal = other is not None and all(obj[k] == other[k] for k in ('shape', 'format', 'layout', 'bytes'))
            record = {'v2_id': oid, 'v3_id': nid, 'metadata_equal': metadata_equal}
            if metadata_equal:
                previous = cached.get(oid)
                if previous and previous['v3_id'] == nid:
                    record.update(previous)
                    record['reused_completed_stream_comparison'] = True
                else:
                    record.update(compare_range(a, b, old_start + obj['offset'], new_start + other['offset'], obj['bytes']))
                total_bytes += obj['bytes']
            if not metadata_equal or not record.get('equal'):
                differences.append(record)
            else:
                matched += 1
            append(records, record)
            if matched % 100 == 0:
                print(f'artifact tensors compared: {matched}; bytes: {total_bytes}', flush=True)
        for oi,ni in sorted(auxiliary_pairs):
            if mapping[oi] == ni: continue
            obj,other=old_by_id[oi],new_by_id[ni]
            metadata_equal=all(obj[k]==other[k] for k in ('shape','format','layout','bytes'))
            record={'v2_id':oi,'v3_id':ni,'metadata_equal':metadata_equal,'duplicated_auxiliary':True}
            if metadata_equal:
                record.update(compare_range(a,b,old_start+obj['offset'],new_start+other['offset'],obj['bytes']))
                total_bytes+=obj['bytes']
            if not metadata_equal or not record.get('equal'): differences.append(record)
            else: matched+=1
            append(records,record)
        resources = []
        for old_name, new_name in [('frontend/tokenizer.json', 'resource/text/tokenizer.json'),
                                   ('frontend/tokenizer_config.json', 'resource/text/tokenizer_config.json'),
                                   ('frontend/chat_template.jinja', 'resource/text/chat_template.jinja')]:
            x, y = old_by_id[old_name], new_by_id[new_name]
            same_length = x['bytes'] == y['bytes']
            r = {'name': old_name, 'v2_bytes': x['bytes'], 'v3_bytes': y['bytes']}
            if same_length:
                r.update(compare_range(a, b, old_start + x['offset'], new_start + y['offset'], x['bytes']))
            else:
                r['equal'] = False
            resources.append(r)
    result = {'passed': not differences and not binding_errors,
              'tensors_equal': matched, 'tensor_bytes_compared': total_bytes,
              'unmapped_new_tensors': sorted(set(o['id'] for o in new['objects'] if o['kind'] == 'tensor') - set(mapping.values()) - {ni for oi,ni in auxiliary_pairs}),
              'binding_count': len(normalized['bindings']), 'binding_errors': binding_errors,
              'extra_new_bindings': sorted(set(new['bindings']) - set(normalized['bindings'])),
              'differences': differences, 'resources': resources, 'activation_policy_errors':use_errors,
              'scope': 'All stored tensor bytes, including encoded codes, scales and scalar divisors; logical bindings checked using pinned v2 upgrade mapping. No unquantized-model claim.',
              'mapping_source_sha256': sha(upgrade_path), 'v2_directory_sha256': hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest(),
              'v3_directory_sha256': hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()}
    result['passed'] &= not result['unmapped_new_tensors'] and not result['extra_new_bindings'] and not use_errors
    write_json(root / 'provenance/artifact-comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('differences', 'resources')}, indent=2), flush=True)


def quality_cases() -> list[dict]:
    rng = random.Random(1729)
    cases = []
    def add(domain, prompt, expected, grader='exact', **extra):
        cases.append({'id': f'{domain}-{sum(x["domain"] == domain for x in cases):03d}',
                      'domain': domain, 'prompt': prompt, 'expected': expected, 'grader': grader, **extra})
    for i in range(128):
        a, b = rng.randrange(10, 990), rng.randrange(1, 99)
        city = ['London', '杭州', 'Paris', 'Oslo'][i % 4]
        variants = [
            ('Create JSON for a transfer tool. Required keys: tool="transfer", arguments={from_account:string,to_account:string,amount:integer}. Transfer AMOUNT from FROM to TO.',
             {'tool': 'transfer', 'arguments': {'from_account': f'A{a}', 'to_account': f'B{b}', 'amount': a + b}}),
            ('Create JSON for a weather tool. Required keys: tool="weather", arguments={city:string,units:string,days:integer}. CITY, celsius, DAYS days.',
             {'tool': 'weather', 'arguments': {'city': city, 'units': 'celsius', 'days': b % 7 + 1}}),
            ('Return JSON only with keys name (string), enabled (boolean), count (integer), tags (array of strings). NAME is disabled, count COUNT, tags TAG1 and TAG2 in that order.',
             {'name': f'job-{a}', 'enabled': False, 'count': b, 'tags': [city, str(a)]}),
            ('Translate this command to a JSON tool call: schedule TITLE at HOUR:MINUTE UTC. Keys: tool="schedule", arguments={title:string,hour:integer,minute:integer,timezone:"UTC"}.',
             {'tool': 'schedule', 'arguments': {'title': f'review-{a}', 'hour': b % 24, 'minute': a % 60, 'timezone': 'UTC'}}),
            ('Return JSON {"tool":"search","arguments":{"query":string,"limit":integer,"include_archived":boolean}} for query QUERY, limit LIMIT, including archived records.',
             {'tool': 'search', 'arguments': {'query': f'{city} item {a}', 'limit': b, 'include_archived': True}}),
            ('Return JSON with keys ordered (integer array), unique_count (integer), null_value (null). Sort VALUES ascending and count distinct values.',
             {'ordered': sorted([a, b, a, -b]), 'unique_count': len({a, b, -b}), 'null_value': None}),
            ('Return JSON {"tool":"read_file","arguments":{"path":string,"offset":integer,"length":integer}}. Read LENGTH bytes from offset OFFSET in PATH.',
             {'tool': 'read_file', 'arguments': {'path': f'/tmp/{city}/file-{a}.txt', 'offset': a, 'length': b}}),
            ('Return JSON with exactly keys user (string), active (boolean), score (integer), metadata (object with region:string). USER is active; score SCORE; region CITY.',
             {'user': f'u{a}', 'active': True, 'score': a - b, 'metadata': {'region': city}}),
        ]
        prompt, expected = variants[i % len(variants)]
        replacements = {'AMOUNT': str(a+b), 'FROM': f'A{a}', 'TO': f'B{b}', 'CITY': city, 'DAYS': str(b%7+1),
                        'NAME': f'job-{a}', 'COUNT': str(b), 'TAG1': city, 'TAG2': str(a), 'TITLE': f'review-{a}',
                        'HOUR': str(b%24), 'MINUTE': str(a%60), 'QUERY': f'{city} item {a}', 'LIMIT': str(b),
                        'VALUES': str([a,b,a,-b]), 'LENGTH': str(b), 'OFFSET': str(a), 'PATH': f'/tmp/{city}/file-{a}.txt',
                        'USER': f'u{a}', 'SCORE': str(a-b)}
        prompt = re.sub(r'\b(' + '|'.join(replacements) + r')\b', lambda m: replacements[m[0]], prompt)
        add('tool_json', prompt, expected, 'json')
    algorithms = [
        ('Return the longest common prefix of the strings xs; empty list gives an empty string.', ['xs'], [([['flower','flow','flight']], 'fl'), ([[]], ''), ([['猫咪','猫头鹰']], '猫')]),
        ('Return whether text is a palindrome after retaining only alphanumeric characters and ignoring case.', ['text'], [(['A man, a plan, a canal: Panama'], True), (['race a car'], False), ([''], True)]),
        ('Return a list of integers xs with duplicates removed, preserving first occurrence order.', ['xs'], [([[3,1,3,2,1]], [3,1,2]), ([[]], []), ([[-1,-1,0]], [-1,0])]),
        ('Merge touching or overlapping closed integer intervals xs and return sorted [start,end] lists.', ['xs'], [([[[5,7],[1,3],[3,4],[9,10]]], [[1,4],[5,7],[9,10]]), ([[]], []), ([[[1,9],[2,3]]], [[1,9]])]),
        ('Return the nth Fibonacci number, using F(0)=0 and F(1)=1.', ['n'], [([0],0),([1],1),([10],55),([30],832040)]),
        ('Return the index of target in sorted xs, or -1 if absent. Use binary search.', ['xs','target'], [([[1,3,7,9],7],2),([[],1],-1),([[1],0],-1)]),
        ('Return balanced-parentheses validity for a string s of (), [], {} only.', ['s'], [(['([]{})'],True),(['([)]'],False),([''],True),(['('],False)]),
        ('Return the maximum sum of a nonempty contiguous subarray of xs. Empty xs returns 0.', ['xs'], [([[-2,1,-3,4,-1,2,1,-5,4]],6),([[-3,-1]],-1),([[]],0)]),
        ('Return the positive greatest common divisor of integers a and b; gcd(0,0)=0.', ['a','b'], [([48,18],6),([-21,14],7),([0,0],0),([0,5],5)]),
        ('Return run-length encoding of string s as a list of [character,count] lists.', ['s'], [(['aaabbc'],[['a',3],['b',2],['c',1]]),([''],[]),(['猫猫a'],[['猫',2],['a',1]])]),
        ('Return all primes strictly less than integer n, sorted ascending.', ['n'], [([2],[]),([12],[2,3,5,7,11]),([0],[]),([20],[2,3,5,7,11,13,17,19])]),
        ('Return the edit distance between strings a and b using insertion, deletion and substitution, each costing 1.', ['a','b'], [(['kitten','sitting'],3),(['','abc'],3),(['same','same'],0),(['猫','狗'],1)]),
    ]
    for i in range(96):
        description, params, tests = algorithms[i % 12]
        name = f'solve_{i}'
        add('code', f'Write Python 3 code only, with no markdown or explanation. Define {name}({", ".join(params)}). {description} Do not read input or print.', tests,
            'code', function=name)
    for i in range(64):
        a,b,c = rng.randint(11,999),rng.randint(2,99),rng.randint(2,49)
        kind = i % 4
        if kind == 0: prompt, answer = f'Compute ({a} * {b}) - {c}. Return the integer only.', a*b-c
        elif kind == 1: prompt, answer = f'Compute ({a} + {b}) * {c}. Return the integer only.', (a+b)*c
        elif kind == 2: prompt, answer = f'A warehouse has {a} boxes, each with {b} items. It ships {c} boxes. How many items remain? Return the integer only.', (a-c)*b
        else: prompt, answer = f'求 {a*b+c} 除以 {b} 的整数商（向下取整）。只输出整数。', (a*b+c)//b
        add('arithmetic', prompt, str(answer))
    for i in range(64):
        entries = [(f'key-{j:03}', f'{rng.getrandbits(32):08x}') for j in range(100)]
        target = (i*37) % len(entries)
        text = '\n'.join(f'{k}: {v}' for k,v in entries)
        question = f'What value is associated with {entries[target][0]}? Return only the value.' if i%2==0 else f'{entries[target][0]} 对应的值是什么？只输出该值。'
        add('retrieval_en' if i%2==0 else 'retrieval_zh', f'Data:\n{text}\n\n{question}', entries[target][1])
    for i in range(32):
        initial, revised = f'{rng.getrandbits(24):06x}', f'{rng.getrandbits(24):06x}'
        history = [{'role':'user','content':f'The current project code is {initial}.'},
                   {'role':'assistant','content':f'Project code recorded: {initial}.'},
                   {'role':'user','content':f'Correction: replace that project code with {revised}.'},
                   {'role':'assistant','content':f'Updated to {revised}.'}]
        add('history', 'What is the current project code? Return only the code.', revised, history=history)
    assert len(cases) == 384
    return cases


def prepare(root: Path) -> None:
    if (root / 'fixtures/freeze.json').exists():
        raise ValueError('fixtures already frozen; use a fresh campaign directory')
    root.mkdir(parents=True, exist_ok=True)
    sources = ENGINES['vendor']['source'] / 'eval/corpora'
    streams = []
    for corpus, subset in [('perplexity-heldout-2026-09', None), ('perplexity-1m', 'quick')]:
        manifest = json.loads((sources / corpus / 'manifest.json').read_text())
        selected = set(manifest['modes'][subset]) if subset else None
        for stream in manifest['streams']:
            if selected is not None and stream['id'] not in selected:
                continue
            path = sources / corpus / stream['path']
            sid = ('heldout-' if subset is None else 'legacy-') + stream['id']
            # Freeze source text now. Token IDs and exact 8,193-token slices are frozen
            # by the CPU tokenizer probe before the first measurement.
            dest = root / 'fixtures/corpus' / f'{sid}.txt'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
            streams.append({'id': sid, 'domain': stream['domain'], 'path': str(dest),
                            'source': str(path), 'sha256': sha(dest), 'target_tokens': 8192})
    if len(streams) != 16:
        raise ValueError(f'expected 12 held-out and 4 legacy streams, found {len(streams)}')
    write_json(root / 'fixtures/corpus.json', streams)
    cases = quality_cases()
    write_json(root / 'fixtures/quality.json', cases)
    write_json(root / 'fixtures/settings.json', SETTINGS)
    # This fixed interleave, shared across engines, may be truncated only by a
    # time budget, never by observed correctness or likelihood.
    by_domain = {}
    for case in cases:
        by_domain.setdefault(case['domain'], []).append(case['id'])
    order = []
    for i in range(max(map(len, by_domain.values()))):
        order.extend(ids[i] for ids in by_domain.values() if i < len(ids))
    write_json(root / 'fixtures/quality-order.json', order)
    write_json(root / 'fixtures/mtp-subset.json', order[:64])
    write_json(root / 'fixtures/native-subset.json', order[:24])
    provenance = {}
    for key, value in ENGINES.items():
        actual = subprocess.check_output(['git','-C',str(value['source']),'rev-parse','HEAD'], text=True).strip()
        status = subprocess.check_output(['git','-C',str(value['source']),'status','--porcelain'], text=True)
        provenance[key] = {**{k:str(v) for k,v in value.items()}, 'actual_revision':actual,'workspace_status':status}
        if actual != value['revision']:
            raise ValueError(f'{key} revision mismatch: {actual}')
    provenance['active'] = os.readlink(ENGINE / 'runtime/active')
    provenance['config_sha256'] = sha(ROOT / 'llama-swap.yaml')
    write_json(root / 'provenance/engines.json', provenance)
    files = {str(p.relative_to(root)):sha(p) for p in sorted((root/'fixtures').rglob('*')) if p.is_file()}
    write_json(root / 'fixtures/freeze.json', {'created':time.time(),'files':files,'token_ids_frozen':False,
                                             'coverage_order':'fixed round-robin domains; same order in all engines'})
    print(f'Prepared {len(streams)} corpus streams and {len(cases)} scored cases in {root}', flush=True)


def verify_freeze(root: Path) -> None:
    freeze = json.loads((root / 'fixtures/freeze.json').read_text())
    for name, expected in freeze['files'].items():
        if sha(root / name) != expected:
            raise ValueError(f'frozen fixture changed: {name}')


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered)-1)*q
    lo = int(pos)
    return ordered[lo] + (ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(pos-lo)


def paired_bootstrap(pairs: list[tuple[float,float,int]], transform= lambda x:x,
                     replicates: int = 20000, alpha: float = .05/6) -> dict:
    if not pairs:
        return {'verdict':'inconclusive','reason':'no paired observations'}
    if any(n <= 0 or not all(math.isfinite(x) for x in (a,b)) for a,b,n in pairs):
        raise ValueError('invalid paired observation')
    point = transform(sum((a-b)*n for a,b,n in pairs)/sum(n for a,b,n in pairs))
    rng = random.Random(1729)
    samples = []
    for _ in range(replicates):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(transform(sum((a-b)*n for a,b,n in chosen)/sum(n for a,b,n in chosen)))
    return {'estimate':point,'ci':[percentile(samples,alpha/2),percentile(samples,1-alpha/2)],
            'confidence':1-alpha,'units':len(pairs),'resampling':'whole paired documents/cases', 'replicates':replicates}


def verdict(ci: list[float], margin: float, lower_is_better: bool) -> str:
    lo,hi = ci
    if lo >= -margin and hi <= margin: return 'retained'
    if hi < -margin: return 'gained' if lower_is_better else 'lost'
    if lo > margin: return 'lost' if lower_is_better else 'gained'
    return 'inconclusive'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('artifact-check')
    sub.add_parser('verify-freeze')
    build_parser = sub.add_parser('build-probe')
    build_parser.add_argument('--engine', choices=ENGINES, required=True)
    sub.add_parser('stop-vendor-build')
    sub.add_parser('build-all')
    sub.add_parser('build-wide')
    sub.add_parser('prepare-wide')
    wide_parser=sub.add_parser('run-wide');wide_parser.add_argument('--engine',choices=ENGINES,required=True)
    sub.add_parser('reuse-vendor-objects')
    sub.add_parser('prepare-tokens')
    sub.add_parser('manifest')
    sub.add_parser('prepare-images')
    sub.add_parser('prepare-operator-counts')
    sub.add_parser('amend-boundaries')
    sub.add_parser('prepare-long')
    sub.add_parser('freeze-protocol')
    sub.add_parser('prepare-pilot-repair')
    sub.add_parser('prepare-package-repair')
    sub.add_parser('prepare-native-control')
    sub.add_parser('prepare-resume')
    sub.add_parser('resume-controls')
    session_parser=sub.add_parser('session')
    session_parser.add_argument('--phase',choices=['pilot','pilot-controls','main','operator-wide','resume'],required=True)
    sub.add_parser('analyze')
    sub.add_parser('analyze-distributions')
    sub.add_parser('build-cpu')
    sub.add_parser('analyze-operators')
    sub.add_parser('report')
    sub.add_parser('finalize')
    acceptance_parser=sub.add_parser('analyze-acceptance');acceptance_parser.add_argument('--pilot',action='store_true')
    states_parser=sub.add_parser('analyze-states');states_parser.add_argument('--pilot',action='store_true')
    sub.add_parser('status')
    for command in ['run-probe','package']:
        run_parser=sub.add_parser(command)
        run_parser.add_argument('--engine',choices=ENGINES,required=True)
        run_parser.add_argument('--k',type=int,default=0)
        if command=='package': run_parser.add_argument('--images',action='store_true')
        if command=='run-probe': run_parser.add_argument('--suite',required=True)
    args = parser.parse_args()
    if args.command=='status':
        ledger_path=args.root/'control/budget.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        sessions=ledger.get('sessions',[]);latest=sessions[-1] if sessions else {}
        active_elapsed=time.time()-latest['started'] if latest and latest.get('finished') is None else 0
        coverage={key:{p.parent.name:len(rows(p)) for p in sorted((args.root/'runs'/key).glob('*/records.jsonl'))
                       if not any(tag in p.parent.name for tag in ['initial','before-','rejected-'])} for key in ENGINES}
        continuation_path=args.root/'control/resume-budget.json'
        continuation=json.loads(continuation_path.read_text()) if continuation_path.exists() else None
        continuation_sessions=continuation.get('sessions',[]) if continuation else []
        continuation_latest=continuation_sessions[-1] if continuation_sessions else {}
        continuation_elapsed=(time.time()-continuation_latest['started'] if continuation_latest and continuation_latest.get('finished') is None else 0)
        continuation_total=(continuation.get('used_seconds',0) if continuation else 0)+continuation_elapsed
        print(json.dumps({'budget_seconds':ledger.get('budget_seconds'),'completed_interruption_seconds':ledger.get('used_seconds'),
            'active_session_elapsed_seconds':active_elapsed,'total_interruption_seconds':ledger.get('used_seconds',0)+active_elapsed,
            'latest_session':latest,'continuation_ledger':continuation,
            'continuation_interruption_seconds':continuation_total,
            'total_interruption_seconds_including_continuation':ledger.get('used_seconds',0)+active_elapsed+continuation_total,
            'completed_records':coverage},indent=2))
    elif args.command in ['build-probe','build-all','build-wide','stop-vendor-build','reuse-vendor-objects']:
        import ninfer_precision_build
        if args.command=='stop-vendor-build': ninfer_precision_build.stop_vendor_build(args.root)
        elif args.command=='reuse-vendor-objects': ninfer_precision_build.reuse_vendor_objects(args.root)
        elif args.command=='build-wide':ninfer_precision_build.build_wide(args.root)
        elif args.command=='build-all':
            for key in ENGINES: ninfer_precision_build.build(args.root,key)
        else: ninfer_precision_build.build(args.root, args.engine)
    elif args.command in ['analyze','analyze-distributions','build-cpu','analyze-operators','analyze-acceptance','analyze-states','report','finalize']:
        import ninfer_precision_analysis as analysis
        if args.command=='analyze': analysis.summarize(args.root)
        elif args.command=='build-cpu':print(analysis.build_cpu(args.root))
        elif args.command=='analyze-operators':analysis.analyze_operators(args.root)
        elif args.command=='report':analysis.report(args.root)
        elif args.command=='finalize':analysis.finalize(args.root)
        elif args.command=='analyze-acceptance':analysis.analyze_acceptance(args.root,args.pilot)
        elif args.command=='analyze-states':analysis.state_comparisons(args.root,args.pilot)
        else: analysis.analyze_distributions(args.root)
    elif args.command in ['prepare-tokens','prepare-images','prepare-operator-counts','prepare-long','freeze-protocol','prepare-pilot-repair','prepare-package-repair','prepare-native-control','prepare-wide','run-wide','prepare-resume','resume-controls','amend-boundaries','manifest','run-probe','package','session']:
        import ninfer_precision_run as runner
        if args.command=='run-probe': runner.run_probe(args.root,args.engine,args.suite,args.k)
        elif args.command=='prepare-wide':runner.prepare_wide(args.root)
        elif args.command=='run-wide':runner.run_wide(args.root,args.engine)
        elif args.command=='prepare-resume':runner.prepare_resume(args.root)
        elif args.command=='resume-controls':runner.resume_controls(args.root)
        elif args.command=='package': runner.package(args.root,args.engine,args.k,args.images)
        elif args.command=='session': runner.session(args.root,args.phase)
        else: {'prepare-tokens':runner.prepare_tokens,'prepare-images':runner.prepare_images,'prepare-operator-counts':runner.prepare_operator_counts,'prepare-long':runner.prepare_long,'freeze-protocol':runner.freeze_protocol,'prepare-pilot-repair':runner.prepare_pilot_repair,'prepare-package-repair':runner.prepare_package_repair,'prepare-native-control':runner.prepare_native_control,'amend-boundaries':runner.amend_boundaries,'manifest':runner.manifest}[args.command](args.root)
    else:
        {'prepare':prepare,'artifact-check':artifact_check,'verify-freeze':verify_freeze}[args.command](args.root)


if __name__ == '__main__':
    main()
