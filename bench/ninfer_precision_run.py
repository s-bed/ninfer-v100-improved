"""Fixture tokenization, sequential probe/package jobs and campaign manifests."""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zlib
import struct
import ninfer_precision as c
import ninfer_tune as tune
from ninfer_precision_grade import grade


def probe_binary(root,key):
    info=json.loads((root/'probes'/key/'build-provenance.json').read_text())
    path=Path(info['binary'])
    if c.sha(path)!=info['binary_sha256']: raise ValueError('probe binary changed after provenance')
    return path


def run_process(command, directory, timeout=None, extra_env=None):
    directory.mkdir(parents=True,exist_ok=True)
    c.write_json(directory/'command.json',command)
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=':'.join(['/home/simon/ninfer-build-deps/usr/lib/x86_64-linux-gnu',
                                   '/usr/local/cuda-12.9/lib64','/usr/lib/x86_64-linux-gnu'])
    if extra_env: env.update(extra_env)
    proc=None
    with (directory/'process.log').open('a') as log:
        try:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            tune.server_group_event(proc.pid,'start')
            status=proc.wait(timeout=timeout)
            if status: raise RuntimeError(f'probe exited {status}: {directory / "process.log"}')
        finally:
            if proc and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
            if proc: tune.server_group_event(proc.pid,'done')


def render(case):
    if not case.get('history'):
        return c.SETTINGS['template'].format(prompt=case['prompt'])
    text='<|im_start|>system\nFollow the request exactly. Give only the requested answer.\n<|im_end|>\n'
    for message in case['history']+[{'role':'user','content':case['prompt']}]:
        text+=f'<|im_start|>{message["role"]}\n{message["content"]}\n<|im_end|>\n'
    return text+'<|im_start|>assistant\n<think>\n\n</think>\n\n'


def prepare_tokens(root):
    c.verify_freeze(root)
    if (root/'fixtures/tokens-freeze.json').exists(): raise ValueError('tokens already frozen')
    resources=root/'fixtures/tokenizer'
    resources.mkdir(parents=True,exist_ok=True)
    directory,start=c.artifact_directory(c.ENGINES['vendor']['artifact'])
    by_id={o['id']:o for o in directory['objects']}
    with c.ENGINES['vendor']['artifact'].open('rb') as artifact:
        for name in ['tokenizer.json','tokenizer_config.json','generation_config.json']:
            obj=by_id['resource/text/'+name]
            artifact.seek(start+obj['offset'])
            (resources/name).write_bytes(artifact.read(obj['bytes']))
    jobs=root/'fixtures/tokenize-jobs.jsonl'
    streams=json.loads((root/'fixtures/corpus.json').read_text())
    for stream in streams:
        c.append(jobs,{'id':stream['id'],'path':stream['path'],'max_tokens':8193})
    cases={case['id']:case for case in json.loads((root/'fixtures/quality.json').read_text())}
    for case in cases.values():
        c.append(jobs,{'id':case['id'],'text':render(case)})
    # Two fixed real-corpus prefixes, with known retrieval sentinels at separated positions.
    for i,stream in enumerate([streams[0],streams[6]]):
        text=Path(stream['path']).read_text()
        c.append(jobs,{'id':f'long-source-{i}','text':text,'max_tokens':240000})
    for name,text in [('needle-first','\nProject KESTREL passphrase is maple-7c90.\n'),
                      ('needle-last','\nBEACON maintenance code is 519204.\n'),
                      ('needle-question','\nReturn only the project KESTREL passphrase, a space, then the BEACON maintenance code.\n'),
                      ('decode-question','\nRepeat this exact sentence 100 times, with no introduction: The blue kite flies above the quiet lake.\n'),
                      ('decode-script','The blue kite flies above the quiet lake. '*200)]:
        c.append(jobs,{'id':name,'text':text})
    binary=probe_binary(root,'r5')
    output=root/'fixtures/tokens.jsonl'
    run_process([str(binary),'tokenize',str(resources),str(jobs),str(output)],root/'preparation/tokenize',timeout=900)
    tokens={row['id']:row['tokens'] for row in c.rows(output)}
    for stream in streams:
        if len(tokens[stream['id']])!=8193: raise ValueError(f'stream too short: {stream["id"]}')
    # Serialize exact score windows. Every target index is scored once.
    score_jobs=[]
    for start in [0,4096,6144,8192]:
        end=min(8193,4096 if start==0 else start+2048)
        begin=max(0,end-4096)
        first=1 if start==0 else start-begin
        for stream in streams:
            score_jobs.append({'id':f'{stream["id"]}/w{start}', 'document':stream['id'],
                'domain':stream['domain'],'input_begin':begin,'target_begin':begin+first,'target_end':end,
                'first_target':first,'tokens':tokens[stream['id']][begin:end]})
    c.write_json(root/'fixtures/score-windows.json',score_jobs)
    repeat=[]
    for i,stream in enumerate(streams[:8]):
        for repetition in range(2 if i<4 else 1):
            repeat.append({'id':f'repeat-{i}-{repetition}','document':stream['id'],
                'tokens':tokens[stream['id']][:2049],'first_target':1})
    c.write_json(root/'fixtures/repeatability.json',repeat)
    boundary=[]
    for width in c.WIDTHS:
        for i in range(2):
            source=tokens[streams[i*6]['id']]
            boundary.append({'id':f'boundary-{i}-{width}','width':width,'first_target':2049,
                             'tokens':source[:2049+width]})
    c.write_json(root/'fixtures/boundaries.json',boundary)
    freeze_files=[output,root/'fixtures/score-windows.json',root/'fixtures/repeatability.json',root/'fixtures/boundaries.json']
    c.write_json(root/'fixtures/tokens-freeze.json',{'created':time.time(),'files':{str(p.relative_to(root)):c.sha(p) for p in freeze_files},
         'scored_targets':sum(len(x['tokens'])-x['first_target'] for x in score_jobs),
         'public_vocab':c.rows(output)[0]['public_vocab'], 'numerical_inputs':'identical token IDs; no chat template',
         'quality_template':c.SETTINGS['template'],'selection':'fixed stream order and domain round robin; no result-dependent selection'})
    print('TOKENS FROZEN',len(tokens),'score targets',sum(len(x['tokens'])-x['first_target'] for x in score_jobs),flush=True)


def token_map(root):
    c.verify_freeze(root)
    freeze=json.loads((root/'fixtures/tokens-freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if c.sha(root/name)!=digest: raise ValueError('token fixture changed: '+name)
    return {row['id']:row['tokens'] for row in c.rows(root/'fixtures/tokens.jsonl')}


def amend_boundaries(root):
    # Preparation correction only; forbidden once measurement records exist.
    if any((root/'runs').glob('*/**/records.jsonl')): raise ValueError('measurements already started; cannot amend frozen boundaries')
    tokens=token_map(root)
    path=root/'fixtures/boundaries.json'
    previous=root/'fixtures/boundaries-v1-off-by-one.json'
    if previous.exists(): raise ValueError('boundary correction already recorded')
    previous.write_bytes(path.read_bytes())
    streams=json.loads((root/'fixtures/corpus.json').read_text())
    windows=[]
    for width in c.WIDTHS:
        for i in range(2):
            source=tokens[streams[i*6]['id']]
            windows.append({'id':f'boundary-{i}-{width}','width':width,'first_target':2049,
                            'tokens':source[:2049+width],
                            'predictor_prefix':2048,'actual_prefill_tail':width})
    c.write_json(path,windows)
    freeze=root/'fixtures/tokens-freeze.json'
    (root/'fixtures/tokens-freeze-v1.json').write_bytes(freeze.read_bytes())
    data=json.loads(freeze.read_text());data['files']['fixtures/boundaries.json']=c.sha(path)
    data['premeasurement_amendment']={'created':time.time(),
        'reason':'Causal scoring executes N-1 predictor tokens. Add the final target to exercise exactly the requested narrow prefill width; no results inspected.'}
    c.write_json(freeze,data)
    print('Boundary windows corrected and refrozen before any GPU measurement',flush=True)


def run_probe(root,key,suite,k=0):
    token_ids=token_map(root)
    directory=root/'runs'/key/f'{suite}-k{k}'
    if (directory/'records.jsonl').exists(): raise ValueError('probe job already has records')
    directory.mkdir(parents=True,exist_ok=True)
    cases={case['id']:case for case in json.loads((root/'fixtures/quality.json').read_text())}
    if suite in ['likelihood','repeatability','boundary','operators']:
        name={'likelihood':'score-windows','repeatability':'repeatability','boundary':'boundaries','operators':'repeatability'}[suite]
        jobs=json.loads((root/'fixtures'/f'{name}.json').read_text())
        if suite=='operators': jobs=jobs[:1]
        for job in jobs:
            job['observer_dir']=str(directory/'observations'/job['id'])
        mode='score'
        context=4096
    elif suite in ['quality','mtp-quality','native']:
        selection=json.loads((root/'fixtures'/({'quality':'quality-order','mtp-quality':'mtp-subset','native':'native-subset'}[suite]+'.json')).read_text())
        jobs=[]
        for cid in selection:
            case=cases[cid]
            job={'id':cid,'tokens':token_ids[cid],'max_tokens':512 if case['domain']=='code' else 128,'reuse':False}
            if suite=='native': job['messages']=[{'role':'system','content':'Follow the request exactly. Give only the requested answer.'}]+case.get('history',[])+[{'role':'user','content':case['prompt']}]
            jobs.append(job)
        mode='native' if suite=='native' else 'generate'
        context=8192
    elif suite in ['decode','observer-pilot']:
        jobs=[]
        script=token_ids['decode-script'][:130]
        for depth in ([2048] if suite=='observer-pilot' else [8192,32768,65536]):
            for i in range(1 if suite=='observer-pilot' else 2):
                source=token_ids[f'long-source-{i}']
                if not source: raise ValueError('empty long-context source')
                fill=(source*((depth//len(source))+1))[:depth-len(token_ids['decode-question'])]
                job={'id':f'decode-{depth}-{i}','tokens':fill+token_ids['decode-question'],
                     'max_tokens':129,'forced_tokens':script,'observer_dir':str(directory/'observations'/f'decode-{depth}-{i}')}
                jobs.append(job)
        mode='generate'
        context=4096 if suite=='observer-pilot' else 73728
    elif suite=='long':
        freeze=json.loads((root/'fixtures/long-freeze.json').read_text())
        if c.sha(root/'fixtures/long-jobs.json')!=freeze['jobs_sha256']: raise ValueError('long fixture changed')
        jobs=json.loads((root/'fixtures/long-jobs.json').read_text())
        for job in jobs: job['observer_dir']=str(directory/'observations'/job['id'])
        mode='generate'
        context=237568
    else: raise ValueError('unsupported suite')
    jobs_file=directory/'jobs.jsonl'
    for job in jobs: c.append(jobs_file,job)
    c.write_json(directory/'settings.json',{'engine':key,'suite':suite,'k':k,'context':context,'jobs':len(jobs),
        'prefill_chunk':2048,'prefill_chunk_requested':2048,
        'prefill_chunk_normalized':1024 if mode=='score' else 2048,
        'prefill_align_env':os.environ.get('NINFER_PREFILL_ALIGN'),
        'prefill_chunk_effective_source_derived':(
            (1024 if mode=='score' else 2048) if key=='v100' or os.environ.get('NINFER_PREFILL_ALIGN','').startswith('0')
            else (1152 if mode=='score' else 1920)),
        'device_profile':'off','kv':'int8','state':'existing FP32','production_cache':False})
    monitor=tune.GpuMonitor()
    monitor.start()
    started=time.monotonic()
    extra_env=None
    if suite=='mtp-quality':extra_env={'NINFER_PRECISION_DISABLE_LOOKUP':'1'}
    if suite=='operators':
        if key!='v100':
            freeze=json.loads((root/'fixtures/operator-freeze.json').read_text())
            for relative,digest in freeze['files'].items():
                if c.sha(root/relative)!=digest: raise ValueError('operator activation fixture changed')
        extra_env={'NINFER_PRECISION_OPERATOR_OUT':str(directory/'operators'),
                   'NINFER_PRECISION_OPERATOR_FIXTURES':str(root/'fixtures/operator-activations'),
                   'NINFER_PRECISION_OPERATOR_COUNTS':str(root/'fixtures/operator-counts.json'),
                   'NINFER_PRECISION_OPERATOR_CAPTURE':'1' if key=='v100' else '0'}
    try:
        run_process([str(probe_binary(root,key)),mode,str(c.ENGINES[key]['artifact']),str(jobs_file),
                     str(directory/'records.jsonl'),str(k),str(context)],directory,extra_env=extra_env)
    finally:
        monitor.stop()
        c.write_json(directory/'gpu.json',monitor.samples)
        c.write_json(directory/'timing.json',{'seconds':time.monotonic()-started,'gpu':monitor.stats(started,time.monotonic())})
    if suite in ['decode','observer-pilot']:
        expected=token_ids['decode-script'][:129]
        violations=[]
        for record in c.rows(directory/'records.jsonl'):
            if record.get('tokens')!=expected:
                violations.append({'id':record['id'],'reason':'output does not match forced 129-token history'})
        for job in jobs:
            for state in c.rows(Path(job['observer_dir'])/'states.jsonl'):
                if state['ledger_frontier']!=len(job['tokens'])+state['script_cursor'] or state['execution_frontier']+1!=state['ledger_frontier']:
                    violations.append({'id':job['id'],'reason':'script/ledger/execution frontier mismatch','state':state})
            if k:
                for proposal in c.rows(Path(job['observer_dir'])/'proposal-inputs.jsonl'):
                    if proposal['width']!=k or proposal['extent']>k:
                        violations.append({'id':job['id'],'reason':'actual verification width differs from requested K','proposal':proposal})
        c.write_json(directory/'control-validation.json',{'passed':not violations,'violations':violations})
        if violations: raise RuntimeError('forced execution observer validation failed; measurements are not qualified')
    if suite in ['quality','mtp-quality','native']:
        for record in c.rows(directory/'records.jsonl'):
            result={'id':record['id'],'engine':key,'suite':suite}
            if 'error' in record: result.update(infrastructure_error=record['error'],correct=None)
            else: result.update(grade(cases[record['id']],record['content']))
            c.append(directory/'grades.jsonl',result)
    if suite=='operators' and key=='v100':
        files={str(p.relative_to(root)):c.sha(p) for p in sorted((root/'fixtures/operator-activations').rglob('*')) if p.is_file()}
        files['fixtures/operator-counts.json']=c.sha(root/'fixtures/operator-counts.json')
        c.write_json(root/'fixtures/operator-freeze.json',{'files':files,'created':time.time(),
            'selection':'first, middle and last text-layer occurrence per weight format/shape/operator; fixed widths; no result-dependent selection'})
    if suite=='long' and key=='r5' and (root/'control/operator-wide-manifest.json').exists():
        # The model process has exited. Reuse the same exclusive GPU session for
        # the alignment-corrected boundaries and frozen three-engine replay,
        # then let the outer guard restore once.
        correct_boundary_alignment(root)
        for wide_key in c.ENGINES:
            c.append(root/'control/inline-wide.jsonl',{'engine':wide_key,'event':'start','created':time.time()})
            run_wide(root,wide_key)
            c.append(root/'control/inline-wide.jsonl',{'engine':wide_key,'event':'complete','created':time.time()})


def package_flags(directory,key,k=3,images=False):
    flags={'--max-context':'8192','--kv-capacity':'8192','--max-concurrency':'1','--kv-dtype':'int8',
           '--prefill-chunk':'2048','--pending-timeout-ms':'600000','--preserve-thinking':'','--vision':''}
    if not images:flags['--no-prefix-reuse']=''
    if k: flags.update({'--spec':'mtp','--draft-tokens':str(k),'--lm-head-draft':''})
    else: flags['--spec']='none'
    if key=='v100':
        if images: flags.update({'--host-state-slots':'0','--host-kv-mib':'0','--device-state-slots':'2'})
    else:
        flags.update({'--device-profile':'off','--ngram-draft-tokens':'0'})
        if images:
            flags.update({'--use-alt-prefix-caching':'','--host-cache-mib':'1024','--device-snapshot-slots':'2',
                          '--prefix-cache-file':str(directory/'prefix.cache')})
        if key=='r5': flags['--v100-kernels']='ported'
    return flags


def package(root,key,k=3,images=False):
    token_map(root)
    directory=root/'runs'/key/('package-images' if images else f'package-native-k{k}')
    directory.mkdir(parents=True,exist_ok=True)
    flags=package_flags(directory,key,k,images)
    args=argparse.Namespace(runtime='original' if key=='v100' else 'all',
          launcher=str(c.ENGINES[key]['bundle']/'ninfer-serve-run'),artifact=str(c.ENGINES[key]['artifact']),
          shutdown_timeout=150,cache_dir=str(root/'runs'))
    server=tune.Server(directory,flags,args)
    cases={case['id']:case for case in json.loads((root/'fixtures/quality.json').read_text())}
    selected=json.loads((root/'fixtures/native-subset.json').read_text())
    server.start()
    try:
        ready=server.wait('ready',240)
        c.write_json(directory/'ready.json',ready)
        if not ready['ok']: raise RuntimeError('package startup failed')
        iterable=json.loads((root/'fixtures/images.json').read_text()) if images else [cases[cid] for cid in selected]
        for case in iterable:
            cid=case['id']
            if images:
                uri='data:image/png;base64,'+base64.b64encode((root/case['path']).read_bytes()).decode()
                messages=[{'role':'user','content':[{'type':'text','text':case['prompt']},
                            {'type':'image_url','image_url':{'url':uri}}]}]
            else:
                messages=[{'role':'system','content':'Follow the request exactly. Give only the requested answer.'}]+case.get('history',[])+[{'role':'user','content':case['prompt']}]
            payload={'model':'qwen3.8-27b','messages':messages,'temperature':0,'top_k':0,'top_p':1,
                     'min_p':0,'presence_penalty':0,'frequency_penalty':0,'seed':1729,
                     'max_tokens':512 if case['domain']=='code' else 128,
                     'chat_template_kwargs':{'enable_thinking':False},'stream':False}
            begin=time.monotonic()
            status,response=tune.http_json(server.port,'POST','/v1/chat/completions',payload,timeout=300)
            record={'id':cid,'status':status,'response':response,'seconds':time.monotonic()-begin}
            if status==200:
                record['grade']=grade(case,response['choices'][0]['message'].get('content') or '')
            c.append(directory/'records.jsonl',record)
            print('PACKAGE',key,cid,status,flush=True)
            if images and cid.endswith('-0') and status==200:
                # Same image and history with an appended turn; explicit reuse evidence.
                payload['messages']=messages+[{'role':'assistant','content':response['choices'][0]['message'].get('content') or ''},
                                               {'role':'user','content':'Give that same color name again, with no other text.'}]
                again_status,again=tune.http_json(server.port,'POST','/v1/chat/completions',payload,timeout=300)
                continuation={'id':cid,'status':again_status,'response':again}
                if again_status==200:
                    continuation['grade']=grade(case,again['choices'][0]['message'].get('content') or '')
                    usage=again.get('usage',{})
                    continuation['prompt_tokens']=usage.get('prompt_tokens')
                    continuation['cached_tokens']=usage.get('prompt_tokens_details',{}).get('cached_tokens')
                c.append(directory/'cache-continuations.jsonl',continuation)
    finally:
        server.stop()
        c.write_json(directory/'shutdown.json',server.shutdown)


def manifest(root):
    token_map(root)
    jobs=[]
    python=sys.executable
    command=[python,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root)]
    # Suite-major order means every baseline reaches each diagnostic before the next
    # suite is attempted. No result-dependent reorder or dropping is permitted.
    for suite,k in [('repeatability',0),('observer-pilot',0),('observer-pilot',3),('native',3)]:
        for key in c.ENGINES:
            jobs.append(command+['run-probe','--engine',key,'--suite',suite,'--k',str(k)])
            if suite=='native': jobs.append(command+['package','--engine',key,'--k',str(k)])
    c.write_json(root/'control/pilot-manifest.json',jobs)
    jobs=[]
    for suite,k in [('operators',0),('likelihood',0),('boundary',0),('decode',0),('decode',1),('decode',3),('decode',5),
                    ('quality',0),('mtp-quality',3),('long',3)]:
        for key in c.ENGINES:
            jobs.append(command+['run-probe','--engine',key,'--suite',suite,'--k',str(k)])
            if suite=='mtp-quality': jobs.append(command+['package','--engine',key,'--k','3','--images'])
    c.write_json(root/'control/main-manifest.json',jobs)
    print('Wrote sequential pilot and main manifests',flush=True)


def prepare_images(root):
    if (root/'fixtures/images-freeze.json').exists(): raise ValueError('images already frozen')
    colors={'red':(230,15,15),'blue':(15,15,230),'green':(15,190,15),'yellow':(240,240,10),
            'purple':(150,20,180),'cyan':(10,230,230),'black':(0,0,0),'white':(255,255,255)}
    fixtures=[]
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    for name,rgb in colors.items():
        for variant in range(3):
            width=height=256
            image=bytearray()
            for y in range(height):
                image.append(0)
                for x in range(width):
                    selected=variant==0 or (variant==1 and 32<=x<224 and 32<=y<224) or (variant==2 and y<128)
                    image.extend(rgb if selected else (130,130,130))
            relative=Path('fixtures/images')/f'{name}-{variant}.png'
            path=root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+
                             chunk(b'IDAT',zlib.compress(image))+chunk(b'IEND',b''))
            question=['Name the color filling this image.','Name the color of the central square, ignoring the gray border.',
                      'Name the color of the upper half, ignoring the gray lower half.'][variant]
            fixtures.append({'id':f'image-{name}-{variant}','path':str(relative),'domain':'image','grader':'exact',
                             'prompt':question+' Return one lowercase English color name only.','expected':name})
    c.write_json(root/'fixtures/images.json',fixtures)
    files={f['path']:c.sha(root/f['path']) for f in fixtures}
    files['fixtures/images.json']=c.sha(root/'fixtures/images.json')
    c.write_json(root/'fixtures/images-freeze.json',{'created':time.time(),'files':files,
        'scope':'24 synthetic color and spatial-position sentinels; no broad vision-equivalence inference'})
    prepare_operator_counts(root)


def prepare_operator_counts(root):
    if (root/'fixtures/operator-freeze.json').exists(): raise ValueError('operator fixtures already frozen')
    old,_=c.artifact_directory(c.ENGINES['v100']['artifact'])
    counts={}
    for obj in old['objects']:
        if not obj.get('name','').startswith('text/layers/') or obj.get('format') not in ['NVFP4','FP8_E4M3FN_ROW_BF16S'] or len(obj['shape'])!=2: continue
        q=7 if obj['format']=='NVFP4' else 8
        n,k=obj['shape']
        name=f'q{q}-n{n}-k{k}'
        counts[name]=counts.get(name,0)+1
    c.write_json(root/'fixtures/operator-counts.json',counts)


def prepare_long(root):
    tokens=token_map(root)
    if (root/'fixtures/long-freeze.json').exists(): raise ValueError('long fixture already frozen')
    tokenizer_jobs=root/'fixtures/long-tokenize-jobs.jsonl'
    c.append(tokenizer_jobs,{'id':'known-answer','text':'maple-7c90 519204\n'})
    additions=root/'fixtures/long-token-additions.jsonl'
    run_process([str(probe_binary(root,'r5')),'tokenize',str(root/'fixtures/tokenizer'),str(tokenizer_jobs),str(additions)],
                root/'preparation/tokenize-long',timeout=60)
    answer=c.rows(additions)[0]['tokens']
    source=tokens['long-source-0'];full=(source*((229376//len(source))+1))[:229376]
    for offset,label in [(1024,'needle-first'),(64000,'needle-last')]:
        needle=tokens[label];full[offset:offset+len(needle)]=needle
    question=tokens['needle-question']
    first=full[:131072]+question
    second=first+answer+full[131072:]+question
    jobs=[{'id':'long-131072','tokens':first,'max_tokens':64,'reuse':True},
          {'id':'long-229376','tokens':second,'max_tokens':64,'reuse':True}]
    c.write_json(root/'fixtures/long-jobs.json',jobs)
    c.write_json(root/'fixtures/long-freeze.json',{'created':time.time(),'jobs_sha256':c.sha(root/'fixtures/long-jobs.json'),
        'prompt_lengths':[len(first),len(second)],'expected':['maple-7c90','519204'],
        'scope':'One frozen growing text history. The second input extends the first with the known correct reply, fixed corpus suffix and repeated question. It can retrieve the prior reply; this is a sentinel, not an independent second blind needle test.',
        'known_answer_tokens_sha256':c.sha(additions)})


def freeze_protocol(root):
    token_map(root)
    path=root/'control/protocol-freeze.json'
    if path.exists(): raise ValueError('protocol already frozen')
    files={str(p.relative_to(root)):c.sha(p) for p in sorted((root/'fixtures').rglob('*')) if p.is_file()}
    sources={str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted((c.ROOT/'bench').glob('ninfer_precision*')) if p.is_file()}
    inputs=json.loads((root/'provenance/engines.json').read_text())
    if c.sha(c.ROOT/'llama-swap.yaml')!=inputs['config_sha256'] or os.readlink(c.ENGINE/'runtime/active')!=inputs['active']:
        raise ValueError('production configuration changed during preparation')
    caches={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino} for p in (c.ENGINE/'cache').glob('*.cache') if p.is_file()}
    c.write_json(path,{'created':time.time(),'fixtures':files,'sources':sources,'production_caches':caches,
        'primary_comparisons':c.PAIRS,'primary_claims':6,'alpha_per_claim':.05/6,
        'accuracy_resampling':'Cluster repeated code algorithms, tool schemas and arithmetic operations; retrieval/history by whole case. Preserve token/case weighting.',
        'likelihood_resampling':'Whole paired frozen corpus streams; 8192 target tokens each.',
        'decode_measurement':'129 output tokens: first forced prefill anchor plus 128 ordinary decode transitions; MTP commits the same scripted history at K=1/3/5.',
        'budget_seconds':21600,'restoration_reserve_seconds':1800,
        'balanced_coverage':'Fixed suite-major engine order, fixed domain interleave, no result-dependent selection. Missing required coverage forces an inconclusive conclusion.'})
    print('Protocol frozen; GPU measurement may begin',flush=True)


def prepare_pilot_repair(root):
    import shutil
    ledger=json.loads((root/'control/budget.json').read_text())
    if ledger['sessions'][-1]['finished'] is None or not ledger['sessions'][-1]['restored']:
        raise ValueError('restore inference before preparing pilot corrections')
    archive=root/'probes/v100/before-lookup-control'
    if archive.exists():raise ValueError('pilot repair already prepared')
    archive.mkdir()
    binary=probe_binary(root,'v100')
    shutil.copy2(binary,archive/binary.name)
    shutil.copy2(root/'probes/v100/build-provenance.json',archive/'build-provenance.json')
    source=root/'probes/v100/source/src/targets/qwen3_6/impl/runtime'
    for name in ['program_impl.h','decode_impl.h','mtp_impl.h']:
        shutil.copy2(source/name,archive/name)
    moves=[]
    for key,suite,suffix in [('v100','observer-pilot-k3','context-copy-initial'),
                             ('v100','native-k3','before-lookup-control'),
                             ('v100','package-native-k3','reuse-enabled-initial'),
                             ('vendor','package-native-k3','rejected-flags')]:
        old=root/'runs'/key/suite
        if old.exists():
            dest=old.with_name(old.name+'-'+suffix)
            old.rename(dest);moves.append({'before':str(old),'preserved':str(dest)})
    command=[sys.executable,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root)]
    jobs=[command+['run-probe','--engine','v100','--suite','observer-pilot','--k','3'],
          command+['run-probe','--engine','v100','--suite','native','--k','3'],
          command+['package','--engine','v100','--k','3'],
          command+['package','--engine','vendor','--k','3'],
          command+['run-probe','--engine','r5','--suite','native','--k','3'],
          command+['package','--engine','r5','--k','3']]
    c.write_json(root/'control/pilot-controls-manifest.json',jobs)
    c.append(root/'control/protocol-amendments.jsonl',{'created':time.time(),'stage':'pilot, before primary suites',
        'reasons':['Original V100 automatically selected 15-token context-copy verification during a nominal K3 test; disable the optional proposer only in forced diagnostics and the controlled MTP subset.',
                   'Server harness added a persistence file while the package profile removed the Host tier; use a private per-engine cache and a 1 GiB Host tier.',
                   'Disable reuse in package/probe agreement cases so both execute fresh identical native-template histories. Image continuation checks keep reuse enabled.'],
        'preserved_records':moves,'original_probe_archive':str(archive),
        'primary_cases_or_expected_answers_changed':False,'precision_or_equivalence_margins_changed':False})
    print('Pilot correction manifest prepared; original evidence preserved',flush=True)


def prepare_package_repair(root):
    ledger=json.loads((root/'control/budget.json').read_text())
    if not ledger['sessions'][-1]['restored']:raise ValueError('restore service first')
    marker=root/'control/package-controls-repair.json'
    if marker.exists():raise ValueError('package repair already prepared')
    moves=[]
    for key in c.ENGINES:
        for suite in ['native-k3','package-native-k3']:
            old=root/'runs'/key/suite
            if old.exists():
                dest=old.with_name(old.name+'-before-cache-control')
                old.rename(dest);moves.append({'before':str(old),'preserved':str(dest)})
    command=[sys.executable,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root)]
    jobs=[]
    for key in c.ENGINES:
        jobs.extend([command+['run-probe','--engine',key,'--suite','native','--k','3'],
                     command+['package','--engine',key,'--k','3']])
    c.write_json(root/'control/pilot-controls-manifest.json',jobs)
    amendment={'created':time.time(),'stage':'pilot, before primary suites',
        'reasons':['CLI forbids explicit cache-capacity flags with --no-prefix-reuse. Remove capacity, cache mode and persistence flags from fresh package agreement tests; disable context cache in matching native probe mode. Image checks keep valid private caching.',
                   'External production config change observed: startup preload now ninfer-all-v100. Preserve that change; campaign restoration readiness will follow it.'],
        'preserved_records':moves,'primary_cases_or_expected_answers_changed':False,
        'precision_or_equivalence_margins_changed':False,'config_sha256':c.sha(c.ROOT/'llama-swap.yaml')}
    c.append(root/'control/protocol-amendments.jsonl',amendment);c.write_json(marker,amendment)
def prepare_native_control(root):
    command=[sys.executable,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root)]
    main=root/'control/main-manifest.json'
    if (root/'control/main-manifest-before-native-control.json').exists():raise ValueError('native control already prepared')
    if any((root/'runs').glob('*/quality-k0/records.jsonl')):raise ValueError('primary measurement already began')
    existing=json.loads(main.read_text())
    (root/'control/main-manifest-before-native-control.json').write_bytes(main.read_bytes())
    additions=[command+['run-probe','--engine',key,'--suite','native','--k','0'] for key in c.ENGINES]
    c.write_json(main,additions+existing)
    c.append(root/'control/protocol-amendments.jsonl',{'created':time.time(),
        'reason':'Add the same frozen 24-case native-template subset without MTP to separate template effects from speculation. Added before primary-suite measurement, without changing any cases or margins.'})


def record_prefill_alignment_review(root):
    """CPU-only correction of option normalization versus actual planner widths."""
    path=root/'provenance/effective-engine-options.json'
    previous=root/'provenance/effective-engine-options-before-alignment-review.json'
    if path.exists() and not previous.exists():previous.write_bytes(path.read_bytes())
    # Mirror the pinned planner's deterministic wave-efficiency calculation.
    # The PG500 has 72 SMs, this attention geometry has 24 query heads and
    # default fast_prefill_kernel=false selects 64 query rows per CTA.
    def aligned(requested):
        def efficiency(width):
            ctas=((width+63)//64)*24
            return ctas/(((ctas+71)//72)*72)
        best=requested;filled=efficiency(best)
        for width in range(max(128,requested//2//128*128),requested+257,128):
            value=efficiency(width)
            if value>filled+1e-9 or value>filled-1e-9 and abs(width-requested)<abs(best-requested):
                best=width;filled=value
        return best
    values={str(requested):aligned(requested) for requested in [1024,2048,4096]}
    if values!={'1024':1152,'2048':1920,'4096':4224}:raise ValueError('unexpected source-derived prefill alignment')
    c.write_json(path,{'causal_scoring':{'prefill_chunk_requested':2048,'prefill_chunk_normalized':1024,
        'prefill_chunk_effective_by_engine':{'v100':1024,'vendor':1152,'r5':1152},
        'graphs':False,'cache':False,'speculation':'none','context':4096,'stride':2048,'kv':'int8 group64'},
        'generation':{'prefill_chunk_requested':2048,
            'prefill_chunk_effective_source_derived_by_engine':{'v100':2048,'vendor':1920,'r5':1920},
            'graphs':'actual launches observed','kv':'int8 group64','state':'existing FP32 recurrent and BF16 convolution'},
        'planner':{'all_default':'NINFER_PREFILL_ALIGN enabled','disable':'NINFER_PREFILL_ALIGN=0',
            'source_derived_requested_to_effective':values,'multiprocessors':72,'query_heads':24,
            'rows_per_CTA':64,'fast_prefill_kernel':False,
            'observed_evidence':'All 21 vendor/R5 operator metadata records have actual_capture_columns=1152; V100 has1024.'},
        'boundary_correction':{'engines':['vendor','r5'],'environment':{'NINFER_PREFILL_ALIGN':'0'},
            'effective_chunk':1024,'source_derived_predictor_chunks':'1024,1024,then requested tail width',
            'frozen_inputs_unchanged':True,'original_all_runs_preserved_as':'boundary-native-aligned-k0'},
        'source_evidence':{'all_normalization':'src/runtime/engine/model_instance.cpp:normalize_engine_options',
            'all_planner':'src/models/qwen3_5/program/planning/startup.cpp:effective_prefill_chunk',
            'wave_selection':'src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:causal_softmax_attention_prompt_aligned_chunk'},
        'note':'Original all boundary tails were896+width, not1..128. Corrected boundary runs disable alignment. Existing likelihood/decode/quality remain valid matched-token end-to-end native-alignment comparisons, with a scheduling confound versus original V100; do not interpret them as identical-chunk arithmetic isolation.'})
    marker=root/'provenance/prefill-alignment-review.json'
    if not marker.exists():
        entry={'created':time.time(),'reason':'Planner alignment follows public option normalization: all scoring executes1152-column chunks, not1024; fixed2K-prefix boundary tails therefore missed the intended narrow widths. Preserve original all boundary records and rerun the same frozen windows with alignment disabled after R5 long execution, before wide replay.',
               'primary_cases_or_answers_changed':False,'numerical_tolerances_or_equivalence_margins_changed':False,
               'generation_scheduling_confound':'All generation requested2048 but effective1920; original V100 remains2048.',
               'source_derived_requested_to_effective':values}
        c.write_json(marker,entry);c.append(root/'control/protocol-amendments.jsonl',entry)


def correct_boundary_alignment(root):
    """Replay the identical frozen boundary windows at the actual intended widths."""
    record_prefill_alignment_review(root)
    import shutil
    expected={job['id']:job['width'] for job in json.loads((root/'fixtures/boundaries.json').read_text())}
    marker=root/'control/boundary-alignment-correction.json'
    info=json.loads(marker.read_text()) if marker.exists() else {'created':time.time(),'relocations':[],'engines':{},
        'environment':{'NINFER_PREFILL_ALIGN':'0'},'effective_chunk':1024,
        'tail_width_evidence':'Source-derived from the disabled-alignment planner and the frozen predictor_count=2048+width; not a CUDA event counter.'}
    for key in ['vendor','r5']:
        directory=root/'runs'/key/'boundary-k0';archive=root/'runs'/key/'boundary-native-aligned-k0'
        if not archive.exists():
            if not directory.is_dir():raise ValueError('missing original native-aligned boundary run: '+key)
            shutil.move(str(directory),str(archive))
            info['relocations'].append({'engine':key,'before':str(directory),'preserved':str(archive)})
            info['engines'][key]={'status':'native-aligned records preserved'}
            c.write_json(marker,info)
        def complete():
            output=directory/'records.jsonl';settings=directory/'settings.json'
            if not output.exists() or not settings.exists():return False
            metadata=json.loads(settings.read_text());rows=c.rows(output)
            return (metadata.get('prefill_align_env')=='0' and len(rows)==len(expected) and
                    {row['id'] for row in rows}==set(expected) and
                    all('error' not in row and len(row.get('logprobs',[]))==expected[row['id']] for row in rows))
        if not complete():
            if directory.exists():raise ValueError('incomplete corrected boundary evidence needs review before resuming: '+str(directory))
            info['engines'][key]={'status':'running'};c.write_json(marker,info)
            c.append(root/'control/inline-wide.jsonl',{'engine':key,'event':'boundary-alignment-start','created':time.time()})
            previous=os.environ.get('NINFER_PREFILL_ALIGN')
            os.environ['NINFER_PREFILL_ALIGN']='0'
            try:run_probe(root,key,'boundary',0)
            finally:
                if previous is None:os.environ.pop('NINFER_PREFILL_ALIGN',None)
                else:os.environ['NINFER_PREFILL_ALIGN']=previous
            if not complete():raise RuntimeError('corrected boundary replay is incomplete: '+key)
        info['engines'][key]={'status':'complete','cases':len(expected),'widths':sorted(set(expected.values())),
            'effective_chunk_source_derived':1024,'input_fixture_sha256':c.sha(root/'fixtures/boundaries.json')}
        c.write_json(marker,info)
        c.append(root/'control/inline-wide.jsonl',{'engine':key,'event':'boundary-alignment-complete','created':time.time()})


def prepare_wide(root):
    marker=root/'control/operator-wide-manifest.json'
    if marker.exists():raise ValueError('wide replay already frozen')
    command=[sys.executable,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root)]
    jobs=[]
    for key in c.ENGINES:
        directory=root/'runs'/key/'operators-wide-k0';directory.mkdir(parents=True,exist_ok=True)
        operators=root/'runs'/key/'operators-k0/operators'
        weights=sorted(operators.glob('*/weight.json'))
        if len(weights)!=21:raise ValueError('expected 21 frozen first/middle/last operator instances')
        c.write_json(directory/'jobs.json',[{'operator_dir':str(p.parent),'weight_metadata_sha256':c.sha(p)} for p in weights])
        jobs.append(command+['run-wide','--engine',key])
    c.write_json(marker,jobs)
    c.append(root/'control/protocol-amendments.jsonl',{'created':time.time(),
        'reason':'Actual scoring capture used 1024 columns. Add balanced wide replay of the same frozen activations after main-session restoration if interruption budget remains. Validate reconstruction against frozen widths 1 and 128 before width 1024. Selection is independent of errors and primary results.',
        'new_measurement_inputs':False,'primary_claims_changed':False})


def wide_completed(root,key):
    directory=root/'runs'/key/'operators-wide-k0'
    jobs=json.loads((directory/'jobs.json').read_text())
    records={r['operator']:r for r in c.rows(directory/'records.jsonl')} if (directory/'records.jsonl').exists() else {}
    completed={}
    for job in jobs:
        operator=Path(job['operator_dir']);name=operator.name;record=records.get(name,{})
        metadata=json.loads((operator/'weight.json').read_text())
        widths=record.get('wide_columns',[])
        if (record.get('narrow_1_and_128_exact') is True and isinstance(widths,list) and
            all(width in widths for width in [1024,2048]) and
            all((operator/f'width-{width}.bin').is_file() and
                (operator/f'width-{width}.bin').stat().st_size==metadata['output_rows']*width*metadata['output_element_bytes']
                for width in [1024,2048])):
            completed[name]=record
    return jobs,completed


def run_wide(root,key):
    info=json.loads((root/'probes'/key/'wide-build-provenance.json').read_text());binary=Path(info['binary'])
    if c.sha(binary)!=info['binary_sha256']:raise ValueError('wide probe changed')
    directory=root/'runs'/key/'operators-wide-k0';jobs=directory/'jobs.json'
    for job in json.loads(jobs.read_text()):
        if c.sha(Path(job['operator_dir'])/'weight.json')!=job['weight_metadata_sha256']:raise ValueError('frozen operator metadata changed')
    ordered,completed=wide_completed(root,key)
    pending=[job for job in ordered if Path(job['operator_dir']).name not in completed]
    if not pending:
        print('WIDE COMPLETE',key,len(completed),'qualified operators; reused without GPU work',flush=True)
        return
    import shutil
    attempts=directory/'attempts';attempts.mkdir(exist_ok=True)
    number=1
    while (attempts/f'attempt-{number:03d}').exists():number+=1
    attempt=attempts/f'attempt-{number:03d}';attempt.mkdir()
    if (directory/'records.jsonl').exists():shutil.copy2(directory/'records.jsonl',attempt/'previous-qualification-index.jsonl')
    preserved=[]
    for job in pending:
        operator=Path(job['operator_dir'])
        for width in [1024,2048]:
            path=operator/f'width-{width}.bin'
            if path.exists():
                destination=attempt/'preserved-unqualified'/operator.name/path.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(path),str(destination))
                preserved.append({'before':str(path),'preserved':str(destination),'bytes':destination.stat().st_size})
    c.write_json(attempt/'jobs.json',pending)
    c.write_json(attempt/'provenance.json',{'binary_sha256':info['binary_sha256'],
        'original_frozen_jobs_sha256':c.sha(jobs),'selection':'Only operators lacking a complete qualification record and complete output files; no numerical error or quality outcome selection.',
        'reused_qualified':list(completed),'preserved_unqualified_outputs':preserved})
    try:
        run_process([str(binary),str(attempt/'jobs.json'),str(attempt/'records.jsonl')],attempt)
    finally:
        # The canonical file is an index of completed qualifications. Every raw
        # attempt remains separate, including an interrupted final JSONL record.
        if (attempt/'records.jsonl').exists():
            for record in c.rows(attempt/'records.jsonl'):
                name=record.get('operator')
                if name not in {Path(job['operator_dir']).name for job in pending}:raise ValueError('unexpected wide qualification identity')
                completed[name]={**record,'source_attempt':str(attempt/'records.jsonl')}
        target=directory/'records.jsonl';temporary=directory/'records.jsonl.next'
        with temporary.open('w') as output:
            for job in ordered:
                record=completed.get(Path(job['operator_dir']).name)
                if record is not None:output.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        os.replace(temporary,target)
    _,qualified=wide_completed(root,key)
    if len(qualified)!=len(ordered):raise RuntimeError('wide replay ended without all required qualifications: '+key)


def _quarantine_resume(directory):
    """Keep incomplete attempt bytes intact before a complete fresh replay."""
    if not directory.exists():return None
    number=1
    while directory.with_name(directory.name+f'-before-resume-attempt-{number:03d}').exists():number+=1
    destination=directory.with_name(directory.name+f'-before-resume-attempt-{number:03d}')
    directory.rename(destination)
    return {'before':str(directory),'preserved':str(destination),'reason':'Incomplete collection; rerun the full frozen job, including both growing long-context requests.'}


def _resume_expected(root,argv):
    key=argv[argv.index('--engine')+1]
    if 'package' in argv:
        images='--images' in argv
        name='package-images' if images else 'package-native-k'+argv[argv.index('--k')+1]
        source='images' if images else 'native-subset'
        values=json.loads((root/'fixtures'/f'{source}.json').read_text())
        ids=[value['id'] if isinstance(value,dict) else value for value in values]
        return key,name,ids
    suite=argv[argv.index('--suite')+1];k=argv[argv.index('--k')+1]
    if suite in ['native','quality','mtp-quality']:
        source={'native':'native-subset','quality':'quality-order','mtp-quality':'mtp-subset'}[suite]
        ids=json.loads((root/'fixtures'/f'{source}.json').read_text())
    elif suite=='decode':ids=[f'decode-{depth}-{i}' for depth in [8192,32768,65536] for i in range(2)]
    elif suite=='observer-pilot':ids=['decode-2048-0']
    else:
        source={'operators':'repeatability','repeatability':'repeatability','likelihood':'score-windows','boundary':'boundaries','long':'long-jobs'}[suite]
        rows=json.loads((root/'fixtures'/f'{source}.json').read_text())
        ids=[row['id'] for row in (rows[:1] if suite=='operators' else rows)]
    return key,f'{suite}-k{k}',ids


def _resume_measurement(root,argv):
    key,name,expected=_resume_expected(root,argv)
    directory=root/'runs'/key/name
    # The frozen main boundary job measured native alignment. Once corrected
    # data occupy the canonical path, its original attempt lives in the archive.
    if name=='boundary-k0' and (directory.with_name('boundary-native-aligned-k0')).exists():
        directory=directory.with_name('boundary-native-aligned-k0')
    rows=c.rows(directory/'records.jsonl') if (directory/'records.jsonl').exists() else []
    ids=[row.get('id') for row in rows]
    complete=len(ids)==len(expected) and len(set(ids))==len(ids) and set(ids)==set(expected)
    missing=sorted(set(expected)-set(ids))
    if complete and name=='package-images':
        required={row['id'] for row in rows if row['id'].endswith('-0') and row.get('status')==200}
        follows=c.rows(directory/'cache-continuations.jsonl') if (directory/'cache-continuations.jsonl').exists() else []
        complete=required<={row['id'] for row in follows}
        if not complete:missing+=['cache-continuation:'+cid for cid in sorted(required-{row['id'] for row in follows})]
    return {'engine':key,'suite':name,'directory':str(directory),'completed':complete,
            'records':len(rows),'required_records':len(expected),'missing_ids':missing,
            'recorded_errors':sum('error' in row or row.get('status',200)!=200 for row in rows),
            'selection':'Collection completeness only; correctness, acceptance and numerical-error outcomes are not used.'}


def _boundary_correction_complete(root,key):
    directory=root/'runs'/key/'boundary-k0'
    if not (root/'runs'/key/'boundary-native-aligned-k0').exists():return False
    if not (directory/'settings.json').exists() or not (directory/'records.jsonl').exists():return False
    metadata=json.loads((directory/'settings.json').read_text())
    rows=c.rows(directory/'records.jsonl')
    expected={row['id']:row['width'] for row in json.loads((root/'fixtures/boundaries.json').read_text())}
    return (metadata.get('prefill_align_env')=='0' and len(rows)==len(expected) and {row['id'] for row in rows}==set(expected)
            and all('error' not in row and len(row.get('logprobs',[]))==expected[row['id']] for row in rows))


def _recover_forced_validation(directory,k):
    """Finish interrupted CPU validation without regenerating complete GPU data."""
    path=directory/'control-validation.json'
    if path.exists():return
    jobs=c.rows(directory/'jobs.jsonl')
    if not jobs or not jobs[0].get('forced_tokens'):raise ValueError('missing frozen forced history for CPU recovery')
    expected=jobs[0]['forced_tokens'][:129];violations=[]
    for record in c.rows(directory/'records.jsonl'):
        if record.get('tokens')!=expected:violations.append({'id':record['id'],'reason':'output does not match forced 129-token history'})
    for job in jobs:
        state_file=Path(job['observer_dir'])/'states.jsonl'
        for state in c.rows(state_file) if state_file.exists() else []:
            if state['ledger_frontier']!=len(job['tokens'])+state['script_cursor'] or state['execution_frontier']+1!=state['ledger_frontier']:
                violations.append({'id':job['id'],'reason':'script/ledger/execution frontier mismatch','state':state})
        proposal_file=Path(job['observer_dir'])/'proposal-inputs.jsonl'
        if k and proposal_file.exists():
            for proposal in c.rows(proposal_file):
                if proposal['width']!=k or proposal['extent']>k:
                    violations.append({'id':job['id'],'reason':'actual verification width differs from requested K','proposal':proposal})
    c.write_json(path,{'passed':not violations,'violations':violations,'recovered_by_cpu_only_continuation':True})


def prepare_resume(root):
    """Prepare continuation only after every existing session has restored."""
    for name in ['budget.json','resume-budget.json']:
        path=root/'control'/name
        if path.exists():
            ledger=json.loads(path.read_text())
            if any(row.get('finished') is None or not row.get('restored') for row in ledger['sessions']):
                raise ValueError('Cannot prepare continuation while a session is unfinished or unrestored: '+str(path))
    manifest=root/'control/main-manifest.json';original=json.loads(manifest.read_text())
    selected=[];coverage=[];relocations=[]
    for argv in original:
        status=_resume_measurement(root,argv);coverage.append(status)
        if status['completed']:
            if status['suite'].startswith(('decode-k','observer-pilot-k')):
                _recover_forced_validation(Path(status['directory']),int(status['suite'].rsplit('-k',1)[1]))
            continue
        if status['suite']=='operators-k0' and status['engine']=='v100':
            raise ValueError('Incomplete canonical activation capture requires explicit review; continuation will not replace frozen operator fixtures')
        relocation=_quarantine_resume(Path(status['directory']))
        if relocation:relocations.append(relocation)
        selected.append(argv)
    post_needed=False
    for key in ['vendor','r5']:
        if not _boundary_correction_complete(root,key):
            post_needed=True
            directory=root/'runs'/key/'boundary-k0'
            if directory.with_name('boundary-native-aligned-k0').exists() and directory.exists():
                relocation=_quarantine_resume(directory)
                if relocation:relocations.append(relocation)
    for key in c.ENGINES:
        jobs,qualified=wide_completed(root,key)
        post_needed|=len(qualified)!=len(jobs)
    if post_needed:selected.append([sys.executable,'-u',str(c.ROOT/'bench/ninfer_precision.py'),'--root',str(root),'resume-controls'])
    number=1
    while (root/'control'/f'resume-manifest-{number:03d}.json').exists():number+=1
    target=root/'control'/f'resume-manifest-{number:03d}.json'
    c.write_json(target,selected);c.write_json(root/'control/resume-manifest.json',selected)
    plan={'created':time.time(),'manifest':str(target),'original_main_manifest_sha256':c.sha(manifest),
        'pending_main_jobs':sum('resume-controls' not in argv for argv in selected),'pending_controls':post_needed,
        'frozen_job_order_preserved':True,'coverage':coverage,'preserved_incomplete_attempts':relocations,
        'ledger':str(root/'control/resume-budget.json'),'budget_seconds':7200,'restoration_reserve_seconds':600,
        'reason':'User prioritizes full laboratory coverage over interruption duration; continue missing frozen measurements without resetting or modifying the original six-hour ledger.'}
    c.write_json(root/'control'/f'resume-plan-{number:03d}.json',plan)
    c.write_json(root/'control/resume-plan.json',plan)
    c.append(root/'control/protocol-amendments.jsonl',plan)
    print('RESUME PREPARED',len(selected),'sequential jobs;',len(relocations),'incomplete attempts preserved;',target,flush=True)
    return plan


def resume_controls(root):
    correct_boundary_alignment(root)
    for key in c.ENGINES:run_wide(root,key)


def session(root,phase):
    manifest_file=root/'control'/f'{phase}-manifest.json'
    if phase=='resume':
        plan=json.loads((root/'control/resume-plan.json').read_text());manifest_file=Path(plan['manifest'])
    if not manifest_file.exists(): raise ValueError('prepare manifests first')
    if not json.loads(manifest_file.read_text()):
        print('No missing GPU jobs; service left running',flush=True);return
    ledger=root/'control'/('resume-budget.json' if phase=='resume' else 'budget.json')
    command=[sys.executable,'-u',str(c.ROOT/'bench/ninfer_session.py'),'--program-manifest',str(manifest_file),
             '--budget-ledger',str(ledger),'--budget-seconds','7200' if phase=='resume' else '21600',
             '--reserve-seconds','600' if phase=='resume' else '1800',
             '--restore-timeout','600']
    # The stack's flow-style preload entry is deliberately kept as a simple
    # stdlib parse; reject ambiguity rather than silently restoring another model.
    import re
    match=re.search(r'^    preload: \[([^\],]+)\]\s*$',(c.ROOT/'llama-swap.yaml').read_text(),re.M)
    if not match:raise ValueError('cannot identify configured single startup preload')
    command+=['--ready-model',match[1].strip().strip("\"'")]
    if phase.startswith('pilot'): command+=['--deadline',str(time.time()+1200)]
    try:
        subprocess.run(command,cwd=c.ROOT,check=True)
    finally:
        status,models=tune.http_json(8082,'GET','/v1/models',timeout=30)
        restoration_name=f'restored-{manifest_file.stem}.json' if phase=='resume' else f'restored-{phase}.json'
        c.write_json(root/'control'/restoration_name,{'created':time.time(),'http_status':status,'models':models,
            'gpu':tune.gpu_state(),'active':os.readlink(c.ENGINE/'runtime/active'),
            'config_sha256':c.sha(c.ROOT/'llama-swap.yaml')})
