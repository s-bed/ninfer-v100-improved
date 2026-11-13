#!/usr/bin/env python3.11
"""Small native-cache agentic performance comparison; preparation is CPU-only.

prepare -> session -> analyze. GPU runs are refused until precision finalization.
No production config, selected bundle, production cache or source is modified.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import struct
import subprocess
import sys
import time
import zlib

import ninfer_tune as tune

REPO=Path(__file__).resolve().parent.parent
ENGINE=Path('/models/engine/iamwavecut-ninfer-all')
DONOR=Path('/models/engine/geoffwatts-ninfer-v100')
PRECISION=ENGINE/'bench-runs/precision-20261030'
DEFAULT_ROOT=PRECISION/'agentic'
ENGINES={
    'v100':{'revision':'b37d0dd3e1163b9d802d8bccfa89918bf68d793e',
            'source':DONOR/'source/standard','bundle':DONOR/'runtime/standard',
            'model':'ninfer/qwen3.8-27b','runtime':'original'},
    'vendor':{'revision':'33163f05fb54459c19e2574607f5d252f4be2ba4',
              'source':ENGINE/'source/vendor-33163f05fb54','bundle':ENGINE/'runtime/vendor-33163f05fb54/current',
              'model':'ninfer-all/qwen3.8-27b-hybrid','runtime':'all'},
    'r5':{'revision':'ec94ca8d97aebcae2e5429b83925b2c08252143a',
          'source':ENGINE/'source/ported','bundle':ENGINE/'runtime/ported-33163f0-r5',
          'model':'ninfer-all/qwen3.8-27b-hybrid','runtime':'all'},
}
MEASURED=['new-lineage','identical','append-tool','followup','late-result-edit',
          'tools-reordered','tool-added','tool-schema-changed','vision-turn','vision-followup']
CHECKS=['imports','unit','schema','lint','cache','vision']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.next')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def rows(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def model_command(config,model):
    """Read only the named folded command from this stack's simple YAML shape."""
    lines=config.splitlines()
    start=next((i for i,line in enumerate(lines) if re.fullmatch(r'  [\"\']?'+re.escape(model)+r'[\"\']?:\s*',line)),None)
    if start is None:raise ValueError('model entry missing: '+model)
    end=next((i for i in range(start+1,len(lines)) if re.match(r'^  [^ #\s].*:\s*$',lines[i])),len(lines))
    cmd=next((i for i in range(start+1,end) if re.fullmatch(r'    cmd: >\s*',lines[i])),None)
    if cmd is None:raise ValueError('folded command missing: '+model)
    body=[]
    for line in lines[cmd+1:end]:
        if line and not line.startswith('      '):break
        if line.strip() and not line.lstrip().startswith('#'):body.append(line.strip())
    tokens=shlex.split(' '.join(body))
    if len(tokens)<3 or not tokens[0].startswith('${ninfer'):raise ValueError('unexpected engine command')
    return tokens


def source_flags(config,key):
    tokens=model_command(config,ENGINES[key]['model'])
    artifact=tokens[1];flags=tune.parse_flags(tokens[2:])
    for name in ['--host','--port','--request-log-jsonl']:flags.pop(name,None)
    if key=='vendor':flags.pop('--v100-kernels',None)
    return artifact,flags


def make_tools():
    tools=[]
    names=['read_file','search_code','run_tests','write_file','list_files','inspect_image','read_docs','git_diff']
    for name in names:
        tools.append({'type':'function','function':{'name':name,
            'description':f'Workspace {name} operation. Results are returned as text; preserve call identifiers and exact arguments.',
            'parameters':{'type':'object','properties':{'path':{'type':'string'},'query':{'type':'string'}},'required':['path']}}})
    return tools


def call(identifier,name,arguments):
    return {'role':'assistant','content':None,'tool_calls':[{'id':identifier,'type':'function',
        'function':{'name':name,'arguments':json.dumps(arguments,separators=(',',':'))}}]}


def question(vision=False):
    text=('Use the latest recorded run_tests AgenticResult. Return one JSON object only, without Markdown or explanation, '
          'with keys result (the exact AgenticResult string), status ("PASS"), checks '
          '(the array ["imports","unit","schema","lint","cache","vision"]), and next ("review"). '
          'All evidence is already supplied: do not call tools.')
    if vision:text+=' Also include color: the dominant image color, as one lowercase English word.'
    return text


def expected(repeat,suffix='A',color=None):
    result={'result':f'lab-{repeat:02d}-{suffix}','status':'PASS','checks':CHECKS,'next':'review'}
    if color:result['color']=color
    return result


def repository_text(repeat):
    # Frozen synthetic code, no private repository contents or model downloads.
    parts=['Repository evidence snapshot: src/packet_rules.py\n']
    for i in range(100):
        parts.append(f'def rule_{i:03d}(packet):\n'
            f'    """Validate packet group {i} for revision {repeat}."""\n'
            f'    total = packet.get("base", {i+repeat})\n'
            '    values = packet.get("values", [])\n'
            '    return total + sum(int(value) for value in values if value is not None)\n\n')
    return ''.join(parts)


def payload(messages,tools):
    return {'model':'qwen3.8-27b','messages':messages,'tools':tools,'tool_choice':'auto',
        'temperature':0,'top_k':0,'top_p':1,'min_p':0,'presence_penalty':0,'frequency_penalty':0,
        'seed':1729,'max_tokens':128,'stream':True,'stream_options':{'include_usage':True},
        'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False},'return_progress':True}


def png(width,height,rgb):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=(b'\0'+bytes(rgb)*width)*height
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+
            chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def fixtures(root):
    cases=[]
    for repeat,(color,rgb) in enumerate([('blue',(12,25,235)),('green',(15,200,25)),('red',(230,15,20))]):
        image=f'fixtures/image-{repeat}.png';(root/image).write_bytes(png(512,512,rgb))
        messages=[{'role':'system','content':f'Agentic lab lineage {repeat:02d}. You are reviewing recorded repository evidence. Use only supplied results.'},
            {'role':'user','content':'Inspect src/packet_rules.py and the unit-test transcript. The tool calls below have already executed.'},
            call(f'call_read_{repeat}','read_file',{'path':'src/packet_rules.py'}),
            {'role':'tool','tool_call_id':f'call_read_{repeat}','content':repository_text(repeat)},
            call(f'call_test_{repeat}','run_tests',{'path':'tests/test_packet_rules.py'}),
            {'role':'tool','tool_call_id':f'call_test_{repeat}','content':f'AgenticResult: lab-{repeat:02d}-A\nAll recorded checks PASS. No additional tool invocation is needed.'},
            {'role':'user','content':question()}]
        base=payload(messages,make_tools());mutations={}
        late=copy.deepcopy(base);late['messages'][-2]['content']=f'AgenticResult: lab-{repeat:02d}-C\nAll recorded checks PASS. This replaces the earlier result.'
        mutations['late-result-edit']=late
        reorder=copy.deepcopy(base);reorder['tools'].reverse();mutations['tools-reordered']=reorder
        added=copy.deepcopy(base);added['tools'].append({'type':'function','function':{'name':'git_status',
            'description':'Report repository status without changing files.',
            'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}})
        mutations['tool-added']=added
        changed=copy.deepcopy(base);changed['tools'][0]['function']['parameters']['properties']['include_context']={'type':'boolean','description':'Include surrounding source lines when requested.'}
        mutations['tool-schema-changed']=changed
        cases.append({'repeat':repeat,'base':base,'mutations':mutations,'image':image,'color':color,
            'expected_base':expected(repeat),'expected_append':expected(repeat,'B'),
            'expected_late':expected(repeat,'C'),'expected_vision':expected(repeat,color=color)})
    return cases


def prepare(root):
    if (root/'fixtures/freeze.json').exists():raise ValueError('already frozen; choose a new root for a new run')
    (root/'fixtures').mkdir(parents=True,exist_ok=True)
    config=(REPO/'llama-swap.yaml').read_text();settings={}
    for key,engine in ENGINES.items():
        revision=subprocess.check_output(['git','-C',str(engine['source']),'rev-parse','HEAD'],text=True).strip()
        if revision!=engine['revision']:raise ValueError('unexpected source version: '+key)
        artifact,flags=source_flags(config,key)
        profiles={}
        for flag in ['--context-cost-presets','--device-profile-path']:
            if flag in flags:
                original=Path(flags[flag]);destination=root/'fixtures'/(key+'-'+original.name)
                destination.write_bytes(original.read_bytes());profiles[flag]={'source':str(original),'sha256':sha(original)}
                flags[flag]=str(destination)
        flags.pop('--prefix-cache-file',None)
        settings[key]={'revision':revision,'launcher':str(engine['bundle']/'ninfer-serve-run'),
            'launcher_sha256':sha(engine['bundle']/'ninfer-serve-run'),'artifact':artifact,'flags':flags,
            'binary':str(engine['bundle']/'ninfer-serve'),'binary_sha256':sha(engine['bundle']/'ninfer-serve'),
            'backend_sha256':{str(p):sha(p) for p in sorted((engine['bundle']/'backends').glob('**/*.so*')) if p.is_file()},
            'profiles':profiles,'native_command':model_command(config,engine['model'])}
    cases=fixtures(root);write(root/'fixtures/cases.json',cases);write(root/'fixtures/settings.json',settings)
    (root/'provenance').mkdir(exist_ok=True)
    (root/'provenance/ninfer_agentic.py').write_bytes(Path(__file__).read_bytes())
    write(root/'fixtures/protocol.json',{'repetitions':3,'measured_stages':MEASURED,'max_tokens':128,
        'seed':1729,'greedy':True,'tool_choice':'auto','actual_tool_execution':False,
        'depth':'Approximately 8K logical-history tokens; report actual native rendered token counts, not a fixed equal-token claim.',
        'branch_control':'Re-prime the same frozen ancestor before each mutation and vision branch. Setup re-primes are excluded from headline stage medians.',
        'live_continuations':'Append actual assistant content/reasoning/call IDs and scripted new tool results; runtime payloads may differ across engines and are saved.',
        'cold_definition':'New lineage changes the leading system nonce. Report actual cache_n; shared template/tool-prefix hits may remain.',
        'correctness':'Parse the requested JSON (allow a whole-response JSON fence), compare exact typed content and reject unexpected tool calls. No model judge.',
        'comparison_scope':'Native configurations and templates, same frozen logical requests. Descriptive medians/ranges of n=3; no formal equivalence or pure kernel-speed attribution.',
        'native_cache_difference':'Original separate HostKV16GiB plus 8 Host state slots and 3 extra Device slots vs vendor/R5 unified16GiB Host and 2 Device snapshots. Deliberate native configuration difference.',
        'prefill_alignment':'Preserve native NInfer-all default prefill alignment; do not set NINFER_PREFILL_ALIGN=0. On this 72SM/24-query-head/int8/default slow-prompt-kernel geometry, original requested2048 stays2048; vendor/R5 requested4096 becomes4224 by source calculation. Final tails can be narrower. Preserve progress samples and startup/request logs; source-derived width is not a measured launch count.',
        'vendor_adaptation':'Same llama-swap all settings and byte-identical calibration as R5, omitting only unsupported --v100-kernels and using its pinned artifact/bundle.',
        'target_gpu_minutes':[15,25],'must_run_after_precision_finalization':True})
    files={str(p.relative_to(root)):sha(p) for p in sorted((root/'fixtures').glob('*')) if p.is_file()}
    write(root/'fixtures/freeze.json',{'created':time.time(),'config_sha256':sha(REPO/'llama-swap.yaml'),
        'files':files,'harness_sha256':sha(Path(__file__)),'active':os.readlink(ENGINE/'runtime/active')})
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root)]
    write(root/'control/manifest.json',[command+['run','--engine',key] for key in ENGINES])
    print('Prepared frozen agentic comparison:',root,flush=True)


def verify(root):
    freeze=json.loads((root/'fixtures/freeze.json').read_text())
    for relative,digest in freeze['files'].items():
        if sha(root/relative)!=digest:raise ValueError('fixture changed: '+relative)
    config=(REPO/'llama-swap.yaml').read_text();settings=json.loads((root/'fixtures/settings.json').read_text())
    for key,setting in settings.items():
        if model_command(config,ENGINES[key]['model'])!=setting['native_command']:
            raise ValueError('selected llama-swap command changed after freezing: '+key)
        if sha(setting['launcher'])!=setting['launcher_sha256']:raise ValueError('serving launcher changed: '+key)
        if sha(setting['binary'])!=setting['binary_sha256']:raise ValueError('serving binary changed: '+key)
        for path,digest in setting['backend_sha256'].items():
            if sha(path)!=digest:raise ValueError('serving backend changed: '+path)
        for profile in setting['profiles'].values():
            if sha(profile['source'])!=profile['sha256']:raise ValueError('native profile changed after freezing: '+profile['source'])
    if os.readlink(ENGINE/'runtime/active')!=freeze['active']:raise ValueError('active selection changed')
    return settings


def precision_done():
    for name in ['acceptance','states','distributions','operators']:
        if not (PRECISION/f'analysis/{name}-complete.json').exists():raise ValueError('finish precision finalization first: '+name)
    report=PRECISION/'NINFER_PRECISION.txt'
    if not report.exists() or 'IN PROGRESS' in report.read_text():raise ValueError('precision report is not finalized')
    ledger=json.loads((PRECISION/'control/budget.json').read_text())
    if any(s.get('finished') is None or not s.get('restored') for s in ledger['sessions']):
        raise ValueError('precision GPU session is not restored')


def typed_equal(a,b):
    return type(a) is type(b) and (a.keys()==b.keys() and all(typed_equal(a[k],b[k]) for k in a) if isinstance(a,dict)
        else len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b)) if isinstance(a,list) else a==b)


def grade(result,want):
    if not result.ok:return {'correct':False,'reason':'HTTP/stream error'}
    if result.calls():return {'correct':False,'reason':'unexpected tool call'}
    text=result.text.strip()
    match=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.I|re.S)
    if match:text=match[1]
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('duplicate key')
            value[key]=item
        return value
    try:return {'correct':typed_equal(json.loads(text,object_pairs_hook=unique),want),'reason':'exact typed JSON content'}
    except (ValueError,TypeError):return {'correct':False,'reason':'invalid or duplicate-key JSON'}


def append_assistant(messages,result,tools=None):
    calls=result.calls();identifiers=[item['id'] for item in calls]
    if len(set(identifiers))!=len(identifiers) or any(not identifier for identifier in identifiers):return False
    names={tool['function']['name'] for tool in tools} if tools is not None else None
    for item in calls:
        if names is not None and item['function']['name'] not in names:return False
        try:
            def invalid_constant(value):raise ValueError('non-JSON number')
            arguments=json.loads(item['function']['arguments'],parse_constant=invalid_constant)
            if not isinstance(arguments,dict):return False
        except (ValueError,TypeError):return False
    message={'role':'assistant','content':result.text or None}
    if result.thinking:message['reasoning_content']=result.thinking
    if calls:message['tool_calls']=calls
    elif not result.text:message['content']=''
    messages.append(message)
    for item in calls:
        messages.append({'role':'tool','tool_call_id':item['id'],'content':'No external tool was executed. Use only the already recorded evidence.'})
    return True


def run(root,key):
    precision_done();settings=verify(root);setting=settings[key];directory=root/'runs'/key
    if not os.environ.get('NINFER_SESSION_SERVER_GROUPS'):raise ValueError('run only under ninfer_session.py restoration guard')
    if os.environ.get('NINFER_PREFILL_ALIGN','1').startswith('0'):raise ValueError('native comparison must not inherit precision-only NINFER_PREFILL_ALIGN=0')
    if abs(tune.gpu_state()['power_limit_w']-150)>.5:raise ValueError('expected the unchanged 150W power limit')
    if (directory/'records.jsonl').exists():raise ValueError('engine already has records; preserve them and use a new root')
    directory.mkdir(parents=True,exist_ok=True);(directory/'payloads').mkdir(exist_ok=True)
    flags=copy.deepcopy(setting['flags'])
    if key!='v100':
        cache=directory/'private-prefix.cache'
        if cache.exists():raise ValueError('private cache already exists; new run must start without prior persistence')
        flags['--prefix-cache-file']=str(cache)
    args=argparse.Namespace(runtime=ENGINES[key]['runtime'],launcher=setting['launcher'],artifact=setting['artifact'],
        cache_dir=str(root/'runs'),shutdown_timeout=150)
    server=tune.Server(directory,flags,args);monitor=tune.GpuMonitor();monitor.start()
    write(directory/'native-options.json',{'flags':flags,'source':setting['native_command'],'profiles':setting['profiles']})
    started=time.monotonic()
    try:
        server.start()
        ready=server.wait('ready',300);write(directory/'ready.json',ready)
        if not ready['ok']:raise RuntimeError('engine startup failed; see ready.json and server.log')
        log=tune.JsonlTail(directory/'server.jsonl');number=0
        def request(spec,repeat,stage,want,setup=False):
            nonlocal number
            name=f'{number:03d}-r{repeat}-{stage}';number+=1
            write(directory/'payloads'/f'{name}.json',spec);log.skip_to_end()
            progress=[]
            result=tune.chat_stream(server.port,spec,on_progress=lambda item:progress.append(item))
            record={'engine':key,'repeat':repeat,'stage':stage,'setup':setup,
                'payload':f'payloads/{name}.json','payload_sha256':sha(directory/'payloads'/f'{name}.json'),
                'status':result.status,'error':result.error,'content':result.text,'reasoning':result.thinking,
                'prompt_progress':progress,
                'tool_calls':result.calls(),'usage':result.usage,'timings':result.timings,
                'client_ttft_s':result.ttft,'wall_s':result.t_end-result.t0,**tune.metrics(result)}
            done=log.next_done(timeout=20) if result.ok else None
            if done:
                record['logged_metrics']=tune.logged_metrics(done)
                record.update({k:v for k,v in record['logged_metrics'].items() if v is not None})
                record['prefix_reuse_path']=tune.find_key(done,'prefix_reuse_path')
            prompt=record.get('prompt_tokens');cached=record.get('cache_n')
            if cached is None:cached=result.usage.get('prompt_tokens_details',{}).get('cached_tokens');record['cache_n']=cached
            record['cache_fraction']=cached/prompt if prompt and cached is not None else None
            record['actual_new_lineage_cache_fraction']=record['cache_fraction'] if stage=='new-lineage' else None
            record['grade']=grade(result,want);append(directory/'records.jsonl',record)
            print('AGENTIC',key,f'r{repeat}',stage,'cache',cached,'/',prompt,'ttft',result.ttft,'correct',record['grade']['correct'],flush=True)
            if not result.ok:raise RuntimeError('request failed: '+name)
            return result
        def skipped(repeat,stage):
            append(directory/'records.jsonl',{'engine':key,'repeat':repeat,'stage':stage,'setup':False,
                'skipped_invalid_history':True,'status':None,'error':'Prior assistant call arguments/name/IDs could not form a valid continuation.',
                'payload_sha256':None,'wall_s':None,'client_ttft_s':None,'cache_fraction':None,
                'grade':{'correct':False,'reason':'invalid dependency; no request sent'}})
            print('AGENTIC',key,f'r{repeat}',stage,'SKIPPED invalid prior tool call',flush=True)
        for case in json.loads((root/'fixtures/cases.json').read_text()):
            repeat=case['repeat'];base=case['base'];want=case['expected_base']
            request(base,repeat,'new-lineage',want)
            ancestor=request(base,repeat,'identical',want)
            growing=copy.deepcopy(base)
            if append_assistant(growing['messages'],ancestor,growing['tools']):
                growing['messages']+=[{'role':'user','content':'The repository changed. Here is the recorded test rerun.'},
                    call(f'call_append_{repeat}','run_tests',{'path':'tests/test_packet_rules.py'}),
                    {'role':'tool','tool_call_id':f'call_append_{repeat}','content':f'AgenticResult: lab-{repeat:02d}-B\nAll recorded checks PASS.'},
                    {'role':'user','content':question()}]
                result=request(growing,repeat,'append-tool',case['expected_append'])
                follow=copy.deepcopy(growing)
                if append_assistant(follow['messages'],result,follow['tools']):
                    follow['messages'].append({'role':'user','content':'Repeat the same final JSON, preserving the latest recorded result. Do not call tools.'})
                    request(follow,repeat,'followup',case['expected_append'])
                else:skipped(repeat,'followup')
            else:
                skipped(repeat,'append-tool');skipped(repeat,'followup')
            for stage,spec in case['mutations'].items():
                request(base,repeat,'reprime-for-'+stage,want,setup=True)
                request(spec,repeat,stage,case['expected_late'] if stage=='late-result-edit' else want)
            ancestor=request(base,repeat,'reprime-for-vision',want,setup=True)
            vision=copy.deepcopy(base)
            if not append_assistant(vision['messages'],ancestor,vision['tools']):
                skipped(repeat,'vision-turn');skipped(repeat,'vision-followup');continue
            uri='data:image/png;base64,'+base64.b64encode((root/case['image']).read_bytes()).decode()
            vision['messages'].append({'role':'user','content':[{'type':'text','text':question(True)},
                {'type':'image_url','image_url':{'url':uri}}]})
            result=request(vision,repeat,'vision-turn',case['expected_vision'])
            continuation=copy.deepcopy(vision)
            if append_assistant(continuation['messages'],result,continuation['tools']):
                continuation['messages'].append({'role':'user','content':'Repeat that same JSON, including the same image color. Do not call tools.'})
                request(continuation,repeat,'vision-followup',case['expected_vision'])
            else:skipped(repeat,'vision-followup')
    finally:
        try:server.stop()
        finally:
            monitor.stop();write(directory/'shutdown.json',server.shutdown);write(directory/'gpu.json',monitor.samples)
            write(directory/'timing.json',{'seconds':time.monotonic()-started,'gpu':monitor.stats(started,time.monotonic())})
            write(directory/'prefill-alignment.json',{'requested_prefill_chunk':flags.get('--prefill-chunk'),
                'native_effective_chunk_source_derived':2048 if key=='v100' else 4224,
                'environment_NINFER_PREFILL_ALIGN':os.environ.get('NINFER_PREFILL_ALIGN'),
                'note':'Unset keeps native defaults. Effective width derives from pinned startup/causal_attention source, 72SM/24 query heads/int8/fast_prompt=false. Progress samples may corroborate processed-token increments; requested-chunk log fields are not launch measurements.',
                'startup_lines':[line for line in ((directory/'server.log').read_text(errors='replace') if (directory/'server.log').exists() else '').splitlines()
                    if re.search('prefill|chunk|capacity',line,re.I)][:40]})


def session(root):
    precision_done();verify(root)
    config=(REPO/'llama-swap.yaml').read_text()
    match=re.search(r'^    preload: \[([^\],]+)\]\s*$',config,re.M)
    if not match:raise ValueError('cannot identify single configured preload')
    command=[sys.executable,'-u',str(REPO/'bench/ninfer_session.py'),'--program-manifest',str(root/'control/manifest.json'),
        '--budget-ledger',str(root/'control/budget.json'),'--budget-seconds','3600','--reserve-seconds','600',
        '--ready-model',match[1].strip().strip('"\''),'--restore-timeout','600']
    try:subprocess.run(command,cwd=REPO,check=True)
    finally:
        evidence={'created':time.time(),'active':os.readlink(ENGINE/'runtime/active'),'config_sha256':sha(REPO/'llama-swap.yaml')}
        try:
            status,models=tune.http_json(8082,'GET','/v1/models',timeout=30)
            evidence.update(http_status=status,models=models,gpu=tune.gpu_state())
        except Exception as error:evidence['error']=repr(error)
        write(root/'control/restored.json',evidence)
    analyze(root)


def stats(values):
    clean=[x for x in values if x is not None and math.isfinite(x)]
    return {'n':len(clean),'median':statistics.median(clean),'min':min(clean),'max':max(clean)} if clean else {'n':0}


def analyze(root):
    result={'engines':{},'comparisons':{},'scope':json.loads((root/'fixtures/protocol.json').read_text())}
    records={key:rows(root/'runs'/key/'records.jsonl') for key in ENGINES}
    for key,data in records.items():
        stages={}
        for stage in MEASURED:
            sample=[r for r in data if r['stage']==stage and not r['setup']]
            stages[stage]={'n':len(sample),'measured_n':sum(not r.get('skipped_invalid_history',False) for r in sample),
                'correct':sum(r['grade']['correct'] for r in sample),
                **{metric:stats([r.get(metric) for r in sample]) for metric in ['client_ttft_s','engine_ttft_s','wall_s',
                    'prompt_tokens','cache_n','cache_fraction','prompt_n','prefill_tps','decode_tps','completion_tokens']}}
        rendering=[]
        for repeat in range(3):
            bystage={r['stage']:r for r in data if r['repeat']==repeat and not r['setup']}
            base=bystage.get('identical',{}).get('prompt_tokens')
            for stage in ['tool-added','tool-schema-changed']:
                value=bystage.get(stage,{}).get('prompt_tokens')
                rendering.append({'repeat':repeat,'stage':stage,'prompt_token_delta':value-base if value is not None and base is not None else None,
                    'observed_nonzero_delta':value is not None and base is not None and value!=base})
        result['engines'][key]={'stages':stages,'tool_render_checks':rendering,
            'correctness_failures':[r for r in data if not r['grade']['correct']],
            'shutdown':json.loads((root/'runs'/key/'shutdown.json').read_text()) if (root/'runs'/key/'shutdown.json').exists() else None}
    for left,right in [('r5','v100'),('r5','vendor'),('vendor','v100')]:
        paired={}
        for stage in MEASURED:
            a={r['repeat']:r for r in records[left] if r['stage']==stage and not r['setup']}
            b={r['repeat']:r for r in records[right] if r['stage']==stage and not r['setup']}
            pairs=[]
            for repeat in sorted(a.keys()&b.keys()):
                x,y=a[repeat],b[repeat]
                if x.get('skipped_invalid_history') or y.get('skipped_invalid_history'):continue
                pairs.append({'repeat':repeat,'identical_payload':x['payload_sha256']==y['payload_sha256'],
                    'left_correct':x['grade']['correct'],'right_correct':y['grade']['correct'],
                    'left_prompt_tokens':x.get('prompt_tokens'),'right_prompt_tokens':y.get('prompt_tokens'),
                    'wall_ratio':x['wall_s']/y['wall_s'] if y.get('wall_s') else None,
                    'ttft_ratio':x['client_ttft_s']/y['client_ttft_s'] if x.get('client_ttft_s') is not None and y.get('client_ttft_s') else None,
                    'cache_fraction_difference':x['cache_fraction']-y['cache_fraction'] if x.get('cache_fraction') is not None and y.get('cache_fraction') is not None else None})
            both_correct=[p for p in pairs if p['left_correct'] is True and p['right_correct'] is True]
            paired[stage]={'paired_n':len(pairs),'pairs':pairs,
                'wall_ratio':stats([p['wall_ratio'] for p in pairs]),'ttft_ratio':stats([p['ttft_ratio'] for p in pairs]),
                'both_correct':{'paired_n':len(both_correct),'pairs':both_correct,
                    'wall_ratio':stats([p['wall_ratio'] for p in both_correct]),
                    'ttft_ratio':stats([p['ttft_ratio'] for p in both_correct])}}
        result['comparisons'][left+'_vs_'+right]=paired
    write(root/'analysis/summary.json',result)
    lines=['Small native-cache agentic performance comparison','Evidence: '+str(root),'',
        'Three frozen logical lineages; 10 measured stages and 5 equal-ancestor setup re-primes per lineage/engine.',
        'Native templates/settings intentionally differ. Actual prompt tokens are reported; live continuation payloads preserve actual assistant history and can differ across engines.',
        'Requests go directly to isolated loopback engine servers using the frozen llama-swap native options. This measures engine/request/cache behavior, not llama-swap proxy overhead or model-switch latency.',
        'Descriptive medians and min/max ranges, n=3 per stage. No formal equivalence or pure kernel-performance claim.',
        'TTFT is first visible streaming content/tool/reasoning event; end-to-end is measured at the client. Decode rates use actual completion counts and short outputs; interpret cautiously.','',
        'Engine | Stage | n | Correct | Prompt tokens median | Cache % median | TTFT median [min,max] s | End-to-end median [min,max] s | Output tokens median | Decode t/s median']
    def fmt(value):
        return f"{value['median']:.3f} [{value['min']:.3f},{value['max']:.3f}]" if value['n'] else 'unavailable'
    for key,engine in result['engines'].items():
        for stage,record in engine['stages'].items():
            prompt=record['prompt_tokens'];cache=record['cache_fraction'];out=record['completion_tokens'];rate=record['decode_tps']
            lines.append(f"{key} | {stage} | {record['measured_n']}/{record['n']} | {record['correct']} | {prompt.get('median')} | {100*cache['median'] if cache['n'] else None} | {fmt(record['client_ttft_s'])} | {fmt(record['wall_s'])} | {out.get('median')} | {rate.get('median')}")
    lines+=['','Paired R5/baseline TTFT and end-to-end ratios (less than1 is faster):']
    for name,comparison in result['comparisons'].items():
        for stage,record in comparison.items():
            qualified=record['both_correct']
            lines.append(f"{name} {stage}: all paired n={record['paired_n']}, TTFT n={record['ttft_ratio']['n']} {fmt(record['ttft_ratio'])}, end-to-end n={record['wall_ratio']['n']} {fmt(record['wall_ratio'])}; both-correct secondary subset n={qualified['paired_n']}, TTFT n={qualified['ttft_ratio']['n']} {fmt(qualified['ttft_ratio'])}, end-to-end n={qualified['wall_ratio']['n']} {fmt(qualified['wall_ratio'])}.")
    lines+=['','Configuration and limits:',
        'All exact source commands and adapted flags: fixtures/settings.json and runs/<engine>/native-options.json. Only benchmark host/port/logs/private persistence/profile-copy paths differ; vendor omits unsupported --v100-kernels.',
        'Original uses separate16GiB Host KV plus8 Host state slots,3 extra Device states, requested/effective chunk2048. Vendor/R5 use hybrid unified16GiB Host,2 Device snapshots, requested chunk4096, MTP3/ngram15 and auto KV/headroom256. Native NInfer-all alignment is preserved; pinned-source calculation gives effective4224 on this geometry, with narrower final tails. This source-derived width is not a measured launch count; progress samples and logs are preserved. Native-cache budget/planner differences are intentional.',
        'New-lineage means a changed leading nonce, not guaranteed zero cache. Actual cached tokens/fraction are reported; identical/reprime requests are separate from cold and live appended turns.',
        'Tool choice is auto so definitions render. Add/schema changes must change actual prompt-token count; see tool_render_checks. A missing check invalidates attribution of tooling-prefix effects.',
        'Three unique512px color images are used, one per lineage; the immediate vision continuation preserves the same image and actual assistant reply. Prefix reuse does not by itself prove a media-encoder cache hit.',
        'Tools are scripted history only; no external commands/files are executed. Correctness uses exact typed JSON fields after optional whole-response JSON-fence removal, including known vision colors. No model judge.',
        'Actual assistant reasoning/call IDs are retained. Unexpected tool calls fail grading and receive deterministic no-execution results; malformed/truncated arguments, unknown tools or duplicate/missing call IDs cause dependent stages to be recorded as skipped rather than sending invalid histories. Table n is measured/recorded, with target3.',
        'This small test does not measure cache eviction pressure, concurrent agent sessions, near-limit capacity, disk restart reuse or general vision quality.',
        'All raw request payloads, responses, logs, per-request timing/usage, GPU samples and shutdowns are retained. Failures remain in the summary and are not excluded from latency records.',
        'All-paired timing ratios include wrong answers and are not evidence of useful agentic speed by themselves. The explicitly labeled secondary both-correct subset reports only pairs where both deterministic grades passed, with its sample count; no cases are removed from the full results.',
        'Service restoration evidence: control/restored.json; interruption/session accounting: control/budget.json. Accepted selection and production caches/config remain unchanged.']
    text='\n'.join(lines)+'\n';(REPO/'bench/NINFER_AGENTIC.txt').write_text(text);(root/'NINFER_AGENTIC.txt').write_text(text)
    print('Agentic report:',REPO/'bench/NINFER_AGENTIC.txt',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['prepare','verify','session','analyze']:sub.add_parser(command)
    run_parser=sub.add_parser('run');run_parser.add_argument('--engine',choices=ENGINES,required=True)
    args=parser.parse_args()
    if args.command=='run':run(args.root,args.engine)
    else:globals()[args.command](args.root)


if __name__=='__main__':main()
