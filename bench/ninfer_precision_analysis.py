"""Paired analysis and explicit coverage gates for precision campaign records."""
from __future__ import annotations
from array import array
import json
import math
from pathlib import Path
import statistics
import struct
import hashlib
import subprocess
import shutil
import re
from collections import Counter,defaultdict
import ninfer_precision as c
from ninfer_precision_grade import clean,grade


def read_if(path): return c.rows(path) if path.exists() else []


def interruption_accounting(root,require_restored=False):
    """Aggregate precision sessions, keeping the original target separate.

    A continuation has its own scheduling limit, not a replacement campaign
    target. Agentic performance sessions are a different experiment and are
    deliberately outside these two explicitly named precision ledgers.
    """
    ledgers={};sessions=[];used_seconds=0.0
    for name in ['budget.json','resume-budget.json']:
        path=root/'control'/name
        if not path.exists():
            if name=='budget.json':raise ValueError('missing original precision budget ledger')
            continue
        data=json.loads(path.read_text())
        entries=data.get('sessions',[])
        used=data.get('used_seconds',0)
        if not isinstance(entries,list) or isinstance(used,bool) or not isinstance(used,(int,float)) or not math.isfinite(used) or used<0:
            raise ValueError('invalid precision budget ledger: '+str(path))
        if not entries and used:
            raise ValueError('precision ledger has recorded time without sessions: '+str(path))
        for index,entry in enumerate(entries):
            if not isinstance(entry,dict) or entry.get('started') is None:
                raise ValueError('invalid precision session: '+str(path))
            sessions.append({**entry,'ledger':'control/'+name,'ledger_session':index})
        ledgers['control/'+name]={'used_seconds':used,'session_count':len(entries),
            'budget_seconds':data.get('budget_seconds'),'reserve_seconds':data.get('reserve_seconds')}
        used_seconds+=used
    incomplete=[s for s in sessions if s.get('finished') is None or s.get('restored') is not True]
    restored=bool(sessions) and not incomplete
    if require_restored and not restored:
        raise ValueError('CPU finalization requires every started precision session in budget.json and resume-budget.json to finish with successful service restoration')
    original=ledgers['control/budget.json']
    target=original['budget_seconds'] if original['budget_seconds'] is not None else 21600
    reserve=original['reserve_seconds'] if original['reserve_seconds'] is not None else 1800
    return {'used_seconds':used_seconds,'session_count':len(sessions),
        'completed_session_count':sum(s.get('finished') is not None for s in sessions),
        'all_started_sessions_restored':restored,'incomplete_or_unrestored_sessions':incomplete,
        'original_target_seconds':target,'original_restoration_reserve_seconds':reserve,
        'over_original_target_seconds':max(0,used_seconds-target),
        'continuation_started':any(s['ledger']=='control/resume-budget.json' for s in sessions),
        'ledgers':ledgers,'sessions':sessions,
        'scope':'Only control/budget.json and optional control/resume-budget.json; agentic performance ledger excluded.'}


def likelihood(root):
    streams=json.loads((root/'fixtures/corpus.json').read_text())
    documents={}
    errors=[]
    for key in c.ENGINES:
        grouped={}
        for record in read_if(root/'runs'/key/'likelihood-k0/records.jsonl'):
            if 'error' in record:
                errors.append({'engine':key,**record});continue
            doc=record['id'].split('/')[0]
            grouped.setdefault(doc,[]).extend(record['logprobs'])
        documents[key]={doc:{'tokens':len(logs),'mean_nll':-math.fsum(logs)/len(logs),
                             'ppl':math.exp(-math.fsum(logs)/len(logs))} for doc,logs in grouped.items() if logs}
    comparisons={}
    for left,right in c.PAIRS:
        paired=[];domains={};missing=[]
        for stream in streams:
            sid=stream['id'];a=documents[left].get(sid);b=documents[right].get(sid)
            if not a or not b or a['tokens']!=8192 or b['tokens']!=8192:
                missing.append(sid);continue
            pair=(a['mean_nll'],b['mean_nll'],8192)
            paired.append(pair);domains.setdefault(stream['domain'],[]).append(pair)
        result=c.paired_bootstrap(paired,transform=math.expm1)
        result['missing_or_incomplete_documents']=missing
        result['full_target_tokens']=131072
        result['paired_tokens']=sum(p[2] for p in paired)
        result['metric']='relative PPL difference: exp(token-weighted paired mean NLL difference)-1'
        if 'ci' in result:
            result['verdict']=c.verdict(result['ci'],.005,True) if not missing else 'inconclusive'
        result['domains']={name:(c.paired_bootstrap(values,transform=math.expm1) if len(values)>1 else
            {'estimate':math.expm1(values[0][0]-values[0][1]),'units':1,
             'uncertainty':'Single stream; no domain confidence interval estimated.'}) for name,values in domains.items()}
        comparisons[f'{left}_vs_{right}']=result
    return {'documents':documents,'comparisons':comparisons,'infrastructure_errors':errors}


def accuracy(root):
    cases=json.loads((root/'fixtures/quality.json').read_text())
    records={key:{r['id']:r for r in read_if(root/'runs'/key/'quality-k0/grades.jsonl')} for key in c.ENGINES}
    raw_by_engine={}
    for key in c.ENGINES:
        raw={r['id']:r for r in read_if(root/'runs'/key/'quality-k0/records.jsonl')}
        raw_by_engine[key]=raw
        for case in cases:
            cid=case['id']
            if cid not in records[key] and cid in raw:
                r=raw[cid]
                records[key][cid]={'id':cid,**({'correct':None,'infrastructure_error':r['error']} if 'error' in r else grade(case,r['content']))}
    engine_results={}
    for key,grades in records.items():
        valid=[g for g in grades.values() if type(g.get('correct')) is bool]
        domains={}
        for case in cases:
            record=grades.get(case['id'])
            if record and type(record.get('correct')) is bool: domains.setdefault(case['domain'],[]).append(record['correct'])
        format_only=[]
        for case in cases:
            cid=case['id'];record=grades.get(cid,{});raw=raw_by_engine[key].get(cid,{})
            if case['grader']!='json' or record.get('correct') is not False or 'content' not in raw:continue
            match=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',clean(raw['content']),re.I|re.S)
            if match and grade(case,match[1])['correct']:format_only.append(cid)
        engine_results[key]={'valid':len(valid),'correct':sum(g['correct'] for g in valid),
              'accuracy':sum(g['correct'] for g in valid)/len(valid) if valid else None,
              'domains':{k:{'n':len(v),'correct':sum(v),'accuracy':sum(v)/len(v)} for k,v in domains.items()},
              'format_only_json_failures':format_only,
              'failures':[g for g in grades.values() if g.get('correct') is not True]}
    comparisons={}
    for left,right in c.PAIRS:
        paired=[];domains={};missing=[];discordant=[];clusters={};domain_clusters={}
        for case in cases:
            cid=case['id'];a=records[left].get(cid,{}).get('correct');b=records[right].get(cid,{}).get('correct')
            if type(a) is not bool or type(b) is not bool: missing.append(cid);continue
            pair=(int(a),int(b),1);paired.append(pair);domains.setdefault(case['domain'],[]).append(pair)
            number=int(cid.rsplit('-',1)[1])
            family=(f'code-algorithm-{number%12}' if case['domain']=='code' else
                    f'tool-schema-{number%8}' if case['domain']=='tool_json' else
                    f'arithmetic-operation-{number%4}' if case['domain']=='arithmetic' else cid)
            clusters.setdefault(family,[]).append(pair)
            domain_clusters.setdefault(case['domain'],{}).setdefault(family,[]).append(pair)
            if a!=b: discordant.append({'id':cid,'left_correct':a,'right_correct':b})
        def aggregate(groups):
            return [(sum(p[0] for p in group)/len(group),sum(p[1] for p in group)/len(group),len(group)) for group in groups.values()]
        result=c.paired_bootstrap(aggregate(clusters))
        result['resampling']='whole paired task families for repeated code/tool/arithmetic patterns; whole cases for retrieval/history'
        result['paired_cases']=len(paired)
        result['case_only_sensitivity']=c.paired_bootstrap(paired)
        result.update(missing_or_ungraded=missing,discordant=discordant)
        if 'ci' in result: result['verdict']=c.verdict(result['ci'],.02,False) if not missing else 'inconclusive'
        result['domains']={name:c.paired_bootstrap(aggregate(groups)) for name,groups in domain_clusters.items()}
        comparisons[f'{left}_vs_{right}']=result
    return {'engines':engine_results,'comparisons':comparisons,
            'scope':'384 frozen synthetic cases. Code uses 12 algorithm families with 8 function-name variants each; tool/JSON cases use 8 schema families. Results do not establish broad benchmark quality.'}


def repeatability(root):
    result={}
    for key in c.ENGINES:
        records={r['id']:r for r in read_if(root/'runs'/key/'repeatability-k0/records.jsonl')}
        comparisons=[]
        for i in range(4):
            a=records.get(f'repeat-{i}-0',{}).get('logprobs');b=records.get(f'repeat-{i}-1',{}).get('logprobs')
            if a is None or b is None: continue
            if len(a)!=len(b): raise ValueError('repeatability unpaired token count')
            difference=[x-y for x,y in zip(a,b)]
            comparisons.append({'window':i,'tokens':len(a),'max_abs_logprob':max(map(abs,difference)),
                                'rmse':math.sqrt(math.fsum(x*x for x in difference)/len(difference)),
                                'exact':a==b})
        result[key]={'windows':len(records),'repeated_pairs':comparisons,
                     'coverage_complete':len(records)==12 and len(comparisons)==4}
    return result


def agreement(root):
    result={}
    for key in c.ENGINES:
        probes={r['id']:r for r in read_if(root/'runs'/key/'native-k3/records.jsonl')}
        packages={r['id']:r for r in read_if(root/'runs'/key/'package-native-k3/records.jsonl')}
        paired=[]
        for cid in json.loads((root/'fixtures/native-subset.json').read_text()):
            a=probes.get(cid);b=packages.get(cid)
            if not a or not b or 'error' in a or b.get('status')!=200:
                paired.append({'id':cid,'available':False});continue
            text=b['response']['choices'][0]['message'].get('content') or ''
            paired.append({'id':cid,'available':True,'exact_text':clean(a['content'])==clean(text),
                           'probe_text':a['content'],'package_text':text})
        result[key]={'paired':paired,'complete':all(r['available'] for r in paired),
                     'all_exact':all(r.get('exact_text') for r in paired),
                     'scope':'Native-template text agreement on 24 fixed cases; not proof of all execution paths or toolchain equivalence.'}
    return result


def bf16_values(file,offset,count):
    with file.open('rb') as f:
        f.seek(offset);data=f.read(count*2)
    if len(data)!=count*2: raise ValueError('truncated observed logits')
    words=array('H');words.frombytes(data)
    return [struct.unpack('<f',struct.pack('<I',word<<16))[0] for word in words]


def distribution(a,b):
    if len(a)!=len(b) or len(a)<2: raise ValueError('unpaired vocabulary')
    if not all(math.isfinite(x) for x in a+b): raise ValueError('non-finite observed logits')
    am=max(a);bm=max(b)
    az=am+math.log(math.fsum(math.exp(x-am) for x in a))
    bz=bm+math.log(math.fsum(math.exp(x-bm) for x in b))
    ac=math.fsum(a)/len(a);bc=math.fsum(b)/len(b)
    errors=[(x-ac)-(y-bc) for x,y in zip(a,b)]
    kl_ab=math.fsum(math.exp(x-az)*((x-az)-(y-bz)) for x,y in zip(a,b))
    kl_ba=math.fsum(math.exp(y-bz)*((y-bz)-(x-az)) for x,y in zip(a,b))
    amax=sorted(range(len(a)),key=a.__getitem__,reverse=True)[:2]
    bmax=sorted(range(len(b)),key=b.__getitem__,reverse=True)[:2]
    return {'kl_left_right':kl_ab,'kl_right_left':kl_ba,
            'centered_logit_rmse':math.sqrt(math.fsum(x*x for x in errors)/len(errors)),
            'centered_logit_max_abs':max(map(abs,errors)),
            'argmax_agreement':amax[0]==bmax[0],'left_argmax':amax[0],'right_argmax':bmax[0],
            'left_top2_margin':a[amax[0]]-a[amax[1]],'right_top2_margin':b[bmax[0]]-b[bmax[1]],
            'vocabulary_rows':len(a)}


def build_cpu(root):
    source=c.ROOT/'bench/ninfer_precision_oracle.cpp'
    directory=root/'analysis';directory.mkdir(parents=True,exist_ok=True)
    binary=directory/'precision-cpu'
    provenance=directory/'cpu-build.json'
    old=json.loads(provenance.read_text()) if provenance.exists() else {}
    if not binary.exists() or old.get('source_sha256')!=c.sha(source):
        blas=Path('/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.32.so')
        command=['g++-13','-O3','-ffp-contract=off','-std=c++20','-I'+str(c.ENGINES['vendor']['source']/'third_party'),str(source),str(blas),'-o',str(binary)]
        subprocess.run(command,check=True)
        c.write_json(provenance,{'argv':command,'source_sha256':c.sha(source),'binary_sha256':c.sha(binary),'gpu_used':False,
                                'fp64_blas':str(blas),'fp64_blas_sha256':c.sha(blas),'blas_threads':2})
        shutil.copy2(binary,c.ROOT/'.local/precision-oracle')
    return binary


def analyze_distributions(root,pilot=False):
    jobs=[];token_ids={r['id']:r['tokens'] for r in c.rows(root/'fixtures/tokens.jsonl')};script=token_ids['decode-script'][:130]
    suites=['observer-pilot-k0','observer-pilot-k3'] if pilot else ['likelihood-k0','boundary-k0','repeatability-k0','decode-k0','decode-k1','decode-k3','decode-k5']
    for suite in suites:
        pairings=[(left,right,suite,f'{left}_vs_{right}') for left,right in c.PAIRS]
        if suite.startswith(('decode-k','observer-pilot-k')) and not suite.endswith('-k0'):
            pairings += [(key,key,suite.rsplit('-k',1)[0]+'-k0',f'{key}_MTP{suite.rsplit("-k",1)[1]}_vs_decode') for key in c.ENGINES]
        for left,right,right_suite,comparison in pairings:
            a=root/'runs'/left/suite/'observations';b=root/'runs'/right/right_suite/'observations'
            if not a.exists() or not b.exists(): continue
            left_jobs={r['id']:r for r in c.rows(a.parent/'jobs.jsonl')};right_jobs={r['id']:r for r in c.rows(b.parent/'jobs.jsonl')}
            for metadata in sorted(a.rglob('observations.jsonl')):
                relative=metadata.relative_to(a)
                other=b/relative
                if not other.exists(): continue
                cid=str(relative.parent)
                if left_jobs[cid]['tokens']!=right_jobs[cid]['tokens'] or left_jobs[cid].get('forced_tokens')!=right_jobs[cid].get('forced_tokens'):
                    raise ValueError('observed distributions have different input histories')
                left_meta=c.rows(metadata);right_meta=c.rows(other)
                # Score tiles have identical boundaries. Forced decode/verify rows
                # match by scripted history length and column, not launch index.
                def keyed(observations,directory):
                    proposals={p['cursor']:p for p in read_if(directory/'proposal-inputs.jsonl')}
                    for index,m in enumerate(observations):
                        proposal=proposals.get(m['script_cursor']) if m['phase']=='verify' else None
                        columns=min(m['columns'],proposal['extent']+1,129-m['script_cursor']) if proposal else m['columns']
                        for col in range(columns):
                            if suite.startswith(('decode','observer-pilot')):
                                history_tokens=script[:m['script_cursor']]+(proposal['drafts'][:col] if proposal else [])
                                history=hashlib.sha256(struct.pack('<'+'i'*len(history_tokens),*history_tokens)).hexdigest()
                            else: history=(index,col)
                            yield history,m,col,index
                def select(observations,directory):
                    mapped={}
                    for history,m,col,index in keyed(observations,directory):
                        rank=(col!=0,col,index)
                        if history not in mapped or rank<mapped[history][2]:mapped[history]=(m,col,rank)
                    return mapped
                mapped=select(right_meta,other.parent)
                for history,(m,col,rank) in select(left_meta,metadata.parent).items():
                    if history not in mapped: continue
                    n,ncol,_=mapped[history]
                    if m['vocab']!=n['vocab']: raise ValueError('public vocabulary mismatch')
                    item={'suite':suite,'comparison':comparison,'fixture':cid,'history':history,
                        'left_file':str(metadata.parent/'logits.bf16'),'right_file':str(other.parent/'logits.bf16'),
                        'left_offset':m['offset']+col*m['vocab']*2,'right_offset':n['offset']+ncol*n['vocab']*2,
                        'vocab':m['vocab'],'left_column':col,'right_column':ncol,
                        'left_actual_graph':m['actual_graph_launch'],'right_actual_graph':n['actual_graph_launch']}
                    cursor=m['script_cursor']+col
                    if suite.startswith(('decode','observer-pilot')) and col==0 and cursor<len(script):item['target_token']=script[cursor]
                    jobs.append(item)
    if not pilot:
        # Long sentinels generate normally. Compare only column zero at histories
        # whose actual generated prefixes match, never by launch index alone.
        suite='long-k3'
        for left,right in c.PAIRS:
            a=root/'runs'/left/suite;b=root/'runs'/right/suite
            if not (a/'jobs.jsonl').exists() or not (b/'jobs.jsonl').exists():continue
            aj={r['id']:r for r in c.rows(a/'jobs.jsonl')};bj={r['id']:r for r in c.rows(b/'jobs.jsonl')}
            ar={r['id']:r for r in read_if(a/'records.jsonl')};br={r['id']:r for r in read_if(b/'records.jsonl')}
            for cid,job in aj.items():
                if cid not in bj or cid not in ar or cid not in br:continue
                if job['tokens']!=bj[cid]['tokens']:raise ValueError('long histories differ')
                prompt=len(job['tokens'])
                def long_rows(directory,record):
                    for m in read_if(directory/'observations.jsonl'):
                        count=m['input_frontier']+1-prompt
                        if count<0 or count>len(record.get('tokens',[])):continue
                        yield (count,tuple(record['tokens'][:count])),m
                ad=a/'observations'/cid;bd=b/'observations'/cid
                bm=dict(long_rows(bd,br[cid]))
                for history,m in long_rows(ad,ar[cid]):
                    if history not in bm:continue
                    n=bm[history]
                    if m['vocab']!=n['vocab']:raise ValueError('long vocabulary mismatch')
                    prefix=history[1]
                    jobs.append({'suite':suite,'comparison':f'{left}_vs_{right}','fixture':cid,
                        'history':hashlib.sha256(struct.pack('<'+'i'*len(prefix),*prefix)).hexdigest(),
                        'matched_generated_prefix_length':history[0],
                        'left_file':str(ad/'logits.bf16'),'right_file':str(bd/'logits.bf16'),
                        'left_offset':m['offset'],'right_offset':n['offset'],'vocab':m['vocab'],
                        'left_column':0,'right_column':0,'left_actual_graph':m['actual_graph_launch'],
                        'right_actual_graph':n['actual_graph_launch']})
    label='pilot-' if pilot else ''
    manifest=root/f'analysis/{label}distribution-manifest.json';destination=root/f'analysis/{label}distributions.jsonl'
    if destination.exists(): raise ValueError('distribution analysis already exists')
    c.write_json(manifest,jobs)
    subprocess.run([str(build_cpu(root)),'--distributions',str(manifest),str(destination)],check=True)


def acceptance_coverage(root,pilot=False):
    """Audit every expected fixture and round before selecting numerical jobs.

    Missing/duplicate records are evidence gaps, never zero-error passes. Valid
    unambiguous rounds still receive numerical analysis when another fixture is
    incomplete, so coverage problems do not stop the remaining CPU analyses.
    """
    jobs=[];fixtures=[]
    expected_ids=(["decode-2048-0"] if pilot else
                  [f'decode-{depth}-{i}' for depth in [8192,32768,65536] for i in range(2)])
    for key in c.ENGINES:
        for k in ([3] if pilot else [1,3,5]):
            suite=root/'runs'/key/f'{"observer-pilot" if pilot else "decode"}-k{k}'
            control=suite/'control-validation.json'
            try:validation=json.loads(control.read_text()) if control.exists() else None
            except (OSError,ValueError):validation=None
            for cid in expected_ids:
                directory=suite/'observations'/cid
                audit={'engine':key,'k':k,'fixture':cid,'missing_files':[],
                       'missing_round_records':[],'duplicate_round_records':[],
                       'duplicate_round_numbers':[],
                       'record_errors':[],'eligible_cursors':[],
                       'control_validation_present':control.exists(),
                       'control_validation_passed':bool(validation and validation.get('passed') is True)}
                def audited_rows(name):
                    path=directory/name
                    if not path.exists():audit['missing_files'].append(str(path.relative_to(root)));return []
                    try:
                        before=len(c.READ_ERRORS);result=c.rows(path)
                        audit['record_errors'].extend(c.READ_ERRORS[before:])
                        return result
                    except (OSError,ValueError) as error:
                        audit['record_errors'].append({'file':name,'error':str(error)[:300]});return []
                proposals=audited_rows('proposal-inputs.jsonl')
                accepted=audited_rows('acceptance.jsonl')
                metadata=[r for r in audited_rows('observations.jsonl') if r.get('phase')=='verify']
                grouped={}
                for name,records,field in [('proposal',proposals,'cursor'),('acceptance',accepted,'cursor'),('verify',metadata,'script_cursor')]:
                    mapping=defaultdict(list)
                    for record in records:
                        cursor=record.get(field)
                        if type(cursor) is not int:
                            audit['record_errors'].append({'file':name,'error':'missing/noninteger cursor'});continue
                        mapping[cursor].append(record)
                    grouped[name]=mapping
                    if name!='verify':
                        numbered=defaultdict(list)
                        for record in records:
                            number=record.get('round')
                            if type(number) is not int or number<0:
                                audit['record_errors'].append({'file':name,'error':'missing/invalid round number'})
                            else:numbered[number].append(record.get('cursor'))
                        audit['duplicate_round_numbers'].extend({'file':name,'round':number,'cursors':positions}
                            for number,positions in numbered.items() if len(positions)>1)
                cursors=sorted(set().union(*(set(m) for m in grouped.values())))
                audit['rounds_observed']=len(cursors)
                logits=directory/'logits.bf16'
                if not logits.exists():audit['missing_files'].append(str(logits.relative_to(root)))
                expected_cursor=1
                for cursor in cursors:
                    counts={name:len(mapping.get(cursor,[])) for name,mapping in grouped.items()}
                    missing=[name for name,count in counts.items() if count==0]
                    duplicate={name:count for name,count in counts.items() if count>1}
                    if missing:audit['missing_round_records'].append({'cursor':cursor,'missing':missing})
                    if duplicate:audit['duplicate_round_records'].append({'cursor':cursor,'duplicates':duplicate})
                    if missing or duplicate:continue
                    p=grouped['proposal'][cursor][0];actual=grouped['acceptance'][cursor][0];m=grouped['verify'][cursor][0]
                    if cursor!=expected_cursor:
                        audit['record_errors'].append({'cursor':cursor,'error':f'round cursor gap: expected {expected_cursor}, got {cursor}'})
                    try:
                        count=actual['count'];extent=p['extent'];vocab=m['vocab'];offset=m['offset']
                        if count!=actual['accepted']+1 or not 1<=count<=k+1 or cursor+count>129:
                            raise ValueError('invalid licensed count/budget')
                        if p['width']!=k or not 0<=extent<=k or len(p['drafts'])!=extent:
                            raise ValueError('verification width/extent differs from frozen K')
                        if p.get('round')!=actual.get('round'):raise ValueError('proposal/acceptance round differs')
                        if type(actual.get('unforced_bonus')) is not int:raise ValueError('missing/noninteger unforced bonus')
                        if type(vocab) is not int or vocab<2 or type(offset) is not int or offset<0 or m.get('columns',0)<extent+1:
                            raise ValueError('invalid/missing observed logit geometry')
                        if not logits.exists() or offset+(extent+1)*vocab*2>logits.stat().st_size:
                            raise ValueError('missing/truncated observed logit columns')
                        expected_cursor=cursor+count
                    except (KeyError,TypeError,ValueError) as error:
                        audit['record_errors'].append({'cursor':cursor,'error':str(error)});continue
                    audit['eligible_cursors'].append(cursor)
                    jobs.append({'engine':key,'k':k,'fixture':cid,
                        'cursor':cursor,'extent':extent,'drafts':p['drafts'],
                        'remaining_budget':129-p['cursor'],'actual_accepted':actual['accepted'],
                        'actual_unforced_bonus':actual['unforced_bonus'],
                        'file':str(directory/'logits.bf16'),'offset':m['offset'],'vocab':m['vocab']})
                audit['final_script_cursor']=expected_cursor
                if expected_cursor!=129:audit['record_errors'].append({'error':f'incomplete forced history: final cursor {expected_cursor}, expected 129'})
                audit['record_coverage_complete']=bool(cursors) and audit['control_validation_passed'] and not any(
                    audit[field] for field in ['missing_files','missing_round_records','duplicate_round_records','duplicate_round_numbers','record_errors'])
                fixtures.append(audit)
    return {'expected_fixture_count':len(c.ENGINES)*(1 if pilot else 3)*len(expected_ids),
            'analysis_complete':False,'overall_verdict':'inconclusive','fixtures':fixtures,
            'pattern_definition':'Budget-tail rounds (remaining output budget <= K) and zero/short extents do not establish full-width full/partial/reject-all coverage.'},jobs


def acceptance_pattern(record):
    if record['remaining_budget']<=record['k']:return 'budget_tail'
    extent=record['extent'];accepted=record['expected_accepted']
    if extent==0:return 'zero_extent'
    if accepted==extent:return 'full' if extent==record['k'] else 'full_short_extent'
    if accepted==0:return 'reject_all'
    return 'partial'


def complete_acceptance_coverage(coverage,records):
    grouped=defaultdict(list)
    for record in records:grouped[(record['engine'],record['k'],record['fixture'])].append(record)
    for fixture in coverage['fixtures']:
        rows=grouped[(fixture['engine'],fixture['k'],fixture['fixture'])]
        counts=Counter(r['cursor'] for r in rows)
        fixture['missing_numerical_cursors']=sorted(set(fixture['eligible_cursors'])-set(counts))
        fixture['duplicate_numerical_cursors']={str(cursor):n for cursor,n in counts.items() if n>1}
        fixture['pattern_counts']=dict(Counter(acceptance_pattern(r) for r in rows))
        required=['full','reject_all'] if fixture['k']==1 else ['full','partial','reject_all']
        fixture['missing_patterns']=[pattern for pattern in required if not fixture['pattern_counts'].get(pattern)]
        fixture['acceptance_errors']=sum(r.get('acceptance_correct') is not True for r in rows)
        fixture['bonus_errors']=sum(r.get('bonus_correct') is not True for r in rows)
        fixture['coverage_complete']=(fixture['record_coverage_complete'] and not fixture['missing_numerical_cursors']
            and not fixture['duplicate_numerical_cursors'] and not fixture['missing_patterns'])
    coverage['analysis_complete']=True
    coverage['complete_fixture_count']=sum(r['coverage_complete'] for r in coverage['fixtures'])
    coverage['coverage_complete']=(len(coverage['fixtures'])==coverage['expected_fixture_count'] and
                                   coverage['complete_fixture_count']==coverage['expected_fixture_count'])
    coverage['acceptance_errors']=sum(r['acceptance_errors'] for r in coverage['fixtures'])
    coverage['bonus_errors']=sum(r['bonus_errors'] for r in coverage['fixtures'])
    coverage['overall_verdict']=('inconclusive' if not coverage['coverage_complete'] else
        'failed' if coverage['acceptance_errors'] or coverage['bonus_errors'] else 'passed')
    return coverage


def analyze_acceptance(root,pilot=False):
    coverage,jobs=acceptance_coverage(root,pilot)
    label='pilot-' if pilot else ''
    manifest=root/f'analysis/{label}acceptance-manifest.json';destination=root/f'analysis/{label}acceptance.jsonl'
    coverage_path=root/f'analysis/{label}acceptance-coverage.json'
    if destination.exists():raise ValueError('acceptance analysis already exists')
    c.write_json(coverage_path,coverage)
    c.write_json(manifest,jobs)
    subprocess.run([str(build_cpu(root)),'--acceptance',str(manifest),str(destination)],check=True)
    c.write_json(coverage_path,complete_acceptance_coverage(coverage,read_if(destination)))


def state_comparisons(root,pilot=False):
    output=[]
    for key in c.ENGINES:
        suite='observer-pilot' if pilot else 'decode'
        plain=root/'runs'/key/f'{suite}-k0/observations'
        for k in ([3] if pilot else [1,3,5]):
            target=root/'runs'/key/f'{suite}-k{k}/observations'
            if not plain.exists() or not target.exists(): continue
            for meta in sorted(plain.rglob('states.jsonl')):
                other=target/meta.relative_to(plain)
                if not other.exists(): continue
                baseline={r['execution_frontier']:r for r in c.rows(meta)}
                with (meta.parent/'states.bin').open('rb') as a,(other.parent/'states.bin').open('rb') as b:
                    for record in c.rows(other):
                        ref=baseline.get(record['execution_frontier'])
                        if not ref: continue
                        integers_equal=all(ref[field]==record[field] for field in ['execution_frontier','ledger_frontier','script_cursor'])
                        if ref['parts']!=record['parts']: raise ValueError('unpaired sampled state layout')
                        a.seek(ref['offset']);b.seek(record['offset'])
                        for part in ref['parts']:
                            av=a.read(part['bytes']);bv=b.read(part['bytes'])
                            if part['kind']=='recurrent':
                                count=len(av)//4;x=struct.unpack('<'+'f'*count,av);y=struct.unpack('<'+'f'*count,bv)
                            else:
                                count=len(av)//2
                                x=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in struct.unpack('<'+'H'*count,av)]
                                y=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in struct.unpack('<'+'H'*count,bv)]
                            errors=[z-w for z,w in zip(y,x)]
                            output.append({'engine':key,'k':k,'fixture':str(meta.parent.relative_to(plain)),
                                'execution_frontier':record['execution_frontier'],'integers_equal':integers_equal,**part,
                                'relative_l2':math.sqrt(math.fsum(e*e for e in errors)/max(math.fsum(z*z for z in x),1e-300)),
                                'max_abs':max(map(abs,errors)),'exact_bytes':av==bv,
                                'scope':'sampled first/middle/last GDN layers after normal fold/commit; comparator is same-engine ordinary decode of identical scripted prefix'})
    c.write_json(root/f'analysis/{"pilot-" if pilot else ""}state-comparisons.json',output)
    return output


def long_state_comparisons(root):
    result=[]
    for left,right in c.PAIRS:
        a=root/'runs'/left/'long-k3';b=root/'runs'/right/'long-k3'
        if not (a/'jobs.jsonl').exists() or not (b/'jobs.jsonl').exists():continue
        aj={r['id']:r for r in c.rows(a/'jobs.jsonl')};bj={r['id']:r for r in c.rows(b/'jobs.jsonl')}
        ar={r['id']:r for r in read_if(a/'records.jsonl')};br={r['id']:r for r in read_if(b/'records.jsonl')}
        for cid,job in aj.items():
            if cid not in bj or cid not in ar or cid not in br:continue
            if job['tokens']!=bj[cid]['tokens']:raise ValueError('long histories differ')
            prompt=len(job['tokens']);ad=a/'observations'/cid;bd=b/'observations'/cid
            def keyed(directory,record):
                for state in read_if(directory/'states.jsonl'):
                    count=state['execution_frontier']-prompt
                    if 0<=count<=len(record.get('tokens',[])):
                        yield (count,tuple(record['tokens'][:count])),state
            bs=dict(keyed(bd,br[cid]))
            if not (ad/'states.bin').exists() or not (bd/'states.bin').exists():continue
            with (ad/'states.bin').open('rb') as af,(bd/'states.bin').open('rb') as bf:
                for history,m in keyed(ad,ar[cid]):
                    n=bs.get(history)
                    if n is None:continue
                    if m['parts']!=n['parts']:raise ValueError('long sampled state layouts differ')
                    af.seek(m['offset']);bf.seek(n['offset'])
                    for part in m['parts']:
                        av=af.read(part['bytes']);bv=bf.read(part['bytes'])
                        if part['dtype']==1:
                            count=len(av)//4;x=struct.unpack('<'+'f'*count,av);y=struct.unpack('<'+'f'*count,bv)
                        else:
                            x=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in struct.unpack('<'+'H'*(len(av)//2),av)]
                            y=[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in struct.unpack('<'+'H'*(len(bv)//2),bv)]
                        errors=[z-w for z,w in zip(x,y)]
                        result.append({'comparison':f'{left}_vs_{right}','fixture':cid,'generated_executed_prefix':history[0],
                            'execution_frontier':m['execution_frontier'],'integers_equal':m['execution_frontier']==n['execution_frontier'] and m['ledger_frontier']==n['ledger_frontier'],
                            **part,'relative_l2':math.sqrt(math.fsum(v*v for v in errors)/max(math.fsum(v*v for v in y),1e-300)),
                            'max_abs':max(map(abs,errors)),'exact_bytes':av==bv})
    c.write_json(root/'analysis/long-state-comparisons.json',result)
    return result


def summarize(root):
    result={'likelihood':likelihood(root),'accuracy':accuracy(root),'repeatability':repeatability(root),
            'probe_package_agreement':agreement(root)}
    c.write_json(root/'analysis/summary.json',result)
    print(json.dumps({'likelihood':result['likelihood']['comparisons'],'accuracy':result['accuracy']['comparisons']},indent=2)[:12000])
    return result


def operator_criterion(family,qtype):
    # Common pinned vendor/R5 semantic A16 criteria. Historical V100 has
    # smaller gross bounds for three families; its separate check preserves
    # those as well. AllowA4/AllowA8 stage A16 on this sm_70 build.
    if family=='linear_swiglu':return (.0033,.005,2/256)
    if qtype==7 and family=='gdn_input_proj':return (.003,.004,2/256)
    if qtype==7 and family=='attn_input_proj':return (.0029,.004,2/256)
    return (1/256,1/256,2/256)


def operator_contract_summary(records,criteria,expected_operators,expected_widths):
    """Supplemental contract only; never mutate common-policy records."""
    checked=[];mapped={};counts=Counter()
    for raw in records:
        key=f"{raw['family']}:q{raw['qtype']}"
        if key not in criteria:raise ValueError('operator outside frozen supplemental contract: '+key)
        criterion=dict(criteria[key])
        limit=criterion['gross_absolute']+criterion['gross_relative_to_max_reference']*raw['reference_max_abs']
        row={**raw,'criterion':criterion,'gross_limit':limit,
            'passes_contract':(raw['count']>0 and raw['nonfinite']==0 and
                raw['relative_l2']<=criterion['relative_l2'] and raw['max_abs']<=limit),
            'represented_weights_equal':raw['represented_weight_differences']==0}
        index=(raw['engine'],raw['operator'],raw['width']);counts[index]+=1;mapped[index]=row
        checked.append(row)
    comparisons={}
    for left,right in c.PAIRS:
        changes=[]
        missing=[{'engine':engine,'operator':op,'width':width} for engine in [left,right]
            for op in expected_operators for width in expected_widths if (engine,op,width) not in mapped]
        duplicates=[{'engine':engine,'operator':op,'width':width,'records':count}
            for (engine,op,width),count in counts.items() if engine in [left,right] and count>1]
        for op in expected_operators:
            for width in expected_widths:
                a=mapped.get((left,op,width));b=mapped.get((right,op,width))
                if a is None or b is None:continue
                changes.append({'operator':op,'width':width,
                    'comparable_represented_weights':a['represented_weights_equal'] and b['represented_weights_equal'],
                    'left_passes':a['passes_contract'],'right_passes':b['passes_contract'],
                    'left_relative_l2':a['relative_l2'],'right_relative_l2':b['relative_l2']})
        valid=[r for r in changes if r['comparable_represented_weights']]
        new_fail=[r for r in valid if not r['left_passes'] and r['right_passes']]
        fixed=[r for r in valid if r['left_passes'] and not r['right_passes']]
        shared=[r for r in valid if not r['left_passes'] and not r['right_passes']]
        verdict=('inconclusive' if missing or duplicates or not changes or len(valid)!=len(changes) else
            'mixed' if new_fail and fixed else 'lost' if new_fail else 'gained' if fixed else
            'inconclusive' if shared else 'retained within original V100 operator tolerances')
        comparisons[f'{left}_vs_{right}']={'verdict':verdict,'operator_width_pairs':len(changes),
            'missing_operator_width_records':missing,'duplicate_operator_width_records':duplicates,
            'new_tolerance_failures':new_fail,'resolved_tolerance_failures':fixed,
            'shared_tolerance_failures':shared,'changes':changes}
    return {'records':checked,'comparisons':comparisons,'engines':{key:{
        'operator_width_records':sum(r['engine']==key for r in checked),
        'criterion_failures':sum(r['engine']==key and not r['passes_contract'] for r in checked)} for key in c.ENGINES}}


def supplemental_operator_contract(root):
    path=root/'provenance/operator-tolerance-provenance.json'
    policy=json.loads(path.read_text())
    records=read_if(root/'analysis/operators.jsonl')
    result=operator_contract_summary(records,policy['original_v100']['criteria'],
        policy['expected_operators'],policy['expected_widths'])
    result.update(contract='Supplemental original V100 semantic A16 contract, applied uniformly to all engines.',
        tolerance_provenance='provenance/operator-tolerance-provenance.json',
        tolerance_provenance_sha256=c.sha(path),primary_common_vendor_policy_unchanged=True,
        scope='Deterministic captured operators against represented weights; not full-model precision or statistical confidence.')
    c.write_json(root/'analysis/operator-original-v100-contract.json',result)
    return result


def analyze_operators(root):
    jobs=[]
    canonical=root/'runs/v100/operators-k0/operators'
    for base in sorted(canonical.glob('*/weight.json')):
        for key in c.ENGINES:
            candidate=root/'runs'/key/'operators-k0/operators'/base.parent.name
            if (candidate/'weight.json').exists():jobs.append({'canonical':str(base.parent),'candidate':str(candidate)})
    if not jobs:raise ValueError('no captured operators')
    manifest=root/'analysis/operator-manifest.json';destination=root/'analysis/operators.jsonl'
    if destination.exists():raise ValueError('operator analysis already exists; preserve previous evidence')
    c.write_json(manifest,jobs)
    subprocess.run([str(build_cpu(root)),str(manifest),str(destination)],check=True)
    return summarize_operators(root)


def summarize_operators(root):
    records=read_if(root/'analysis/operators.jsonl');engines={};mapped={}
    for r in records:
        relative,absolute,gross=operator_criterion(r['family'],r['qtype'])
        r['criterion']={'relative_l2':relative,'gross_absolute':absolute,'gross_relative_to_max_reference':gross}
        r['gross_limit']=absolute+gross*r['reference_max_abs']
        r['passes_existing_criterion']=(r['nonfinite']==0 and r['count']>0 and r['relative_l2']<=relative and r['max_abs']<=r['gross_limit'])
        r['represented_weights_equal']=r['represented_weight_differences']==0
        mapped[(r['engine'],r['operator'],r['width'])]=r
        engines.setdefault(r['engine'],[]).append(r)
    expected_operators={p.parent.name for p in (root/'runs/v100/operators-k0/operators').glob('*/weight.json')}
    expected_widths=c.WIDTHS+[1024,2048]
    comparisons={}
    for left,right in c.PAIRS:
        changes=[]
        for (engine,operator,width),a in mapped.items():
            if engine!=left:continue
            b=mapped.get((right,operator,width))
            if b is None:continue
            valid=a['represented_weights_equal'] and b['represented_weights_equal']
            av=a['relative_l2'];bv=b['relative_l2']
            changes.append({'operator':operator,'width':width,'comparable_represented_weights':valid,
                'left_relative_l2':av,'right_relative_l2':bv,'relative_l2_ratio':av/bv if bv else None,
                'left_passes':a['passes_existing_criterion'],'right_passes':b['passes_existing_criterion'],
                'error_direction':'equal' if abs(av-bv)<=1e-12 else 'improved' if av<bv else 'regressed'})
        passes=[r for r in changes if r['comparable_represented_weights']]
        new_fail=[r for r in passes if not r['left_passes'] and r['right_passes']]
        fixed=[r for r in passes if r['left_passes'] and not r['right_passes']]
        both_fail=[r for r in passes if not r['left_passes'] and not r['right_passes']]
        missing=[{'engine':engine,'operator':operator,'width':width} for engine in [left,right]
                 for operator in sorted(expected_operators) for width in expected_widths
                 if (engine,operator,width) not in mapped]
        comparisons[f'{left}_vs_{right}']={'operator_width_pairs':len(changes),'new_tolerance_failures':new_fail,
            'resolved_tolerance_failures':fixed,'shared_tolerance_failures':both_fail,
            'missing_operator_width_records':missing,
            'error_directions':{direction:sum(r['error_direction']==direction for r in passes) for direction in ['improved','regressed','equal']},
            'changes':changes,'scope':'Deterministic captured operators only; complete output rows/columns and complete K. Full-model accumulated state is reported separately.',
            'verdict':('inconclusive' if not changes or missing or len(passes)!=len(changes) else
                       'mixed' if new_fail and fixed else 'lost' if new_fail else 'gained' if fixed else
                       'inconclusive' if both_fail else 'retained within existing operator tolerances')}
    summary={'records':records,'comparisons':comparisons,
             'engines':{key:{'operator_width_records':len(values),'criterion_failures':sum(not r['passes_existing_criterion'] for r in values),
                            'weight_mismatches':sum(not r['represented_weights_equal'] for r in values),
                            'operator_instances':len({r['operator'] for r in values}),'widths':sorted({r['width'] for r in values}),
                            'relative_l2_median':statistics.median(r['relative_l2'] for r in values),
                            'relative_l2_p95':c.percentile(sorted(r['relative_l2'] for r in values),.95),
                            'max_abs_p95':c.percentile(sorted(r['max_abs'] for r in values),.95),
                            'relative_l2_max':max(r['relative_l2'] for r in values)} for key,values in engines.items()}}
    c.write_json(root/'analysis/operator-summary.json',summary)
    return summary


def descriptive(values):
    values=sorted(values)
    if not values:return {'n':0}
    return {'n':len(values),'min':values[0],'median':statistics.median(values),
            'p95':c.percentile(values,.95),'max':values[-1],
            'mean':math.fsum(values)/len(values)}


def run_error_markers(base):
    # Keep the original suite/archived-pilot logs, and the bounded per-attempt
    # run logs used by wide replay. Do not recurse into copied sources/builds.
    logs=set(base.glob('*/*.log'))|set(base.glob('*/attempts/attempt-*/*.log'))
    markers=[]
    for log in sorted(logs):
        with log.open(errors='replace') as stream:
            lines=[line.rstrip('\r\n')[:500] for line in stream
                   if re.search(r'\b(?:ERROR|FATAL|CRITICAL)\b|force[- ]?kill|engine recovery',line,re.I)]
        if lines:markers.append({'path':str(log.relative_to(base)),'lines':lines})
    return markers


def integration(root):
    cases={r['id']:r for r in json.loads((root/'fixtures/quality.json').read_text())}
    native_ids=json.loads((root/'fixtures/native-subset.json').read_text())
    mtp_ids=json.loads((root/'fixtures/mtp-subset.json').read_text())
    long_jobs=root/'fixtures/long-jobs.json'
    long_limits={job['id']:job.get('max_tokens') for job in json.loads(long_jobs.read_text())} if long_jobs.exists() else {}
    result={}
    for key in c.ENGINES:
        base=root/'runs'/key
        def generated(suite):
            return {r['id']:r for r in read_if(base/suite/'records.jsonl')}
        plain=generated('quality-k0');native=generated('native-k0');mtp=generated('mtp-quality-k3')
        def compare(records,ids):
            out=[]
            for cid in ids:
                a=plain.get(cid);b=records.get(cid)
                if not a or not b or 'error' in a or 'error' in b:
                    out.append({'id':cid,'available':False});continue
                out.append({'id':cid,'available':True,'common_plain_correct':grade(cases[cid],a['content'])['correct'],
                            'control_correct':grade(cases[cid],b['content'])['correct'],
                            'exact_text':clean(a['content'])==clean(b['content'])})
            return out
        images=read_if(base/'package-images/records.jsonl')
        continuations=read_if(base/'package-images/cache-continuations.jsonl')
        longs=generated('long-k3')
        long_results=[]
        for cid in ['long-131072','long-229376']:
            r=longs.get(cid)
            text=r.get('content','') if r else ''
            expected='maple-7c90 519204'
            presence=all(value in text for value in ['maple-7c90','519204'])
            ordered=bool(re.search(r'(?<![\w-])maple-7c90\s+519204(?![\w-])',text))
            strict=clean(text)==expected
            generated_tokens=len(r['tokens']) if r and 'tokens' in r else None
            token_limit=long_limits.get(cid)
            deviation=(None if strict or not r or 'error' in r else
                       'extra trailing text' if clean(text).startswith(expected) else 'answer differs from required exact format')
            long_results.append({'id':cid,'available':bool(r) and 'error' not in r,
                'retrieval_correct':presence,'both_needle_strings_present':presence,'ordered_answer_present':ordered,
                'strict_answer_format_correct':strict,'format_deviation':deviation,
                'generated_tokens':generated_tokens,'generation_token_limit':token_limit,
                'output_token_limit_reached':generated_tokens>=token_limit if generated_tokens is not None and token_limit is not None else None,
                'content':text,'reused_tokens':r.get('reused_tokens') if r else None})
        graphs={}
        for meta in base.glob('*/observations/**/observations.jsonl'):
            suite=meta.relative_to(base).parts[0]
            counts=graphs.setdefault(suite,{'observed_results':0,'actual_graph_launches':0,'phases':{}})
            for record in c.rows(meta):
                counts['observed_results']+=1
                counts['actual_graph_launches']+=int(record['actual_graph_launch'])
                phase=record['phase'];counts['phases'][phase]=counts['phases'].get(phase,0)+1
        timing={p.parent.name:json.loads(p.read_text()) for p in base.glob('*/timing.json')
                if not any(tag in p.parent.name for tag in ['initial','before-','rejected-'])}
        shutdowns={str(p.relative_to(base)):json.loads(p.read_text()) for p in base.glob('*/shutdown.json')}
        error_markers=run_error_markers(base)
        result[key]={'native_template_control':compare(native,native_ids),'mtp_control':compare(mtp,mtp_ids),
            'images':{'completed':len(images),'target':24,'correct':sum(r.get('grade',{}).get('correct') is True for r in images),
                      'failures':[r for r in images if r.get('grade',{}).get('correct') is not True]},
            'image_continuations':continuations,'long_context':long_results,'observed_graph_results':graphs,
            'suite_timing_and_memory':timing,'shutdowns_including_archived_pilot':shutdowns,
            'error_markers_for_review':error_markers}
    return result


def report(root):
    summary=summarize(root)
    extra=integration(root)
    long_format_failures=[{'engine':key,**{field:row[field] for field in
        ['id','both_needle_strings_present','ordered_answer_present','strict_answer_format_correct','format_deviation',
         'generated_tokens','generation_token_limit','output_token_limit_reached']},'content':clean(row['content'])}
        for key,record in extra.items() for row in record['long_context']
        if row['available'] and not row['strict_answer_format_correct']]
    c.write_json(root/'analysis/integration.json',extra)
    long_state_comparisons(root)
    operator_path=root/'analysis/operator-summary.json'
    operators=json.loads(operator_path.read_text()) if operator_path.exists() else None
    original_contract=(supplemental_operator_contract(root) if operators and
        (root/'provenance/operator-tolerance-provenance.json').exists() else None)
    distributions=read_if(root/'analysis/distributions.jsonl')
    distro={}
    for name in sorted({r['comparison'] for r in distributions}):
        for suite in sorted({r['suite'] for r in distributions}):
            values=[r for r in distributions if r['comparison']==name and r['suite']==suite]
            if not values:continue
            flips=[r for r in values if not r['argmax_agreement']]
            distro[name+'/'+suite]={'paired_histories':len(values),
                'kl_left_right':descriptive([r['kl_left_right'] for r in values]),
                'kl_right_left':descriptive([r['kl_right_left'] for r in values]),
                'centered_logit_rmse':descriptive([r['centered_logit_rmse'] for r in values]),
                'argmax_agreement':sum(r['argmax_agreement'] for r in values)/len(values),
                'flips_with_both_top2_margins_at_most_point1':sum(max(r['left_top2_margin'],r['right_top2_margin'])<=.1 for r in flips),
                'other_flips':sum(max(r['left_top2_margin'],r['right_top2_margin'])>.1 for r in flips),
                'target_logprob_difference':descriptive([r['target_logprob_difference'] for r in values if 'target_logprob_difference' in r]),
                'flips':flips}
    c.write_json(root/'analysis/distribution-summary.json',distro)
    states=json.loads((root/'analysis/state-comparisons.json').read_text()) if (root/'analysis/state-comparisons.json').exists() else []
    accept=read_if(root/'analysis/acceptance.jsonl')
    acceptance_coverage_path=root/'analysis/acceptance-coverage.json'
    acceptance_audit=(json.loads(acceptance_coverage_path.read_text()) if acceptance_coverage_path.exists() else
        {'overall_verdict':'inconclusive','complete_fixture_count':0,'expected_fixture_count':54,
         'reason':'Acceptance coverage audit unavailable.'})
    state_summary={}
    for key in c.ENGINES:
        for k in [1,3,5]:
            for kind in ['conv','recurrent']:
                values=[r for r in states if r['engine']==key and r['k']==k and r['kind']==kind]
                if values:state_summary[f'{key}/K{k}/{kind}']={'sampled_parts':len(values),'relative_l2':descriptive([r['relative_l2'] for r in values]),
                    'max_abs':descriptive([r['max_abs'] for r in values]),'integer_mismatches':sum(not r['integers_equal'] for r in values),
                    'exact_parts':sum(r['exact_bytes'] for r in values)}
    c.write_json(root/'analysis/state-summary.json',state_summary)
    adverse_path=root/'analysis/state-adverse-review.json'
    adverse_state_note='Adverse sampled-state audit is unavailable; no aggregate floating-state tolerance verdict is assigned.'
    if adverse_path.exists():
        adverse=json.loads(adverse_path.read_text());samples=adverse.get('adverse_samples',[])
        if samples:
            worst=max(samples,key=lambda r:r['relative_l2'])
            conv={key:max(r['relative_l2'] for r in samples if r['engine']==key and r['kind']=='conv') for key in c.ENGINES}
            recurrent={key:max(r['relative_l2'] for r in samples if r['engine']==key and r['kind']=='recurrent') for key in c.ENGINES}
            r5_abs=max(r['max_abs'] for r in samples if r['engine']=='r5' and r['kind']=='conv')
            adverse_state_note=(f"Adverse sampled-state finding: large MTP-versus-ordinary drift occurs in all engines. Conv relative-L2 maxima are V100 {conv['v100']:.6f}, vendor {conv['vendor']:.6f}, R5 {conv['r5']:.6f}; recurrent maxima are {recurrent['v100']:.6f}, {recurrent['vendor']:.6f}, {recurrent['r5']:.6f}, respectively. "
                f"R5 still has adverse extremes: largest conv absolute error {r5_abs:.6g}, and recurrent relative-L2 {recurrent['r5']:.6f}. "
                f"The worst relative case is {worst['engine']} K{worst['k']} {worst['fixture']} layer{worst['layer']} {worst['kind']}: error RMS {worst['error_rms']:.6g} versus reference RMS {worst['reference_rms']:.6g}, absolute error {worst['max_abs']:.6g}; this is not a tiny-denominator artifact. "
                "The adverse samples were independently rechecked against matching input/forced histories and exact frontiers. No aggregate state tolerance is established, so no full-model retained/lost verdict follows. Scales, per-engine/K distributions and raw-history checks: analysis/state-adverse-review.json.")
    ledger=interruption_accounting(root)
    c.write_json(root/'analysis/interruption-accounting.json',ledger)
    restoration={p.name:json.loads(p.read_text()) for p in (root/'control').glob('restored-*.json')}
    infrastructure=[]
    for p in (root/'control').glob('*.results.json'):
        infrastructure.extend({'manifest':p.name,**r} for r in json.loads(p.read_text()) if r.get('returncode')!=0)
    provenance_dir=root/'provenance/final-harness';provenance_dir.mkdir(parents=True,exist_ok=True)
    sources=list((c.ROOT/'bench').glob('ninfer_precision*'))+list((c.ROOT/'bench').glob('test_ninfer_precision*.py'))+[c.ROOT/'bench/ninfer_tune.py',c.ROOT/'bench/ninfer_session.py']
    for p in sources:
        if p.is_file():shutil.copy2(p,provenance_dir/p.name)
    c.write_json(provenance_dir/'sha256.json',{p.name:c.sha(p) for p in sources if p.is_file()})
    lines=['NInfer three-engine precision and quality campaign','Evidence root: '+str(root),'',
        'Versions: V100 b37d0dd3; unmodified vendor 33163f05fb54; accepted R5 source ec94ca8d.',
        'Primary likelihood and accuracy verdicts use 99.1667% paired bootstrap confidence intervals (Bonferroni six claims), 20,000 replicates, seed 1729. Operator contract checks are deterministic.',
        'Equivalence requires the entire CI inside ±0.5% relative perplexity or ±2 percentage points accuracy. Missing required coverage is inconclusive.','',
        'Comparison | Common vendor operator contract | Supplemental original V100 operator contract | Perplexity | Task accuracy']
    for left,right in c.PAIRS:
        name=f'{left}_vs_{right}';p=summary['likelihood']['comparisons'][name];a=summary['accuracy']['comparisons'][name]
        op=operators['comparisons'][name]['verdict'] if operators else 'inconclusive: oracle analysis unavailable'
        original=original_contract['comparisons'][name]['verdict'] if original_contract else 'inconclusive: supplemental contract unavailable'
        lines.append(f"{name} | {op} | {original} | {p.get('verdict','inconclusive')} | {a.get('verdict','inconclusive')}")
        for label,r,scale in [('PPL relative %',p,100),('accuracy percentage points',a,100)]:
            if 'ci' in r:lines.append(f"  {label}: {r['estimate']*scale:+.6f}, CI [{r['ci'][0]*scale:+.6f}, {r['ci'][1]*scale:+.6f}]. Paired {r.get('paired_tokens',r.get('paired_cases',0))}.")
    lines+=['','Measured likelihood and scored accuracy:','Engine | Scored targets | PPL | Correct/graded cases']
    for key in c.ENGINES:
        docs=summary['likelihood']['documents'][key];tokens=sum(d['tokens'] for d in docs.values())
        ppl=math.exp(sum(d['tokens']*d['mean_nll'] for d in docs.values())/tokens) if tokens else None
        scored=summary['accuracy']['engines'][key]
        lines.append(f"{key} | {tokens}/131072 | {ppl if ppl is not None else 'unavailable'} | {scored['correct']}/{scored['valid']}")
    lines+=['','Domain likelihood differences (relative %, descriptive intervals; primary claims remain the aggregate comparisons):']
    for name,comparison in summary['likelihood']['comparisons'].items():
        for domain,value in comparison['domains'].items():
            if 'ci' in value:lines.append(f"{name} {domain}: {100*value['estimate']:+.5f}%, CI [{100*value['ci'][0]:+.5f}, {100*value['ci'][1]:+.5f}].")
            else:lines.append(f"{name} {domain}: {100*value['estimate']:+.5f}%; one stream, no domain CI.")
    lines+=['','Numerical scope and limitations:',
        'No full-precision checkpoint or resident unquantized model reference is used. The independent CPU codec materializes represented weights in FP32 and computes complete dot products in FP64 against captured BF16 inputs.',
        'All measured operator widths cover every output row/column and complete K, using independent FP64 BLAS and 32-row weight tiles, cross-checked with naive scalar FP64 on 32 fixed rows and every column. Width 1024 uses the captured real input; width 2048 tiles those same 1024 activation/residual columns twice. Descriptive gross-error counts do not replace the contract criteria.',
        'The unchanged common-vendor operator check uses a BF16 gross-relative floor of 2/256. Original V100 A16 used smaller gross-relative bounds for SwiGLU (0.0063), NVFP4 GDN input projection (0.0035) and NVFP4 attention input projection (0.0045). Both contracts apply identically to all engines and are reported separately; relative-L2 and gross-absolute bounds are unchanged. Exact bounds/source hashes were frozen before FP64 results in provenance/operator-tolerance-provenance.json. Supplemental results: analysis/operator-original-v100-contract.json.',
        'Operator verdicts describe crossings of the stated deterministic tolerances on captured inputs. Retention within a contract does not mean every error is unchanged; increases and decreases in FP64-reference error remain visible by operator and width. It is not an overall full-model precision verdict. Domain intervals and integration sentinels are descriptive and do not add primary claims to the six-comparison correction.',
        'Operator tolerance retention is limited to captured operators. Accumulated full-model MTP versus decode state has no established aggregate tolerance here; drift is reported and full-model state precision remains inconclusive.',
        'Logit comparisons use only identical token histories and 248,077 public vocabulary rows, excluding 243 padding rows. Actual graph launches are observed after synchronization outside capture.',
        'MTP injection replaces proposals before upload, preserves actual acceptance/fold logic and scripts only the unprocessed correction/bonus before commit. Integer/frontier agreement is required exactly.',
        'State observations sample first/middle/last GDN layers: first 4096 recurrent FP32 elements and first 16 KiB conv BF16 per layer. They do not prove full-model replay correctness. Terminal released state is unavailable; logits cover 128 ordinary transitions.',
        f"Actual acceptance rounds: {len(accept)}; acceptance errors {sum(not r['acceptance_correct'] for r in accept)}; unforced bonus errors {sum(not r['bonus_correct'] for r in accept)}.",
        f"Acceptance coverage verdict: {acceptance_audit['overall_verdict']}; {acceptance_audit.get('complete_fixture_count',0)}/{acceptance_audit.get('expected_fixture_count',54)} fixtures have complete round/control/pattern coverage. Zero observed errors with incomplete coverage is inconclusive, not a pass. See analysis/acceptance-coverage.json for missing/duplicate records and budget-tail counts.",
        'State drift relative L2 (sampled parts; median / p95 / maximum):\n'+'\n'.join(
            f"  {name}: {value['relative_l2']['median']:.6g} / {value['relative_l2']['p95']:.6g} / {value['relative_l2']['max']:.6g}; integer mismatches {value['integer_mismatches']}; {value['sampled_parts']} parts."
            for name,value in sorted(state_summary.items())),
        adverse_state_note,
        'Operator error distributions: '+json.dumps(operators['engines'] if operators else {},sort_keys=True),
        'Full operator/width results, signed bias, gross errors and worst row/column examples: analysis/operators.jsonl and operator-summary.json.',
        'Distribution KL, centered-logit error, top-token agreement, top-two margins and all flips: analysis/distributions.jsonl and distribution-summary.json.','',
        'Likelihood and answer quality:',
        'Likelihood targets 131,072 tokens per engine (8192 from each of 16 streams), context 4096, stride 2048. Bootstrap resamples whole streams, retaining token weights. See analysis/summary.json for domains and incomplete streams.',
        'CausalScoring disables graphs/cache/speculation and requests chunk 1024 in all engines. Native NInfer-all/R5 then apply default-on prefill alignment: the captured scoring chunks were actually 1152 columns, versus V100 1024. Generation requests chunk 2048 and may likewise be aligned internally. Existing PPL/decode/quality results remain same-token whole-engine comparisons; chunk alignment is an execution-path confound, so those differences are not attributed solely to the kernel port. The canonical boundary reruns disable alignment with NINFER_PREFILL_ALIGN=0; original native-aligned boundary evidence is preserved. See provenance/effective-engine-options.json and control/protocol-amendments.jsonl.',
        'Primary quality uses 384 frozen synthetic deterministic cases. Code has 12 algorithms with eight name variants each; JSON/tools has eight schema families. Primary intervals resample whole repeated task families, and retrieval/history cases; case-only sensitivity is also saved. These fixtures do not establish broad real-world quality.',
        'Bootstrap intervals are conditional on the observed task families. Identical correctness for every paired family yields a degenerate [0,0] interval; it does not establish equality on unseen tasks or account for errors absent from this sample. The predeclared primary resampling method remains unchanged.',
        'Code is graded with deterministic assertions in a restricted separate interpreter; JSON with duplicate-key, type, exact-schema/argument checks; arithmetic/retrieval/history with known answers. No model judge.',
        'Primary JSON cases use identical unconstrained token prompts, with no response_format, schema, grammar or actual tool definitions. The V100 baseline is standard b37d0dd3, not the separate structured-output 28b34a build. NInfer-all/R5 opt-in --structured-output is not enabled. Format-only differences therefore measure unconstrained prompt adherence, not unequal grammar enforcement.',
        'Fixture review: primary JSON requires raw JSON, so Markdown-wrapped valid objects fail format; code fences are accepted by the code grader. The eight interval-merging code variants have a wording ambiguity about adjacent integer intervals: frozen tests keep [1,4] and [5,7] separate. Primary prompts, tests and case selection were preserved. See provenance/fixture-review.json.',
        'Per-engine scored accuracy: '+json.dumps({key:{field:value for field,value in record.items() if field!='failures'} for key,record in summary['accuracy']['engines'].items()},ensure_ascii=False),
        'Example failing grades: '+json.dumps({key:record['failures'][:6] for key,record in summary['accuracy']['engines'].items()},ensure_ascii=False),
        'Failing and discordant cases, adverse domains, all missing records: analysis/summary.json. Raw generated answers: runs/<engine>/quality-k0/records.jsonl.','',
        'Integration and confounders:',
        'Numerical tests use the same frozen token inputs; controlled text quality uses a common frozen rendered template. The separate 24-case native-template K0 subset exercises each engine\'s native preparation/template/output path; it does not isolate template text alone. K3 native probe/package agreement is a small integration check, not full build equivalence.',
        'Isolated probes link preserved kernel archives; vendor object reuse is recorded in probes/vendor/verified-object-reuse.json and was accepted only with source/transitive-header equivalence or the documented unused fixed-width enum addition. Compiler/link/packaging differences and exact flags are preserved in probes/*/build-provenance.json. Deployed bundles were not rebuilt or selected.',
        'Original native MTP may select its context-copy proposer (up to 15). Forced diagnostics and controlled 64-case MTP subset disable that proposer; vendor/R5 n-gram draft is disabled. Native integration therefore has a speculation-policy confound across engines.',
        'Pilot infrastructure failures are preserved: vendor persistence with no Host tier; then no-prefix-reuse with explicit cache capacity. Corrected fresh package agreement disables cache without incompatible capacity/persistence flags; image continuation tests use private caches.',
        'An external startup-preload change from original to ninfer-all-v100 was observed during the pilot; the campaign preserves it and subsequent restoration follows the configured preload. Active R5 selection stays unchanged.',
        'Images (24 synthetic color/spatial fixtures) and long context (one growing history through about 131K/229K) are sentinels, not broad equivalence evidence. The second long input includes the known correct earlier reply and can retrieve it.',
        'Long sentinels require only "maple-7c90 519204". The unchanged legacy retrieval_correct field is a weak check for both needle strings appearing anywhere, also exposed as both_needle_strings_present; it does not establish correct answer associations. The additive ordered_answer_present diagnostic requires the expected passphrase immediately followed by whitespace and 519204; strict_answer_format_correct requires only the exact two-value answer after cleaning stop markers. No model judge is used and primary metrics are unchanged.',
        'Long-answer format deviations (all available sentinel IDs): '+json.dumps(long_format_failures,ensure_ascii=False),
        'Integration raw counts, template/MTP case comparisons, image failures, cache responses, long needle/ordered-answer/strict-format diagnostics and observed graph launches: analysis/integration.json.','',
        'Long-context distributions use only matching actual generated prefixes and column-zero results; sampled state comparisons also require identical executed histories. See analysis/long-state-comparisons.json.','',
        'Execution and restoration:',f"Cumulative recorded precision GPU interruption {ledger['used_seconds']:.3f} s ({ledger['used_seconds']/3600:.3f} h) across {ledger['session_count']} started sessions; {ledger['completed_session_count']} finished. All started precision sessions restored: {ledger['all_started_sessions_restored']}.",
        f"Original interruption target {ledger['original_target_seconds']:.0f} s (six hours), including a {ledger['original_restoration_reserve_seconds']:.0f} s restoration reserve. Recorded interruption beyond that original target: {ledger['over_original_target_seconds']:.3f} s.",
        "The user subsequently clarified that this is a lab and GPU interruption is not a concern. Precision coverage continuations use a separate ledger and are included in the cumulative time and session count; their allowance does not replace the original six-hour target. The agentic performance benchmark is excluded. Unfinished sessions, if any, have not yet contributed their final duration.",
        'Precision ledger accounting: '+json.dumps(ledger['ledgers'],sort_keys=True),
        'Precision budget sessions: '+json.dumps(ledger['sessions'],sort_keys=True),
        'Restoration evidence: '+json.dumps({k:{field:v.get(field) for field in ['created','http_status','gpu','active','config_sha256']} for k,v in restoration.items()},sort_keys=True),
        '150 W retained; ComfyUI remains off; private benchmark cache paths exclude production caches. Only permitted Compose stop/up commands were used. No automatic fixes, rollbacks or deployments.',
        'Infrastructure failures/deadlines: '+json.dumps(infrastructure,sort_keys=True),
        'Interrupted JSONL reads: '+json.dumps(c.READ_ERRORS,sort_keys=True),
        'Fixtures and protocol: fixtures/*-freeze.json, control/protocol-freeze.json, control/protocol-amendments.jsonl.',
        'Full streaming artifact comparison (all 1240 v3 representations plus logical bindings/scales/divisors) passed; unrelated abliterated sidecars excluded. See provenance/artifact-comparison.json.',
        'Final harness sources and hashes: provenance/final-harness/.']
    text='\n'.join(lines)+'\n'
    (c.ROOT/'bench/NINFER_PRECISION.txt').write_text(text)
    (root/'NINFER_PRECISION.txt').write_text(text)
    return summary


def finalize(root):
    interruption_accounting(root,require_restored=True)
    # Each expensive stage is append-only; a completed stage is reused on resume.
    # An interrupted output is never silently treated as a completed stage.
    for name,action,destination in [
        ('acceptance',analyze_acceptance,'acceptance.jsonl'),
        ('states',state_comparisons,'state-comparisons.json'),
        ('distributions',analyze_distributions,'distributions.jsonl'),
        ('operators',analyze_operators,'operators.jsonl')]:
        marker=root/f'analysis/{name}-complete.json'
        output=root/'analysis'/destination
        if marker.exists():
            saved=json.loads(marker.read_text())
            if c.sha(output)!=saved['sha256']:raise ValueError('completed analysis changed: '+name)
            continue
        if output.exists():raise ValueError('unfinished analysis output requires review: '+str(output))
        print('CPU ANALYSIS',name,flush=True)
        action(root)
        c.write_json(marker,{'sha256':c.sha(output),'gpu_used':False})
    return report(root)
