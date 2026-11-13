"""CPU-only isolated probe builds. Reuse frozen archives, never a CMake cache.

The vendor has no public C++ symbols in its distributed shared object, so that
probe is rebuilt in a fresh directory. Its package remains the quality baseline.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import ninfer_precision as campaign

HERE = Path(__file__).resolve().parent
DEPS = Path('/home/simon/ninfer-build-deps/usr')
CUDA = Path('/usr/local/cuda-12.9')
ARCHIVES = {
    'v100': campaign.DONOR / 'archive/build-v100',
    'r5': campaign.ENGINE / 'builds/ported-33163f0-r1',
}


def execute(command, cwd, env, log):
    campaign.append(log.with_suffix('.commands.jsonl'), {'argv':command,'cwd':str(cwd)})
    print('BUILD', shlex.join(command)[:400], flush=True)
    with log.open('a') as out:
        subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)


def stop_vendor_build(root):
    pattern='^cmake --build '+str(root/'probes/vendor/build')+' --target ninfer-precision-probe'
    result=subprocess.run(['pgrep','-f',pattern],capture_output=True,text=True)
    import signal
    for value in result.stdout.split():
        os.kill(int(value),signal.SIGINT)
        print('Interrupted isolated vendor build',value,flush=True)


def augment_sources(source,key):
    if key=='v100':
        base=source/'src/targets/qwen3_6/impl/runtime'
        prefill=base/'program_impl.h';commit=prefill
        decode=base/'decode_impl.h';mtp=base/'mtp_impl.h'
        width='verify_k';pool='state_images->linear()'
    else:
        base=source/'src/models/qwen3_5/program'
        prefill=base/'prefill.cpp';commit=base/'transactions/commit.cpp'
        decode=base/'decode.cpp';mtp=base/'speculative/mtp.cpp'
        width='verify_window';pool='state_images->linear(0)'
    if key=='v100':
        for path in [prefill,decode,mtp]:
            before=path.read_text()
            text=before.replace('state.execution.model.resources().public_token_count','TextConfig::token_domain').replace('model.resources().public_token_count','TextConfig::token_domain')
            if text!=before:path.write_text(text)
        text=prefill.read_text()
        if 'NINFER_PRECISION_DISABLE_LOOKUP' not in text:
            needle='    const std::uint32_t verify_k =\n        use_lookup ? qwen3_6::kMtpLookupMaximumDrafts : draft_window;'
            if text.count(needle)!=1: raise ValueError('original context-copy control point changed')
            text=text.replace(needle,'    if (std::getenv("NINFER_PRECISION_FORCED") || std::getenv("NINFER_PRECISION_DISABLE_LOOKUP")) use_lookup = false;\n'+needle)
            prefill.write_text(text)
    text=prefill.read_text()
    needle='validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));'
    if 'ninfer_precision::force_prefill' not in text:
        if text.count(needle)!=1: raise ValueError('prefill forcing insertion point changed')
        text=text.replace(needle,'ninfer_precision::force_prefill(host_tokens);\n    '+needle)
        if 'ninfer_precision_observer.h' not in text: text=f'#include "{HERE / "ninfer_precision_observer.h"}"\n'+text
        prefill.write_text(text)
    text=decode.read_text()
    if 'ninfer_precision::force_ordinary' not in text:
        needle='executable != nullptr);'
        text=text.replace(needle,needle+'\n    ninfer_precision::force_ordinary(state);')
        decode.write_text(text)
    text=mtp.read_text()
    if 'ninfer_precision::proposals' not in text:
        needle='    run_prepared(state, executable, body);'
        text=text.replace(needle,f'    ninfer_precision::proposals(state, {width});\n'+needle)
        needle='executable != nullptr);'
        text=text.replace(needle,needle+f'\n    ninfer_precision::force_bonus(state, {width});')
        mtp.write_text(text)
    text=commit.read_text()
    if 'if (ninfer_precision::forcing())' in text:
        text=text.replace('if (ninfer_precision::forcing())','if (ninfer_precision::observing_state())')
        commit.write_text(text)
    if 'ninfer_precision::state(' not in text:
        needle='        timing.resume_post();\n        pending_transaction_.reset();'
        if text.count(needle)!=1: raise ValueError('post-fold insertion point changed')
        inserted='''
        if (ninfer_precision::observing_state()) {
            device.synchronize();
            for (std::size_t row=0; row<row_count; ++row) {
                if (terminal[row] || cancelled[row]) continue;
                auto& observed=active_sequence(lanes[row]);
                ninfer_precision::state(POOL, state_selectors(observed).source,
                    observed.execution_frontier, observed.ledger_frontier);
            }
        }
'''.replace('POOL',pool)
        text=text.replace(needle,needle+inserted)
        if 'ninfer_precision_observer.h' not in text: text=f'#include "{HERE / "ninfer_precision_observer.h"}"\n'+text
        commit.write_text(text)
    for path,field in [(decode,'cache_positions'),(mtp,'base_frontiers')]:
        text=path.read_text()
        needle='executable != nullptr);'
        if needle in text:
            text=text.replace(needle,f'executable != nullptr, state.host_ingress.{field}[0]);')
            path.write_text(text)
    operators=[]
    for relative,family,weight,output in [('linear/linear.cpp','linear','w','out'),
            ('wrapper/linear_add.cpp','linear_add','w','residual_out'),
            ('wrapper/linear_swiglu.cpp','linear_swiglu','gate_up_weight','out')]:
        path=source/'src/ops'/relative
        text=path.read_text()
        if 'ninfer_precision::OperatorCapture' not in text:
            pattern=r'(void '+family+r'\(const Tensor& x, const Weight& '+weight+r', Tensor& '+output+r', LinearPolicy policy,\s*WorkspaceArena& \w+, cudaStream_t stream\) \{)'
            if len(re.findall(pattern,text))!=1: raise ValueError('operator insertion point changed: '+relative)
            cap=family+'_workspace_capacity_bytes'
            body=f'''
    ninfer_precision::OperatorCapture precision_operator(x, {weight}, {output}, stream, "{family}", int(policy),
        [{weight}, policy](int width) {{return {cap}({weight}.qtype, {weight}.n, {weight}.k, policy, width, width);}},
        [{weight}, policy, stream](const Tensor& input, Tensor& result, WorkspaceArena& scratch) {{
            {family}(input, {weight}, result, policy, scratch, stream);
        }});
'''
            text=re.sub(pattern,lambda m:m[0]+body,text)
            text=f'#include "{HERE / "ninfer_precision_ops.h"}"\n'+text
            path.write_text(text)
        text=path.read_text()
        if 'NINFER_PRECISION_JSON_HEADER' not in text:
            path.write_text(f'#define NINFER_PRECISION_JSON_HEADER "{source / "third_party/nlohmann/json.hpp"}"\n'+text)
        operators.append(path)
    for family,weight in [('gdn_input_proj','query_key_value_z_weight'),('attn_input_proj','query_key_gate_value_weight')]:
        path=source/'src/ops/wrapper'/f'{family}.cpp'
        text=path.read_text()
        if 'ninfer_precision::OperatorCapture' not in text:
            pattern=r'(void '+family+r'\(const Tensor& x, const Weight& '+weight+r', Tensor& (?:qkv|q),[^{]+?LinearPolicy policy,\s*WorkspaceArena& workspace, cudaStream_t stream\) \{)'
            if len(re.findall(pattern,text))!=1: raise ValueError('projection insertion point changed: '+family)
            if family=='gdn_input_proj':
                replay='''
            DeviceBuffer first(std::size_t(10240)*input.ne[1]*2),second(std::size_t(6144)*input.ne[1]*2);
            Tensor qkv(first.p,DType::BF16,{10240,int(input.ne[1])}),z(second.p,DType::BF16,{6144,int(input.ne[1])});
            gdn_input_proj(input, WEIGHT, qkv, z, policy, scratch, stream);
            ninfer_precision::pack_part(qkv,result,0,stream);
            ninfer_precision::pack_part(z,result,10240,stream);
'''
            else:
                replay='''
            DeviceBuffer first(std::size_t(6144)*input.ne[1]*2),second(std::size_t(1024)*input.ne[1]*2),third(std::size_t(6144)*input.ne[1]*2),fourth(std::size_t(1024)*input.ne[1]*2);
            Tensor q(first.p,DType::BF16,{6144,int(input.ne[1])}),key(second.p,DType::BF16,{1024,int(input.ne[1])}),gate(third.p,DType::BF16,{6144,int(input.ne[1])}),value(fourth.p,DType::BF16,{1024,int(input.ne[1])});
            attn_input_proj(input, WEIGHT, q, gate, key, value, policy, scratch, stream);
            ninfer_precision::pack_part(q,result,0,stream);
            ninfer_precision::pack_part(key,result,6144,stream);
            ninfer_precision::pack_part(gate,result,7168,stream);
            ninfer_precision::pack_part(value,result,13312,stream);
'''
            body=f'''
    Tensor precision_output(x.data,DType::BF16,{{{weight}.n,int(x.ne[1])}});
    ninfer_precision::OperatorCapture precision_operator(x, {weight}, precision_output, stream, "{family}", int(policy),
        [{weight},policy](int width) {{return {family}_workspace_capacity_bytes({weight}.qtype,{weight}.n,{weight}.k,policy,width,width);}},
        [{weight},policy,stream](const Tensor& input, Tensor& result, WorkspaceArena& scratch) {{{replay.replace('WEIGHT',weight)}
        }});
'''
            text=re.sub(pattern,lambda m:m[0]+body,text)
            path.write_text(f'#include "{HERE / "ninfer_precision_ops.h"}"\n'+text)
        text=path.read_text()
        if 'NINFER_PRECISION_JSON_HEADER' not in text:
            path.write_text(f'#define NINFER_PRECISION_JSON_HEADER "{source / "third_party/nlohmann/json.hpp"}"\n'+text)
        operators.append(path)
    if key=='v100': return [source/'src/targets/qwen3_6_27b/impl/variant.cpp',*operators]
    return [*[base/p for p in ['program_impl.cpp','decode.cpp','speculative/mtp.cpp','prefill.cpp','transactions/commit.cpp']],*operators]


def reuse_vendor_objects(root):
    """Reuse only objects whose complete recorded dependencies and flags match.

    A new configure is retained. No historical CMake cache or dependency record is
    transplanted. A phony edge makes the verified frozen object an explicit input.
    """
    old=ARCHIVES['r5'];new=root/'probes/vendor/build'
    source=root/'probes/vendor/source'
    ninja=(new/'build.ninja').read_text()
    oldninja=(old/'build.ninja').read_text()
    dependencies=subprocess.check_output(['ninja','-C',str(old),'-t','deps'],text=True)
    deps={}
    current=None
    for line in dependencies.splitlines():
        if line and not line.startswith(' '):
            current=line.split(': #deps ')[0]
            deps[current]=[]
        elif current and line.startswith('    '): deps[current].append(line.strip())
    def blocks(text):
        return {m[1]:m for m in re.finditer(r'^build ([^\n:]+\.o): (?:CXX|CUDA)_COMPILER[^\n]*\n(?:  [^\n]*\n)+',text,re.M)}
    originals=blocks(oldninja)
    def options(block):
        values=[]
        for key in ['DEFINES','FLAGS','INCLUDES']:
            match=re.search(r'^  '+key+r' = (.*)$',block,re.M)
            values.append(match[1] if match else '')
        return '\n'.join(values)
    def normalize(text):
        return text.replace(str(campaign.ENGINES['r5']['source']),'<SOURCE>').replace(str(source),'<SOURCE>').replace(str(old),'<BUILD>').replace(str(new),'<BUILD>')
    cache={}
    def digest(path):
        if path not in cache:
            cache[path]=campaign.sha(path) if path.is_file() else None
        return cache[path]
    records=[];replacements=[]
    enum_addition=b'    // Device-only donor QPN permutation; 4 is already RowSplitPanel.\n    VoltaQpnPrepacked = 7,\n'
    for target,match in blocks(ninja).items():
        if (new/target).exists() or target not in originals or not (old/target).is_file(): continue
        if not deps.get(target): continue
        old_options=normalize(options(originals[target][0]));new_options=normalize(options(match[0]))
        compression_normalization=('-compress-mode=speed' in old_options and '-compress-mode=size' in new_options)
        if compression_normalization:new_options=new_options.replace('-compress-mode=size','-compress-mode=speed')
        cutlass_include=str(campaign.DONOR/'archive/build-v100/_deps/cutlass-src/include')
        include_normalization=False
        if old_options!=new_options and not any(d.startswith(cutlass_include+'/') for d in deps[target]):
            old_options=old_options.replace(' -isystem '+cutlass_include,'')
            include_normalization=True
        if old_options!=new_options: continue
        checked=[]
        enum_normalizations=[]
        for dependency in deps[target]:
            left=Path(dependency) if dependency.startswith('/') else old/dependency
            right=Path(str(left).replace(str(campaign.ENGINES['r5']['source']),str(source)).replace(str(old),str(new)))
            if digest(left) is None: break
            if digest(left)!=digest(right):
                if (str(left).endswith('/src/core/weight.h') and right.is_file() and
                        left.read_bytes().count(enum_addition)==1 and
                        left.read_bytes().replace(enum_addition,b'')==right.read_bytes()):
                    enum_normalizations.append({'old':str(left),'new':str(right),'old_sha256':digest(left),
                        'new_sha256':digest(right),'difference':enum_addition.decode(),
                        'justification':'Additional unused QuantLayout enumerator; fixed uint16_t underlying type, unchanged existing values and Weight layout.'})
                else: break
            checked.append({'old':str(left),'new':str(right),'sha256':digest(left)})
        else:
            destination=new/target
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old/target,destination)
            records.append({'target':target,'source_object':str(old/target),'sha256':digest(old/target),
                            'flags':normalize(options(match[0])),'dependencies':checked,
                            'unused_include_removed':include_normalization,'enum_normalizations':enum_normalizations,
                            'fatbinary_compression_only_difference':compression_normalization})
            replacements.append((match.start(),match.end(),f'build {target}: phony {old/target}\n'))
    for start,end,replacement in reversed(replacements): ninja=ninja[:start]+replacement+ninja[end:]
    (new/'build.ninja').write_text(ninja)
    recordfile=root/'probes/vendor/verified-object-reuse.json'
    previous=json.loads(recordfile.read_text()) if recordfile.exists() else []
    campaign.write_json(recordfile,previous+records)
    print('Reused',len(records),'additional objects; exact differences and flags recorded',flush=True)


def isolate_reused_objects(root):
    path=root/'probes/vendor/verified-object-reuse.json'
    if not path.exists():return
    for record in json.loads(path.read_text()):
        source=Path(record['source_object']);destination=root/'probes/vendor/build'/record['target']
        if destination.exists() and os.path.samefile(source,destination):
            temporary=destination.with_name(destination.name+'.independent')
            shutil.copy2(source,temporary)
            os.replace(temporary,destination)


def patch_sources(source: Path, key: str):
    observer = str(HERE / 'ninfer_precision_observer.h')
    if key == 'v100':
        base = source / 'src/targets/qwen3_6/impl/runtime'
        program = base / 'program_impl.h'
        decode = base / 'decode_impl.h'
        mtp = base / 'mtp_impl.h'
        public = 'TextConfig::token_domain'
        score_public = 'TextConfig::token_domain'
        compile_files = [source/'src/targets/qwen3_6_27b/impl/variant.cpp']
    else:
        base = source / 'src/models/qwen3_5/program'
        program = base / 'program_impl.cpp'
        decode = base / 'decode.cpp'
        mtp = base / 'speculative/mtp.cpp'
        public = 'state.execution.parameters.model.resources().public_token_count'
        score_public = 'parameters.model.resources().public_token_count'
        compile_files = [program,decode,mtp]
    text = program.read_text()
    needle = '            const auto* host = static_cast<const float*>(score_logprobs_host->data());'
    if text.count(needle) != 1: raise ValueError('score observer insertion point changed')
    text = text.replace(needle, f'            ninfer_precision::logits(logits, device.stream, "score", {score_public});\n'+needle)
    program.write_text(f'#include "{observer}"\n'+text)
    for file, tensor, phase in [(decode,'state.frame.logits.slice(1, 0, batch_size)','decode'),
                                (mtp,'state.frame.target_logits.slice(2, 0, batch_size)','verify')]:
        text=file.read_text()
        needle='    run_prepared(state, executable, body);'
        if text.count(needle) != 1: raise ValueError('graph observer insertion point changed')
        text=text.replace(needle,needle+f'\n    ninfer_precision::logits({tensor}, state.execution.device.stream, "{phase}", {public}, executable != nullptr);')
        file.write_text(f'#include "{observer}"\n'+text)
    return compile_files


def build(root: Path, key: str):
    dest = root/'probes'/key
    dest.mkdir(parents=True,exist_ok=True)
    source=dest/'source'
    if not source.exists():
        shutil.copytree(campaign.ENGINES[key]['source'],source,
                        ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
        patched=patch_sources(source,key)
        campaign.write_json(dest/'observer-patch.json',{'files':[{ 'path':str(p.relative_to(source)), 'sha256':campaign.sha(p)} for p in patched],
                                                      'observer_header_sha256':campaign.sha(HERE/'ninfer_precision_observer.h')})
    else:
        patched=([source/'src/targets/qwen3_6_27b/impl/variant.cpp'] if key=='v100' else
                 [source/'src/models/qwen3_5/program'/p for p in ['program_impl.cpp','decode.cpp','speculative/mtp.cpp']])
    patched=augment_sources(source,key)
    env=dict(os.environ)
    env['PKG_CONFIG_PATH']=str(DEPS/'lib/x86_64-linux-gnu/pkgconfig')
    env['LD_LIBRARY_PATH']=':'.join([str(DEPS/'lib/x86_64-linux-gnu'),str(CUDA/'lib64'),'/usr/lib/x86_64-linux-gnu'])
    env['NINFER_BUILD_ID']='precision-'+key+'-isolated'
    log=dest/'build.log'
    if key=='vendor':
        isolate_reused_objects(root)
        cmake=source/'apps/CMakeLists.txt'
        text=cmake.read_text()
        if 'ninfer-precision-probe' not in text:
            text += f'\nadd_executable(ninfer-precision-probe "{HERE / "ninfer_precision_probe.cpp"}")\ntarget_link_libraries(ninfer-precision-probe PRIVATE ninfer_engine ninfer_product_logging ninfer::json)\nninfer_internal_includes(ninfer-precision-probe)\n'
            cmake.write_text(text)
        builddir=dest/'build'
        cutlass=campaign.DONOR/'archive/build-v100/_deps/cutlass-src'
        configure=['cmake','-S',str(source),'-B',str(builddir),'-G','Ninja',
                   '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=/usr/bin/gcc-13',
                   '-DCMAKE_CXX_COMPILER=/usr/bin/g++-13',f'-DCMAKE_CUDA_COMPILER={CUDA}/bin/nvcc',
                   '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13','-DCMAKE_CUDA_ARCHITECTURES=70',
                   '-DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/15',
                   '-DNINFER_BUILD_APPS=ON','-DBUILD_TESTING=OFF','-DNINFER_BUILD_BENCHMARKS=OFF',
                   '-DNINFER_NVCC_SPLIT_COMPILE=1','-DNINFER_CUDA_COMPRESSION_MODE=speed',
                   '-DNINFER_SHARED_RUNTIME=OFF','-DNINFER_MULTICALL=OFF',
                   '-DNINFER_SFU_SIGMOID_SILU=OFF','-DNINFER_SFU_SOFTPLUS=OFF','-DNINFER_BF16_RESIDUAL_ADD=OFF',
                   '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',f'-DFETCHCONTENT_SOURCE_DIR_CUTLASS={cutlass}',
                   f'-DNINFER_V100_CUTLASS_SOURCE={cutlass}']
        if not (builddir/'CMakeCache.txt').exists(): execute(configure,dest,env,log)
        if not (dest/'verified-object-reuse.json').exists(): reuse_vendor_objects(root)
        execute(['cmake','--build',str(builddir),'--target','ninfer-precision-probe','--parallel','2'],dest,env,log)
        binary=builddir/'apps/ninfer-precision-probe'
    else:
        archive=ARCHIVES[key]
        ninja=(archive/'build.ninja').read_text()
        objects=[]
        for i, path in enumerate(patched):
            relative=str(path.relative_to(source))
            # Extract flags from the actual original compile rule; rewrite relocated paths.
            pattern=r'^build [^\n]+: CXX_COMPILER[^\n]* '+re.escape('/'+relative)+r'(?: [^\n]*)?\n((?:  [^\n]*\n)+)'
            matches=[]
            for m in re.finditer(r'^build [^\n]+: CXX_COMPILER[^\n]*\n((?:  [^\n]*\n)+)',ninja,re.M):
                if ('/'+relative+' ') in m[0].split('\n')[0] or m[0].split('\n')[0].endswith('/'+relative): matches.append(m)
            if not matches: raise ValueError(f'compile rule absent: {relative}')
            block=matches[0][1]
            options=[]
            for field in ['DEFINES','FLAGS','INCLUDES']:
                match=re.search(r'^  '+field+r' = (.*)$',block,re.M)
                if match: options.extend(shlex.split(match[1]))
            options=[x.replace('/work/src',str(source)).replace(str(campaign.ENGINES[key]['source']),str(source)) for x in options]
            if key=='v100': options.append('-DPRECISION_V100=1')
            obj=dest/f'observer-{i}.o'
            command=['/usr/bin/g++-13',*options,'-c',str(path),'-o',str(obj)]
            inputs=[*patched,HERE/'ninfer_precision_observer.h',HERE/'ninfer_precision_ops.h']
            if key=='v100': inputs += [source/'src/targets/qwen3_6/impl/runtime'/p for p in ['program_impl.h','decode_impl.h','mtp_impl.h']]
            previous_commands=campaign.rows(log.with_suffix('.commands.jsonl')) if log.with_suffix('.commands.jsonl').exists() else []
            same_command=any(row['argv']==command for row in previous_commands)
            if not (obj.exists() and same_command and obj.stat().st_mtime>=max(p.stat().st_mtime for p in inputs)):
                execute(command,dest,env,log)
            else: print('REUSE isolated observer object',obj.name,flush=True)
            objects.append(str(obj))
        includes=['-I'+str(source/p) for p in ['include','src','third_party','third_party/utf8proc']]
        if key=='v100': includes+=['-DPRECISION_V100=1','-DNINFER_VOLTA_BUILD=1']
        else: includes+=['-DPRECISION_R5=1','-DNINFER_SM8X_COMPAT=1','-DNINFER_TIER_VOLTA=1']
        obj=dest/'probe.o'
        execute(['/usr/bin/g++-13','-O3','-DNDEBUG','-std=c++20',*includes,'-c',str(HERE/'ninfer_precision_probe.cpp'),'-o',str(obj)],dest,env,log)
        block=re.search(r'^build apps/ninfer-serve:[^\n]*\n((?:  [^\n]*\n)+)',ninja,re.M)[1]
        libs=shlex.split(re.search(r'^  LINK_LIBRARIES = (.*)$',block,re.M)[1])
        libs=[str(archive/x) if not x.startswith(('/','-')) else x for x in libs]
        # Host extracted development libraries replace old container-only paths.
        for i,x in enumerate(libs):
            if x.startswith('/usr/lib/x86_64-linux-gnu/lib') and Path(x).name.startswith(('libav','libsw','libcurl')):
                libs[i]=str(DEPS/'lib/x86_64-linux-gnu'/Path(x).name)
        binary=dest/'ninfer-precision-probe'
        execute(['/usr/bin/g++-13','-O3','-L/usr/lib/gcc/x86_64-linux-gnu/15',
                 '-L'+str(CUDA/'lib64'),'-L'+str(CUDA/'lib64/stubs'),
                 '-o',str(binary),str(obj),*objects,*libs],dest,env,log)
    execute([str(binary),'--help'],dest,env,log)
    campaign.write_json(dest/'build-provenance.json',{'engine':key,'binary':str(binary),'binary_sha256':campaign.sha(binary),
        'source_revision':campaign.ENGINES[key]['revision'],'compiler':subprocess.check_output(['/usr/bin/g++-13','--version'],text=True),
        'method':'fresh vendor configure/build with recorded equivalent object reuse' if key=='vendor' else 'relinked existing kernel archives plus isolated observer translation units',
        'object_reuse_record':str(dest/'verified-object-reuse.json') if key=='vendor' else None,
        'confounder':'Host GCC 13 probe and test observers differ from packaged executable; package agreement must be checked. Vendor object reuse records exact dependencies, an unused fixed-width enum addition, unused include paths and binary-compression differences.',
        'observer_sources':{str(p.relative_to(source)):campaign.sha(p) for p in patched},
        'observer_header_sha256':campaign.sha(HERE/'ninfer_precision_observer.h'),
        'operator_header_sha256':campaign.sha(HERE/'ninfer_precision_ops.h'),
        'probe_main_sha256':campaign.sha(HERE/'ninfer_precision_probe.cpp'),
        'production_modified':False})
    print('PROBE READY',key,binary,flush=True)


def build_wide(root):
    for key in campaign.ENGINES:
        dest=root/'probes'/key;source=dest/'source';obj=dest/'wide.o';binary=dest/'ninfer-precision-wide'
        includes=['-I'+str(source/p) for p in ['include','src','third_party']]+['-I'+str(CUDA/'include')]
        if key=='v100':includes+=['-DPRECISION_V100=1','-DNINFER_VOLTA_BUILD=1']
        elif key=='r5':includes+=['-DPRECISION_R5=1','-DNINFER_SM8X_COMPAT=1','-DNINFER_TIER_VOLTA=1']
        else:includes+=['-DNINFER_SM8X_COMPAT=1','-DNINFER_TIER_VOLTA=1']
        env=dict(os.environ,LD_LIBRARY_PATH=':'.join([str(DEPS/'lib/x86_64-linux-gnu'),str(CUDA/'lib64'),'/usr/lib/x86_64-linux-gnu']))
        log=dest/'wide-build.log'
        execute(['/usr/bin/g++-13','-O3','-std=c++20',*includes,'-c',str(HERE/'ninfer_precision_wide.cpp'),'-o',str(obj)],dest,env,log)
        if key=='vendor':
            cwd=dest/'build'
            commands=subprocess.check_output(['ninja','-C',str(cwd),'-t','commands','apps/ninfer-precision-probe'],text=True).splitlines()
            line=next(line for line in reversed(commands) if ' -o apps/ninfer-precision-probe ' in line)
            command=shlex.split(line.removeprefix(': && ').removesuffix(' && :'))
            command=[x for x in command if not x.startswith('-Wl,--dependency-file=')]
            command=[str(obj) if x.endswith('ninfer_precision_probe.cpp.o') else str(binary) if x=='apps/ninfer-precision-probe' else x for x in command]
        else:
            cwd=dest
            previous=campaign.rows(dest/'build.commands.jsonl')
            command=next(row['argv'] for row in reversed(previous) if '-o' in row['argv'] and str(dest/'ninfer-precision-probe') in row['argv'])
            command=[str(obj) if x==str(dest/'probe.o') else str(binary) if x==str(dest/'ninfer-precision-probe') else x for x in command]
        execute(command,cwd,env,log)
        execute([str(binary),'--help'],dest,env,log)
        campaign.write_json(dest/'wide-build-provenance.json',{'binary':str(binary),'binary_sha256':campaign.sha(binary),
            'source_sha256':campaign.sha(HERE/'ninfer_precision_wide.cpp'),'gpu_used_for_build':False,
            'qualification':'Before each wide replay, reconstruct the frozen Op and require byte-exact output agreement at widths 1 and 128.'})
