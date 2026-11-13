"""Deterministic graders; generated code executes with a restricted AST/builtins.

The worker has no file, process, network, reflection, unrestricted-import or dynamic-code API.
It runs in a separate interpreter with a wall-time and address-space bound.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

SAFE_BUILTINS = ('abs','all','any','bool','dict','enumerate','filter','float','frozenset',
                 'int','isinstance','len','list','map','max','min','next','pow','range',
                 'reversed','round','set','slice','sorted','str','sum','tuple','zip',
                 'ValueError','TypeError','IndexError','KeyError','StopIteration')
SAFE_METHODS = {'append','extend','pop','insert','remove','discard','add','update','get',
                'items','keys','values','sort','reverse','count','index','lower','upper',
                'casefold','strip','lstrip','rstrip','split','join','replace','isalnum',
                'isalpha','isdigit','startswith','endswith','copy','clear','setdefault','fromkeys',
                'gcd','lcm','sqrt','isqrt','ceil','floor','factorial','prod','comb','perm','inf',
                'Counter','defaultdict','OrderedDict','deque','popleft','appendleft','heappush','heappop',
                'heapify','nsmallest','nlargest','bisect','bisect_left','bisect_right','insort',
                'reduce','cache','lru_cache','accumulate','groupby','chain','combinations','permutations',
                'List','Dict','Set','Tuple','Optional','Any','Iterable','Sequence','Deque','Union',
                # Pure regex operations support ordinary palindrome implementations.
                # This does not expose Python's builtin compile/eval/exec functions.
                'sub','compile','IGNORECASE','I'}
SAFE_MODULES = {'math','collections','heapq','itertools','functools','bisect','typing','re'}


def typed_equal(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(typed_equal(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b


def json_equal(a,b):
    # JSON Schema's integer accepts a number with zero fractional part; bool is
    # still a distinct schema type and must never compare equal to numeric one.
    if type(b) is int and type(a) in (int,float): return a==b
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(json_equal(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(json_equal(x,y) for x,y in zip(a,b))
    return a==b


def clean(text):
    text=text.strip()
    for suffix in ('<|im_end|>','<|endoftext|>','<|fim_suffix|>'):
        if text.endswith(suffix): text=text[:-len(suffix)].rstrip()
    return text


def code_worker(case,text):
    import builtins
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(3,3))
    resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    if text.startswith('```'):
        match=re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```',text,re.S)
        if not match: return {'correct':False,'reason':'invalid code fence'}
        text=match[1]
    tree=ast.parse(text)
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.Assign,ast.AnnAssign,ast.Expr,ast.Import,ast.ImportFrom)):
            return {'correct':False,'reason':'unsupported top-level statement'}
        if isinstance(node,ast.Expr) and not isinstance(node.value,ast.Constant):
            return {'correct':False,'reason':'top-level side effect'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith)):
            return {'correct':False,'reason':'code requires an unavailable capability'}
        if isinstance(node,ast.Import) and any(alias.name not in SAFE_MODULES for alias in node.names):
            return {'correct':False,'reason':'unavailable module'}
        if isinstance(node,ast.ImportFrom) and (node.module not in SAFE_MODULES or node.level or
                any(alias.name.startswith('_') or alias.name not in SAFE_METHODS for alias in node.names)):
            return {'correct':False,'reason':'unavailable module member'}
        # Python's builtin compile is absent from exposed builtins; allowing its
        # identifier lets `from re import compile` bind only the safe regex API.
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in ('eval','exec','open','input','globals','locals','vars','getattr','setattr','delattr','breakpoint','help','type','object','super')):
            return {'correct':False,'reason':'forbidden runtime capability'}
        if isinstance(node,ast.Attribute) and node.attr not in SAFE_METHODS:
            return {'correct':False,'reason':'unavailable attribute: '+node.attr}
    def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name not in SAFE_MODULES or level: raise ImportError('unavailable module')
        return builtins.__import__(name,globals,locals,fromlist,level)
    exposed={k:getattr(builtins,k) for k in SAFE_BUILTINS}
    exposed['__import__']=safe_import
    namespace={'__builtins__':exposed}
    exec(compile(tree,'<model-code>','exec'),namespace)
    function=namespace.get(case['function'])
    if not callable(function): return {'correct':False,'reason':'missing required function'}
    for i,(args,expected) in enumerate(case['expected']):
        got=function(*args)
        if not typed_equal(got,expected):
            return {'correct':False,'reason':f'test {i} mismatch','got':repr(got)[:300]}
    return {'correct':True,'reason':'all deterministic tests passed'}


def grade(case,text):
    text=clean(text)
    if case['grader']=='json':
        try:
            def no_duplicates(pairs):
                out={}
                for k,v in pairs:
                    if k in out: raise ValueError('duplicate key')
                    out[k]=v
                return out
            got=json.loads(text,object_pairs_hook=no_duplicates)
            return {'correct':json_equal(got,case['expected']),'reason':'typed schema and exact arguments'}
        except (ValueError,TypeError): return {'correct':False,'reason':'invalid or duplicate-key JSON'}
    if case['grader']=='exact':
        return {'correct':text==case['expected'],'reason':'exact known answer'}
    if case['grader']=='code':
        try:
            result=subprocess.run([sys.executable,'-I','-S',str(Path(__file__).resolve()),'--worker'],
                input=json.dumps({'case':case,'text':text}),text=True,capture_output=True,timeout=5)
            if result.returncode: return {'correct':False,'reason':'code worker failed','error':result.stderr[-500:]}
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired: return {'correct':False,'reason':'code execution timed out'}
    raise ValueError('unknown grader')


if __name__=='__main__':
    try:
        data=json.load(sys.stdin)
        print(json.dumps(code_worker(data['case'],data['text'])))
    except BaseException as e:
        print(json.dumps({'correct':False,'reason':type(e).__name__+': '+str(e)[:400]}))
