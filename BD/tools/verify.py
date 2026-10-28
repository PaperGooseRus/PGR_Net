#!/usr/bin/env python3
"""Compile/configure/run PGR synthetic fixtures and emit ground-truth observations.

No pip dependencies. Operates strictly inside the given data root and a temporary build
area. Never executes unknown code from outside the synthetic corpus. The included
fixtures are deliberately safe, yet only run them in a disposable test environment.
"""
from __future__ import annotations
import argparse, concurrent.futures, json, os, re, shutil, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

def rows(root):
    with (root/'catalog'/'projects.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def scrub(s,root,scratch,pid):
    s=s or ''
    s=s.replace(str(scratch),'<BUILD_DIR>').replace(str(root),'<DATA_ROOT>')
    # Do not leak the project ID into training text (IDs encode no labels, but are nuisance features).
    return s.replace(pid,'<PROJECT>').replace('\\','/')[:24000].strip()

def call(cmd,cwd,timeout):
    try:
        kw={'cwd':cwd,'capture_output':True,'text':True,'errors':'replace','timeout':timeout,'stdin':subprocess.DEVNULL}
        if os.name=='posix':
            def safe_limits():
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
                    resource.setrlimit(resource.RLIMIT_CPU,(min(timeout+1,30),min(timeout+2,31)))
                except Exception:pass
            kw['preexec_fn']=safe_limits
        p=subprocess.run(cmd,**kw)
        return p.returncode,p.stdout or '',p.stderr or ''
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b'').decode('utf-8','replace') if isinstance(e.stdout,bytes) else (e.stdout or ''),\
                    'TIMEOUT: '+str(timeout)+' seconds'
    except OSError as e:
        return 125,'','PROCESS_ERROR: '+str(e)

def one(root,row,compiler,bin_c,bin_cpp,cmake,ninja):
    path=root/row['relative_path']
    pid=row['project_id']
    actual_sources=[path/x for x in row['sources']]
    assert path.is_dir() and path.resolve().is_relative_to((root/'projects').resolve())
    assert all(p.exists() and p.resolve().is_relative_to(path.resolve()) for p in actual_sources)
    name=bin_c if row['language']=='c' else bin_cpp
    t0=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='pgrds_') as d:
        d=Path(d)
        observed='success'; out=''; err=''; rc=0; stage_cmd=''
        if row['expected_stage']=='configure_error' or (row['expected_stage']=='success' and row['build_system']=='cmake'):
            cmd=[cmake,'-S',str(path),'-B',str(d/'build')]
            if ninja:cmd+=['-G','Ninja']
            cmd+=['-DCMAKE_C_COMPILER='+bin_c,'-DCMAKE_CXX_COMPILER='+bin_cpp]
            rc,out,err=call(cmd,path,20);stage_cmd='cmake configure'
            if rc!=0:observed='configure_error'
            else:
                rc,out2,err2=call([cmake,'--build',str(d/'build'),'--parallel','1'],path,20)
                out+='\n'+out2;err+='\n'+err2;stage_cmd='cmake build'
                if rc!=0:observed='compile_error'
        else:
            binpath=d/('case.exe' if os.name=='nt' else 'case')
            cmd=[name,'-std=c11' if row['language']=='c' else '-std=c++17',
                 '-Wall','-Wextra','-Werror','-pedantic','-O0',
                 *map(str,actual_sources),'-o',str(binpath)]
            if row['defect_mode']=='library':cmd+=['-lpgr_missing_'+pid[:8]]
            rc,out,err=call(cmd,path,15);stage_cmd='compile+link'
            if rc!=0:
                # Determine the phase from the actual diagnostic, not the expected label.
                low=(out+'\n'+err).lower()
                observed='link_error' if any(tok in low for tok in (
                    'undefined reference','multiple definition','cannot find -l',
                    'ld returned','undefined symbol','duplicate symbol',
                    'unresolved external symbol','lnk2019','lnk2005','collect2: error',
                    'symbol(s) not found','undefined symbols for architecture')) else 'compile_error'
            else:
                rc,runout,runerr=call([str(binpath)],path,4)
                out+='\n'+runout;err+='\n'+runerr
                stage_cmd='run'
                if rc!=0:observed='runtime_error'
        result={'project_id':pid,'pair_id':row['pair_id'],'split':row['split'],'language':row['language'],
          'fault_code':row['fault_code'],'expected_stage':row['expected_stage'],
          'observed_stage':observed,'match':observed==row['expected_stage'],
          'exit_code':rc,'compiler':compiler,'command_phase':stage_cmd,
          'diagnostic':scrub(err,root,d,pid),'stdout':scrub(out,root,d,pid),
          'relative_path':row['relative_path'],'elapsed_ms':int((time.monotonic()-t0)*1000)}
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--compiler',choices=['gcc','clang'],default='gcc')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--sample-per-family',type=int,default=0,help='0 = all fixtures')
    p.add_argument('--split',choices=['train','val','test','all'],default='all')
    p.add_argument('--strict',action='store_true',help='nonzero exit code on any unexpected result')
    p.add_argument('--output-dir',type=Path,help='separate collection directory; default is observations/<compiler>')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='allow replacing an existing observation file')
    args=p.parse_args()
    root=args.root.resolve();catalog=list(rows(root))
    if args.split!='all':catalog=[r for r in catalog if r['split']==args.split]
    if args.sample_per_family:
        grouped=defaultdict(list)
        for r in catalog:grouped[r['fault_code'] if not r['expected_ok'] else 'CLEAN'].append(r)
        # Include matching clean side of each defective pair (not only one generic clean case).
        keep=[]
        bads=[r for r in catalog if not r['expected_ok']]
        grouped_bad=defaultdict(list)
        for r in bads:grouped_bad[r['fault_code']].append(r)
        selected_pairs={r['pair_id'] for a in grouped_bad.values() for r in sorted(a,key=lambda r:r['project_id'])[:args.sample_per_family]}
        catalog=[r for r in catalog if r['pair_id'] in selected_pairs]
    binaries=(('gcc','g++') if args.compiler=='gcc' else ('clang','clang++'))
    missing=[x for x in binaries+('cmake',) if not shutil.which(x)]
    if missing:raise SystemExit('Missing tools: '+', '.join(missing))
    cmake=shutil.which('cmake');ninja=shutil.which('ninja')
    dest=(args.output_dir or root/'observations'/args.compiler).resolve()
    if (dest/'observations.jsonl').exists() and not args.overwrite:
        raise SystemExit('Observations already exist: choose --output-dir or use explicit --overwrite')
    dest.mkdir(parents=True,exist_ok=True)
    results=[];t0=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(args.jobs,12))) as pool:
        futures={pool.submit(one,root,r,args.compiler,*binaries,cmake,ninja):r for r in catalog}
        for ix,fut in enumerate(concurrent.futures.as_completed(futures),1):
            r=futures[fut]
            try:res=fut.result()
            except Exception as exc:
                res={'project_id':r['project_id'],'pair_id':r['pair_id'],'split':r['split'],
                'fault_code':r['fault_code'],'expected_stage':r['expected_stage'],
                'observed_stage':'runner_error','match':False,'diagnostic':str(exc),'compiler':args.compiler}
            results.append(res)
            if not args.quiet and (ix%250==0 or ix==len(catalog)):
                print(f'{ix}/{len(catalog)} checked, mismatches={sum(not x["match"] for x in results)}, elapsed={time.monotonic()-t0:.1f}s',flush=True)
    results.sort(key=lambda x:x['project_id'])
    obs=dest/'observations.jsonl'
    with obs.open('w',encoding='utf-8',newline='\n') as h:
        for r in results:h.write(json.dumps(r,ensure_ascii=False)+'\n')
    failures=[r for r in results if not r['match']]
    report={'total':len(results),'matched':len(results)-len(failures),'mismatches':len(failures),
      'compiler':args.compiler,'seconds':round(time.monotonic()-t0,2),
      'observed_stages':dict(Counter(r['observed_stage'] for r in results)),
      'mismatch_examples':failures[:200]}
    report_path=(root/'reports'/'collection'/f'{args.compiler}_report.json') if args.output_dir is None else dest/'report.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='mismatch_examples'},ensure_ascii=False))
    print('RESULTS:',obs)
    if failures:
        print('MISMATCHES (first 12):',[(x['fault_code'],x['project_id'],x['observed_stage'],x.get('diagnostic','')[:130]) for x in failures[:12]])
    if args.strict and failures:sys.exit(1)

if __name__=='__main__':main()
