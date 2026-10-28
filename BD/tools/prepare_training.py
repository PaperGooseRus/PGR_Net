#!/usr/bin/env python3
"""Export observed C/C++ projects to independent JSONL train/val/test datasets."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--include-sources',action='store_true',help='embed source files (recommended)')
    p.add_argument('--compiler',choices=['gcc','clang','all'],default='all')
    args=p.parse_args();root=args.root.resolve()
    catalog={r['project_id']:r for line in (root/'catalog'/'projects.jsonl').read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
    out=root/'datasets';out.mkdir(exist_ok=True)
    total=0
    for compiler in ('gcc','clang') if args.compiler=='all' else (args.compiler,):
        source=root/'observations'/compiler/'observations.jsonl'
        if not source.is_file():
            print('SKIP missing observations:',source);continue
        bysplit={s:[] for s in ('train','val','test')}
        for line in source.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if not r['match']:continue
            meta=catalog[r['project_id']]
            inp={'language':meta['language'],'compiler':compiler,
                 'build_system':meta['build_system'],'build_exit_code':r['exit_code'],
                 'command_phase':r['command_phase'],'diagnostic':r['diagnostic']}
            if args.include_sources:
                d=root/meta['relative_path']
                inp['files']={f.name:f.read_text(encoding='utf-8') for f in sorted(d.iterdir()) if f.is_file()}
            sample={'id':r['project_id'],'pair_id':r['pair_id'],'input':inp,
                    'target':{'stage':meta['expected_stage'],'fault_code':meta['fault_code'],
                             'build_ok':meta['expected_ok']}}
            bysplit[r['split']].append(sample)
        for split,samples in bysplit.items():
            dest=out/compiler/f'{split}.jsonl'
            dest.parent.mkdir(parents=True,exist_ok=True)
            samples.sort(key=lambda x:x['id'])
            with dest.open('w',encoding='utf-8',newline='\n') as f:
                for x in samples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
            total+=len(samples)
            print(dest.name,len(samples))
    print('Total observations:',total)

if __name__=='__main__':main()
