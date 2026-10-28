#!/usr/bin/env python3
"""Build paired repair examples as unified diffs, preserving train/val/test splits."""
from __future__ import annotations
import argparse, difflib, json
from collections import defaultdict
from pathlib import Path

def main(root:Path):
    root=root.resolve()
    groups=defaultdict(dict)
    for line in (root/'catalog'/'projects.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);groups[r['pair_id']]['bad' if not r['expected_ok'] else 'good']=r
    dest=root/'repairs'/'repair_pairs.jsonl'
    count=0
    with dest.open('w',encoding='utf-8',newline='\n') as h:
        for key, pair in sorted(groups.items()):
            assert set(pair)=={'bad','good'},(key,pair)
            b,g=pair['bad'],pair['good'];assert b['split']==g['split']
            before=root/b['relative_path'];after=root/g['relative_path']
            paths=sorted({p.relative_to(before).as_posix() for p in before.iterdir() if p.is_file()} |
                {p.relative_to(after).as_posix() for p in after.iterdir() if p.is_file()})
            diffs=[]
            for path in paths:
                bp=before/path;gp=after/path
                old=bp.read_text(encoding='utf-8').splitlines(keepends=True) if bp.exists() else []
                new=gp.read_text(encoding='utf-8').splitlines(keepends=True) if gp.exists() else []
                diffs.extend(difflib.unified_diff(old,new,fromfile='faulty/'+path if bp.exists() else '/dev/null',
                            tofile='corrected/'+path if gp.exists() else '/dev/null'))
            row={'pair_id':key,'split':b['split'],'fault_code':b['fault_code'],
                 'broken_project_id':b['project_id'],'fixed_project_id':g['project_id'],
                 'patch':''.join(diffs)}
            if not row['patch']:
                raise AssertionError('Unexpected identical projects: '+key)
            h.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
    print('Repair pairs:',count,'->',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    main(p.parse_args().root)
