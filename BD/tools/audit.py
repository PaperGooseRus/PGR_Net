#!/usr/bin/env python3
"""Проверить каталог, пары и наблюдения BD; ошибки прогонов показать отдельно."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def need(condition,text):
    if not condition:raise ValueError(text)

def audit(root,strict=False):
    root=Path(root).resolve();rows=read(root/'catalog/projects.jsonl')
    catalog={r['project_id']:r for r in rows}
    need(len(catalog)==len(rows),'Duplicate project_id')
    pairs=defaultdict(list)
    for r in rows:
        pairs[r['pair_id']].append(r)
        path=(root/r['relative_path']).resolve()
        need(path.is_relative_to(root/'projects') and path.is_dir(),'Invalid project path')
        need((path/'CMakeLists.txt').is_file(),'Missing CMakeLists.txt')
        need(bool(r['sources']),'Missing source list')
        for f in r['sources']:
            source=(path/f).resolve()
            need(source.is_relative_to(path) and source.is_file(),'Missing/invalid source')
    for rs in pairs.values():
        need(len(rs)==2 and {r['expected_ok'] for r in rs}=={True,False},'Invalid pair')
        need(len({r['split'] for r in rs})==1,'Pair split leakage')
    for split in ('train','val','test'):
        actual=read(root/'splits'/(split+'.jsonl'))
        ids=[r['project_id'] for r in actual]
        expected={r['project_id'] for r in rows if r['split']==split}
        need(len(ids)==len(set(ids)) and set(ids)==expected,'Invalid split '+split)
    observations={}
    for compiler in ('gcc','clang'):
        p=root/'observations'/compiler/'observations.jsonl'
        if p.is_file():
            obs=read(p);seen=set()
            for r in obs:
                need(r['project_id'] in catalog and r['project_id'] not in seen,'Invalid observation id')
                seen.add(r['project_id'])
            observations[compiler]={'rows':len(obs),'mismatches':sum(not r['match'] for r in obs),'missing_projects':len(set(catalog)-seen)}
    repairs=root/'repairs/repair_pairs.jsonl'
    if repairs.is_file():need(len(read(repairs))==len(pairs),'Invalid repairs count')
    result={'structure':'passed','projects':len(rows),'pairs':len(pairs),
            'splits':dict(Counter(r['split'] for r in rows)),
            'fault_families':len({r['fault_code'] for r in rows if not r['expected_ok']}),
            'observations':observations}
    folder=root/'reports/validation';folder.mkdir(parents=True,exist_ok=True)
    (folder/'audit_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    bad=sum(r['mismatches']+r['missing_projects'] for r in observations.values())
    if bad:print('NOTE: collection problems exist; inspect observations and collection reports. No labels were changed.')
    return 1 if strict and bad else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--strict-observations',action='store_true')
    args=p.parse_args()
    try:raise SystemExit(audit(args.root,args.strict_observations))
    except (OSError,ValueError,KeyError) as exc:raise SystemExit('AUDIT ERROR: '+str(exc))
