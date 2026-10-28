"""Единый источник путей структурированного корпуса BD v2."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('projects','catalog','splits','repairs','observations','datasets','reports/collection','reports/validation','manifests')

def create_layout(root):
    root=Path(root)
    for name in DIRECTORIES:
        (root/name).mkdir(parents=True,exist_ok=True)
    (root/'layout.json').write_text(json.dumps({'layout_version':2,'corpus_contract':'PGR synthetic v1','paths_version':'BD v2'},indent=2)+'\n',encoding='utf-8')

def catalog(root): return Path(root)/'catalog/projects.jsonl'
def observation(root,compiler): return Path(root)/'observations'/compiler/'observations.jsonl'
def collection_report(root,compiler): return Path(root)/'reports/collection'/(compiler+'_report.json')
def training(root,compiler,split): return Path(root)/'datasets'/compiler/(split+'.jsonl')
