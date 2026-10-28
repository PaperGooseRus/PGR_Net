#!/usr/bin/env python3
"""Независимая проверка целостности пользовательских импортов и разделения данных."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from quality_gate import read_jsonl, write_report
from ingest_real import digest, select_split

ROOT = Path(__file__).resolve().parents[1]


def audit(root: Path):
    root = root.resolve()
    cat = root / 'catalog/real_cases.jsonl'
    obs = root / 'observations/real/observations.jsonl'
    if not cat.exists() or not obs.exists():
        return {'status': 'empty', 'cases': 0, 'ready': 0, 'quarantine': 0}
    catalog = list(read_jsonl(cat))
    obs_rows = list(read_jsonl(obs))
    meta = {row['case_id']: row for row in catalog}
    evidence = {row['case_id']: row for row in obs_rows}
    if len(meta) != len(catalog) or len(evidence) != len(obs_rows) or set(meta) != set(evidence):
        raise ValueError('Mismatched/duplicate catalog and observations')
    group_splits = {}
    input_splits = {}
    source_splits = {}
    count = Counter()
    taxonomy = {t['code']: t for t in json.loads((root/'catalog/fault_taxonomy.json').read_text(encoding='utf-8'))}
    for cid, row in meta.items():
        inp = evidence[cid]['input']
        input_hash = digest(inp)
        group = row['source_group_hash']
        if not isinstance(group, str) or row['input_fingerprint'] != input_hash:
            raise ValueError('Input changed without review: ' + cid)
        if cid != 'r_' + digest([group, input_hash])[:20]:
            raise ValueError('Invalid case id: ' + cid)
        split = row['split']
        if split != select_split(group):
            raise ValueError('Unstable split: ' + cid)
        fp_source = digest(['sources', inp['files']]) if inp.get('files') else None
        if fp_source != row.get('source_fingerprint'):
            raise ValueError('Source files were changed: ' + cid)
        for key, value, index in [('group', group, group_splits),
                                  ('input', input_hash, input_splits),
                                  ('sources', fp_source, source_splits)]:
            if value and value in index and index[value] != split:
                raise ValueError('Cross-split leakage by ' + key)
            if value:
                index[value] = split
        label = row.get('label')
        status = row.get('status')
        if status not in ('ready', 'quarantine'):
            raise ValueError('Invalid review status')
        if status == 'ready' and not label:
            raise ValueError('Missing reviewed label')
        if label:
            code, stage = label['fault_code'], label['stage']
            if stage == 'success':
                if code != 'CLEAN' or inp['build_exit_code'] != 0:
                    raise ValueError('Invalid success label')
            else:
                if (code not in taxonomy or taxonomy[code]['stage'] != stage
                        or taxonomy[code]['language'] != inp['language']
                        or inp['build_exit_code'] == 0):
                    raise ValueError('Invalid taxonomy label')
        if row.get('compiler') != inp.get('compiler') or row.get('language') != inp.get('language'):
            raise ValueError('Metadata/input mismatch')
        count[status] += 1
    return {'status': 'passed', 'cases': len(catalog),
            'ready': count['ready'], 'quarantine': count['quarantine'],
            'splits': dict(sorted(Counter(x['split'] for x in catalog).items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        report = audit(args.root)
        write_report(args.root/'reports/validation/real_audit.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit('REAL AUDIT ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    main()
