#!/usr/bin/env python3
"""Выдача только подтверждённых реальных случаев в datasets/real/{train,val,test}.jsonl.

Не меняет каталог, исходные наблюдения и синтетические выборки.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from quality_gate import read_jsonl, write_atomic, write_report
from ingest_real import select_split, SPLITS
from audit_real import audit

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path):
    root = root.resolve()
    # Выпуск обучающих данных запрещён при нарушениях истории импорта.
    audit(root)
    cat = root / 'catalog/real_cases.jsonl'
    obs = root / 'observations/real/observations.jsonl'
    if not cat.exists() or not obs.exists():
        raise ValueError('No imported real data; run ingest first')
    rows = list(read_jsonl(cat))
    observations = {r['case_id']: r for r in read_jsonl(obs)}
    if len(observations) != len(rows) or set(observations) != {r['case_id'] for r in rows}:
        raise ValueError('Unmatched observations/catalog records')
    output = {split: [] for split in SPLITS}
    group_splits = {}
    input_splits = {}
    source_splits = {}
    summary = Counter()
    for meta in rows:
        cid = meta['case_id']
        split = meta['split']
        group = meta['source_group_hash']
        if split not in SPLITS or select_split(group) != split:
            raise ValueError('Invalid split for ' + cid)
        if group in group_splits and group_splits[group] != split:
            raise ValueError('Group leakage')
        group_splits[group] = split
        for key, seen in (('input_fingerprint', input_splits), ('source_fingerprint', source_splits)):
            fingerprint = meta.get(key)
            if fingerprint:
                if fingerprint in seen and seen[fingerprint] != split:
                    raise ValueError('Cross-split duplicate')
                seen[fingerprint] = split
        if meta['status'] != 'ready':
            summary['quarantine'] += 1
            continue
        label = meta.get('label')
        if not label:
            raise ValueError('Ready sample without label')
        inp = observations[cid]['input']
        if any(x in inp for x in ('fault_code', 'stage', 'split', 'label', 'pair_id', 'match')):
            raise ValueError('Target leakage in model input')
        output[split].append({
            'id': cid, 'group_id': group, 'input': inp,
            'target': {'stage': label['stage'], 'fault_code': label['fault_code'],
                       'build_ok': label['stage'] == 'success'},
        })
        summary['ready'] += 1
    for split, records in output.items():
        write_atomic(root / 'datasets/real' / (split + '.jsonl'), sorted(records, key=lambda x: x['id']))
    result = {'source': 'user_import; reviewed labels only', 'total': len(rows),
              'ready': summary['ready'], 'quarantine': summary['quarantine'],
              'splits': {split: len(output[split]) for split in SPLITS},
              'classes': dict(sorted(Counter(x['target']['fault_code'] for a in output.values() for x in a).items()))}
    write_report(root / 'reports/validation/real_dataset.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit('REAL PREPARE ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    main()
