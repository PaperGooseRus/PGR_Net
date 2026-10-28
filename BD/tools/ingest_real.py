#!/usr/bin/env python3
"""Приём вручную проверенных диагностик в существующую структуру BD v2.

Не запускает исходники и компиляторы. Не угадывает разметку.
Вход проверяется целиком до изменения каталогов BD.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import sys

from quality_gate import read_jsonl, write_atomic, STAGES

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ('c', 'cpp')
COMPILERS = ('gcc', 'clang')
SPLITS = ('train', 'val', 'test')
PRIVATE = [
    re.compile(r'(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+'),
    re.compile(r'(?i)/home/[^/\s]+'),
    re.compile(r'(?i)/Users/[^/\s]+'),
    re.compile(r'(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b'),
    re.compile(r'(?i)\b(?:api[_-]?key|password|authorization|secret[_-]?key)\s*[:=]'),
]


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def check_private(text):
    return any(pattern.search(text) for pattern in PRIVATE)


def group_hash(name):
    return digest(['PGR-BD-real-group-v1', name])


def select_split(group_id):
    # Одна семья проекта всегда остаётся в одном split вне зависимости от компилятора.
    value = int(digest(['PGR-BD-real-split-v1', group_id])[:16], 16) % 10
    if value < 8:
        return 'train'
    if value == 8:
        return 'val'
    return 'test'


def validate_case(row, taxonomy):
    if not isinstance(row, dict):
        raise ValueError('Expected JSON object')
    name = row.get('source_group')
    if not isinstance(name, str) or not (1 <= len(name) <= 256):
        raise ValueError('source_group must identify a project family (1-256 chars)')
    if row.get('language') not in LANGUAGES or row.get('compiler') not in COMPILERS:
        raise ValueError('Only c/cpp with gcc/clang are supported')
    if row.get('build_system') not in ('cmake', 'direct+cmake'):
        raise ValueError('Invalid build_system')
    if not isinstance(row.get('command_phase'), str) or not 1 <= len(row['command_phase']) <= 80:
        raise ValueError('Invalid command_phase')
    if type(row.get('exit_code')) is not int or not -1024 <= row['exit_code'] <= 65535:
        raise ValueError('Invalid exit_code')
    diagnostic = row.get('diagnostic')
    if not isinstance(diagnostic, str) or len(diagnostic) > 24000 or check_private(diagnostic):
        raise ValueError('Diagnostic too long or contains possible private data; anonymize manually')
    files = row.get('files', {})
    if not isinstance(files, dict) or len(files) > 32:
        raise ValueError('files must be a small object')
    file_size = 0
    for name, content in files.items():
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError('Invalid file entry')
        path = PurePosixPath(name)
        if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
                or len(name) > 180 or len(content) > 120000 or check_private(content)):
            raise ValueError('Unsafe file path/content; anonymize manually')
        file_size += len(content)
    if file_size > 256000 or (not diagnostic and not files):
        raise ValueError('No diagnostic/source, or excessive source size')
    if type(row.get('reviewed', False)) is not bool:
        raise ValueError('reviewed must be boolean')
    label = row.get('label')
    if label is not None:
        if not isinstance(label, dict):
            raise ValueError('label must be an object')
        stage = label.get('stage')
        code = label.get('fault_code')
        if stage not in STAGES or not isinstance(code, str):
            raise ValueError('Invalid stage/fault_code')
        if stage == 'success':
            if code != 'CLEAN' or row['exit_code'] != 0:
                raise ValueError('Success requires CLEAN and exit_code=0')
        else:
            if code not in taxonomy:
                raise ValueError('Unrecognized fault_code: submit for taxonomy review first')
            item = taxonomy[code]
            if item['stage'] != stage or item['language'] != row['language']:
                raise ValueError('fault_code does not match stage/language in taxonomy')
            if row['exit_code'] == 0:
                raise ValueError('Error stage requires nonzero exit_code')
    gid = group_hash(row['source_group'])
    features = {
        'language': row['language'],
        'compiler': row['compiler'],
        'build_system': row['build_system'],
        'build_exit_code': row['exit_code'],
        'command_phase': row['command_phase'],
        'diagnostic': diagnostic,
        'files': dict(sorted(files.items())),
    }
    fingerprint = digest(features)
    source_fingerprint = digest(['sources', files]) if files else None
    case_id = 'r_' + digest([gid, fingerprint])[:20]
    status = 'ready' if label is not None and row.get('reviewed', False) is True else 'quarantine'
    catalog = {
        'case_id': case_id, 'source_group_hash': gid, 'split': select_split(gid),
        'compiler': row['compiler'], 'language': row['language'],
        'status': status, 'label': label, 'origin': 'user_import',
        'input_fingerprint': fingerprint, 'source_fingerprint': source_fingerprint,
    }
    observation = {'case_id': case_id, 'input': features}
    return catalog, observation


def ingest(root: Path, source: Path, dry_run=False):
    root = root.resolve()
    taxonomy = {r['code']: r for r in json.loads((root / 'catalog/fault_taxonomy.json').read_text(encoding='utf-8'))}
    catalog_path = root / 'catalog/real_cases.jsonl'
    observation_path = root / 'observations/real/observations.jsonl'
    existing_catalog = list(read_jsonl(catalog_path)) if catalog_path.exists() else []
    existing_observations = list(read_jsonl(observation_path)) if observation_path.exists() else []
    catalog_by_id = {x['case_id']: x for x in existing_catalog}
    obs_by_id = {x['case_id']: x for x in existing_observations}
    if len(catalog_by_id) != len(existing_catalog) or set(catalog_by_id) != set(obs_by_id):
        raise ValueError('Inconsistent or duplicate real catalog/observations: repair before import')
    seen_group = {r['source_group_hash']: r['split'] for r in existing_catalog}
    # Проекты, содержащие точно те же исходники, не могут попасть в разные split.
    seen_source = {r['source_fingerprint']: r['split'] for r in existing_catalog if r.get('source_fingerprint')}
    seen_input = {r['input_fingerprint']: r['split'] for r in existing_catalog}
    additions = 0
    examples = list(read_jsonl(source))
    if not examples:
        raise ValueError('Empty import file')
    if len(examples) > 10000:
        raise ValueError('Import batch is too large')
    for index, row in enumerate(examples, 1):
        try:
            meta, obs = validate_case(row, taxonomy)
        except (ValueError, KeyError) as exc:
            raise ValueError(f'Input record {index}: {exc}') from exc
        cid = meta['case_id']
        if cid in catalog_by_id:
            if catalog_by_id[cid] != meta or obs_by_id[cid] != obs:
                raise ValueError(f'Conflicting existing case: {cid}; no silent overwrite')
            continue
        group = meta['source_group_hash']
        split = meta['split']
        if group in seen_group and seen_group[group] != split:
            raise ValueError('Group split mismatch')
        for fingerprint, seen in ((meta['input_fingerprint'], seen_input),
                                  (meta['source_fingerprint'], seen_source)):
            if fingerprint and fingerprint in seen and seen[fingerprint] != split:
                raise ValueError('Potential duplicate leakage across train/val/test')
        seen_group[group] = split
        seen_input[meta['input_fingerprint']] = split
        if meta['source_fingerprint']:
            seen_source[meta['source_fingerprint']] = split
        catalog_by_id[cid] = meta
        obs_by_id[cid] = obs
        additions += 1
    counts = {'input_records': len(examples), 'new': additions, 'total': len(catalog_by_id),
              'ready': sum(v['status'] == 'ready' for v in catalog_by_id.values()),
              'quarantine': sum(v['status'] == 'quarantine' for v in catalog_by_id.values()),
              'dry_run': dry_run}
    if not dry_run:
        write_atomic(catalog_path, [catalog_by_id[k] for k in sorted(catalog_by_id)])
        write_atomic(observation_path, [obs_by_id[k] for k in sorted(obs_by_id)])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path, required=True, help='User-reviewed JSONL evidence')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(ingest(args.root, args.input, args.dry_run), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit('INGEST ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    main()
