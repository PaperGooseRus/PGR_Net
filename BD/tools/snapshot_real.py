#!/usr/bin/env python3
"""Отдельный SHA-256 снимок изменяемых пользовательских данных.

Поставляемый manifests/SHA256SUMS.txt остаётся исходной контрольной точкой BD.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'catalog/real_cases.jsonl', 'observations/real/observations.jsonl',
    'datasets/real/train.jsonl', 'datasets/real/val.jsonl', 'datasets/real/test.jsonl',
    'reports/validation/real_dataset.json',
)


def checksums(root: Path):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in FILES if (root/name).is_file()}


def snapshot(root: Path, write: bool):
    root = root.resolve()
    manifest = root/'manifests/real_snapshot.json'
    if write:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        data = {'manifest_version':1,'files':checksums(root)}
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        return {'created':str(manifest),'files':len(data['files'])}
    if not manifest.exists():
        raise ValueError('No user snapshot; run --write after reviewing imported data')
    data = json.loads(manifest.read_text(encoding='utf-8'))
    current = checksums(root)
    if data.get('manifest_version') != 1 or data.get('files') != current:
        raise ValueError('Real data have changed since last snapshot')
    return {'verified':str(manifest),'files':len(current)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(snapshot(args.root, args.write), indent=2))
    except (ValueError, OSError) as exc:
        raise SystemExit('SNAPSHOT ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    main()
