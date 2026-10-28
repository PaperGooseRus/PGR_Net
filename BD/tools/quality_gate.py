#!/usr/bin/env python3
"""Подготовка проверенных выборок без изменения исходного корпуса BD v2.

Политика consensus: обе стороны пары подтверждены каждым компилятором.
Политика per-compiler: обе стороны пары подтверждены выбранным компилятором.
Проверка match опирается на поставленные observations, а не на новый запуск компилятора.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import tempfile

SPLITS = ('train', 'val', 'test')
COMPILERS = ('gcc', 'clang')
STAGES = {'success', 'compile_error', 'link_error', 'configure_error', 'runtime_error'}


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON: {path}:{number}') from exc


def write_atomic(path: Path, rows):
    """Публикация JSONL через временный файл; не оставляет частично записанной выборки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix='.tmp_', delete=False) as output:
            name = output.name
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
        os.chmod(path, 0o644)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def export_verified(root: Path, policy='consensus'):
    root = root.resolve()
    if policy not in ('consensus', 'per-compiler'):
        raise ValueError('Invalid policy')
    meta = {r['project_id']: r for r in read_jsonl(root / 'catalog/projects.jsonl')}
    pairs = defaultdict(set)
    for r in meta.values():
        pairs[r['pair_id']].add(r['project_id'])
        if r['split'] not in SPLITS:
            raise ValueError('Unknown split')
    if any(len(members) != 2 for members in pairs.values()):
        raise ValueError('Incomplete source pair')

    observations = {}
    good_pairs = {}
    for compiler in COMPILERS:
        obs_rows = list(read_jsonl(root / 'observations' / compiler / 'observations.jsonl'))
        obs = {}
        for row in obs_rows:
            pid = row['project_id']
            if pid in obs or pid not in meta:
                raise ValueError(f'Duplicate/unknown observation: {compiler}:{pid}')
            obs[pid] = row
        observations[compiler] = obs
        good_pairs[compiler] = {
            pair_id for pair_id, members in pairs.items()
            if all(pid in obs and obs[pid].get('match') is True
                   and obs[pid].get('observed_stage') == meta[pid]['expected_stage']
                   and obs[pid].get('pair_id') == pair_id
                   and obs[pid].get('compiler') == compiler
                   and obs[pid].get('split') == meta[pid]['split']
                   for pid in members)
        }
    consensus = good_pairs['gcc'] & good_pairs['clang']
    report = {
        'policy': policy,
        'source': 'BD v2 observations (not reverified on this machine)',
        'source_projects': len(meta),
        'source_pairs': len(pairs),
        'source_mismatches': {
            c: sum(o.get('match') is not True for o in observations[c].values())
            for c in COMPILERS
        },
        'good_pairs_by_compiler': {c: len(good_pairs[c]) for c in COMPILERS},
        'consensus_pairs': len(consensus),
        'exported': {},
        'excluded': {},
        'stage_distribution': {},
        'class_distribution': {},
        'warnings': [
            'All supplied cases are synthetic; verified means consistency with stored observations.',
            'CLEAN is approximately half of the cases; use class-aware evaluation / weighting.',
            'No independent Windows toolchain rerun was performed in this stage.',
        ],
    }
    for compiler in COMPILERS:
        chosen = consensus if policy == 'consensus' else good_pairs[compiler]
        output = {split: [] for split in SPLITS}
        excluded = Counter()
        for pid, metadata in sorted(meta.items()):
            if metadata['pair_id'] not in chosen:
                excluded[metadata['split']] += 1
                continue
            observation = observations[compiler][pid]
            project_path = (root / metadata['relative_path']).resolve()
            if not project_path.is_relative_to(root / 'projects') or not project_path.is_dir():
                raise ValueError('Unsafe/missing project path for ' + pid)
            files = {}
            for item in sorted(project_path.iterdir()):
                if not item.is_file() or item.is_symlink():
                    raise ValueError('Unexpected non-file source in ' + pid)
                files[item.name] = item.read_text(encoding='utf-8')
            sample = {
                'id': pid,
                'pair_id': metadata['pair_id'],
                'input': {
                    'language': metadata['language'],
                    'compiler': compiler,
                    'build_system': metadata['build_system'],
                    'build_exit_code': observation['exit_code'],
                    'command_phase': observation['command_phase'],
                    'diagnostic': observation['diagnostic'],
                    'files': files,
                },
                'target': {
                    'stage': metadata['expected_stage'],
                    'fault_code': metadata['fault_code'],
                    'build_ok': metadata['expected_ok'],
                }
            }
            output[metadata['split']].append(sample)
        dataset_dir = 'verified' if policy == 'consensus' else 'verified_per_compiler'
        for split, records in output.items():
            write_atomic(root / 'datasets' / dataset_dir / compiler / (split + '.jsonl'), records)
        report['exported'][compiler] = {split: len(output[split]) for split in SPLITS}
        report['excluded'][compiler] = dict(excluded)
        report['stage_distribution'][compiler] = {
            split: dict(sorted(Counter(x['target']['stage'] for x in output[split]).items()))
            for split in SPLITS
        }
        report['class_distribution'][compiler] = {
            split: dict(sorted(Counter(x['target']['fault_code'] for x in output[split]).items()))
            for split in SPLITS
        }
    # Набор train/val/test сохранён, пары не распадаются, исходники не изменены.
    report_name = 'quality_gate.json' if policy == 'consensus' else 'quality_gate_per_compiler.json'
    destination = root / 'reports/validation' / report_name
    write_report(destination, report)
    return report


def write_report(path: Path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--policy', choices=['consensus', 'per-compiler'], default='consensus')
    args = parser.parse_args()
    try:
        print(json.dumps(export_verified(args.root, args.policy), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit('QUALITY ERROR: ' + str(error)) from error


if __name__ == '__main__':
    main()
