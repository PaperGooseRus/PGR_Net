#!/usr/bin/env python3
"""Stage 4: построить train-only словарь и разреженные входы для PGR Neural.

Использует исключительно datasets/verified. Исходные файлы не изменяет.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from quality_gate import read_jsonl, write_atomic
from stage4_features import FEATURE_SCHEMA, input_features, encode

COMPILERS = ('gcc', 'clang')
SPLITS = ('train', 'val', 'test')
STAGES = ('success', 'configure_error', 'compile_error', 'link_error', 'runtime_error')
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data):
    write_atomic(path, [data])  # Атомарная запись одного JSON-объекта.


def source_files(root: Path) -> list[Path]:
    return [root/'datasets/verified'/compiler/(split+'.jsonl')
            for compiler in COMPILERS for split in SPLITS]


def validate_and_iter(root: Path, class_labels: set[str]):
    """Проверка пар и исходного разделения ДО публикации результатов."""
    seen_pairs: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    by_split: dict[tuple[str, str], set[str]] = {}
    stage_by_label: dict[str, str] = {}
    for split in SPLITS:
        for compiler in COMPILERS:
            ids = set()
            pair_members = defaultdict(set)
            for row in read_jsonl(root/'datasets/verified'/compiler/(split+'.jsonl')):
                if not isinstance(row, dict) or not {'id','pair_id','input','target'} <= set(row):
                    raise ValueError('Invalid dataset row')
                project_id = row['id']
                pair = row['pair_id']
                if not isinstance(project_id, str) or not isinstance(pair, str) or not project_id or not pair:
                    raise ValueError('Invalid project/pair ID')
                if project_id in ids:
                    raise ValueError('Repeated project in compiler split: ' + project_id)
                ids.add(project_id)
                pair_members[pair].add(project_id)
                old_split = seen_pairs.setdefault(pair, split)
                if old_split != split:
                    raise ValueError('Pair leaks between splits: ' + pair)
                old_id = seen_ids.setdefault(project_id, split)
                if old_id != split:
                    raise ValueError('Project leaks between splits: ' + project_id)
                if row['input'].get('compiler') != compiler:
                    raise ValueError('Compiler mismatch in row')
                label = row['target'].get('fault_code')
                stage = row['target'].get('stage')
                if label not in class_labels or stage not in STAGES:
                    raise ValueError('Unknown label/stage')
                if bool(row['target'].get('build_ok')) != (stage == 'success'):
                    raise ValueError('Stage/build_ok mismatch')
                prior = stage_by_label.setdefault(label, stage)
                if prior != stage:
                    raise ValueError('Inconsistent class stage')
                # Вызов валидирует вход; результат здесь ещё не сохраняем.
                input_features(row['input'])
            if any(len(members) != 2 for members in pair_members.values()):
                raise ValueError('Incomplete pair in ' + compiler + '/' + split)
            by_split[(compiler, split)] = ids
        if by_split[('gcc', split)] != by_split[('clang', split)]:
            raise ValueError('GCC/Clang project sets differ in ' + split)
    return {'unique_projects': len(seen_ids), 'unique_pairs': len(seen_pairs)}


def prepare(root: Path = ROOT, min_df: int = 2, max_features: int = 12000) -> dict:
    root = root.resolve()
    if min_df < 1 or max_features < 10:
        raise ValueError('Invalid min_df/max_features')
    for path in source_files(root):
        if not path.is_file():
            raise ValueError('Missing verified data: ' + str(path))

    taxonomy = json.loads((root/'catalog/fault_taxonomy.json').read_text(encoding='utf-8'))
    labels = sorted({'CLEAN'} | {item['code'] for item in taxonomy})
    if len(labels) != len(taxonomy) + 1:
        raise ValueError('Repeated class in taxonomy')
    class_ids = {name: i for i, name in enumerate(labels)}
    stage_ids = {name: i for i, name in enumerate(STAGES)}
    structure = validate_and_iter(root, set(labels))

    document_frequency = Counter()
    train_count = 0
    for compiler in COMPILERS:
        for row in read_jsonl(root/'datasets/verified'/compiler/'train.jsonl'):
            document_frequency.update(input_features(row['input']).keys())
            train_count += 1
    vocab_terms = [name for name, df in
                   sorted(document_frequency.items(), key=lambda pair: (-pair[1], pair[0]))
                   if df >= min_df][:max_features]
    # Индексы фиксированы, не зависят от порядка файлов в системе.
    vocabulary = {name: index for index, name in enumerate(vocab_terms)}
    if not vocabulary:
        raise ValueError('Empty vocabulary')

    counts = {}
    oov_counts = {}
    output_paths = []
    for split in SPLITS:
        for compiler in COMPILERS:
            out = []
            unknown = 0
            for row in read_jsonl(root/'datasets/verified'/compiler/(split+'.jsonl')):
                indices, values, missing = encode(input_features(row['input']), vocabulary)
                unknown += missing
                out.append({
                    'id': row['id'], 'pair_id': row['pair_id'],
                    'input': {'indices': indices, 'values': values},
                    'target': {
                        'class_id': class_ids[row['target']['fault_code']],
                        'stage_id': stage_ids[row['target']['stage']],
                    },
                })
            path = root/'datasets/encoded'/compiler/(split+'.jsonl')
            write_atomic(path, out)
            output_paths.append(path)
            counts.setdefault(compiler, {})[split] = len(out)
            oov_counts.setdefault(compiler, {})[split] = unknown

    label_file = root/'catalog/training/labels.json'
    vocab_file = root/'catalog/training/vocabulary.json'
    write_json(label_file, {
        'schema': FEATURE_SCHEMA, 'classes': labels, 'stages': list(STAGES),
        'class_to_stage': {key: val for key, val in sorted(
            ((code, ('success' if code=='CLEAN' else next(item['stage'] for item in taxonomy if item['code']==code)))
             for code in labels))},
    })
    write_json(vocab_file, {
        'schema': FEATURE_SCHEMA, 'min_df': min_df,
        'max_features': max_features, 'train_documents': train_count,
        'terms': vocab_terms,
    })
    report = {
        'stage': 4, 'schema': FEATURE_SCHEMA,
        'verified_policy': 'consensus',
        'classes': len(labels), 'stages': len(STAGES), 'features': len(vocab_terms),
        'train_documents': train_count, 'counts': counts,
        'oov_token_counts': oov_counts,
        **structure,
        'data_limitations': [
            'Synthetic input, no independent Windows compiler re-run.',
            'Vocabulary fitted only on train, not val/test.',
            'Project IDs, pair IDs, labels and repair patches excluded from features.',
            'Closely related templates may exist across splits; real-world accuracy is unproven.',
        ],
    }
    report_file = root/'reports/training/stage4_encoding.json'
    write_json(report_file, report)
    # Отдельный список контрольных сумм Stage 4. Исторический SHA256SUMS
    # текущей поставки обслуживается упаковщиком, но не меняется конвейером.
    dependencies = source_files(root) + [root/'catalog/fault_taxonomy.json']
    products = output_paths + [label_file, vocab_file, report_file]
    snapshot = {
        'schema': FEATURE_SCHEMA,
        'inputs': {str(p.relative_to(root)).replace('\\','/'): sha256(p) for p in dependencies},
        'outputs': {str(p.relative_to(root)).replace('\\','/'): sha256(p) for p in products},
    }
    write_json(root/'manifests/stage4_snapshot.json', snapshot)
    return report


def verify_snapshot(root: Path = ROOT) -> dict:
    root = root.resolve()
    snapshot_path = root/'manifests/stage4_snapshot.json'
    if not snapshot_path.is_file():
        raise ValueError('Stage 4 snapshot is missing; run prepare-stage4')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    if snapshot.get('schema') != FEATURE_SCHEMA:
        raise ValueError('Stage 4 schema mismatch')
    checked = 0
    for kind in ('inputs', 'outputs'):
        entries = snapshot.get(kind)
        if not isinstance(entries, dict):
            raise ValueError('Invalid snapshot')
        for name, expected in entries.items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Unsafe/missing snapshot file: '+name)
            if sha256(path) != expected:
                raise ValueError('Snapshot checksum mismatch: ' + name)
            checked += 1
    return {'stage': 4, 'snapshot': 'passed', 'files': checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--min-df', type=int, default=2)
    parser.add_argument('--max-features', type=int, default=12000)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        result = (verify_snapshot(args.root) if args.verify else
                  prepare(args.root, args.min_df, args.max_features))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit('STAGE4 ERROR: '+str(exc))


if __name__ == '__main__':
    main()
