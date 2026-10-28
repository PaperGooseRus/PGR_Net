#!/usr/bin/env python3
"""Обучить собственный базовый Multinomial Naive Bayes и оценить Stage 4.

Это прозрачная контрольная модель, НЕ нейросеть. Использует только encoded
train для обучения, val/test только для оценки; stdlib Python >= 3.10.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from quality_gate import read_jsonl, write_atomic
from prepare_stage4 import ROOT, sha256, verify_snapshot

SPLITS = ('val', 'test')
COMPILERS = ('gcc', 'clang')


def read_one(path: Path) -> dict:
    rows = list(read_jsonl(path))
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError('Invalid metadata JSON')
    return rows[0]


def load_encoded(root: Path, compiler: str, split: str, vocab_size: int,
                 labels: dict) -> list[dict]:
    rows = list(read_jsonl(root/'datasets/encoded'/compiler/(split+'.jsonl')))
    observed = set()
    for row in rows:
        pid = row['id']
        if pid in observed:
            raise ValueError('Duplicate encoded row: ' + pid)
        observed.add(pid)
        data = row['input']
        indices, values = data['indices'], data['values']
        if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
            raise ValueError('Invalid sparse dimensions: ' + pid)
        if indices != sorted(set(indices)):
            raise ValueError('Unsorted/duplicate indices: ' + pid)
        if any(not isinstance(i, int) or i < 0 or i >= vocab_size for i in indices):
            raise ValueError('Invalid feature index: ' + pid)
        if any(not isinstance(n, int) or n <= 0 or n > 8 for n in values):
            raise ValueError('Invalid feature count: ' + pid)
        target = row['target']
        cls, stage = target['class_id'], target['stage_id']
        if (not isinstance(cls, int) or cls < 0 or cls >= len(labels['classes']) or
                not isinstance(stage, int) or stage < 0 or stage >= len(labels['stages'])):
            raise ValueError('Invalid label index: ' + pid)
        if labels['class_to_stage'][labels['classes'][cls]] != labels['stages'][stage]:
            raise ValueError('Stage/class inconsistent: ' + pid)
    return rows


class MultinomialNB:
    """Мультиномиальная модель без scikit-learn, прозрачная для тестирования."""
    def __init__(self, n_classes: int, n_features: int, alpha: float = 1.0):
        if alpha <= 0 or n_classes < 1 or n_features < 1:
            raise ValueError('Invalid NB parameters')
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.class_counts = [0] * n_classes
        self.token_totals = [0] * n_classes
        self.token_counts = [Counter() for _ in range(n_classes)]

    def fit(self, rows: list[dict], target_key: str):
        for row in rows:
            cls = row['target'][target_key]
            self.class_counts[cls] += 1
            for index, count in zip(row['input']['indices'], row['input']['values']):
                self.token_counts[cls][index] += count
                self.token_totals[cls] += count
        if any(n == 0 for n in self.class_counts):
            raise ValueError('Training set lacks one or more classes')
        return self

    def predict(self, row: dict) -> int:
        # Сумма лог-вероятностей; отсутствующие токены не сглаживаются поштучно.
        total_docs = sum(self.class_counts)
        length = sum(row['input']['values'])
        best_index = 0
        best_score = -math.inf
        for cls in range(self.n_classes):
            denominator = self.token_totals[cls] + self.alpha * self.n_features
            score = (math.log(self.class_counts[cls] / total_docs)
                     - length * math.log(denominator))
            counts = self.token_counts[cls]
            for index, value in zip(row['input']['indices'], row['input']['values']):
                score += value * math.log(counts.get(index, 0) + self.alpha)
            if score > best_score:
                best_index, best_score = cls, score
        return best_index

    def export(self) -> dict:
        return {
            'alpha': self.alpha,
            'class_counts': self.class_counts,
            'token_totals': self.token_totals,
            'token_counts': [dict(sorted(c.items())) for c in self.token_counts],
        }


def metrics(actual: list[int], predicted: list[int], names: list[str]) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError('Invalid eval results')
    matrix = [[0] * len(names) for _ in names]
    for target, guess in zip(actual, predicted):
        matrix[target][guess] += 1
    per_class = {}
    total_correct = 0
    weighted_f1 = 0.0
    for i, name in enumerate(names):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            'support': support,
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            'f1': round(f1, 6),
        }
        total_correct += tp
        weighted_f1 += f1 * support
    n = len(actual)
    return {
        'samples': n,
        'accuracy': round(total_correct / n, 6),
        'macro_f1': round(sum(v['f1'] for v in per_class.values()) / len(names), 6),
        'weighted_f1': round(weighted_f1 / n, 6),
        'per_class': per_class,
        'confusion_matrix': matrix,
    }


def summarize(report: dict) -> dict:
    # Ограничиваем размер краткой сводки. Полная матрица пишется отдельно.
    return {key: report[key] for key in ('samples', 'accuracy', 'macro_f1', 'weighted_f1')}


def evaluate(root: Path = ROOT) -> dict:
    root = root.resolve()
    verify_snapshot(root)  # Запрет использования устаревших закодированных данных.
    labels_path = root/'catalog/training/labels.json'
    vocab_path = root/'catalog/training/vocabulary.json'
    labels, vocab = read_one(labels_path), read_one(vocab_path)
    classes, stages = labels['classes'], labels['stages']
    vocab_size = len(vocab['terms'])
    train = []
    for compiler in COMPILERS:
        train.extend(load_encoded(root, compiler, 'train', vocab_size, labels))
    if len(train) == 0:
        raise ValueError('No train data')
    nb_class = MultinomialNB(len(classes), vocab_size).fit(train, 'class_id')
    nb_stage = MultinomialNB(len(stages), vocab_size).fit(train, 'stage_id')
    majority_class = max(range(len(classes)), key=lambda n: nb_class.class_counts[n])
    majority_stage = max(range(len(stages)), key=lambda n: nb_stage.class_counts[n])

    summary = {'stage': 4, 'baseline': 'MultinomialNB', 'train_samples': len(train),
               'majority_class': classes[majority_class],
               'majority_stage': stages[majority_stage], 'results': {}}
    matrices = {'classes': classes, 'stages': stages, 'results': {}}
    for split in SPLITS:
        summary['results'][split] = {}
        matrices['results'][split] = {}
        for compiler in ('gcc', 'clang', 'both'):
            rows = []
            for source in (COMPILERS if compiler == 'both' else (compiler,)):
                rows.extend(load_encoded(root, source, split, vocab_size, labels))
            true_class = [r['target']['class_id'] for r in rows]
            true_stage = [r['target']['stage_id'] for r in rows]
            pred_class = [nb_class.predict(r) for r in rows]
            pred_stage = [nb_stage.predict(r) for r in rows]
            class_report = metrics(true_class, pred_class, classes)
            stage_report = metrics(true_stage, pred_stage, stages)
            baseline_class = metrics(true_class, [majority_class] * len(rows), classes)
            baseline_stage = metrics(true_stage, [majority_stage] * len(rows), stages)
            summary['results'][split][compiler] = {
                'nb_fault': summarize(class_report),
                'nb_stage': summarize(stage_report),
                'majority_fault': summarize(baseline_class),
                'majority_stage': summarize(baseline_stage),
            }
            matrices['results'][split][compiler] = {
                'fault': class_report, 'stage': stage_report,
            }

    report_dir = root/'reports/training'
    write_atomic(report_dir/'stage4_baseline.json', [summary])
    write_atomic(report_dir/'stage4_confusion.json', [matrices])
    write_atomic(report_dir/'stage4_nb_model.json', [{
        'schema': 'pgr-nb-v1',
        'vocabulary_sha256': sha256(vocab_path),
        'labels_sha256': sha256(labels_path),
        'fault': nb_class.export(), 'stage': nb_stage.export(),
    }])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.root), ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        sys.exit('EVALUATION ERROR: '+str(exc))


if __name__ == '__main__':
    main()
