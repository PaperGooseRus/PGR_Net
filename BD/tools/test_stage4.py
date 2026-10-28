#!/usr/bin/env python3
"""Контрактные и интеграционные тесты Stage 4 без интернета и компиляторов."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from stage4_features import tokenize, input_features, encode, MAX_DIAG_TOKENS
from prepare_stage4 import prepare, verify_snapshot, validate_and_iter
from evaluate_stage4 import MultinomialNB, metrics, read_one
from quality_gate import read_jsonl


def sample(project: str, pair: str, split: str, compiler: str,
           source: str, label='CLEAN', stage='success'):
    return {
        'id': project, 'pair_id': pair,
        'input': {
            'language': 'c', 'compiler': compiler,
            'build_system': 'direct+cmake', 'command_phase': 'run',
            'build_exit_code': 0 if stage == 'success' else 1,
            'diagnostic': '', 'files': {'main.c': source},
        },
        'target': {'fault_code': label, 'stage': stage,
                   'build_ok': stage == 'success'},
    }


def fixture(root: Path, validation_only='validation_only'):
    (root/'catalog').mkdir(parents=True)
    (root/'catalog/fault_taxonomy.json').write_text(json.dumps([
        {'code': 'BAD', 'stage': 'compile_error'},
    ]), encoding='utf-8')
    for split in ('train', 'val', 'test'):
        for compiler in ('gcc', 'clang'):
            directory = root/'datasets/verified'/compiler
            directory.mkdir(parents=True, exist_ok=True)
            marker = validation_only if split != 'train' else 'training_only'
            records = [
                sample(f'{split}_clean', f'{split}_pair', split, compiler, marker),
                sample(f'{split}_fault', f'{split}_pair', split, compiler,
                       f'{marker} illegal_x', 'BAD', 'compile_error'),
            ]
            with (directory/(split+'.jsonl')).open('w', encoding='utf-8') as out:
                for row in records:
                    out.write(json.dumps(row, sort_keys=True)+'\n')


class FeatureTests(unittest.TestCase):
    def test_normalizes_synthetic_ids_paths_and_numbers(self):
        first = tokenize('C:\\Users\\person\\Project\\main.c:15: error: seed_v1234abcd missing 99',100)
        second = tokenize('D:\\Temp\\Project\\main.c:34: error: seed_vabcdef12 missing 72',100)
        self.assertEqual(first, second)
        self.assertLessEqual(len(tokenize('error '*3000, MAX_DIAG_TOKENS)), MAX_DIAG_TOKENS)

    def test_features_cannot_see_target_identifiers(self):
        row = sample('secretid', 'secretpair', 'train', 'gcc', 'int main(){}')
        before = input_features(row['input'])
        row['target'] = {'fault_code': 'SECRET_TARGET', 'stage': 'link_error'}
        row['id'] = 'OTHER_SECRET'
        self.assertEqual(before, input_features(row['input']))
        self.assertFalse(any('secret' in term for term in before))

    def test_encoder_sorted_and_counts_are_bounded(self):
        vec = input_features(sample('x','p','train','gcc','error error error '*100)['input'])
        vocab = {token: i for i, token in enumerate(sorted(vec))}
        indices, values, unknown = encode(vec, vocab)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 < x <= 8 for x in values))
        self.assertEqual(unknown, 0)
        self.assertGreater(encode(vec, {})[2], 0)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_vocabulary_is_fitted_only_on_train(self):
        report = prepare(self.root, min_df=1)
        self.assertEqual(report['classes'], 2)
        vocab = read_one(self.root/'catalog/training/vocabulary.json')
        self.assertIn('src:training_only', vocab['terms'])
        self.assertNotIn('src:validation_only', vocab['terms'])
        self.assertEqual(report['counts']['gcc']['train'], 2)
        enc = next(read_jsonl(self.root/'datasets/encoded/gcc/test.jsonl'))
        self.assertEqual(set(enc['input']), {'indices', 'values'})
        self.assertNotIn('target', enc['input'])

    def test_prepare_is_byte_deterministic(self):
        prepare(self.root, min_df=1)
        before = hashlib.sha256((self.root/'datasets/encoded/gcc/train.jsonl').read_bytes()).hexdigest()
        snapshot = (self.root/'manifests/stage4_snapshot.json').read_bytes()
        prepare(self.root, min_df=1)
        after = hashlib.sha256((self.root/'datasets/encoded/gcc/train.jsonl').read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(snapshot, (self.root/'manifests/stage4_snapshot.json').read_bytes())

    def test_snapshot_detects_modification(self):
        prepare(self.root, min_df=1)
        self.assertEqual(verify_snapshot(self.root)['snapshot'], 'passed')
        path = self.root/'datasets/encoded/gcc/val.jsonl'
        path.write_text(path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_snapshot(self.root)

    def test_rejects_pair_leak_before_writing_output(self):
        path = self.root/'datasets/verified/gcc/val.jsonl'
        rows = list(read_jsonl(path))
        rows[0]['pair_id'] = 'train_pair'
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'Pair leaks'):
            prepare(self.root, min_df=1)
        self.assertFalse((self.root/'catalog/training/vocabulary.json').exists())

    def test_rejects_inconsistent_classes(self):
        path = self.root/'datasets/verified/gcc/train.jsonl'
        rows = list(read_jsonl(path))
        rows[0]['target']['build_ok'] = False
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'Stage/build_ok'):
            validate_and_iter(self.root, {'CLEAN', 'BAD'})


class BaselineTests(unittest.TestCase):
    def test_nb_and_majority_metrics(self):
        def row(k, cls):
            return {'input': {'indices':[k], 'values':[3]},
                    'target': {'class_id':cls, 'stage_id':cls}}
        rows = [row(0,0), row(0,0), row(1,1), row(1,1)]
        model = MultinomialNB(2,2).fit(rows, 'class_id')
        self.assertEqual([model.predict(x) for x in rows], [0,0,1,1])
        self.assertEqual(metrics([0,0,1,1],[0,0,1,1],['a','b'])['macro_f1'], 1.0)
        self.assertLess(metrics([0,0,1,1],[0,0,0,0],['a','b'])['macro_f1'],1.0)

    def test_report_and_model_match_current_dataset(self):
        snapshot = verify_snapshot(ROOT)
        report = read_one(ROOT/'reports/training/stage4_encoding.json')
        baseline = read_one(ROOT/'reports/training/stage4_baseline.json')
        self.assertEqual(report['classes'], 117)
        self.assertEqual(report['unique_pairs'], 3684)
        self.assertEqual(baseline['train_samples'], 11044)
        self.assertEqual(baseline['results']['test']['both']['nb_fault']['samples'], 1848)
        self.assertGreater(snapshot['files'], 10)

    def test_generated_codes_do_not_contain_labels_in_input(self):
        for compiler in ('gcc','clang'):
            rows = list(read_jsonl(ROOT/'datasets/encoded'/compiler/'test.jsonl'))
            self.assertEqual(len(rows), 924)
            for row in rows:
                self.assertEqual(set(row['input']), {'indices','values'})
                self.assertEqual(set(row['target']), {'class_id','stage_id'})

    def test_help_and_report_format(self):
        for command in ('stage4','prepare-stage4','evaluate-stage4', 'verify-stage4'):
            cp = subprocess.run([sys.executable,str(ROOT/'bd.py'),command,'--help'],
                                capture_output=True, text=True)
            self.assertEqual(cp.returncode,0,cp.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
