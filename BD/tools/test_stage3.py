#!/usr/bin/env python3
"""Регрессионные тесты этапа подготовки обучающих данных BD v2.1."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_gate import read_jsonl, export_verified
from ingest_real import ingest, group_hash, select_split
from prepare_real import prepare
from snapshot_real import snapshot

ROOT = Path(__file__).resolve().parents[1]


class VerifiedDatasetTests(unittest.TestCase):
    def test_verified_sets_align_and_do_not_leak_pairs(self):
        catalog = {r['project_id']: r for r in read_jsonl(ROOT/'catalog/projects.jsonl')}
        project_sets = {}
        for compiler in ('gcc', 'clang'):
            by_split = {}
            pairs = defaultdict(set)
            for split, count in [('train', 5522), ('val', 922), ('test', 924)]:
                rows = list(read_jsonl(ROOT/'datasets/verified'/compiler/(split+'.jsonl')))
                self.assertEqual(len(rows), count)
                by_split[split] = {r['id'] for r in rows}
                for r in rows:
                    pid = r['id']
                    self.assertEqual(r['pair_id'], catalog[pid]['pair_id'])
                    self.assertEqual(catalog[pid]['split'], split)
                    self.assertEqual(r['input']['compiler'], compiler)
                    self.assertEqual(r['target']['fault_code'], catalog[pid]['fault_code'])
                    self.assertNotIn('fault_code', r['input'])
                    self.assertNotIn('expected_stage', r['input'])
                    pairs[r['pair_id']].add(pid)
            self.assertTrue(all(len(v) == 2 for v in pairs.values()))
            self.assertFalse(by_split['train'] & by_split['val'])
            self.assertFalse(by_split['train'] & by_split['test'])
            self.assertFalse(by_split['val'] & by_split['test'])
            project_sets[compiler] = by_split
        self.assertEqual(project_sets['gcc'], project_sets['clang'])

    def test_quality_report_keeps_source_disagreements(self):
        report = json.loads((ROOT/'reports/validation/quality_gate.json').read_text(encoding='utf-8'))
        self.assertEqual(report['consensus_pairs'], 3684)
        self.assertEqual(report['source_mismatches'], {'gcc':28,'clang':0})
        self.assertEqual(report['source_pairs']-report['consensus_pairs'], 28)

    def test_existing_original_files_not_modified(self):
        mapping=json.loads((ROOT/'manifests/source_mapping.json').read_text(encoding='utf-8'))['entries']
        legacy=[x for x in mapping if x['original'].startswith(('projects/','labels/','observations/','training/'))]
        for x in legacy:
            with self.subTest(x['destination']):
                self.assertEqual(hashlib.sha256((ROOT/x['destination']).read_bytes()).hexdigest(),
                                 x['original_sha256'])


class RealIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'BD'
        (self.root/'catalog').mkdir(parents=True)
        (self.root/'catalog/fault_taxonomy.json').write_bytes((ROOT/'catalog/fault_taxonomy.json').read_bytes())
        self.source = Path(self.temp.name)/'input.jsonl'

    def record(self, group='test-group', reviewed=True, label=True):
        return {'source_group':group,'language':'c','compiler':'gcc',
                'build_system':'direct+cmake','command_phase':'compile+link','exit_code':1,
                'diagnostic':'main.c:2: error: undeclared identifier',
                'files':{'main.c':'int main(void) { return missing_symbol; }\n'},
                'reviewed': reviewed, 'label':{'stage':'compile_error','fault_code':'C_UNDECLARED'} if label else None}

    def write_input(self, rows):
        self.source.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')

    def test_import_idempotent_and_export(self):
        self.write_input([self.record()])
        report = ingest(self.root, self.source)
        self.assertEqual(report['ready'], 1)
        self.assertEqual(report['new'], 1)
        self.assertEqual(ingest(self.root, self.source)['new'], 0)
        self.assertEqual(prepare(self.root)['ready'], 1)
        rows=sum((list(read_jsonl(self.root/'datasets/real'/ (sp+'.jsonl'))) for sp in ('train','val','test')),[])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['target']['fault_code'],'C_UNDECLARED')
        self.assertNotIn('fault_code',rows[0]['input'])
        self.assertNotIn('label',rows[0]['input'])
        self.assertNotIn('source_group',rows[0]['input'])

    def test_snapshot_detects_change(self):
        self.write_input([self.record()])
        ingest(self.root, self.source)
        prepare(self.root)
        snapshot(self.root, True)
        snapshot(self.root, False)
        with (self.root/'observations/real/observations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaises(ValueError):
            snapshot(self.root, False)

    def test_unreviewed_goes_to_quarantine(self):
        self.write_input([self.record(reviewed=False)])
        self.assertEqual(ingest(self.root,self.source)['quarantine'],1)
        self.assertEqual(prepare(self.root)['ready'],0)

    def test_invalid_batch_writes_nothing(self):
        self.write_input([self.record(), dict(self.record(), label={'stage':'link_error','fault_code':'C_UNDECLARED'})])
        with self.assertRaises(ValueError):
            ingest(self.root,self.source)
        self.assertFalse((self.root/'catalog/real_cases.jsonl').exists())

    def test_sensitive_data_is_rejected(self):
        self.write_input([dict(self.record(), diagnostic=r'C:\Users\Someone\Documents\secret.c: error')])
        with self.assertRaises(ValueError):
            ingest(self.root,self.source)

    def test_exact_duplicates_cannot_cross_split(self):
        first = self.record(group='real-project-1')
        a = select_split(group_hash(first['source_group']))
        for i in range(100):
            candidate = self.record(group='real-project-'+str(i+2))
            if select_split(group_hash(candidate['source_group'])) != a:
                break
        else:
            self.fail('Could not find alternate deterministic split')
        self.write_input([first, candidate])
        with self.assertRaises(ValueError):
            ingest(self.root,self.source)

    def test_capture_creates_unreviewed_template(self):
        log = Path(self.temp.name)/'stderr.txt'
        log.write_text('main.c: error: unknown type\n',encoding='utf-8')
        out = Path(self.temp.name)/'new_case.jsonl'
        cmd = [sys.executable, str(ROOT/'bd.py'), 'capture',
               '--log', str(log), '--output', str(out), '--source-group', 'group1',
               '--compiler', 'gcc', '--language', 'c', '--exit-code', '1']
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = next(read_jsonl(out))
        self.assertIsNone(row['label'])
        self.assertIs(row['reviewed'], False)
        self.assertEqual(row['diagnostic'], 'main.c: error: unknown type\n')

    def test_dry_run_does_not_change_library(self):
        self.write_input([self.record()])
        self.assertEqual(ingest(self.root,self.source,dry_run=True)['new'],1)
        self.assertFalse((self.root/'catalog/real_cases.jsonl').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
