"""Проверить новую раскладку и генерацию на небольшом временном корпусе."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from layout import ROOT

class LayoutTests(unittest.TestCase):
    def test_no_old_labels_or_training_folders(self):
        self.assertFalse((ROOT/'labels').exists())
        self.assertFalse((ROOT/'training').exists())
        self.assertTrue((ROOT/'catalog/projects.jsonl').is_file())
        self.assertTrue((ROOT/'datasets/gcc/train.jsonl').is_file())

    def test_source_data_preserved(self):
        entries=json.loads((ROOT/'manifests/source_mapping.json').read_text())['entries']
        immutable=[r for r in entries if r['original'].startswith(('projects/','labels/','observations/','training/'))]
        for r in immutable:
            with self.subTest(path=r['destination']):
                actual=hashlib.sha256((ROOT/r['destination']).read_bytes()).hexdigest()
                self.assertEqual(actual,r['original_sha256'])

    def test_fresh_generation_and_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'corpus'
            a=subprocess.run([sys.executable,str(ROOT/'bd.py'),'generate','--out',str(dest),'--copies','1'],capture_output=True)
            self.assertEqual(a.returncode,0,a.stderr.decode('utf-8',errors='replace'))
            self.assertTrue((dest/'catalog/projects.jsonl').is_file())
            self.assertTrue((dest/'splits/train.jsonl').is_file())
            b=subprocess.run([sys.executable,str(ROOT/'bd.py'),'repairs','--root',str(dest)],capture_output=True)
            self.assertEqual(b.returncode,0,b.stderr.decode('utf-8',errors='replace'))
            count=sum(1 for line in (dest/'repairs/repair_pairs.jsonl').read_text().splitlines() if line)
            self.assertEqual(count,116)
            audit=subprocess.run([sys.executable,str(ROOT/'bd.py'),'audit','--root',str(dest)],capture_output=True)
            self.assertEqual(audit.returncode,0,audit.stderr.decode('utf-8',errors='replace'))

    def test_help_for_all_tools(self):
        for command in ('generate','collect','prepare','repairs','audit'):
            p=subprocess.run([sys.executable,str(ROOT/'bd.py'),command,'--help'],capture_output=True)
            self.assertEqual(p.returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
