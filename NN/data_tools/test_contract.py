"""Граничные проверки контракта на маленьких искусственных записях, не копиях BD."""
import json
from pathlib import Path
import tempfile
import unittest
from inspect_bd import Corpus,inspect


def fixture(root):
    def write(name,value,jsonl=False):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        content=''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value)
        p.write_text(content,encoding='utf-8')
    write('labels/fault_taxonomy.json',[{'code':'EXAMPLE_FAULT'}])
    rows=[{'project_id':'p_a','pair_id':'pair_a','split':'train','fault_code':'EXAMPLE_FAULT','expected_stage':'compile_error','expected_ok':False},
          {'project_id':'p_b','pair_id':'pair_a','split':'train','fault_code':'CLEAN','expected_stage':'success','expected_ok':True}]
    write('labels/projects.jsonl',rows,True)
    training=[{'id':r['project_id'],'pair_id':r['pair_id'],'input':{'diagnostic':'example' if not r['expected_ok'] else '', 'compiler':'gcc'},
               'target':{'fault_code':r['fault_code'],'stage':r['expected_stage'],'build_ok':r['expected_ok']}} for r in rows]
    write('training/train_gcc.jsonl',training,True)


class ContractTests(unittest.TestCase):
    def test_valid_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);c=Corpus(root)
            self.assertEqual(inspect(c)['projects'],2)

    def test_pair_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);p=root/'labels/projects.jsonl'
            rows=[json.loads(x) for x in p.read_text().splitlines()];rows[1]['split']='test'
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):
                inspect(Corpus(root))

    def test_target_in_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);p=root/'training/train_gcc.jsonl'
            rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['input']['fault_code']='EXAMPLE_FAULT'
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):
                inspect(Corpus(root))

    def test_target_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);p=root/'training/train_gcc.jsonl'
            rows=[json.loads(x) for x in p.read_text().splitlines()];rows[0]['target']['stage']='success'
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):
                inspect(Corpus(root))


if __name__=='__main__':
    unittest.main()
