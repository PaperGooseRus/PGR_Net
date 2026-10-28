"""Регрессионные и граничные проверки; внешний компилятор не нужен.

Все примеры здесь разработческие, а не независимые данные для метрики качества.
Для реальных сборок отдельно запускается run_suite.py.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from classifier_v2 import classify
from evaluation import compare
from io_utils import clean, read_jsonl

ROOT=Path(__file__).resolve().parent


class BaselineTests(unittest.TestCase):
    def test_recorded_windows_regressions(self):
        rows=[]
        for p in (ROOT/'data').glob('*.jsonl'):
            rows+=read_jsonl(p)
        _,summary=compare(rows)
        self.assertEqual(summary['evaluated_errors'],48)
        self.assertEqual(summary['v2_matches'],48)
        self.assertEqual(summary['regressions'],[])

    def test_header_needs_matching_include(self):
        for name in ('config','with spaces','dir/detail','sensor.HPP'):
            message='main.cpp:3:10: fatal error: '+name+': No such file or directory'
            for newline in ('\n','\r\n'):
                raw=message+newline+'    3 | #include "'+name+'"'+newline+'      |          ^'
                row={'diagnostic':message,'stage':'compile','raw_stderr':raw}
                self.assertEqual(classify(row).label,'header_missing')
                for wrong in ('',raw.replace('3 |','4 |'),raw.replace('#include','#define'),raw.replace(name+'"','different"')):
                    self.assertEqual(classify(dict(row,raw_stderr=wrong)).label,'unknown')

    def test_header_from_other_error_does_not_count(self):
        m='main.cpp:2:10: fatal error: board: No such file or directory'
        raw=m+'\nother.cpp:4:1: fatal error: another: No such file or directory\n4 | #include "board"\n'
        self.assertEqual(classify({'diagnostic':m,'raw_stderr':raw,'stage':'compile'}).label,'unknown')

    def test_sources_or_labels_do_not_leak_into_prediction(self):
        row={'diagnostic':"x.cpp:1:1: error: expected ')' before '{' token",'stage':'compile'}
        a=classify(row)
        b=classify(dict(row,label='unknown_option',expected_marker='anything',notes='header_missing'))
        self.assertEqual(a,b)
        self.assertEqual(a.label,'syntax_error')

    def test_unknown_operating_systems_tools_and_build_errors(self):
        # Синтетические контрактные строки, не журналы реальных прогонов.
        cases=[
            ('ld: cannot find -lboard: No such file or directory','link'),
            ('ld: cannot open output file app.exe: Permission denied','link'),
            ('ld: region FLASH overflowed by 4096 bytes','link'),
            ('ld: file format not recognized; treating as linker script','link'),
            ('ld: skipping incompatible libboard.a when searching for -lboard','link'),
            ('ld: relocation truncated to fit: R_ARM_THM_CALL','link'),
            ('ld: section .text will not fit in region FLASH','link'),
            ('collect2: error: ld returned 1 exit status','link'),
            ('g++: fatal error: Killed signal terminated program cc1plus','compile'),
            ('cc1plus: fatal error: absent.hpp: No such file or directory','compile'),
            ('x.cpp:3:2: error: redefinition of int x','compile'),
            ('x.cpp:3:2: error: use of deleted function void f()','compile'),
            ('x.cpp:3:2: error: static assertion failed: cannot convert profile','compile'),
            ('x.cpp:3:2: error: #error undefined reference to startup','compile'),
            ("x.cpp:3:2: error: #error 'x' was not declared in this scope",'compile'),
            ('x.cpp:3:2: error: #error unrecognized command-line option','compile'),
            ('x.cpp:3:2: error: call of overloaded f(int) is ambiguous','compile'),
            ('x.cpp:3:2: error: int Board::pin is private within this context','compile'),
            ('x.cpp:3:2: warning: invalid conversion from int to int*','compile'),
            ('ld: warning: undefined reference to optional_symbol','link'),
            ('ld: note: multiple definition of optional_symbol','link'),
            ('ninja: build stopped: subcommand failed.','unknown'),
            ('make: *** [all] Error 2','unknown'),
            ('CMake Error: Could not find Qt6','configure'),
            ('MSBUILD : error MSB1009: Project file does not exist.','unknown'),
            ('main.cpp(2): error C2065: value: undeclared identifier','compile'),
            ('LINK : fatal error LNK1104: cannot open file app.exe','link'),
            ('fatal error: config file not found','compile'),
            ('error: не найден файл заголовка','compile'),
            ('Build finished successfully.','unknown')]
        for message,stage in cases:
            with self.subTest(message=message):
                self.assertEqual(classify({'diagnostic':message,'stage':stage}).label,'unknown')

    def test_stage_mismatch_and_tool_mismatch(self):
        m="file.cpp:1:1: error: 'x' was not declared in this scope"
        for stage in ('link','configure','invocation',None,123):
            self.assertEqual(classify({'diagnostic':m,'stage':stage}).label,'unknown')
        for tool in ('clang','cl','ninja',None):
            self.assertEqual(classify({'diagnostic':m,'stage':'compile','tool':tool}).label,'unknown')

    def test_bounds_and_ambiguous_messages(self):
        for value in (None,{}, {'diagnostic':None}, {'diagnostic':'x'*16385},
                      {'diagnostic':"a\nb"}, {'diagnostic':'ld: undefined reference to x: multiple definition of y','stage':'link'}):
            self.assertEqual(classify(value).label,'unknown')

    def test_cp1252_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'report'
            cmd=[sys.executable,str(ROOT/'evaluate.py'),'--output',str(out)]
            run=subprocess.run(cmd,capture_output=True,env=dict(os.environ,PYTHONIOENCODING='cp1252'))
            self.assertEqual(run.returncode,0,run.stderr.decode('utf-8',errors='replace'))
            self.assertIn('Совпадения v2:',run.stdout.decode('utf-8'))
            before=(out/'summary.json').read_bytes()
            again=subprocess.run(cmd,capture_output=True)
            self.assertEqual(again.returncode,2)
            self.assertEqual(before,(out/'summary.json').read_bytes())

    def test_bad_json_and_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.jsonl'
            for content in ('','bad json','[]','{"id":"a"}\n{"id":"a"}\n'):
                p.write_text(content,encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_jsonl(p)

    def test_failed_collection_is_skipped(self):
        results,s=compare([{'id':'failed','label':'syntax_error','status':'timeout','diagnostic':''}])
        self.assertFalse(results[0]['evaluated'])
        self.assertEqual(s['collection_problems'],1)
        self.assertEqual(s['evaluated_errors'],0)

    def test_missing_compiler(self):
        run=subprocess.run([sys.executable,str(ROOT/'run_suite.py'),'--compiler','no_such_pgr_compiler_999'],capture_output=True)
        self.assertEqual(run.returncode,2)

    def test_windows_temp_masking(self):
        self.assertEqual(clean(r'ld: C:\Users\name\Temp\ccABC.o: x'), 'ld: <TEMP_OBJECT>: x')


if __name__=='__main__':
    unittest.main()
