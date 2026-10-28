#!/usr/bin/env python3
"""Единая точка запуска BD. Все команды выполняются относительно этого каталога."""
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT=Path(__file__).resolve().parent
COMMANDS={'generate':'generate.py','collect':'verify.py','prepare':'prepare_training.py',
          'repairs':'make_repairs.py','audit':'audit.py','check':'check_integrity.py','test':'test_layout.py',
          'quality':'quality_gate.py','ingest':'ingest_real.py','export-real':'prepare_real.py',
          'test-stage3':'test_stage3.py','audit-real':'audit_real.py',
          'capture':'capture_template.py','snapshot-real':'snapshot_real.py',
          'stage4':'stage4_pipeline.py', 'prepare-stage4':'prepare_stage4.py',
          'evaluate-stage4':'evaluate_stage4.py', 'verify-stage4':'prepare_stage4.py',
          'test-stage4':'test_stage4.py'}


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            if stream.isatty():stream.reconfigure(errors='backslashreplace')
            else:stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        print('BD v2 commands: info, check, audit, test, generate, collect, repairs, prepare, quality, capture, ingest, audit-real, export-real, snapshot-real, test-stage3, stage4, prepare-stage4, evaluate-stage4, verify-stage4, test-stage4')
        print('Example: python bd.py collect --compiler gcc --sample-per-family 1 --strict')
        print('Command help: python bd.py <command> --help')
        return 0
    command=sys.argv[1]
    if command=='info':
        data=json.loads((ROOT/'catalog/summary.json').read_text(encoding='utf-8'))
        print(json.dumps({'root':str(ROOT),'layout_version':2,'corpus':data},ensure_ascii=False,indent=2))
        return 0
    if command=='verify-stage4':
        sys.argv.insert(2, '--verify')
    if command not in COMMANDS:
        print('Unknown command: '+command,file=sys.stderr);return 2
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable,str(ROOT/'tools'/COMMANDS[command]),*sys.argv[2:]],cwd=ROOT,env=env).returncode


if __name__=='__main__':sys.exit(main())
