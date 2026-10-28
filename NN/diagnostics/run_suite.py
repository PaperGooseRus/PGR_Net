#!/usr/bin/env python3
"""Последовательно собрать 54 включённых проекта; сохранить логи и сравнить v1/v2.

Учебные программы не запускаются. Изменение проектов или PGR_Make не требуется.
При прерывании уже собранные записи остаются в results.jsonl.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from evaluation import write_report
from io_utils import clean, configure_output

ROOT=Path(__file__).resolve().parent


def compiler_info(compiler, flag):
    p=subprocess.run([compiler,flag],capture_output=True,text=True,errors='replace',timeout=20)
    if p.returncode or not p.stdout.strip():
        raise ValueError('Не удалось получить '+flag+': '+p.stderr)
    return p.stdout.strip().splitlines()[0]


def run_case(compiler,meta,folder,env,timeout):
    stdout,stderr,code='','',None
    status='unexpected_result'
    with tempfile.TemporaryDirectory(prefix='pgr_baseline_') as tmp:
        work=Path(tmp)/'project'
        shutil.copytree(folder,work)
        command=[compiler,'-std=c++17','-fdiagnostics-color=never','-fmax-errors=1',*meta['arguments']]
        try:
            p=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,errors='replace',timeout=timeout)
            stdout,stderr,code=clean(p.stdout,work),clean(p.stderr,work),p.returncode
            if meta['label']=='build_success':
                if code==0:
                    status='control_success'
            elif code!=0 and meta['expected_marker'] in stderr:
                status='expected_marker_found'
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else (value or '')
            stdout,stderr=clean(decode(exc.stdout),work),clean(decode(exc.stderr),work)
            status='timeout'
        except OSError as exc:
            stderr=str(exc)
            status='launch_error'
    diagnostic=next((line for line in stderr.splitlines()
                     if re.search(r'fatal error:|\berror:|undefined reference|multiple definition|cannot find',line)), '')
    return dict(meta,origin='synthetic_regression',split='unassigned',status=status,
                return_code=code,diagnostic=diagnostic,raw_stdout=stdout,raw_stderr=stderr)


def main():
    configure_output()
    parser=argparse.ArgumentParser(description='54 учебных сборки и отчёт сравнения правил.')
    parser.add_argument('--compiler',default='g++')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--timeout',type=int,default=30)
    args=parser.parse_args()
    if args.timeout<1:
        parser.error('--timeout должен быть положительным')
    compiler=shutil.which(args.compiler)
    if not compiler:
        print('g++ не найден. Укажите --compiler или настройте PATH.',file=sys.stderr)
        return 2
    try:
        version=compiler_info(compiler,'--version')
        target=compiler_info(compiler,'-dumpmachine')
        cases=[]
        ids=set()
        for manifest in sorted((ROOT/'projects').glob('*/case.json')):
            meta=json.loads(manifest.read_text(encoding='utf-8'))
            if not re.fullmatch(r'[a-z_]+[0-9]+',meta['id']) or meta['id'] in ids:
                raise ValueError('Недопустимый/дублированный id в '+str(manifest))
            if not isinstance(meta['arguments'],list) or not all(isinstance(a,str) for a in meta['arguments']):
                raise ValueError('Аргументы должны быть списком строк')
            ids.add(meta['id'])
            cases.append((manifest.parent,meta))
        if len(cases)!=54:
            raise ValueError('Ожидалось 54 проекта, найдено '+str(len(cases)))
        output=args.output or ROOT/('final_results_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        output.mkdir(exist_ok=False)
        (output/'logs').mkdir()
        rows=[]
        env=dict(os.environ,LC_ALL='C',LANG='C',GCC_COLORS='')
        # Запись и flush после каждого проекта сохраняют промежуточный результат.
        with (output/'results.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for folder,meta in cases:
                row=run_case(compiler,meta,folder,env,args.timeout)
                row.update(tool='g++',tool_version=version,host_os=platform.system(),target_triplet=target)
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
                (output/'logs'/(meta['id']+'.txt')).write_text('STDOUT:\n'+row['raw_stdout']+'\nSTDERR:\n'+row['raw_stderr'],encoding='utf-8')
                print(meta['id']+': '+row['status'])
        summary=write_report(output,rows)
        # Версии исходников правил фиксируются вместе с новым результатом.
        (output/'environment.json').write_text(json.dumps({
            'compiler_version':version,'target_triplet':target,'host_os':platform.system(),
            'python':sys.version,'classifier_v1_sha256':hashlib.sha256((ROOT/'baseline_v1/classifier.py').read_bytes()).hexdigest(),
            'classifier_v2_sha256':hashlib.sha256((ROOT/'classifier_v2.py').read_bytes()).hexdigest()
        },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Результаты:',output.resolve())
        return 1 if summary['collection_problems'] else 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as exc:
        print('Ошибка: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
