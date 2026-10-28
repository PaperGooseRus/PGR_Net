#!/usr/bin/env python3
"""Сборка C++17-прототипа через g++ без зависимости от установленного CMake.

Создаёт только НОВУЮ папку результатов, компилирует собственные исходники,
запускает численные тесты и демонстрацию, сохраняет их вывод в UTF-8.
Пользователь явно запускает этот скрипт; это не выполнение действий моделью.
"""
import argparse
import csv
import math
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def configure_output():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            if stream.isatty():
                stream.reconfigure(errors='backslashreplace')
            else:
                stream.reconfigure(encoding='utf-8',errors='backslashreplace')


def main():
    configure_output()
    parser=argparse.ArgumentParser(description='Собрать прототип, запустить тесты и пример.')
    parser.add_argument('--compiler',default='g++')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    compiler=shutil.which(args.compiler)
    if not compiler:
        print('g++ не найден: укажите --compiler или настройте PATH.',file=sys.stderr)
        return 2
    out=(args.output or ROOT/('nn_results_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    try:
        out.mkdir(exist_ok=False)
    except OSError as exc:
        print('Нельзя создать новую папку: '+str(exc),file=sys.stderr)
        return 2
    records=[]
    # Полный путь к g++ может быть вне PATH. Его bin нужен также для DLL
    # MinGW при запуске собранных тестов на Windows.
    env=dict(os.environ, LC_ALL='C', LANG='C', GCC_COLORS='')
    env['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+env.get('PATH','')
    suffix='.exe' if os.name=='nt' else ''

    def run(name,command,timeout=120):
        p=subprocess.run([str(x) for x in command],capture_output=True,text=True,
                         encoding='utf-8',errors='replace',timeout=timeout,cwd=ROOT,env=env)
        (out/(name+'.log')).write_text('STDOUT:\n'+p.stdout+'\nSTDERR:\n'+p.stderr,encoding='utf-8')
        records.append({'step':name,'return_code':p.returncode,'command':[str(x) for x in command]})
        print(name+': '+('OK' if p.returncode==0 else 'FAILED'))
        if p.returncode:
            raise RuntimeError('Смотрите '+str(out/(name+'.log')))
        if name in ('tests','demo','gradient_tests','gradient_demo','training_tests','training_demo',
                    'storage_tests','model_demo','storage_cli_tests'):
            print(p.stdout,end='')
        return p.stdout

    status='failed'
    try:
        run('compiler_version',[compiler,'--version'],20)
        flags=['-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-Werror',
               '-I'+str(ROOT/'math/include'),'-I'+str(ROOT/'model/include'),
               '-I'+str(ROOT/'training/include'),'-I'+str(ROOT/'storage/include')]
        # Каждый модуль компилируется один раз. shell=True не используется.
        objects=[]
        for name,source in [('math','math/src/math.cpp'),('network','model/src/network.cpp'),('example','model/src/example.cpp'),('gradients','training/src/gradients.cpp'),('initialization','training/src/initialization.cpp'),('sgd','training/src/sgd.cpp'),('trainer','training/src/trainer.cpp'),('model_io','storage/src/model_io.cpp'),('file_io','storage/src/file_io.cpp')]:
            obj=out/(name+'.o')
            run('compile_'+name,[compiler,*flags,'-c',ROOT/source,'-o',obj])
            objects.append(obj)
        for name,source in [('pgr_neural_tests','tests/test_forward.cpp'),('pgr_neural_demo','demo/main.cpp'),('pgr_neural_gradient_tests','tests/test_gradients.cpp'),('pgr_neural_gradient_demo','demo/gradient_main.cpp'),('pgr_neural_training_tests','tests/test_training.cpp'),('pgr_neural_training_demo','demo/training_main.cpp'),('pgr_neural_storage_tests','tests/test_storage.cpp'),('pgr_neural_model_demo','demo/model_main.cpp')]:
            platform_libraries=['-lshell32'] if os.name=='nt' else []
            run('link_'+name,[compiler,*flags,ROOT/source,*objects,*platform_libraries,'-o',out/(name+suffix)])
        run('tests',[out/('pgr_neural_tests'+suffix)],30)
        run('demo',[out/('pgr_neural_demo'+suffix)],30)
        run('gradient_tests',[out/('pgr_neural_gradient_tests'+suffix)],30)
        run('gradient_demo',[out/('pgr_neural_gradient_demo'+suffix)],30)
        run('training_tests',[out/('pgr_neural_training_tests'+suffix)],30)
        run('storage_tests',[out/('pgr_neural_storage_tests'+suffix)],60)
        run('training_demo',[out/('pgr_neural_training_demo'+suffix),'--output',out/'training_demo'],30)
        # Проверка созданных файлов, а не только кода завершения демо.
        metrics=json.loads((out/'training_demo/run.json').read_text(encoding='utf-8'))
        with (out/'training_demo/history.csv').open(encoding='utf-8',newline='') as stream:
            history=list(csv.DictReader(stream))
        if len(history)!=81 or metrics['updates']!=160 or not math.isfinite(metrics['final_training_loss']):
            raise RuntimeError('Некорректный отчёт учебного обучения')
        if float(history[-1]['training_loss'])!=metrics['final_training_loss']:
            raise RuntimeError('CSV и JSON расходятся')
        if not metrics['weights_saved'] or not metrics['model_roundtrip_exact']:
            raise RuntimeError('Обученные веса не сохранены или не проверены')
        loaded=json.loads(run('model_demo',[out/('pgr_neural_model_demo'+suffix),
            '--input',out/'training_demo/model.pgrnn','--tokens','1,2','--compiler','0','--stage','0'],30))
        if any(loaded[key]!=metrics['probe'][key] for key in ('tokens','compiler_index','stage_index','logits','probabilities')):
            raise RuntimeError('Новый процесс вернул другие результаты сохранённой модели')
        run('storage_cli_tests',[sys.executable,ROOT/'tests/test_model_cli.py','--bin-dir',out],90)
        status='passed' 
    except (OSError,subprocess.TimeoutExpired,RuntimeError,ValueError,KeyError) as exc:
        print('Ошибка: '+str(exc),file=sys.stderr)
    finally:
        (out/'summary.json').write_text(json.dumps({'status':status,'steps':records,
            'scope':'SGD toy training and PGRNN v1 inference weights; no diagnostic corpus training, exact training resume or PGR_Make integration'},
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Результаты:',out)
    return 0 if status=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
