#!/usr/bin/env python3
"""Создать непромаркированный шаблон для ручной проверки из существующего build log.

Не исполняет файл и не импортирует его в BD. Логи могут содержать персональные
сведения; их необходимо просмотреть и обезличить до команды ingest.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True, help='Текстовый журнал компилятора UTF-8')
    parser.add_argument('--output', type=Path, required=True, help='Новый JSONL; перезапись запрещена')
    parser.add_argument('--source-group', required=True, help='Общая группа версий одного проекта')
    parser.add_argument('--language', choices=['c','cpp'], required=True)
    parser.add_argument('--compiler', choices=['gcc','clang'], required=True)
    parser.add_argument('--build-system', choices=['cmake','direct+cmake'], default='cmake')
    parser.add_argument('--phase', default='compile+link')
    parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Output already exists: refuses to overwrite')
        text = args.log.read_text(encoding='utf-8-sig')
        if len(text) > 24000:
            raise ValueError('Log exceeds 24000 characters: review and shorten')
        result = {'source_group':args.source_group,'language':args.language,
                  'compiler':args.compiler,'build_system':args.build_system,
                  'exit_code':args.exit_code,'command_phase':args.phase,
                  'diagnostic':text,'files':{},'reviewed':False,'label':None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        print('UNREVIEWED template:', args.output)
        print('Next: review/redact the log, set a taxonomy label, and mark reviewed=true only after verification.')
    except (OSError, ValueError) as exc:
        raise SystemExit('CAPTURE ERROR: ' + str(exc)) from exc


if __name__ == '__main__':
    main()
