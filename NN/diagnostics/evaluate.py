#!/usr/bin/env python3
"""Сравнение по сохранённым сообщениям: никакие программы не собирает."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from io_utils import configure_output, read_jsonl
from evaluation import compare, write_report
ROOT=Path(__file__).resolve().parent


def main():
    configure_output()
    parser=argparse.ArgumentParser(description='Оценить сохранённые данные без компилятора.')
    parser.add_argument('input', nargs='?', type=Path, help='JSONL; без аргумента — все включённые Windows-данные')
    parser.add_argument('--output',type=Path,help='Новый каталог отчёта')
    args=parser.parse_args()
    try:
        paths=[args.input] if args.input else sorted((ROOT/'data').glob('*.jsonl'))
        rows=[]
        for path in paths:
            for row in read_jsonl(path):
                if args.input is None:
                    row['id']=path.stem+'__'+row['id']
                rows.append(row)
        if not rows:
            raise ValueError('Нет данных')
        compare(rows)  # Проверяем вход до создания каталога.
        output=args.output or ROOT/('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        output.mkdir(exist_ok=False)
        write_report(output,rows)
        print('Результаты:',output.resolve())
        return 0
    except (OSError,ValueError,TypeError) as exc:
        print('Ошибка: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
