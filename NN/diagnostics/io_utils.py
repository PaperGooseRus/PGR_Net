"""Общие операции ввода/вывода; внешних пакетов нет."""
import json
import re
import sys
from pathlib import Path


def configure_output():
    """UTF-8 в pipe/файле, безопасная замена символов в старой консоли Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            if stream.isatty():
                stream.reconfigure(errors='backslashreplace')
            else:
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')


def clean(text, work=None):
    if work is not None:
        text = text.replace(str(work), '<WORKDIR>').replace(str(work).replace('\\','/'), '<WORKDIR>')
    text = re.sub(r'[A-Za-z]:[\\/][^\r\n:]*?[\\/]cc[A-Za-z0-9]+\.o', '<TEMP_OBJECT>', text, flags=re.I)
    return re.sub(r'/tmp/cc[A-Za-z0-9]+\.o', '<TEMP_OBJECT>', text)


def read_jsonl(path):
    """Строгий разбор JSONL: не допускает дубли id и смешанные типы записей."""
    rows, ids = [], set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
                raise ValueError('Строка %d: требуется объект с непустым строковым id' % n)
            if row['id'] in ids:
                raise ValueError('Повторный id: '+row['id'])
            ids.add(row['id'])
            rows.append(row)
    if not rows:
        raise ValueError('Пустой набор данных')
    return rows


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
