#!/usr/bin/env python3
"""Проверить SHA-256 поставляемых исходников; результаты опытов не включены.

Это контроль случайной порчи/собственных изменений, не цифровая подпись.
Ничего не удаляет, не восстанавливает и не запускает сборку.
"""
import hashlib
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')
    failures = 0
    checked = 0
    try:
        lines = (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
        if not lines:
            raise ValueError('Empty SHA256SUMS.txt')
        seen = set()
        for line in lines:
            expected, name = line.split('  ', 1)
            relative = PurePosixPath(name)
            if (len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)
                    or relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or name in seen):
                raise ValueError('Invalid or duplicate manifest entry')
            seen.add(name)
            path = (ROOT/relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Manifest path leaves NN')
            digest = hashlib.sha256()
            try:
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError('SHA-256 mismatch')
            except (OSError, ValueError) as error:
                failures += 1
                print('FAILED:', name, str(error), file=sys.stderr)
            checked += 1
    except (OSError, ValueError) as error:
        print('ERROR:', str(error), file=sys.stderr)
        return 1
    print('SHA-256 checked:', checked, 'files; failures:', failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
