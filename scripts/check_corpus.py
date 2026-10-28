#!/usr/bin/env python3
"""Explain whether the optional full project corpus is present.

The supplied archive contains BD metadata, but not BD/projects. Do not mark a
partial corpus as successfully integrity-checked. CI tests dataset-only APIs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    projects = ROOT / 'BD' / 'projects'
    if not projects.is_dir():
        print('NOTICE: BD/projects is absent; full corpus check is NOT run.')
        print('The provided project snapshot contains only BD metadata and datasets.')
        return 0
    import subprocess
    import sys

    print('Full BD/projects available; running corpus integrity checks.')
    return subprocess.call([sys.executable, str(ROOT / 'BD' / 'bd.py'), 'check'])


if __name__ == '__main__':
    raise SystemExit(main())
