#!/usr/bin/env python3
"""Проверить выпущенный снимок BD по SHA-256; не исправляет изменённые файлы."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'manifests/SHA256SUMS.txt'
    if not manifest.is_file():raise SystemExit('Missing manifests/SHA256SUMS.txt')
    checked=0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line:continue
        digest,name=line.split('  ',1)
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():raise SystemExit('Invalid/missing file: '+name)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise SystemExit('Checksum mismatch: '+name)
        checked+=1
    print('SHA-256 verified:',checked,'files. Extra unlisted files are not checked.')

if __name__=='__main__':main()
