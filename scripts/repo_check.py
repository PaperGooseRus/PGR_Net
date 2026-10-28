#!/usr/bin/env python3
"""Dependency-free git hygiene and optional Python syntax validation.

Checks the *staged/tracked* files, not only the ignore rules. An existing Git
index may contain files committed before the .gitignore was introduced.
"""
import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = (
    ROOT / '.editorconfig',
    ROOT / '.clang-format',
    ROOT / '.gitattributes',
    ROOT / '.gitignore',
    ROOT / 'pyproject.toml',
)


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8'
    )


def check_git() -> int:
    inside = run_git('rev-parse', '--is-inside-work-tree')
    if inside.returncode:
        print('NOTICE: no .git directory; Git tracking will be checked after git init/clone.')
        return 0
    ignored = run_git('ls-files', '-ci', '--exclude-standard', '-z')
    if ignored.returncode:
        print(ignored.stderr, file=sys.stderr)
        return 2
    bad = [item for item in ignored.stdout.split('\0') if item]
    if bad:
        print('ERROR: files already tracked despite .gitignore:', file=sys.stderr)
        for item in bad[:40]:
            print('  ' + item, file=sys.stderr)
        if len(bad) > 40:
            print(f'  ... and {len(bad) - 40} more', file=sys.stderr)
        print('Remove unwanted paths from Git INDEX only with git rm --cached.', file=sys.stderr)
        return 1
    print('Git: no ignored files remain tracked')
    return 0


def check_syntax() -> int:
    errors = 0
    count = 0
    for root in (ROOT / 'BD' / 'tools', ROOT / 'NN', ROOT / 'scripts'):
        if not root.exists():
            continue
        for path in root.rglob('*.py'):
            if any(part in {'__pycache__', 'build'} or part.startswith('nn_results_')
                   for part in path.relative_to(ROOT).parts):
                continue
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            except (SyntaxError, UnicodeError) as error:
                print(f'Invalid Python source: {path.relative_to(ROOT)}: {error}', file=sys.stderr)
                errors += 1
            count += 1
    print(f'Python syntax: checked {count}; failures {errors}')
    return 1 if errors else 0


def check_policy() -> int:
    failures = 0
    for path in (*POLICY_FILES, *sorted((ROOT / 'scripts').glob('*.py'))):
        data = path.read_bytes()
        if b'\r\n' in data or not data.endswith(b'\n') or b'\t' in data:
            print('Style: invalid LF/newline/indent in', path.name, file=sys.stderr)
            failures += 1
        if any(line.rstrip(b' \t') != line for line in data.splitlines()):
            print('Style: trailing whitespace in', path.name, file=sys.stderr)
            failures += 1
    print(f'Formatting policy: {"OK" if not failures else "FAILED"}')
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--syntax', action='store_true', help='Parse project Python modules')
    args = parser.parse_args()
    return check_syntax() if args.syntax else max(check_policy(), check_git())


if __name__ == '__main__':
    raise SystemExit(main())
