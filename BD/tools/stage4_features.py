#!/usr/bin/env python3
"""Чистые функции Stage 4. Кодирует только input, никогда target/id/pair_id.

Никаких внешних библиотек, сетевых вызовов и запуска компилятора.
"""
from __future__ import annotations
from collections import Counter
import re
from typing import Mapping

# Нормализуем изменчивые пути, идентификаторы и числа, не лишая
# диагностику ключевых слов компилятора и операторов C/C++.
_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_WINDOWS_PATH = re.compile(r'(?<!\w)[A-Za-z]:[\\/](?:[^\s\"\'<>|]+[\\/])*[^\s\"\'<>|:]+')
_SYNTH_PATH = re.compile(r'<DATA_ROOT>(?:/[a-zA-Z0-9_<>.-]+)+')
_HEX = re.compile(r'\b0x[0-9a-f]+\b', re.I)
_SUFFIX = re.compile(r'\b[a-z_][a-z0-9_]*_v[0-9a-f]{6,}\b', re.I)
_UUID = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)
_NUMBER = re.compile(r'(?<![a-z_])\d+(?:\.\d+)?(?![a-z_])', re.I)
_TOKENS = re.compile(r'[a-z_][a-z_0-9]*|::|->|==|!=|<=|>=|&&|\|\||[{}()\[\];,+*/%<>=!&|.#?:-]', re.I)

FEATURE_SCHEMA = 'pgr-features-v1'
MAX_DIAG_TOKENS = 600
MAX_SOURCE_TOKENS = 1200
MAX_PER_FILE = 800


def tokenize(value: str, limit: int) -> list[str]:
    """Разбить текст на воспроизводимые токены с ограничением длины."""
    if not isinstance(value, str):
        raise ValueError('Diagnostic/source must be a string')
    text = _ANSI.sub('', value).lower()
    text = _SYNTH_PATH.sub(' path_token ', text)
    text = _WINDOWS_PATH.sub(' path_token ', text)
    text = _UUID.sub(' uuid_token ', text)
    text = _SUFFIX.sub(' symbol_token ', text)
    text = _HEX.sub(' hex_token ', text)
    text = _NUMBER.sub(' num_token ', text)
    return _TOKENS.findall(text)[:limit]


def input_features(sample_input: Mapping) -> Counter[str]:
    """Признаки только из разрешённых наблюдаемых полей input.

    Исключено: target, id, pair_id, split, исправления, имена каталогов
    проектов и точное численное значение кода завершения.
    """
    if not isinstance(sample_input, dict):
        raise ValueError('Missing input object')
    required = ('language', 'compiler', 'build_system', 'command_phase',
                'build_exit_code', 'diagnostic', 'files')
    if any(field not in sample_input for field in required):
        raise ValueError('Missing input field')
    if sample_input['language'] not in ('c', 'cpp'):
        raise ValueError('Unknown language')
    if sample_input['compiler'] not in ('gcc', 'clang'):
        raise ValueError('Unknown compiler')
    if not isinstance(sample_input['build_exit_code'], int):
        raise ValueError('Exit status must be integer')
    if not isinstance(sample_input['files'], dict):
        raise ValueError('Files must be an object')
    if not isinstance(sample_input['build_system'], str) or not isinstance(sample_input['command_phase'], str):
        raise ValueError('Build metadata must be strings')

    # Метаданные инструмента разрешены, но исходная метка и имена проектов нет.
    features = Counter({
        'meta:compiler=' + sample_input['compiler']: 1,
        'meta:language=' + sample_input['language']: 1,
        'meta:system=' + sample_input['build_system'].lower(): 1,
        'meta:phase=' + sample_input['command_phase'].lower(): 1,
        'meta:exit=' + ('zero' if sample_input['build_exit_code'] == 0 else 'nonzero'): 1,
    })
    diagnostic = tokenize(sample_input['diagnostic'], MAX_DIAG_TOKENS)
    if not diagnostic:
        features['diag:empty'] = 1
    else:
        features.update('diag:' + term for term in diagnostic)

    remaining = MAX_SOURCE_TOKENS
    for file_name, content in sorted(sample_input['files'].items()):
        if not isinstance(file_name, str) or not isinstance(content, str):
            raise ValueError('File names and contents must be strings')
        if remaining <= 0:
            break
        if file_name.lower() == 'cmakelists.txt':
            features['meta:cmake_file'] += 1
        elif file_name.lower().endswith('.cpp'):
            features['meta:source_cpp'] += 1
        elif file_name.lower().endswith('.c'):
            features['meta:source_c'] += 1
        elif file_name.lower().endswith(('.hpp', '.h')):
            features['meta:header'] += 1
        else:
            features['meta:other_file'] += 1
        terms = tokenize(content, min(MAX_PER_FILE, remaining))
        features.update('src:' + term for term in terms)
        remaining -= len(terms)
    # Ограничение повторов защищает от подавления признаков текстовым шумом.
    for name in list(features):
        features[name] = min(8, features[name])
    return features


def encode(features: Counter[str], vocabulary: Mapping[str, int]) -> tuple[list[int], list[int], int]:
    """Разреженный вектор; незнакомые токены учитываются в OOV, но не изучаются."""
    seen = Counter()
    missing = 0
    for term, count in features.items():
        if term in vocabulary:
            seen[vocabulary[term]] += count
        else:
            missing += count
    ordered = sorted(seen.items())
    return [i for i, _ in ordered], [n for _, n in ordered], missing
