"""Правила v2: семь известных категорий и явный отказ unknown.

Результат описывает симптом, не причину и не способ исправления.
Поля label/expected_marker/notes не читаются: это разметка, а не признаки.
Вход не исполняется, пути из журнала не открываются. Поддержан английский GCC.
"""
from dataclasses import dataclass
import re

VERSION = '2.0.0-baseline'
KNOWN = frozenset(('header_missing','syntax_error','undeclared_identifier','type_mismatch',
                   'undefined_reference','multiple_definition','unknown_option'))


@dataclass(frozen=True)
class Prediction:
    label: str
    rule: str
    explanation: str


def reject(reason):
    return Prediction('unknown', reason, 'Недостаточно поддерживаемых признаков; требуется просмотр.')


def include_evidence(message, stderr, operand):
    """Проверить строку include в непосредственно связанном фрагменте GCC.

    Расширение файла не является доказательством: g++ может не найти даже
    входной файл с суффиксом .h. Используем номер строки диагностического
    сообщения и следующий фрагмент `N | #include ...`. Определённая через
    макрос директива не распознаётся и приводит к отказу. Это эвристика по
    журналу, а не формальная проверка препроцессора.
    """
    loc = re.search(r':(\d+)(?::\d+)?:\s*fatal error:', message)
    if not loc or not isinstance(stderr, str) or len(stderr) > 1048576:
        return False
    lines = stderr.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != message.strip():
            continue
        # GCC показывает исходную строку сразу после сообщения. Не ищем
        # include по всему логу: там может быть другая независимая ошибка.
        for excerpt in lines[i+1:i+3]:
            m = re.match(r'\s*(\d+)\s*\|\s*#\s*include(?:_next)?\s*[<"]([^>"\r\n]+)[>"]', excerpt)
            if m and m.group(1) == loc.group(1) and m.group(2) == operand:
                return True
    return False


def classify(row):
    """Классифицировать одну выделенную строку с необязательным raw_stderr.

    Не извлекает ошибки из произвольного общего лога. Слишком длинный,
    многострочный или неподдерживаемый вход отклоняется. Ограничения защищают
    обработчик от случайных гигантских блоков, не от любых атак на Python.
    """
    if not isinstance(row, dict):
        return reject('invalid_input')
    message = row.get('diagnostic')
    stage = row.get('stage', 'unknown')
    if not isinstance(message, str) or not message.strip():
        return reject('empty')
    if len(message) > 16384 or len(message.splitlines()) != 1:
        return reject('input_limit')
    if not isinstance(stage, str) or stage not in ('compile','link','invocation','unknown'):
        return reject('unsupported_stage')
    tool = row.get('tool', 'g++')
    if not isinstance(tool, str) or tool.casefold() not in ('gcc','g++','c++','gcc.exe','g++.exe'):
        return reject('unsupported_tool')
    text = message.strip().replace('‘', "'").replace('’', "'")
    severity = re.search(r'\b(fatal error|error|warning|note):', text)
    if severity and severity.group(1) in ('warning','note'):
        return reject('not_an_error')
    error = re.search(r'\b(fatal error|error):\s*(.*)$', text)
    matches = []

    def add(label, name):
        matches.append(Prediction(label, name, 'Совпадение с ограниченным шаблоном GCC; причина не установлена.'))

    # Не ищем ключевые слова компоновщика в строке компилятора: текст
    # static_assert/#error может дословно содержать undefined reference.
    if error:
        body = re.sub(r'\s+\[-[^\]\r\n]+\]\s*$', '', error.group(2))
        lower = body.casefold()
        if stage in ('compile','unknown'):
            header = re.fullmatch(r'(.+): No such file or directory', body)
            if error.group(1) == 'fatal error' and header:
                if include_evidence(text, row.get('raw_stderr', ''), header.group(1)):
                    add('header_missing', 'gcc_include_context')
            if re.fullmatch(r"'.+?' (?:was not declared in this scope|has not been declared)", body):
                add('undeclared_identifier','gcc_undeclared')
            # Только явные несовместимые преобразования/ссылки. Общая фраза
            # no matching function не доказывает несовместимость типов.
            if re.match(r'(?:invalid conversion|cannot convert|invalid initialization of reference|cannot bind .*reference|narrowing conversion)', lower):
                add('type_mismatch','gcc_type_conversion')
            if re.match(r"expected (?:(?:'[^']{1,3}'(?: or '[^']{1,3}')?)|primary-expression) (?:before|after|at end of input)", body):
                add('syntax_error','gcc_expected_delimiter')
        if stage in ('invocation','compile','unknown') and lower.startswith('unrecognized command-line option'):
            add('unknown_option','gcc_option')
    elif stage in ('link','unknown'):
        if 'undefined reference to' in text and 'multiple definition of' in text:
            return reject('conflicting_rules')
        # Требуем начало самостоятельного фрагмента после двоеточия.
        # Форматы без такого фрагмента остаются unknown.
        if re.search(r'(?:^|:\s*)undefined reference to\b', text):
            add('undefined_reference','ld_undefined')
        if re.search(r'(?:^|:\s*)multiple definition of\b', text):
            add('multiple_definition','ld_duplicate')
    labels = {x.label for x in matches}
    if len(labels) == 1:
        return matches[0]
    if len(labels) > 1:
        return reject('conflicting_rules')
    return reject('no_supported_rule')
