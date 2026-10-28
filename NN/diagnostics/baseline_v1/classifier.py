"""Простой классификатор диагностики GCC/MinGW по заданным правилам.

Правила экспериментальные. Они анализируют одну строку диагностики и стадию;
если информации недостаточно или найдено несколько категорий, возвращают unknown.
Функция не читает файлы проекта, не выполняет команды и не изменяет данные.
"""
from dataclasses import dataclass
import re

UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Prediction:
    """Категория, название совпавшего правила и краткое объяснение."""
    label: str
    rule: str
    explanation: str


# Строки и регулярные выражения относятся к английским сообщениям GCC/MinGW.
# Приводим текст к нижнему регистру, поэтому шаблоны записаны строчными буквами.
HEADER = re.compile(r"fatal error:\s*[^\r\n:]+\.(?:h|hh|hpp|hxx)\s*:\s*no such file or directory")


def classify(message, stage='unknown'):
    """Вернуть Prediction для одной строки; пустой/неизвестный текст даёт отказ.

    Допустимые стадии у правил ниже ограничены намеренно. Значение unknown
    допускается как отсутствие контекста, но текст всё равно должен совпасть.
    Если текст подходит к разным категориям, ответом будет unknown.
    """
    if not isinstance(message, str) or not message.strip():
        return Prediction(UNKNOWN, 'empty', 'Сообщение отсутствует или пустое.')
    text = message.casefold()
    stage = stage.casefold() if isinstance(stage, str) else 'unknown'
    matches = []

    def add(label, rule, explanation):
        matches.append(Prediction(label, rule, explanation))

    if stage in ('compile', 'unknown') and HEADER.search(text):
        add('header_missing', 'gcc_missing_header', 'Заголовочный файл не найден.')
    if stage in ('link', 'unknown') and 'undefined reference to' in text:
        add('undefined_reference', 'ld_undefined_reference', 'Не найдено определение символа.')
    if stage in ('link', 'unknown') and 'multiple definition of' in text:
        add('multiple_definition', 'ld_multiple_definition', 'Найдено повторное определение символа.')
    if stage in ('invocation', 'compile', 'unknown') and 'unrecognized command-line option' in text:
        add('unknown_option', 'gcc_unrecognized_option', 'Компилятор не распознал параметр.')
    if stage in ('compile', 'unknown') and re.search(r"\berror:.*(?:was not declared in this scope|has not been declared)", text):
        add('undeclared_identifier', 'gcc_undeclared_identifier', 'Имя не объявлено в доступной области.')
    if stage in ('compile', 'unknown') and re.search(r'\berror:.*(?:invalid conversion|cannot convert)', text):
        add('type_mismatch', 'gcc_incompatible_types', 'Обнаружено несовместимое преобразование типов.')
    # Синтаксические правила ограничены тремя формами, встречавшимися в данных.
    # Общий поиск слова expected слишком часто классифицирует другие ошибки.
    if stage in ('compile', 'unknown') and re.search(r"\berror: expected (?:['‘][,;]['’](?: or ['‘][,;]['’])?|primary-expression) before", text):
        add('syntax_error', 'gcc_expected_token', 'В выражении или конструкции ожидается другой элемент.')

    labels = {item.label for item in matches}
    if len(labels) == 1:
        return matches[0]
    if len(labels) > 1:
        return Prediction(UNKNOWN, 'ambiguous', 'Подошло несколько несовместимых правил.')
    return Prediction(UNKNOWN, 'no_rule', 'Подходящее правило не найдено.')
