# Формат датасета PGR Neural (v1)

Все файлы UTF-8, JSON Lines: одна JSON-запись на строку. Не требуется pip.

## labels/projects.jsonl

- `project_id`: нейтральное имя папки `p_<hash>`.
- `pair_id`: одна ошибка и её исправленная версия.
- `split`: `train | val | test`; назначается вместе для пары.
- `language`: `c | cpp`.
- `expected_stage`: `success | compile_error | link_error | configure_error | runtime_error`.
- `fault_code`: `CLEAN` либо один из 116 стабильных кодов.
- `expected_ok`: логическое `true | false`.
- `sources`: имена исходников, `relative_path`: путь относительно корня набора.
- `build_system`: `direct+cmake` либо `cmake` (для CMake-кейсов).

## observations/*_observations.jsonl

`project_id, pair_id, split, language, fault_code, expected_stage, observed_stage, match, exit_code, compiler, command_phase, diagnostic, stdout, relative_path, elapsed_ms`.

`diagnostic`: stderr, приведённый к нейтральным `<DATA_ROOT>`, `<BUILD_DIR>`, `<PROJECT>`; обрезан до 24000 символов. `observed_stage` установлен по фактическому завершению и тексту диагностики. При обнаружении несоответствия инструмент не скрывает результат.

## training/*

`{"id": "...", "pair_id": "...", "input": {"language": "c", "compiler": "gcc", "build_system": "direct+cmake", "build_exit_code": 1, "command_phase": "compile+link", "diagnostic": "...", "files": {"main.c": "..."}}, "target": {"stage": "compile_error", "fault_code": "C_UNDECLARED", "build_ok": false}}`.

`files` присутствует при `prepare_training.py --include-sources`.

## labels/repair_pairs.jsonl

`pair_id, split, fault_code, broken_project_id, fixed_project_id, patch`. `patch` = unified diff. Сопоставления пары и патчи **не допускаются на вход обычного классификатора**, чтобы избежать утечки ответов.

## Воспроизводимость

- Программа-генератор всегда получает фиксированные `--seed` и `--copies`.
- Небинарные синтетические проекты не имеют внешних зависимостей; исполнять их рекомендуется в отдельной тестовой среде.
- Диагностика может отличаться по версиям GCC/Clang/CMake/ОС; классификационные метки соответствуют **ожидаемому этапу**. Скрипт `verify.py --strict` сообщает о расхождениях.
- Дата подготовки: 2026-10-28.
