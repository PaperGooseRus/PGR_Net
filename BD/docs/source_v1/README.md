# PGR Neural · Synthetic Diagnostic Corpus v1

**Назначение:** воспроизводимый набор маленьких C11/C++17/CMake-проектов для обучения и проверки собственного классификатора PGR Neural. **Все 7424 проекта являются синтетическими.** Каждая из 116 категорий имеет 32 преднамеренно ошибочных случая и 32 парные исправные версии. Только стандартная библиотека C/C++, GCC/Clang, CMake, Ninja; нет Qt SDK, внешних пакетов или интернет-запросов.

## Состав

| Элемент | Содержание |
|---|---|
| `projects/p_<hash>/` | Самостоятельные исходники и `CMakeLists.txt`; 7424 минипроекта |
| `labels/projects.jsonl` и `projects.csv` | Машинная разметка каждого проекта: класс, этап, язык, разделение, ID пары |
| `labels/fault_taxonomy.json` | 116 типов ошибок: C, C++, линковка, CMake, выполнение |
| `labels/{train,val,test}.jsonl` | Разбиение по **парам**: 5568 / 928 / 928 проектов |
| `labels/repair_pairs.jsonl` | 3712 патчей «неисправный → исправный» в формате unified diff |
| `observations/{gcc,clang}_observations.jsonl` | Фактически полученные сообщения компилятора, ошибки, коды завершения |
| `observations/{gcc,clang}_report.json` | Проверки ожидаемой стадии и распределение результатов |
| `training/{train,val,test}_{gcc,clang}.jsonl` | Готовые записи для supervised training с исходниками, диагностикой, target |
| `generate.py`, `verify.py`, `make_repairs.py`, `prepare_training.py`, `audit.py` | Python-скрипты без pip-зависимостей |
| `docs/DATA_CONTRACT.md`, `docs/LIMITATIONS.md` | Контракт данных и границы применения |

**Классы по этапам:** 78 ошибок компиляции, 16 компоновки, 10 конфигурации CMake, 12 выполнения; класс `CLEAN` представляет 116 видов корректных программ.

**Баланс:** 3712 с дефектами + 3712 исправных. Для каждой категории: 24 неисправных и 24 исправных в train, 4+4 в val, 4+4 в test. Парные проекты никогда не оказываются в разных выборках. Имена `p_<hash>` не содержат класса ошибки.

## Проверить

Для текущего каталога распакованного архива:

```powershell
python audit.py --root .
python verify.py --root . --compiler gcc --sample-per-family 1 --strict
python verify.py --root . --compiler clang --sample-per-family 1 --strict
```

Для **полной перепроверки всех 7424 проектов** (перезапишет файлы наблюдений):

```powershell
python verify.py --root . --compiler gcc --jobs 4 --strict
python verify.py --root . --compiler clang --jobs 4 --strict
python prepare_training.py --root . --include-sources
```

Требуются `gcc`, `g++`, `clang`, `clang++`, `cmake`, опционально `ninja` в PATH. В MSYS2 UCRT64 можно использовать `python` или `py -3`. Если компилятор не установлен, соответствующую команду не выполняйте. Папку `observations/` из архива можно использовать без пересборки.

**Воспроизвести из генератора:**

```powershell
python generate.py --out PGR_Recreated --copies 32 --seed 20261028
python make_repairs.py --root PGR_Recreated
python verify.py --root PGR_Recreated --compiler gcc --strict
python prepare_training.py --root PGR_Recreated --compiler gcc --include-sources
```

Для увеличения количества пар используйте `--copies 64` (14 848 проектов) или `--copies 128` (29 696 проектов). Увеличение числа копий **не** увеличивает число классов ошибок и само по себе не гарантирует более высокое качество модели.

## Для PGR Rule Classifier / Neural Baseline

1. **Исправность сборки:** бинарная классификация `target.build_ok`.
2. **Этап сбоя:** `target.stage` (`compile_error`, `link_error`, `configure_error`, `runtime_error`, `success`).
3. **Тип диагностики:** `target.fault_code`, 116 категорий + `CLEAN`.
4. **Возможное исправление:** `labels/repair_pairs.jsonl` и исходники исправной пары. Нельзя подавать исправленную версию или patch в входную последовательность классификатора: это **целевой ответ**.

Поля из `labels/` применяются как метки, **не как текст на входе**. Для обучения на **логе ошибки** использовать `input.diagnostic`, `input.build_exit_code` и `input.command_phase`; для анализа исходников дополнительно `input.files`. Не передавайте тестовые данные в подбор гиперпараметров. При использовании обоих компиляторов группировать все наблюдения одного `project_id` в одну выборку.

**Статус:** набор предназначен для машинного обучения, а не для имитации репрезентативности реального ПО. Синтетическая точность не эквивалентна качеству на реальных сборках PGR_Make. Нейросеть должна оставаться консультативным компонентом; PGR Core сохраняет детерминированную проверку.

## Результат независимой проверки (Linux)

На GCC 14.2 проверено **7424/7424**, на Clang 17 проверено **7424/7424**; 0 расхождений (`labels/validation_summary.json`). Логи сохранены для каждого проекта на обоих компиляторах. Поведение заново проверяется на конкретной Windows-машине при необходимости: результаты запуска не переносятся автоматически на иной toolchain/OS.

Целостность после распаковки: `python check_integrity.py`. Манифест `SHA256SUMS.txt` содержит SHA-256 всех файлов корпуса, кроме самого манифеста.
