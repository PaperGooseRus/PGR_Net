# Куда смотреть при работе

| Вопрос | Где искать |
|---|---|
| Какие проекты есть и какие у них метки? | catalog/projects.jsonl или projects.csv |
| Какие категории предусмотрены? | catalog/fault_taxonomy.json |
| Что входит в train/val/test? | splits/ |
| Где исходники конкретного проекта? | projects/ + поле relative_path |
| Что реально ответил GCC? | observations/gcc/observations.jsonl |
| Что реально ответил Clang? | observations/clang/observations.jsonl |
| Какие файлы читает будущий тренер? | datasets/<compiler>/<split>.jsonl |
| Где ожидаемое исправление? | repairs/repair_pairs.jsonl; это target, не input |
| Почему прогон не совпал с ожиданием? | reports/collection/ и diagnostic в observations |
| Все ли файлы сохранены? | python bd.py check |
| Нет ли разрыва пары между выборками? | python bd.py audit |
| Где старые отчёты с прежними утверждениями? | reports/history/ |

Новая нейросеть не реализована внутри BD. Генерация, компиляция, подготовка
данных и обучение — разные этапы. Скрипт prepare обновляет производные
выборки, а исходники/метки/наблюдения сохраняются в собственных каталогах.
Не перемешивайте эти уровни и не корректируйте match только ради зелёного отчёта.

## После этапа 3

| Вопрос | Где искать |
|---|---|
| Какие синтетические пары подтверждены двумя компиляторами? | datasets/verified/{gcc,clang}/ + reports/validation/quality_gate.json |
| Куда добавлять наблюдения с реальных сборок? | python bd.py ingest; catalog/real_cases.jsonl; observations/real/ |
| Что одобрено для обучения из реальных наблюдений? | datasets/real/ (после audit-real и export-real) |
| Как проверить собственные импортированные данные? | python bd.py audit-real; python bd.py snapshot-real --verify |

Подробности: `TRAINING_STAGE3_RU.md`.
