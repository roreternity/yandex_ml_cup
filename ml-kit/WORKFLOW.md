# Порядок работы

## Первые 20 минут

1. Прочитать метрику и формат submission.
2. Открыть `sample_submission.csv`: колонки, id, число строк.
3. Посмотреть `train.csv` и `test.csv`: shape, dtypes, пропуски, target.
4. Сделать простой baseline.
5. Проверить submission валидатором.
6. Записать эксперимент.
7. Только потом улучшать модель.

## Частые команды

Табличная задача:

```bash
python templates/tabular_baseline.py --train data/train.csv --test data/test.csv --target target --id-col id --metric auto --output sub_lgbm_001.csv
```

Текстовая задача:

```bash
python templates/text_tfidf_baseline.py --train data/train.csv --test data/test.csv --text-col text --target target --id-col id --metric auto --output sub_tfidf_001.csv
```

Проверка submission:

```bash
python scripts/validate_submission.py --sample data/sample_submission.csv --submission sub_lgbm_001.csv
```

Blending:

```bash
python scripts/blend_submissions.py --sample data/sample_submission.csv --subs sub1.csv sub2.csv --weights 0.7,0.3 --output blend.csv
```

Repair:

```bash
python scripts/repair_submission.py --sample data/sample_submission.csv --submission broken.csv --clip-min 0 --clip-max 1 --output fixed.csv
```

