# Yandex ML Kit

Локальный набор скриптов для ML-соревнований: быстрый осмотр окружения, baseline для табличных/текстовых/image-задач, проверка submission, blending, repair и логирование экспериментов.

## Быстрый старт

```bash
cd ml-kit
python scripts/check_env.py
```

Если данные лежат в `data/`, типичный порядок такой:

```bash
python templates/tabular_baseline.py --train data/train.csv --test data/test.csv --target target --id-col id --metric auto --output sub_lgbm_001.csv
python scripts/validate_submission.py --sample data/sample_submission.csv --submission sub_lgbm_001.csv
python scripts/log_experiment.py --name lgbm_001 --metric auc --score 0.8123 --notes "raw baseline"
```

## Что внутри

- `templates/tabular_baseline.py` - LightGBM baseline для табличных задач.
- `templates/text_tfidf_baseline.py` - TF-IDF baseline для текстовой классификации/регрессии.
- `templates/image_torch_baseline.py` - простой torchvision baseline для image classification.
- `scripts/validate_submission.py` - проверка CSV submission против `sample_submission.csv`.
- `scripts/adversarial_validation.py` - диагностика train/test shift.
- `scripts/blend_submissions.py` - weighted average нескольких submission.
- `scripts/repair_submission.py` - восстановление порядка id/колонок, заполнение NaN, clipping.
- `scripts/metric_sandbox.py` - быстрая проверка метрики на `truth.csv` и `pred.csv`.
- `scripts/log_experiment.py` - JSONL-лог экспериментов.
- `snippets/README.md` - короткие сниппеты для notebook.

## Правило

Сначала сделать валидный baseline submission. Потом улучшать качество.

