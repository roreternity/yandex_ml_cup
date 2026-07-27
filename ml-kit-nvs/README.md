# ML Kit NVS

Набор для задач типа CV / autonomous driving / Novel View Synthesis, где ответ - папка с изображениями:

```text
submission/<sample_id>/pred.jpg
```

## Быстрый старт

```bash
cd ml-kit-nvs
python scripts/inspect_nvs_dataset.py --dataset dataset
python scripts/nvs_copy_baseline.py --test-dir dataset/test --output submission --mode same_camera_blend
python scripts/validate_image_submission.py --test-dir dataset/test --submission submission --strict-size
python scripts/zip_submission.py --submission submission --output submission.zip
```

## Что делает baseline

Для каждого test sample:

1. Читает `meta.json`.
2. Берёт `target_camera`.
3. Находит эту камеру в `input/t0` и `input/t1`.
4. Делает resize под intrinsics target camera.
5. Сохраняет `submission/<sample_id>/pred.jpg`.

Первый смысл такого baseline - быстро получить валидный submission. Качество улучшать после проверки формата.

