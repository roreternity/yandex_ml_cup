# Промпты для агента

## Осмотр

```text
Работай только с локальной папкой dataset. Определи структуру train/test, sample ids, поля meta.json, размер target image, target_camera и требуемый формат submission. Интернет не использовать. Выведи конкретные пути файлов и возможные точки поломки.
```

## Baseline

```text
Создай самый простой валидный image-folder submission. Для каждого test sample прочитай meta.json, найди target_camera, скопируй или усредни ту же камеру из input/t0 и input/t1, сделай resize до width/height из intrinsics, сохрани submission/<sample_id>/pred.jpg. Потом собери zip.
```

## Проверка

```text
Проверь формат submission. Для каждого test sample должен быть submission/<sample_id>/pred.jpg. Картинки должны читаться как RGB JPEG, размеры должны совпадать с target intrinsics, лишних и пропущенных sample folders быть не должно.
```

