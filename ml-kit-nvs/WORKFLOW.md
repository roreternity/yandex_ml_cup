# Порядок работы для NVS / image-folder задач

## Первые 15 минут

1. Проверить структуру `dataset/train` и `dataset/test`.
2. Открыть один `meta.json`.
3. Понять `target_camera` и размер target image.
4. Сделать `same_camera_blend` baseline.
5. Проверить submission.
6. Собрать zip.

## Baseline modes

- `same_camera_t0` - взять target camera из `input/t0`.
- `same_camera_t1` - взять target camera из `input/t1`.
- `same_camera_blend` - усреднить `t0` и `t1`.
- `front_fallback` - взять front camera, если metadata или camera input сломаны.

## Улучшения после валидного zip

- выбрать `t0` или `t1` по близости target pose;
- warping по геометрии камер;
- lidar projection для depth hints;
- optical flow между `t0` и `t1`;
- inpaint пустых областей после warping.

