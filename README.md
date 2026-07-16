# Yandex ML Cup — Решения

Мои решения трёх треков ML-соревнования Yandex ML Cup.

| Трек | Задача | Папка |
|---|---|---|
| **A** | Адаптивный решатель дискретных головоломок | [`Task_A_Puzzles/`](Task_A_Puzzles/) |
| **B** | Синтез новых ракурсов по нескольким камерам | [`Task_B_ViewSynthesis/`](Task_B_ViewSynthesis/) · данные на [HF](https://huggingface.co/datasets/roreternity/yandex_mlcup_taskb) |
| **C** | Ответы на школьные вопросы с помощью LLM | [`Task_C_LLM_QA/`](Task_C_LLM_QA/) |

В каждой папке — `README.md` с описанием задачи и код решения.

## Большие файлы (хранятся вне GitHub)

GitHub ограничивает размер файлов 100 МБ, поэтому данные и веса моделей лежат отдельно:

| Ресурс | Размер | Ссылка |
|---|---|---|
| Задача B — мультикамерный + LiDAR датасет | ~22 ГБ | [Hugging Face](https://huggingface.co/datasets/roreternity/yandex_mlcup_taskb) |
| Задача C — веса модели Qwen3 | ~1.4 ГБ | [Hugging Face](https://huggingface.co/roreternity/yandex_mlcup_taskc) |

Чтобы запустить задачу, скачайте соответствующий ресурс и поместите в папку задачи
(задача B → `task_b_data/`, задача C → `task_c_solution/weights/`).
