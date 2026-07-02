# Yandex ML Cup — Solutions

My solutions to the three tracks of the Yandex ML Cup ML challenge.

| Track | Task | Where |
|---|---|---|
| **A** | Adaptive discrete-puzzle solver | [`Task_A_Puzzles/`](Task_A_Puzzles/) |
| **B** | Multi-camera novel-view synthesis | [`Task_B_ViewSynthesis/`](Task_B_ViewSynthesis/) · data on [HF](https://huggingface.co/datasets/roreternity/yandex_mlcup_taskb) |
| **C** | School-question answering with an LLM | [`Task_C_LLM_QA/`](Task_C_LLM_QA/) |

Each task folder has a `README.md` with the task statement and the solution code.

## Large assets (hosted outside GitHub)

GitHub caps files at 100 MB, so the big data and model weights live elsewhere:

| Asset | Size | Link |
|---|---|---|
| Task B — multi-camera + LiDAR dataset | ~22 GB | [Hugging Face](https://huggingface.co/datasets/roreternity/yandex_mlcup_taskb) |
| Task C — Qwen3 model weights | ~1.4 GB | [Hugging Face](https://huggingface.co/roreternity/yandex_mlcup_taskc) |

To run a task, download its asset above and drop it back into that task's folder
(Task B → `task_b_data/`, Task C → `task_c_solution/weights/`).
