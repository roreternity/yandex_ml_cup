# Task A — Adaptive Discrete Puzzle Solver

## Task

Build one universal algorithm that, given **50 min to train** and **25 min to solve**,
solves *any* reversible discrete puzzle through a single `gym.py` API — including hidden
puzzles it has never seen. The solver talks to the environment only through
`reset / valid_actions / step / is_solved / encode_state` and is never told which puzzle
it is playing.

Open puzzles: 15-puzzle (`SWAP`+`EMPTY`), Lights Out (`TOGGLE`), Varykon cylinder (`ROTATE`).

## Solution

A structure-aware cascade — exact solvers first, generic search last:

- **Lights Out → GF(2) linear solver** (Gaussian elimination mod 2): exact and instant.
- **15-puzzle → fast-array IDA\*** (Manhattan + linear conflict), then beam search.
- **Rotational / hidden → long beam search**, then A\* guided by a learned value net `V(s)`.

`V(s)` is trained on backward random walks from the solved state (free labeled data) and
reads through `encode_state`, so one model works on every puzzle. Final score: **82**.

## Folders & files

| Item | Description |
|---|---|
| `writeup_task_a.md` | Narrative of the whole approach and how the score went 63 → 82 |
| `baseline/` | The organizers' baseline (gym API + simple `V(s)` A\*) |
| `solution_main_code/` | The main, most-developed solver code (the cascade) |
| `solution_with_experiments/` | An earlier code version **plus** all the `work_*` experiment runs and `experiments.md` decision notes |
| `submission/` | Generated per-instance solution outputs |
| `notebook_task_a_puzzles.ipynb` | Step-by-step walkthrough notebook |
| `*.zip` | Packaged (submittable) copies of the folders above |

Inside a solver folder: `gym.py` (environments), `common.py` (tokenization + backward walks),
`model.py` (`ValueNet`), `search.py` (A\*), `train.py` (trains `V`, builds the reverse-BFS table),
`solve.py` (the cascade → `output_actions.csv`), `generate_states.py`, `check.py` (scoring).
