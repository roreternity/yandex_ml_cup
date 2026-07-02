# Task C — School-Question Answering with an LLM

## Task

Answer school-level questions (in Russian) correctly, briefly, and to the point — fully
offline in a Docker container under a tight compute/time budget.

## Solution

A small **Qwen3** model (~0.6B params) served with **vLLM**, plus hybrid routing:

```
question ─┬─ looks like pure arithmetic? ─ yes ─▶ exact evaluator (Fraction math)
          └─ no ──────────────────────────────▶ vLLM (Qwen3, greedy decoding)
```

- **Arithmetic path** (`try_arithmetic`): conservatively detects pure computations and
  evaluates them exactly with `fractions.Fraction` over a whitelisted AST (never `eval()`).
  Small LLMs get arithmetic wrong; this makes those questions always correct.
- **LLM path**: greedy decoding (`temperature=0.0`), `max_tokens=192`, short `max_model_len`
  for throughput. A system prompt enforces short, correct answers; raw output is cleaned of
  `<think>` blocks and chat-template markers.

## Model weights

The fine-tuned Qwen3 weights (~1.4 GB) are hosted on Hugging Face, not in this repo:

**https://huggingface.co/roreternity/yandex_mlcup_taskc**

Download them into `task_c_solution/weights/` before running the solution.

## Files

| File | Description |
|---|---|
| `task_c_solution/solution.py` | Arithmetic router + vLLM inference + output cleanup |
| `task_c_solution/weights/` | Model weights — download from Hugging Face (link above) |
| `notebook_task_c_llm_qa.ipynb` | Step-by-step walkthrough notebook |
