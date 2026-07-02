import ast
import json
import os
import pickle
import re
import sys
import traceback
from fractions import Fraction

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/weights"
INPUT_PATH = "/workspace/input.pickle"
OUT_DIR = "/workspace/out"
OUTPUT_PATH_OUT = "/workspace/out/output.json"
OUTPUT_PATH_ROOT = "/workspace/output.json"

SYSTEM_PROMPT = (
    "Ты помощник для школьных вопросов. "
    "Отвечай правильно, кратко и по делу. "
    "Если нужно решение — дай короткое понятное решение. "
    "Не пиши лишние рассуждения."
)


def load_items():
    with open(INPUT_PATH, "rb") as f:
        items = pickle.load(f)
    return items if isinstance(items, list) else []


def make_prompt(question) -> str:
    q = str(question).strip()
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{q}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# -------- small deterministic arithmetic, no extra packages --------

def _eval_frac(node):
    if isinstance(node, ast.Expression):
        return _eval_frac(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value, 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_frac(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_frac(node.operand)
    if isinstance(node, ast.BinOp):
        a = _eval_frac(node.left)
        b = _eval_frac(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError
            return a / b
    raise ValueError("bad expr")


def try_arithmetic(question):
    q = str(question).lower().strip()
    if len(q) > 140:
        return None
    if not any(w in q for w in ("сколько", "вычисли", "посчитай", "найди значение", "реши пример")):
        return None
    if not any(op in q for op in ("+", "-", "*", "/", ":")):
        return None

    candidates = re.findall(r"[0-9][0-9\s\+\-\*/:\.,\(\)]{1,90}[0-9\)]", q)
    if not candidates:
        return None
    expr = max(candidates, key=len)
    expr = expr.replace(",", ".").replace(":", "/")
    expr = re.sub(r"\s+", "", expr)
    if not re.fullmatch(r"[0-9\+\-\*/\.\(\)]+", expr):
        return None
    if not any(op in expr for op in ("+", "-", "*", "/")):
        return None
    try:
        val = _eval_frac(ast.parse(expr, mode="eval"))
    except Exception:
        return None
    if val.denominator == 1:
        return f"Ответ: {val.numerator}."
    return f"Ответ: {val.numerator}/{val.denominator}."


def clean_answer(text):
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.replace("<think>", "").replace("</think>", "")
    for marker in ("<|im_start|>user", "<|im_start|>system", "<|im_start|>assistant"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = text.strip()
    if len(text) > 1800:
        cut = text[:1800]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut.strip()
    return text


def write_outputs(items, answers):
    os.makedirs(OUT_DIR, exist_ok=True)
    result = []
    for item, answer in zip(items, answers):
        rid = item.get("rid") if isinstance(item, dict) else None
        result.append({"rid": rid, "answer": clean_answer(answer)})
    for path in (OUTPUT_PATH_OUT, OUTPUT_PATH_ROOT):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
        except Exception:
            pass


def generate_with_vllm(items):
    answers = [None] * len(items)
    llm_items = []
    llm_indices = []

    for i, item in enumerate(items):
        q = item.get("question", "") if isinstance(item, dict) else ""
        det = try_arithmetic(q)
        if det is not None:
            answers[i] = det
        else:
            llm_items.append(item)
            llm_indices.append(i)

    if llm_items:
        prompts = [make_prompt((item.get("question", "") if isinstance(item, dict) else "")) for item in llm_items]
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.88,
            max_model_len=1024,
        )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=192,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = llm.generate(prompts, sampling)
        for idx, out in zip(llm_indices, outputs):
            if out.outputs:
                answers[idx] = clean_answer(out.outputs[0].text)
            else:
                answers[idx] = ""

    return [a if a is not None else "" for a in answers]


def main():
    try:
        items = load_items()
    except BaseException:
        os.makedirs(OUT_DIR, exist_ok=True)
        for path in (OUTPUT_PATH_OUT, OUTPUT_PATH_ROOT):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False)
            except Exception:
                pass
        raise

    try:
        answers = generate_with_vllm(items)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        answers = [try_arithmetic((item.get("question", "") if isinstance(item, dict) else "")) or "" for item in items]

    write_outputs(items, answers)


if __name__ == "__main__":
    main()
