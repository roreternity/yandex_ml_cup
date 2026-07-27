import ast
import gzip
import json
import os
import re
import pickle
import sys
import traceback
from fractions import Fraction

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/weights"
INPUT_PATH = "/workspace/input.pickle"
OUT_DIR = "/workspace/out"
OUTPUT_PATH_OUT = "/workspace/out/output.json"
OUTPUT_PATH_ROOT = "/workspace/output.json"
TRAIN_BANK_PATH = "/workspace/train_bank.jsonl.gz"

SYSTEM_PROMPT = (
    "Ты помощник для школьных вопросов по всем предметам (математика, русский, "
    "литература, история, биология, физика, химия и т.д.). "
    "Отвечай правильно и по существу. "
    "Если это задача или уравнение — покажи короткое решение по шагам и дай "
    "финальный ответ отдельной строкой в формате 'Ответ: ...'. "
    "Если нужно сочинение или развёрнутый ответ — пиши связно и по формату "
    "предмета (литература/история — развёрнуто; математика/физика — формулами). "
    "Не пиши лишних рассуждений и служебных пометок."
)

NEAR_DUP_THRESHOLD = 0.90
FEWSHOT_THRESHOLD = 0.30
FEWSHOT_K = 2
MAX_NEW_TOKENS = 768


# -------- normalization / lexical index over train bank --------

def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_set(text):
    return set(re.findall(r"[a-zA-Zа-яёA-ZА-ЯЁ0-9]+", text))


def char_ngrams(text, n=3):
    text = re.sub(r"\s+", "", text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class TrainBank:
    def __init__(self, path):
        self.entries = []
        self.by_norm = {}
        if not os.path.exists(path):
            return
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                norm = normalize(rec.get("norm") or rec.get("query") or "")
                answer = rec.get("answer", "")
                if not norm or not answer:
                    continue
                entry = {
                    "norm": norm,
                    "answer": answer,
                    "words": word_set(norm),
                    "grams": char_ngrams(norm),
                }
                self.entries.append(entry)
                self.by_norm.setdefault(norm, entry)

    def exact(self, question):
        return self.by_norm.get(normalize(question))

    def search(self, question, k=FEWSHOT_K):
        norm = normalize(question)
        qwords = word_set(norm)
        qgrams = char_ngrams(norm)
        scored = []
        for entry in self.entries:
            wj = jaccard(qwords, entry["words"])
            if wj == 0.0:
                continue
            gj = jaccard(qgrams, entry["grams"])
            score = 0.5 * wj + 0.5 * gj
            if score > 0.05:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# -------- deterministic shortcuts (no LLM needed) --------

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


# bare inequality/equation expressions like "-4<=x<-2" that just need
# LaTeX-style reformatting
_INEQ_CHARS = re.compile(r"^[\s0-9xXхХ\+\-\*/\.,<>=]+$")


def try_bare_inequality(question):
    q = str(question).strip()
    if len(q) > 40 or len(q) < 3:
        return None
    if not _INEQ_CHARS.fullmatch(q):
        return None
    if not any(op in q for op in ("<=", ">=", "<", ">")):
        return None
    if not re.search(r"[xXхХ]", q):
        return None
    out = q.replace("<=", " \\leq ").replace(">=", " \\geq ")
    out = re.sub(r"(?<![<>])<(?!=)", " < ", out)
    out = re.sub(r"(?<![<>])>(?!=)", " > ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return f"${out}$"


def try_linear_equation(question):
    q = str(question).strip().lower()
    if len(q) > 30:
        return None
    q_compact = q.replace(" ", "")
    m = re.fullmatch(r"(-?\d*)x([+\-]\d+)?=(-?\d+)", q_compact)
    if not m:
        return None
    a_str, b_str, c_str = m.groups()
    if a_str in (None, "", "-"):
        a = Fraction(-1 if a_str == "-" else 1)
    else:
        a = Fraction(int(a_str))
    b = Fraction(int(b_str)) if b_str else Fraction(0)
    c = Fraction(int(c_str))
    if a == 0:
        return None
    x = (c - b) / a
    if x.denominator == 1:
        return f"Ответ: x = {x.numerator}."
    return f"Ответ: x = {x.numerator}/{x.denominator}."


def try_deterministic(question):
    for fn in (try_arithmetic, try_linear_equation, try_bare_inequality):
        res = fn(question)
        if res is not None:
            return res
    return None


# -------- prompting --------

def make_prompt(question, fewshot):
    q = str(question).strip()
    parts = ["<|im_start|>system\n", SYSTEM_PROMPT, "\n<|im_end|>\n"]
    for ex_q, ex_a in fewshot:
        parts.append("<|im_start|>user\n")
        parts.append(ex_q)
        parts.append("\n<|im_end|>\n<|im_start|>assistant\n")
        parts.append(ex_a)
        parts.append("\n<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    parts.append(q)
    parts.append("\n<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


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


def load_items():
    with open(INPUT_PATH, "rb") as f:
        items = pickle.load(f)
    return items if isinstance(items, list) else []


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


def generate_with_vllm(items, bank):
    answers = [None] * len(items)
    llm_items = []
    llm_indices = []
    llm_fewshot = []

    for i, item in enumerate(items):
        q = item.get("question", "") if isinstance(item, dict) else ""

        det = try_deterministic(q)
        if det is not None:
            answers[i] = det
            continue

        exact = bank.exact(q)
        if exact is not None:
            answers[i] = exact["answer"]
            continue

        hits = bank.search(q, k=FEWSHOT_K)
        if hits and hits[0][0] >= NEAR_DUP_THRESHOLD:
            answers[i] = hits[0][1]["answer"]
            continue

        fewshot = [(entry["norm"], entry["answer"]) for score, entry in hits if score >= FEWSHOT_THRESHOLD]
        llm_items.append(item)
        llm_indices.append(i)
        llm_fewshot.append(fewshot)

    if llm_items:
        prompts = [
            make_prompt((item.get("question", "") if isinstance(item, dict) else ""), fewshot)
            for item, fewshot in zip(llm_items, llm_fewshot)
        ]
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.88,
            max_model_len=4096,
        )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=MAX_NEW_TOKENS,
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
        bank = TrainBank(TRAIN_BANK_PATH)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        bank = TrainBank.__new__(TrainBank)
        bank.entries = []
        bank.by_norm = {}

    try:
        answers = generate_with_vllm(items, bank)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        answers = [try_deterministic(item.get("question", "") if isinstance(item, dict) else "") or "" for item in items]

    write_outputs(items, answers)


if __name__ == "__main__":
    main()
