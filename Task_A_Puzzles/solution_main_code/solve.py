"""Policy-based inference for adaptive reversible puzzles.

Expected files at solve time:
  - gym.py
  - common.py
  - model.py with PolicyNet
  - optional model.pt produced by train.py
  - input_states.jsonl

Writes output_actions.csv with columns: instance_id,actions
"""

import argparse
import csv
import json
import os
import pickle
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import gym
import common
from model import PolicyNet

MODEL_PATH = "model.pt"
BFS_TABLE_PATH = "bfs_table.pkl"

ACTION_SWAP = getattr(gym, "ACTION_SWAP", 0)
ACTION_ROTATE = getattr(gym, "ACTION_ROTATE", 1)
ACTION_TOGGLE = getattr(gym, "ACTION_TOGGLE", 2)
ACTION_PERMUTE = getattr(gym, "ACTION_PERMUTE", 3)
CONTENT_EMPTY = getattr(gym, "CONTENT_EMPTY", 0)
CONTENT_NUM = getattr(gym, "CONTENT_NUM", 1)
CONTENT_COLOR = getattr(gym, "CONTENT_COLOR", 2)

_FAST_SLIDING_CONTEXT = None
_POLICY_CACHE: Dict[bytes, Dict[str, float]] = {}
POLICY_CACHE_LIMIT = int(os.environ.get("POLICY_CACHE_LIMIT", "120000"))
_TOGGLE_LINALG_CACHE = None
_STATIC_PERM_CACHE = None



def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_state(state):
    return common.to_jsonable(state)


def key(state) -> bytes:
    return common.compact_state_key(state)


def load_bfs_table(path: str = BFS_TABLE_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        table = payload.get("table", payload) if isinstance(payload, dict) else payload
        print(f"loaded bfs table: {len(table):,} states")
        return table
    except Exception as exc:
        print(f"cannot load bfs table: {exc!r}")
        return None


def load_policy_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        print("model.pt not found: policy beam will use heuristic fallback")
        return None, []
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        action_vocab = list(ckpt.get("action_vocab", []))
        num_actions = int(ckpt.get("num_actions", len(action_vocab)))
        if num_actions <= 0:
            return None, []
        model = PolicyNet(num_actions=num_actions)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, action_vocab
    except Exception as exc:
        print(f"cannot load policy model: {exc!r}")
        return None, []


def tokens_to_tensors(tokens: np.ndarray):
    b, n, _ = tokens.shape
    parts = common.split_token_features(tokens.reshape(b * n, -1))
    dense = torch.from_numpy(parts["dense"].reshape(b, n, -1))
    cv = torch.from_numpy(parts["content_value"].reshape(b, n))
    tv = torch.from_numpy(parts["target_value"].reshape(b, n))
    return dense, cv, tv


def make_policy_fn(env, model, action_vocab: Sequence[str], batch_size: int = 64):
    """Return state -> {action: log_probability} scorer.

    The function is batched and cached. Caching matters because beam search often
    sees the same state through different short paths.
    """
    if model is None or not action_vocab:
        def zero_policy(states):
            out = []
            for state in states:
                try:
                    env.set_state(state)
                    actions = env.valid_actions()
                except Exception:
                    actions = []
                out.append({a: 0.0 for a in actions})
            return out
        return zero_policy

    def p_fn(states):
        results = [None] * len(states)
        todo = []
        todo_idx = []
        for i, state in enumerate(states):
            k = key(state)
            cached = _POLICY_CACHE.get(k)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(state)
                todo_idx.append(i)

        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            idxs = todo_idx[start:start + batch_size]
            try:
                tokens = np.stack([common.encode_tokens(env, s) for s in chunk])
                dense, cv, tv = tokens_to_tensors(tokens)
                with torch.no_grad():
                    log_probs = F.log_softmax(model(dense, cv, tv), dim=-1).cpu().numpy()
            except Exception:
                log_probs = np.zeros((len(chunk), len(action_vocab)), dtype=np.float32)

            for row, state, original_i in zip(log_probs, chunk, idxs):
                scores = {a: float(row[j]) for j, a in enumerate(action_vocab) if j < len(row)}
                if len(_POLICY_CACHE) >= POLICY_CACHE_LIMIT:
                    _POLICY_CACHE.clear()
                _POLICY_CACHE[key(state)] = scores
                results[original_i] = scores

        return results

    return p_fn


def validate_actions(env, initial_state, actions: Sequence[str]) -> bool:
    try:
        env.set_state(initial_state)
        for action in actions:
            if action not in env.valid_actions():
                return False
            env.step(action)
        return bool(env.is_solved())
    except Exception:
        return False


def ordered_children(env, state, prev_action=None, action_scores: Optional[Dict[str, float]] = None,
                     top_k: Optional[int] = None):
    try:
        env.set_state(state)
        actions = list(env.valid_actions())
    except Exception:
        return []

    if prev_action is not None and len(actions) > 1:
        try:
            inv = env.inverse_action(prev_action)
            filtered = [a for a in actions if a != inv]
            if filtered:
                actions = filtered
        except Exception:
            pass

    if action_scores is not None:
        actions.sort(key=lambda a: action_scores.get(a, -1e9), reverse=True)
    else:
        actions.sort()

    if top_k is not None:
        actions = actions[:top_k]

    children = []
    for action in actions:
        try:
            env.set_state(state)
            env.step(action)
            ns = as_state(env.get_state())
            children.append((action, ns))
        except Exception:
            continue
    return children


def mismatch_count(env, state) -> int:
    obs = env.encode_state(state)
    ct = np.asarray(obs["content_types"])
    cv = np.asarray(obs["content_values"])
    tt = np.asarray(obs["target_types"])
    tv = np.asarray(obs["target_values"])
    return int(np.sum((ct != tt) | (cv != tv)))


def color_mismatch_count(env, state) -> int:
    """Loose heuristic: exact mismatch for NUM/EMPTY, multiset mismatch for COLOR."""
    try:
        obs = env.encode_state(state)
        ct = np.asarray(obs["content_types"], dtype=np.int64)
        cv = np.asarray(obs["content_values"], dtype=np.int64)
        tt = np.asarray(obs["target_types"], dtype=np.int64)
        tv = np.asarray(obs["target_values"], dtype=np.int64)
    except Exception:
        return 10**9

    exact_mask = (ct != CONTENT_COLOR) | (tt != CONTENT_COLOR)
    exact = int(np.sum(((ct != tt) | (cv != tv)) & exact_mask))
    color_mask = (ct == CONTENT_COLOR) & (tt == CONTENT_COLOR)
    color = int(np.sum((cv != tv) & color_mask))
    return exact + color


def search_score(env, state) -> float:
    sm = sliding_manhattan(env, state)
    if sm is not None:
        return float(sm)
    return float(color_mismatch_count(env, state))


# ---------- BFS table ----------

def extract_table_path(env, state, table, max_steps=10000):
    if not table:
        return None
    path = []
    cur = as_state(state)
    for _ in range(max_steps):
        item = table.get(key(cur))
        if item is None:
            return None
        dist, action = item
        if dist == 0:
            return path
        if action is None:
            return None
        try:
            env.set_state(cur)
            if action not in env.valid_actions():
                return None
            env.step(action)
            cur = as_state(env.get_state())
            path.append(action)
        except Exception:
            return None
    return None


def solve_with_table(env, initial_state, table):
    path = extract_table_path(env, initial_state, table)
    if path is not None and validate_actions(env, initial_state, path):
        return path
    return None


# ---------- exact solver for TOGGLE / Lights Out-like puzzles ----------

def _toggle_linalg_context(env, initial_state):
    """Build A*x=b over GF(2) once for TOGGLE-only puzzles.

    Returns reusable RREF data.  The old solver stopped at one arbitrary
    solution.  This context additionally stores the nullspace, so solve time can
    choose the minimum-Hamming representative of the affine solution space.
    """
    try:
        obs = env.encode_state(initial_state)
        enc = env.encode_actions(state=initial_state)
    except Exception:
        return None

    actions = list(enc.get("actions", []))
    action_types = list(enc.get("action_types", []))
    affected = list(enc.get("affected", []))
    if not actions or any(int(t) != ACTION_TOGGLE for t in action_types):
        return None

    ct = np.asarray(obs["content_types"], dtype=np.int64)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tt = np.asarray(obs["target_types"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)
    if not np.all(ct == tt):
        return None
    if not set(np.unique(cv).tolist()).issubset({0, 1}):
        return None
    if not set(np.unique(tv).tolist()).issubset({0, 1}):
        return None

    n_cells = len(cv)
    n_actions = len(actions)
    if n_actions > 240:  # Python int bitsets are great, but keep hidden traps bounded.
        return None

    rows = []
    affected_sets = [set(map(int, cells)) for cells in affected]
    for cell in range(n_cells):
        mask = 0
        for j, cells in enumerate(affected_sets):
            if cell in cells:
                mask ^= 1 << j
        rows.append(mask)

    # RREF of A, stored as rows with pivot bit + free coefficients.
    pivot_cols = []
    row = 0
    for col in range(n_actions):
        bit = 1 << col
        pivot = None
        for r in range(row, n_cells):
            if rows[r] & bit:
                pivot = r
                break
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        for r in range(n_cells):
            if r != row and (rows[r] & bit):
                rows[r] ^= rows[row]
        pivot_cols.append(col)
        row += 1
        if row == n_cells:
            break

    rank = len(pivot_cols)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n_actions) if c not in pivot_set]
    rref_rows = rows[:rank]

    # Nullspace basis.  For free variable f=1, each pivot variable equals the
    # coefficient of f in its RREF row.
    null_basis = []
    for f in free_cols:
        v = 1 << f
        bit = 1 << f
        for r, pcol in enumerate(pivot_cols):
            if rref_rows[r] & bit:
                v ^= 1 << pcol
        null_basis.append(v)

    return {
        "actions": actions,
        "n_cells": n_cells,
        "n_actions": n_actions,
        "pivot_cols": pivot_cols,
        "rref_rows": rref_rows,
        "rank": rank,
        "null_basis": null_basis,
    }


def _minimize_affine_bitvector(x: int, basis: Sequence[int], deadline=None) -> int:
    """Find/approximate min popcount vector in x + span(basis)."""
    if not basis:
        return x
    k = len(basis)

    # Exact enumeration is worth it up to ~2M candidates: still cheap for 1000
    # instances when nullity is small, and gives true shortest TOGGLE solution.
    if k <= 21:
        best = x
        best_w = x.bit_count()
        cur = 0
        prev_gray = 0
        total = 1 << k
        for i in range(1, total):
            if deadline is not None and (i & 4095) == 0 and time.time() >= deadline:
                break
            gray = i ^ (i >> 1)
            diff = gray ^ prev_gray
            j = diff.bit_length() - 1
            cur ^= basis[j]
            cand = x ^ cur
            w = cand.bit_count()
            if w < best_w:
                best, best_w = cand, w
            prev_gray = gray
        return best

    # Large nullity: do a deterministic local reduction.  Not guaranteed optimal,
    # but it often cuts long arbitrary Gaussian solutions a lot.
    basis = sorted(set(int(v) for v in basis if v), key=lambda v: v.bit_count())
    best = x
    best_w = x.bit_count()
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for v in basis:
            cand = best ^ v
            w = cand.bit_count()
            if w < best_w:
                best, best_w = cand, w
                improved = True
    return best


def solve_toggle_linear(env, initial_state, deadline=None):
    """Exact GF(2) solver for binary TOGGLE puzzles, with min-weight lift.

    For underdetermined systems this is the important algebraic upgrade: solve
    A*x=b, then search x + Null(A) for the shortest action list instead of using
    the first Gaussian-elimination answer.
    """
    global _TOGGLE_LINALG_CACHE
    if _TOGGLE_LINALG_CACHE is None:
        _TOGGLE_LINALG_CACHE = _toggle_linalg_context(env, initial_state)
    ctx = _TOGGLE_LINALG_CACHE
    if ctx is None:
        return None

    try:
        obs = env.encode_state(initial_state)
    except Exception:
        return None
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)
    if len(cv) != ctx["n_cells"]:
        return None
    rhs = (cv ^ tv).astype(np.uint8)

    # Apply exactly the same row operations encoded by the already-built RREF.
    # Since each RREF row is a linear combination with leading pivot, the matching
    # RHS for that row is obtained by dot(row_mask, original_rhs columns of A^T).
    # Easier and safer: rebuild augmented RHS with the same elimination, still
    # only O(cells*actions) and tiny for Lights-Out-like tasks.
    rows = []
    try:
        enc = env.encode_actions(state=initial_state)
        affected_sets = [set(map(int, cells)) for cells in enc.get("affected", [])]
        for cell in range(ctx["n_cells"]):
            mask = 0
            for j, cells in enumerate(affected_sets):
                if cell in cells:
                    mask ^= 1 << j
            rows.append([mask, int(rhs[cell])])
    except Exception:
        return None

    row = 0
    n_cells = ctx["n_cells"]
    n_actions = ctx["n_actions"]
    pivot_cols = []
    for col in range(n_actions):
        bit = 1 << col
        pivot = None
        for r in range(row, n_cells):
            if rows[r][0] & bit:
                pivot = r
                break
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        for r in range(n_cells):
            if r != row and (rows[r][0] & bit):
                rows[r][0] ^= rows[row][0]
                rows[r][1] ^= rows[row][1]
        pivot_cols.append(col)
        row += 1
        if row == n_cells:
            break

    for mask, val in rows:
        if mask == 0 and val:
            return None

    x = 0
    for r, col in enumerate(pivot_cols):
        if rows[r][1]:
            x |= 1 << col

    # If action matrix changed with state, cached nullspace may be unsafe.
    if pivot_cols == ctx["pivot_cols"]:
        x = _minimize_affine_bitvector(x, ctx["null_basis"], deadline=deadline)

    actions = ctx["actions"]
    sol = [actions[j] for j in range(len(actions)) if (x >> j) & 1]
    return sol if validate_actions(env, initial_state, sol) else None


# ---------- algebraic bidirectional search for static permutation puzzles ----------

def _encode_cell_symbols(obs):
    ct = np.asarray(obs["content_types"], dtype=np.int64)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tt = np.asarray(obs["target_types"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)
    cur = tuple((int(t), int(v)) for t, v in zip(ct, cv))
    tgt = tuple((int(t), int(v)) for t, v in zip(tt, tv))
    return cur, tgt


def _static_perm_context(env, state):
    """Detect actions that are state-independent permutations of cells."""
    try:
        obs = env.encode_state(state)
        enc = env.encode_actions(state=state)
    except Exception:
        return None
    actions = list(enc.get("actions", []))
    if not actions or len(actions) > 80:
        return None
    types = [int(t) for t in enc.get("action_types", [])]
    if any(t == ACTION_TOGGLE for t in types):
        return None

    cur, tgt = _encode_cell_symbols(obs)
    base_positions = tuple(tuple(round(float(x), 6) for x in row) for row in np.asarray(obs.get("positions", []), dtype=np.float64))
    n = len(cur)
    if n == 0 or n > 80:
        return None

    perms = []
    inv_perms = []
    inverses = []
    for i, a in enumerate(actions):
        mf = enc.get("map_from", [[]])[i]
        mt = enc.get("map_to", [[]])[i]
        if len(mf) != len(mt):
            return None
        perm = list(range(n))  # new[j] = old[perm[j]]
        for src, dst in zip(mf, mt):
            src = int(src); dst = int(dst)
            if not (0 <= src < n and 0 <= dst < n):
                return None
            perm[dst] = src
        # Must be a real permutation on the affected part.
        if len(set(perm)) != n:
            return None
        inv = [0] * n
        for dst, src in enumerate(perm):
            inv[src] = dst
        # Pure algebraic permutation solver is only safe when cell positions are
        # fixed.  This excludes cylinder-like puzzles where actions move a
        # cursor/top slot encoded through positions, not only through contents.
        try:
            env.set_state(state)
            env.step(a)
            obs_after = env.encode_state(env.get_state())
            pos_after = tuple(tuple(round(float(x), 6) for x in row) for row in np.asarray(obs_after.get("positions", []), dtype=np.float64))
            if pos_after != base_positions:
                return None
        except Exception:
            return None
        try:
            ia = env.inverse_action(a)
        except Exception:
            ia = None
        perms.append(tuple(perm))
        inv_perms.append(tuple(inv))
        inverses.append(ia)

    # Quick sanity: action set should not depend on values after a move.
    try:
        env.set_state(state)
        env.step(actions[0])
        ns = as_state(env.get_state())
        enc2 = env.encode_actions(state=ns)
        actions2 = list(enc2.get("actions", []))
        if set(actions2) != set(actions):
            return None
        by_action = {}
        for i2, a2 in enumerate(actions2):
            perm2 = list(range(n))
            for src, dst in zip(enc2.get("map_from", [[]])[i2], enc2.get("map_to", [[]])[i2]):
                perm2[int(dst)] = int(src)
            by_action[a2] = tuple(perm2)
        for a, perm in zip(actions, perms):
            if by_action.get(a) != perm:
                return None
    except Exception:
        return None

    return {"actions": actions, "perms": perms, "inv_perms": inv_perms, "target": tgt, "n": n}


def _apply_perm_tuple(state_tuple, perm):
    return tuple(state_tuple[i] for i in perm)


def solve_static_perm_bidir(env, initial_state, deadline, max_nodes=70000, max_depth=18):
    """Algebraic bidirectional BFS on the cell permutation group.

    This bypasses env.step() in the inner loop. It helps hidden puzzles whose
    actions are fixed SWAP/ROTATE/PERMUTE generators: rows/rings/layers etc.
    """
    global _STATIC_PERM_CACHE
    if _STATIC_PERM_CACHE is None:
        _STATIC_PERM_CACHE = _static_perm_context(env, initial_state)
    ctx = _STATIC_PERM_CACHE
    if ctx is None:
        return None

    try:
        obs = env.encode_state(initial_state)
        start_tuple, target_tuple = _encode_cell_symbols(obs)
    except Exception:
        return None
    if len(start_tuple) != ctx["n"] or target_tuple != ctx["target"]:
        return None
    if start_tuple == target_tuple:
        return []

    actions = ctx["actions"]
    perms = ctx["perms"]
    inv_perms = ctx["inv_perms"]

    f_seen = {start_tuple: []}
    b_seen = {target_tuple: []}  # state -> path from this state to target
    f_front = [start_tuple]
    b_front = [target_tuple]
    nodes = 2

    for depth in range(max_depth):
        if time.time() >= deadline or nodes >= max_nodes:
            return None

        expand_forward = len(f_front) <= len(b_front)
        if expand_forward:
            new_front = []
            for st in f_front:
                base_path = f_seen[st]
                prev_inv = None
                if base_path:
                    try:
                        prev_inv = env.inverse_action(base_path[-1])
                    except Exception:
                        prev_inv = None
                for a, perm in zip(actions, perms):
                    if a == prev_inv:
                        continue
                    ns = _apply_perm_tuple(st, perm)
                    if ns in f_seen:
                        continue
                    path = base_path + [a]
                    other = b_seen.get(ns)
                    if other is not None:
                        sol = path + other
                        return sol if validate_actions(env, initial_state, sol) else None
                    f_seen[ns] = path
                    new_front.append(ns)
                    nodes += 1
                    if nodes >= max_nodes or time.time() >= deadline:
                        break
                if nodes >= max_nodes or time.time() >= deadline:
                    break
            f_front = new_front
        else:
            new_front = []
            for st in b_front:
                base_path = b_seen[st]
                # predecessor p --a--> st; p = inv_perm_a(st), path(p->target)=a+path(st->target)
                for a, invp in zip(actions, inv_perms):
                    ns = _apply_perm_tuple(st, invp)
                    if ns in b_seen:
                        continue
                    path = [a] + base_path
                    other = f_seen.get(ns)
                    if other is not None:
                        sol = other + path
                        return sol if validate_actions(env, initial_state, sol) else None
                    b_seen[ns] = path
                    new_front.append(ns)
                    nodes += 1
                    if nodes >= max_nodes or time.time() >= deadline:
                        break
                if nodes >= max_nodes or time.time() >= deadline:
                    break
            b_front = new_front

        if not f_front or not b_front:
            return None
    return None

# ---------- sliding puzzle helpers ----------

def sliding_layout(env, state):
    try:
        obs = env.encode_state(state)
    except Exception:
        return None

    pos = np.asarray(obs["positions"], dtype=np.float32)
    ct = np.asarray(obs["content_types"], dtype=np.int64)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tt = np.asarray(obs["target_types"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)

    if int(np.sum(ct == CONTENT_EMPTY)) != 1 or int(np.sum(tt == CONTENT_EMPTY)) != 1:
        return None
    nums = cv[ct == CONTENT_NUM]
    if len(nums) == 0 or len(set(nums.tolist())) != len(nums):
        return None

    ranks = []
    for axis in range(pos.shape[1]):
        vals = sorted({round(float(x), 6) for x in pos[:, axis]})
        ranks.append({v: i for i, v in enumerate(vals)})

    coords = [tuple(ranks[axis][round(float(p[axis]), 6)] for axis in range(pos.shape[1])) for p in pos]
    target_coord = {}
    for i, (typ, val) in enumerate(zip(tt, tv)):
        if int(typ) == CONTENT_NUM:
            target_coord[int(val)] = coords[i]

    tiles = []
    for i, (typ, val) in enumerate(zip(ct, cv)):
        if int(typ) != CONTENT_NUM:
            continue
        tgt = target_coord.get(int(val))
        if tgt is None:
            return None
        tiles.append((int(val), coords[i], tgt))

    active_axes = [axis for axis, r in enumerate(ranks) if len(r) > 1]
    return tiles, active_axes


def lis_length(values: Sequence[int]) -> int:
    tails = []
    for value in values:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < value:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(value)
        else:
            tails[lo] = value
    return len(tails)


def sliding_manhattan(env, state):
    layout = sliding_layout(env, state)
    if layout is None:
        return None
    tiles, _active_axes = layout
    return int(sum(sum(abs(a - b) for a, b in zip(cur, tgt)) for _val, cur, tgt in tiles))


def sliding_heuristic(env, state):
    layout = sliding_layout(env, state)
    if layout is None:
        return None
    tiles, active_axes = layout
    dist = int(sum(sum(abs(a - b) for a, b in zip(cur, tgt)) for _val, cur, tgt in tiles))

    if len(active_axes) == 2:
        ax0, ax1 = active_axes
        conflicts = 0
        for line_axis, order_axis in ((ax0, ax1), (ax1, ax0)):
            groups = {}
            for _val, cur, tgt in tiles:
                if cur[line_axis] == tgt[line_axis]:
                    groups.setdefault(cur[line_axis], []).append((cur[order_axis], tgt[order_axis]))
            for entries in groups.values():
                entries.sort(key=lambda x: x[0])
                seq = [tgt_order for _cur_order, tgt_order in entries]
                conflicts += len(seq) - lis_length(seq)
        dist += 2 * conflicts
    return dist


def state_array_from_env(env, state):
    try:
        obs = env.encode_state(state)
    except Exception:
        return None
    ct = np.asarray(obs["content_types"], dtype=np.int64)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tt = np.asarray(obs["target_types"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)
    if int(np.sum(ct == CONTENT_EMPTY)) != 1 or int(np.sum(tt == CONTENT_EMPTY)) != 1:
        return None
    nums = cv[ct == CONTENT_NUM]
    if len(nums) == 0 or len(set(nums.tolist())) != len(nums):
        return None
    return cv.astype(np.int16).tolist(), tv.astype(np.int16).tolist(), obs


def array_to_state_like(values: Sequence[int], template):
    it = iter(int(v) for v in values)

    def fill(x):
        if isinstance(x, dict):
            return {k: fill(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fill(v) for v in x]
        if isinstance(x, tuple):
            return tuple(fill(v) for v in x)
        if isinstance(x, np.ndarray):
            flat = [next(it) for _ in range(x.size)]
            return np.asarray(flat, dtype=x.dtype).reshape(x.shape)
        if isinstance(x, (int, np.integer)):
            return next(it)
        return x

    return as_state(fill(template))


def make_fast_sliding_context(env, template_state):
    parsed = state_array_from_env(env, template_state)
    if parsed is None:
        return None
    _start_arr, target_arr, obs = parsed
    n = len(target_arr)

    pos = np.asarray(obs["positions"], dtype=np.float32)
    ranks = []
    for axis in range(pos.shape[1]):
        vals = sorted({round(float(x), 6) for x in pos[:, axis]})
        ranks.append({v: i for i, v in enumerate(vals)})
    coords = [tuple(ranks[axis][round(float(p[axis]), 6)] for axis in range(pos.shape[1])) for p in pos]
    active_axes = [axis for axis, r in enumerate(ranks) if len(r) > 1]
    if len(active_axes) != 2:
        return None

    try:
        target_empty = int(np.asarray(obs["target_types"], dtype=np.int64).tolist().index(CONTENT_EMPTY))
    except ValueError:
        return None

    goal_coord = {int(val): coords[idx] for idx, val in enumerate(target_arr) if idx != target_empty}
    dist_table = {
        val: [sum(abs(a - b) for a, b in zip(cur, tgt)) for cur in coords]
        for val, tgt in goal_coord.items()
    }

    moves = [[] for _ in range(n)]
    base = list(target_arr)
    for blank_idx in range(n):
        rep = list(base)
        rep[target_empty], rep[blank_idx] = rep[blank_idx], rep[target_empty]
        state_like = array_to_state_like(rep, template_state)
        try:
            enc = env.encode_actions(state=state_like)
        except Exception:
            return None
        for action, typ, affected in zip(enc.get("actions", []), enc.get("action_types", []), enc.get("affected", [])):
            if int(typ) != ACTION_SWAP or len(affected) != 2:
                return None
            a0, a1 = int(affected[0]), int(affected[1])
            if blank_idx == a0:
                moves[blank_idx].append((a1, action))
            elif blank_idx == a1:
                moves[blank_idx].append((a0, action))

    if any(not m for m in moves):
        return None

    return {
        "n": n,
        "target": tuple(int(v) for v in target_arr),
        "target_empty": target_empty,
        "coords": coords,
        "active_axes": active_axes,
        "goal_coord": goal_coord,
        "dist_table": dist_table,
        "moves": moves,
    }


def fast_sliding_key(arr: Sequence[int]) -> bytes:
    return bytes(int(v) & 0xFF for v in arr)


def fast_sliding_heuristic(ctx, arr: Sequence[int]) -> int:
    dist = 0
    coords = ctx["coords"]
    ax0, ax1 = ctx["active_axes"]
    goal_coord = ctx["goal_coord"]
    row_groups, col_groups = {}, {}

    for idx, raw_val in enumerate(arr):
        val = int(raw_val)
        if val == 0:
            continue
        table = ctx["dist_table"].get(val)
        if table is None:
            return 10**9
        dist += table[idx]
        cur = coords[idx]
        tgt = goal_coord[val]
        if cur[ax0] == tgt[ax0]:
            row_groups.setdefault(cur[ax0], []).append((cur[ax1], tgt[ax1]))
        if cur[ax1] == tgt[ax1]:
            col_groups.setdefault(cur[ax1], []).append((cur[ax0], tgt[ax0]))

    conflicts = 0
    for groups in (row_groups, col_groups):
        for entries in groups.values():
            entries.sort(key=lambda x: x[0])
            seq = [target for _cur, target in entries]
            conflicts += len(seq) - lis_length(seq)
    return int(dist + 2 * conflicts)


def is_sliding_swap_env(env, state) -> bool:
    if sliding_heuristic(env, state) is None:
        return False
    try:
        enc = env.encode_actions(state=state)
        types = enc.get("action_types", [])
        affected = enc.get("affected", [])
        return bool(types) and all(int(t) == ACTION_SWAP for t in types) and all(len(a) == 2 for a in affected)
    except Exception:
        return False


def solve_sliding_fast_idastar(env, initial_state, deadline, max_depth=90):
    global _FAST_SLIDING_CONTEXT
    parsed = state_array_from_env(env, initial_state)
    if parsed is None:
        return None
    start_arr, _target_arr, _obs = parsed

    if _FAST_SLIDING_CONTEXT is None:
        _FAST_SLIDING_CONTEXT = make_fast_sliding_context(env, initial_state)
    ctx = _FAST_SLIDING_CONTEXT
    if ctx is None or len(start_arr) != ctx["n"]:
        return None
    if tuple(start_arr) == ctx["target"]:
        return []

    try:
        blank = start_arr.index(0)
    except ValueError:
        return None

    arr = list(start_arr)
    threshold = fast_sliding_heuristic(ctx, arr)
    path = []
    path_keys = {fast_sliding_key(arr)}

    def dfs(g: int, bound: int, blank_idx: int, prev_blank: int):
        if time.time() >= deadline:
            return None, float("inf")
        h = fast_sliding_heuristic(ctx, arr)
        f = g + h
        if f > bound:
            return None, f
        if h == 0:
            return list(path), f
        if g >= max_depth:
            return None, float("inf")

        children = []
        for nb, action in ctx["moves"][blank_idx]:
            if nb == prev_blank:
                continue
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            k = fast_sliding_key(arr)
            if k not in path_keys:
                children.append((fast_sliding_heuristic(ctx, arr), nb, action, k))
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
        children.sort(key=lambda x: x[0])

        best_next = float("inf")
        for _h, nb, action, k in children:
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            path.append(action)
            path_keys.add(k)
            res, nxt = dfs(g + 1, bound, nb, blank_idx)
            path_keys.remove(k)
            path.pop()
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            if res is not None:
                return res, nxt
            best_next = min(best_next, nxt)
        return None, best_next

    while time.time() < deadline and threshold <= max_depth:
        res, nxt = dfs(0, threshold, blank, -1)
        if res is not None:
            return res if validate_actions(env, initial_state, res) else None
        if nxt == float("inf"):
            return None
        threshold = int(nxt)
    return None


# ---------- generic search engines ----------

def solve_bfs_shallow(env, initial_state, solved_key: bytes, deadline, max_depth=8, max_nodes=30000):
    start = as_state(initial_state)
    if key(start) == solved_key:
        return []

    q = deque([(start, [], None)])
    seen = {key(start)}
    nodes = 0
    while q and nodes < max_nodes and time.time() < deadline:
        state, path, prev = q.popleft()
        if len(path) >= max_depth:
            continue
        for action, ns in ordered_children(env, state, prev):
            nk = key(ns)
            if nk in seen:
                continue
            seen.add(nk)
            npath = path + [action]
            if nk == solved_key:
                return npath
            q.append((ns, npath, action))
            nodes += 1
    return None


def solve_heuristic_beam(env, initial_state, solved_key: bytes, deadline,
                         beam_width=96, max_depth=90, child_top_k=None, bfs_table=None):
    start = as_state(initial_state)
    if key(start) == solved_key:
        return []

    beam = [(search_score(env, start), start, [], None)]
    seen = {key(start)}

    for _depth in range(max_depth):
        if time.time() >= deadline:
            return None
        candidates = []
        for _score, state, path, prev in beam:
            for action, ns in ordered_children(env, state, prev, top_k=child_top_k):
                nk = key(ns)
                if nk in seen:
                    continue
                seen.add(nk)
                npath = path + [action]
                if nk == solved_key:
                    return npath
                if bfs_table is not None and nk in bfs_table:
                    suffix = extract_table_path(env, ns, bfs_table)
                    if suffix is not None:
                        return npath + suffix
                score = search_score(env, ns)
                candidates.append((score, ns, npath, action))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    return None


def solve_policy_beam(env, initial_state, solved_key: bytes, policy_fn, deadline,
                      beam_width=8, max_depth=130, top_k=3,
                      heuristic_weight=0.03, bfs_table=None):
    """Policy-guided beam search.

    The model ranks actions directly. We only simulate top_k moves for every
    beam state, then keep beam_width most promising paths.
    """
    start = as_state(initial_state)
    if key(start) == solved_key:
        return []

    beam = [(0.0, start, [], None)]
    seen = {key(start)}

    for _depth in range(max_depth):
        if time.time() >= deadline:
            return None

        states = [item[1] for item in beam]
        scores_batch = policy_fn(states)
        candidates = []

        for i, (score, state, path, prev) in enumerate(beam):
            action_scores = scores_batch[i]
            for action, ns in ordered_children(env, state, prev, action_scores=action_scores, top_k=top_k):
                nk = key(ns)
                if nk in seen:
                    continue
                seen.add(nk)
                npath = path + [action]
                if nk == solved_key:
                    return npath
                if bfs_table is not None and nk in bfs_table:
                    suffix = extract_table_path(env, ns, bfs_table)
                    if suffix is not None:
                        return npath + suffix

                # Primary signal: policy log-probability. Small heuristic tie-breaker
                # prevents obviously-worse states from surviving when the policy is weak.
                policy_score = score + action_scores.get(action, -30.0)
                h_penalty = heuristic_weight * search_score(env, ns)
                candidates.append((policy_score - h_penalty, ns, npath, action))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    return None


def solve_instance(env, state, solved_key: bytes, policy_fn, deadline, strategy: str, bfs_table=None):
    if time.time() >= deadline:
        return None

    # 1. Precomputed exact reverse table, if train.py produced one.
    if bfs_table is not None:
        sol = solve_with_table(env, state, bfs_table)
        if sol is not None:
            return sol

    # 2. Exact linear algebra for Lights Out / binary TOGGLE puzzles.
    sol = solve_toggle_linear(env, state, deadline=deadline)
    if sol is not None:
        return sol

    # 3. Algebraic bidirectional BFS for static SWAP/ROTATE/PERMUTE puzzles.
    if strategy in ("full", "beam", "heuristic") and time.time() < deadline:
        local_deadline = min(deadline, time.time() + 0.35)
        sol = solve_static_perm_bidir(env, state, local_deadline, max_nodes=45000, max_depth=16)
        if sol is not None:
            return sol

    # 4. Exact-ish IDA* for 15-puzzle-like SWAP+EMPTY tasks.
    if strategy in ("full", "sliding", "sliding_only") and is_sliding_swap_env(env, state):
        local_deadline = min(deadline, time.time() + max(0.15, 0.35 * (deadline - time.time())))
        sol = solve_sliding_fast_idastar(env, state, local_deadline, max_depth=90)
        if sol is not None:
            return sol

    # 5. Shallow BFS catches easy/near-solved states without trusting the model.
    if strategy in ("full", "bfs") and time.time() < deadline:
        local_deadline = min(deadline, time.time() + 0.15)
        sol = solve_bfs_shallow(env, state, solved_key, local_deadline, max_depth=7, max_nodes=20000)
        if sol is not None:
            return sol

    # 6. Main path: policy-guided beam.
    if strategy in ("full", "policy", "beam") and time.time() < deadline:
        sol = solve_policy_beam(env, state, solved_key, policy_fn, deadline,
                                beam_width=20, max_depth=180, top_k=4, bfs_table=bfs_table)
        if sol is not None:
            return sol

    # 7. Fallback: generic heuristic beam. Slower, but no model needed.
    if strategy in ("full", "heuristic", "beam") and time.time() < deadline:
        sol = solve_heuristic_beam(env, state, solved_key, deadline,
                                   beam_width=128, max_depth=150, bfs_table=bfs_table)
        if sol is not None:
            return sol

    return None


def write_output(path: str, instances: Sequence[dict], solutions: Dict[str, Sequence[str]]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "actions"])
        for inst in instances:
            iid = inst["instance_id"]
            actions = solutions.get(iid) or []
            writer.writerow([iid, " ".join(str(a) for a in actions)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int, default=int(os.environ.get("SOLVE_TIME_LIMIT", 24 * 60)))
    parser.add_argument("--strategy", default=os.environ.get("SOLVER_STRATEGY", "full"),
                        choices=["full", "policy", "beam", "heuristic", "bfs", "sliding", "sliding_only"])
    args = parser.parse_args()

    start = time.time()
    hard_limit = min(args.time_limit, 24 * 60)
    deadline = start + hard_limit - 10.0

    torch.set_num_threads(1)

    env = gym.make_env()
    env.reset()
    solved_key = key(env.get_state())

    instances = load_jsonl(args.input)
    model, action_vocab = load_policy_model(MODEL_PATH)
    policy_fn = make_policy_fn(env, model, action_vocab)
    bfs_table = load_bfs_table(BFS_TABLE_PATH)

    solutions: Dict[str, Sequence[str]] = {}
    unsolved: List[dict] = []

    # Pass 1: fast sweep.  Do not let several hard states burn the whole run.
    for idx, inst in enumerate(instances):
        if time.time() >= deadline:
            unsolved.extend(instances[idx:])
            break
        iid = inst["instance_id"]
        state = inst["state"]

        remaining_instances = max(1, len(instances) - idx)
        remaining_time = max(0.01, deadline - time.time())
        per_instance = min(1.35, max(0.04, 1.25 * remaining_time / remaining_instances))
        inst_deadline = min(deadline, time.time() + per_instance)

        sol = solve_instance(env, state, solved_key, policy_fn, inst_deadline, args.strategy, bfs_table=bfs_table)
        if sol is not None and validate_actions(env, state, sol):
            solutions[iid] = sol
        else:
            unsolved.append(inst)

        if (idx + 1) % 50 == 0:
            print(f"pass1 {idx + 1}/{len(instances)}; solved {len(solutions)}; unsolved {len(unsolved)}; t={time.time() - start:.1f}s")

    print(f"pass1 done: solved {len(solutions)}/{len(instances)}; unsolved={len(unsolved)}; t={time.time() - start:.1f}s")

    # Pass 2: spend the remaining time only on misses.  No artificial 2s cap.
    for idx, inst in enumerate(unsolved):
        if time.time() >= deadline:
            break
        iid = inst["instance_id"]
        if iid in solutions:
            continue
        remaining_instances = max(1, len(unsolved) - idx)
        remaining_time = max(0.01, deadline - time.time())
        inst_deadline = min(deadline, time.time() + remaining_time / remaining_instances)

        sol = solve_instance(env, inst["state"], solved_key, policy_fn, inst_deadline, args.strategy, bfs_table=bfs_table)
        if sol is not None and validate_actions(env, inst["state"], sol):
            solutions[iid] = sol

        if (idx + 1) % 50 == 0:
            print(f"pass2 {idx + 1}/{len(unsolved)}; solved {len(solutions)}; t={time.time() - start:.1f}s")

    write_output(args.output, instances, solutions)
    print(f"final: solved {len(solutions)}/{len(instances)}; total time {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
