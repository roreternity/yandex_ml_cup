"""Inference: load model.pt, run A* (with V as heuristic), write CSV."""

import argparse
import csv
import json
import os
import pickle
import time
from collections import deque

import numpy as np
import torch

import gym
import common
from common import state_key
from model import ValueNet
from search import solve_astar


TIME_LIMIT_DEFAULT = 1 * 60
SAFETY_MARGIN_DEFAULT = 10
MODEL_PATH = "model.pt"
BFS_TABLE_PATH = "bfs_table.pkl"
ACTION_TOGGLE = getattr(gym, "ACTION_TOGGLE", 2)
CONTENT_NUM = getattr(gym, "CONTENT_NUM", 1)
CONTENT_EMPTY = getattr(gym, "CONTENT_EMPTY", 0)
_FAST_SLIDING_CONTEXT = None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = ValueNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_bfs_table():
    if not os.path.exists(BFS_TABLE_PATH):
        return None
    try:
        with open(BFS_TABLE_PATH, "rb") as f:
            payload = pickle.load(f)
        table = payload.get("table", payload)
        print(f"bfs table loaded: {len(table):,} states")
        return table
    except Exception as exc:
        print(f"bfs table load failed: {repr(exc)}")
        return None


def make_v_fn(env, model):
    if model is None:
        return lambda states: np.zeros(len(states), dtype=np.float32)

    def v_fn(states):
        tokens = np.stack([common.encode_tokens(env, s) for s in states])
        B, N, _ = tokens.shape
        parts = common.split_token_features(tokens.reshape(B * N, -1))
        dense = torch.from_numpy(parts["dense"].reshape(B, N, -1))
        cv = torch.from_numpy(parts["content_value"].reshape(B, N))
        tv = torch.from_numpy(parts["target_value"].reshape(B, N))
        with torch.no_grad():
            return model(dense, cv, tv).cpu().numpy()

    return v_fn


def validate_actions(env, initial_state, actions):
    try:
        env.set_state(initial_state)
        for a in actions:
            if a not in env.valid_actions():
                return False
            env.step(a)
        return env.is_solved()
    except Exception:
        return False


def extract_table_path(env, state, table, max_steps=10000):
    path = []
    cur = common.to_jsonable(state)
    for _ in range(max_steps):
        key = common.compact_state_key(cur)
        item = table.get(key)
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
            cur = common.to_jsonable(env.get_state())
        except Exception:
            return None
        path.append(action)
    return None


def solve_with_table(env, initial_state, table, deadline, beam_width=256, max_depth=12):
    if not table:
        return None

    direct = extract_table_path(env, initial_state, table)
    if direct is not None:
        return direct if validate_actions(env, initial_state, direct) else None

    start = common.to_jsonable(initial_state)
    start_key = common.compact_state_key(start)
    beam = [(search_score(env, start), start, [], None)]
    seen = {start_key}

    for _depth in range(max_depth):
        if time.time() >= deadline:
            return None
        candidates = []
        for _score, state, path, prev in beam:
            for action, ns in ordered_children(env, state, prev):
                key = common.compact_state_key(ns)
                if key in seen:
                    continue
                seen.add(key)
                npath = path + [action]
                if key in table:
                    suffix = extract_table_path(env, ns, table)
                    if suffix is not None:
                        sol = npath + suffix
                        return sol if validate_actions(env, initial_state, sol) else None
                candidates.append((search_score(env, ns) + 0.03 * len(npath), ns, npath, action))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    return None


def mismatch_count(env, state):
    obs = env.encode_state(state)
    ct = np.asarray(obs["content_types"])
    cv = np.asarray(obs["content_values"])
    tt = np.asarray(obs["target_types"])
    tv = np.asarray(obs["target_values"])
    return int(np.sum((ct != tt) | (cv != tv)))


def sliding_layout(env, state):
    """Return current and target coordinates for NUM+EMPTY sliding puzzles."""
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

    # Convert arbitrary coordinate values into grid ranks per axis.
    ranks = []
    for axis in range(pos.shape[1]):
        vals = sorted({round(float(x), 6) for x in pos[:, axis]})
        ranks.append({v: i for i, v in enumerate(vals)})
    cell_coord = []
    for p in pos:
        cell_coord.append(tuple(ranks[axis][round(float(p[axis]), 6)] for axis in range(pos.shape[1])))

    target_coord = {}
    for i, (typ, val) in enumerate(zip(tt, tv)):
        if typ == CONTENT_NUM:
            target_coord[int(val)] = cell_coord[i]

    tiles = []
    for i, (typ, val) in enumerate(zip(ct, cv)):
        if typ != CONTENT_NUM:
            continue
        tgt = target_coord.get(int(val))
        if tgt is None:
            return None
        cur = cell_coord[i]
        tiles.append((int(val), cur, tgt))

    active_axes = [axis for axis, mp in enumerate(ranks) if len(mp) > 1]
    return tiles, active_axes


def sliding_manhattan(env, state):
    """Generic Manhattan heuristic for NUM tiles plus one EMPTY cell."""
    layout = sliding_layout(env, state)
    if layout is None:
        return None
    tiles, active_axes = layout

    dist = 0
    for _val, cur, tgt in tiles:
        dist += sum(abs(a - b) for a, b in zip(cur, tgt))
    return int(dist)


def sliding_heuristic(env, state):
    """Generic Manhattan + linear conflict heuristic for NUM tiles plus one EMPTY cell."""
    layout = sliding_layout(env, state)
    if layout is None:
        return None
    tiles, active_axes = layout

    dist = 0
    for _val, cur, tgt in tiles:
        dist += sum(abs(a - b) for a, b in zip(cur, tgt))

    # For standard 2D sliding puzzles, add linear conflict: two tiles in the
    # same target row/column but reversed require at least two extra moves.
    # For higher-dimensional layouts, keep the admissible Manhattan term.
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
                target_order = [target for _cur, target in entries]
                conflicts += len(target_order) - lis_length(target_order)
        dist += 2 * conflicts

    return int(dist)


def lis_length(values):
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


def state_array_from_env(env, state):
    obs = env.encode_state(state)
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


def array_to_state_like(values, template):
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

    out = fill(template)
    return common.to_jsonable(out)


def make_fast_sliding_context(env, template_state):
    """Precompute direct array transitions for NUM+EMPTY+SWAP puzzles."""
    parsed = state_array_from_env(env, template_state)
    if parsed is None:
        return None
    start_arr, target_arr, obs = parsed
    n = len(start_arr)

    pos = np.asarray(obs["positions"], dtype=np.float32)
    ranks = []
    for axis in range(pos.shape[1]):
        vals = sorted({round(float(x), 6) for x in pos[:, axis]})
        ranks.append({v: i for i, v in enumerate(vals)})
    coords = [
        tuple(ranks[axis][round(float(p[axis]), 6)] for axis in range(pos.shape[1]))
        for p in pos
    ]
    active_axes = [axis for axis, mp in enumerate(ranks) if len(mp) > 1]
    if len(active_axes) != 2:
        return None

    try:
        target_empty = int(np.asarray(obs["target_types"], dtype=np.int64).tolist().index(CONTENT_EMPTY))
        current_empty = int(np.asarray(obs["content_types"], dtype=np.int64).tolist().index(CONTENT_EMPTY))
    except ValueError:
        return None

    goal_index = {}
    goal_coord = {}
    for idx, val in enumerate(target_arr):
        if idx == target_empty:
            continue
        goal_index[int(val)] = idx
        goal_coord[int(val)] = coords[idx]

    # Distance table: value -> [distance from each cell to value's target cell].
    dist_table = {}
    for val, tgt in goal_coord.items():
        dist_table[val] = [sum(abs(a - b) for a, b in zip(cur, tgt)) for cur in coords]

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
        for action, typ, affected in zip(
            enc.get("actions", []),
            enc.get("action_types", []),
            enc.get("affected", []),
        ):
            if typ != getattr(gym, "ACTION_SWAP", 0) or len(affected) != 2:
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
        "current_empty": current_empty,
        "coords": coords,
        "active_axes": active_axes,
        "goal_coord": goal_coord,
        "dist_table": dist_table,
        "moves": moves,
        "template": template_state,
    }


def fast_sliding_key(arr):
    return bytes(int(v) & 0xFF for v in arr)


def fast_sliding_heuristic(ctx, arr):
    dist = 0
    coords = ctx["coords"]
    ax0, ax1 = ctx["active_axes"]
    goal_coord = ctx["goal_coord"]

    row_groups = {}
    col_groups = {}
    for idx, val in enumerate(arr):
        val = int(val)
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
    return dist + 2 * conflicts


def fast_sliding_manhattan(ctx, arr):
    dist = 0
    for idx, val in enumerate(arr):
        val = int(val)
        if val == 0:
            continue
        table = ctx["dist_table"].get(val)
        if table is None:
            return 10**9
        dist += table[idx]
    return dist


def solve_sliding_fast_manhattan_idastar(env, initial_state, deadline, max_depth=100):
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
    threshold = fast_sliding_manhattan(ctx, arr)
    path = []
    path_keys = {fast_sliding_key(arr)}

    def dfs(g, bound, blank_idx, prev_blank, h):
        if time.time() >= deadline:
            return None, float("inf")
        f = g + h
        if f > bound:
            return None, f
        if h == 0:
            return list(path), f
        if g >= max_depth:
            return None, float("inf")

        best_next = float("inf")
        moves = []
        for nb, action in ctx["moves"][blank_idx]:
            if nb == prev_blank:
                continue
            moved = int(arr[nb])
            table = ctx["dist_table"].get(moved)
            if table is None:
                continue
            nh = h - table[nb] + table[blank_idx]
            moves.append((nh, nb, action, table))

        moves.sort(key=lambda x: x[0])
        for nh, nb, action, _table in moves:
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            key = fast_sliding_key(arr)
            if key in path_keys:
                arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
                continue
            path.append(action)
            path_keys.add(key)
            res, nxt = dfs(g + 1, bound, nb, blank_idx, nh)
            path_keys.remove(key)
            path.pop()
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            if res is not None:
                return res, nxt
            best_next = min(best_next, nxt)
        return None, best_next

    while time.time() < deadline and threshold <= max_depth:
        res, nxt = dfs(0, threshold, blank, -1, threshold)
        if os.environ.get("DEBUG_MANHATTAN_IDA") == "1":
            print(f"debug_manhattan_ida threshold={threshold} nxt={nxt} res={None if res is None else len(res)}")
        if res is not None:
            return res if validate_actions(env, initial_state, res) else None
        if nxt == float("inf"):
            return None
        threshold = int(nxt)
    return None


def solve_sliding_fast_idastar(env, initial_state, solved_key, deadline, max_depth=100):
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
    expanded = 0

    def dfs(g, bound, blank_idx, prev_blank):
        nonlocal expanded
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
        expanded += 1

        children = []
        for nb, action in ctx["moves"][blank_idx]:
            if nb == prev_blank:
                continue
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            key = fast_sliding_key(arr)
            if key not in path_keys:
                children.append((fast_sliding_heuristic(ctx, arr), nb, action, key))
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]

        children.sort(key=lambda x: x[0])
        best_next = float("inf")
        for _h, nb, action, key in children:
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
            path.append(action)
            path_keys.add(key)
            res, nxt = dfs(g + 1, bound, nb, blank_idx)
            if res is not None:
                return res, nxt
            best_next = min(best_next, nxt)
            path_keys.remove(key)
            path.pop()
            arr[blank_idx], arr[nb] = arr[nb], arr[blank_idx]
        return None, best_next

    while time.time() < deadline and threshold <= max_depth:
        res, nxt = dfs(0, threshold, blank, -1)
        if res is not None:
            return res if validate_actions(env, initial_state, res) else None
        if nxt == float("inf"):
            return None
        threshold = int(nxt)
    return None


def search_score(env, state):
    # Beam search needs a very smooth ranking signal. In local tests both
    # Manhattan and linear conflict were too brittle as beam scores; mismatch
    # keeps more useful diversity.
    return mismatch_count(env, state)


def is_sliding_swap_env(env, state):
    h = sliding_heuristic(env, state)
    if h is None:
        return False
    try:
        enc = env.encode_actions(state=state)
        types = enc.get("action_types", [])
        affected = enc.get("affected", [])
        return bool(types) and all(t == 0 for t in types) and all(len(a) == 2 for a in affected)
    except Exception:
        return False


def solve_toggle_linear(env, initial_state):
    """Exact GF(2) solver for binary TOGGLE puzzles."""
    try:
        obs = env.encode_state(initial_state)
        actions = env.valid_actions() if initial_state is None else None
        enc_actions = env.encode_actions(actions=actions, state=initial_state)
    except Exception:
        return None

    action_types = enc_actions.get("action_types", [])
    actions = enc_actions.get("actions", [])
    affected = enc_actions.get("affected", [])
    if not actions or any(t != ACTION_TOGGLE for t in action_types):
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
    rhs = (cv ^ tv).astype(np.uint8)

    # Rows are equations per cell: sum(action toggles cell) = rhs[cell] mod 2.
    rows = []
    for i in range(n_cells):
        mask = 0
        for j, cells in enumerate(affected):
            if i in cells:
                mask ^= 1 << j
        rows.append([mask, int(rhs[i])])

    pivot_cols = []
    row = 0
    for col in range(n_actions):
        pivot = None
        bit = 1 << col
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
    # Free variables stay 0; reduced rows give one valid solution.
    for r, col in enumerate(pivot_cols):
        if rows[r][1]:
            x |= 1 << col
    sol = [actions[j] for j in range(n_actions) if (x >> j) & 1]
    return sol if validate_actions(env, initial_state, sol) else None


def ordered_children(env, state, prev_action=None, v_fn=None):
    try:
        env.set_state(state)
        actions = env.valid_actions()
    except Exception:
        return []
    if prev_action is not None:
        try:
            inv = env.inverse_action(prev_action)
            if len(actions) > 1:
                actions = [a for a in actions if a != inv]
        except Exception:
            pass

    children = []
    for a in actions:
        try:
            env.set_state(state)
            env.step(a)
            ns = common.to_jsonable(env.get_state())
            children.append((a, ns, search_score(env, ns)))
        except Exception:
            continue
    children.sort(key=lambda x: x[2])
    return [(a, ns) for a, ns, _m in children]


def solve_sliding_idastar(env, initial_state, solved_key, deadline, max_depth=80):
    if not is_sliding_swap_env(env, initial_state):
        return None
    fast = solve_sliding_fast_idastar(env, initial_state, solved_key, deadline, max_depth=max_depth)
    if fast is not None:
        return fast
    start = common.to_jsonable(initial_state)
    if common.state_key(start) == solved_key:
        return []

    h0 = sliding_heuristic(env, start)
    if h0 is None:
        return None
    threshold = h0
    path = []
    path_keys = {common.state_key(start)}

    def search(state, g, bound, prev_action):
        if time.time() >= deadline:
            return None, float("inf")
        h = sliding_heuristic(env, state)
        if h is None:
            return None, float("inf")
        f = g + h
        if f > bound:
            return None, f
        sk = common.state_key(state)
        if sk == solved_key:
            return list(path), f
        if g >= max_depth:
            return None, float("inf")

        children = []
        for a, ns in ordered_children(env, state, prev_action):
            nk = common.state_key(ns)
            if nk in path_keys:
                continue
            hh = sliding_heuristic(env, ns)
            if hh is None:
                continue
            children.append((hh, a, ns, nk))
        children.sort(key=lambda x: x[0])

        best_next = float("inf")
        for _hh, a, ns, nk in children:
            path.append(a)
            path_keys.add(nk)
            res, nxt = search(ns, g + 1, bound, a)
            if res is not None:
                return res, nxt
            best_next = min(best_next, nxt)
            path_keys.remove(nk)
            path.pop()
        return None, best_next

    while time.time() < deadline and threshold <= max_depth:
        res, nxt = search(start, 0, threshold, None)
        if res is not None:
            return res if validate_actions(env, initial_state, res) else None
        if nxt == float("inf"):
            return None
        threshold = int(nxt)
    return None


def solve_beam(env, initial_state, solved_key, deadline, beam_width=256, max_depth=80):
    start = common.to_jsonable(initial_state)
    if common.state_key(start) == solved_key:
        return []
    beam = [(search_score(env, start), start, [], None)]
    seen = {common.state_key(start)}

    for _depth in range(max_depth):
        if time.time() >= deadline:
            return None
        candidates = []
        for _score, state, path, prev in beam:
            for a, ns in ordered_children(env, state, prev):
                k = common.state_key(ns)
                if k in seen:
                    continue
                seen.add(k)
                npath = path + [a]
                if k == solved_key:
                    return npath
                score = search_score(env, ns) + 0.03 * len(npath)
                candidates.append((score, ns, npath, a))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
    return None


def solve_bfs_shallow(env, initial_state, solved_key, deadline, max_depth=10, max_nodes=20000):
    start = common.to_jsonable(initial_state)
    if common.state_key(start) == solved_key:
        return []
    q = deque([(start, [], None)])
    seen = {common.state_key(start)}
    nodes = 0
    while q and nodes < max_nodes and time.time() < deadline:
        state, path, prev = q.popleft()
        if len(path) >= max_depth:
            continue
        for a, ns in ordered_children(env, state, prev):
            k = common.state_key(ns)
            if k in seen:
                continue
            seen.add(k)
            npath = path + [a]
            if k == solved_key:
                return npath
            q.append((ns, npath, a))
            nodes += 1
    return None


def solve_instance(env, state, solved_k, v_fn, deadline, strategy, bfs_table=None):
    if strategy == "table_only":
        return solve_with_table(env, state, bfs_table, deadline)
    if strategy == "toggle_only":
        return solve_toggle_linear(env, state)
    if strategy == "sliding_only":
        return solve_sliding_idastar(env, state, solved_k, deadline)
    if strategy == "sliding_manhattan":
        return solve_sliding_fast_manhattan_idastar(env, state, deadline)
    if strategy == "bfs":
        return solve_bfs_shallow(env, state, solved_k, deadline, max_depth=12, max_nodes=50_000)
    if strategy == "greedy":
        return solve_beam(env, state, solved_k, deadline, beam_width=1, max_depth=120)
    if strategy == "beam":
        return solve_beam(env, state, solved_k, deadline, beam_width=256, max_depth=120)
    if strategy == "astar":
        return solve_astar(env, state, solved_k, v_fn, deadline)
    if strategy == "zero_astar":
        return solve_astar(env, state, solved_k, None, deadline)

    # Full cascade, tuned for robust throughput rather than a single solver's purity.
    sol = solve_with_table(env, state, bfs_table, min(deadline, time.time() + 0.5))
    if sol is None:
        sol = solve_toggle_linear(env, state)
    sliding = is_sliding_swap_env(env, state)
    if sol is None and sliding and time.time() < deadline:
        sol = solve_sliding_idastar(env, state, solved_k, deadline)
    if sol is None and sliding and time.time() < deadline:
        sol = solve_beam(env, state, solved_k, deadline, beam_width=256, max_depth=120)
    if sol is None:
        quick_deadline = min(deadline, time.time() + 0.15)
        sol = solve_bfs_shallow(env, state, solved_k, quick_deadline)
    if sol is None and time.time() < deadline:
        quick_deadline = min(deadline, time.time() + 0.35)
        sol = solve_beam(env, state, solved_k, quick_deadline)
    if sol is None and not sliding and time.time() < deadline:
        sol = solve_beam(env, state, solved_k, deadline, beam_width=512, max_depth=160)
    if sol is None and time.time() < deadline:
        sol = solve_astar(env, state, solved_k, v_fn, deadline)
    return sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("SOLVE_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--strategy", default=os.environ.get("SOLVER_STRATEGY", "full"),
                        choices=[
                            "full", "toggle_only", "sliding_only", "sliding_manhattan", "bfs",
                            "greedy", "beam", "astar", "zero_astar", "table_only",
                        ])
    args = parser.parse_args()

    start = time.time()
    safety_margin = min(SAFETY_MARGIN_DEFAULT, max(1.0, 0.05 * args.time_limit))
    deadline = start + args.time_limit - safety_margin
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    env = gym.make_env()
    instances = load_jsonl(args.input)
    print(f"loaded {len(instances)} instances")
    print(f"strategy={args.strategy}")

    model = load_model()
    print(f"model loaded: {model is not None}")
    bfs_table = load_bfs_table()

    env.reset()
    solved_k = state_key(env.get_state())
    v_fn = make_v_fn(env, model)

    n = len(instances)
    solutions = [None] * n
    retry_reserve = min(120.0, max(2.0, 0.08 * (deadline - start)))
    first_pass_deadline = max(start, deadline - retry_reserve)

    for i, inst in enumerate(instances):
        iid = inst["instance_id"]
        if time.time() >= first_pass_deadline:
            continue

        remaining = n - i
        inst_deadline = time.time() + max(0.35, (first_pass_deadline - time.time()) / max(1, remaining))

        try:
            solutions[i] = solve_instance(
                env, inst["state"], solved_k, v_fn, inst_deadline,
                args.strategy, bfs_table=bfs_table,
            )
        except Exception as e:
            print(f"  {iid} failed: {repr(e)}")
            solutions[i] = None

        if (i + 1) % 25 == 0:
            solved_now = sum(1 for sol in solutions if sol)
            print(f"  pass1 {i+1}/{n} solved={solved_now} elapsed={time.time()-start:.0f}s")

    # Second pass: spend leftover time on the failures. This recovers early hard
    # instances that only received the fair-share budget in pass 1.
    retry_indices = [i for i, sol in enumerate(solutions) if not sol]
    for retry_no, i in enumerate(retry_indices):
        if time.time() >= deadline or not retry_indices:
            break
        inst = instances[i]
        remaining = len(retry_indices) - retry_no
        inst_deadline = time.time() + max(0.4, (deadline - time.time()) / max(1, remaining))
        try:
            sol = solve_instance(
                env, inst["state"], solved_k, v_fn, inst_deadline,
                args.strategy, bfs_table=bfs_table,
            )
            if sol:
                solutions[i] = sol
        except Exception as e:
            print(f"  retry {inst['instance_id']} failed: {repr(e)}")

    solved = sum(1 for sol in solutions if sol)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        writer.writeheader()
        for inst, sol in zip(instances, solutions):
            writer.writerow({
                "instance_id": inst["instance_id"],
                "actions": " ".join(sol or []),
            })

    print(f"final: solved {solved}/{n}, time {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
