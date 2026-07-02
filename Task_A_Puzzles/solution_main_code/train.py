"""Train-time adaptation for adaptive reversible puzzles.

Produces:
  - model.pt: PolicyNet classifier P(action | state)
  - bfs_table.pkl: optional reverse table near the solved state

Key fix over the naive policy baseline: action_vocab is collected from many
reachable states, not only from env.valid_actions() at the solved state.
That matters for dynamic-action puzzles such as cylinder/empty-slot puzzles.
"""

import json
import os
import pickle
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
import common
from model import PolicyNet

TIME_LIMIT_DEFAULT = float(os.environ.get("TRAIN_TIME_LIMIT", 50 * 60))
MODEL_PATH = "model.pt"
BFS_TABLE_PATH = "bfs_table.pkl"
META_PATH = "meta.json"

ACTION_TOGGLE = getattr(gym, "ACTION_TOGGLE", 2)


def to_tensors(tokens: np.ndarray):
    b, n, _ = tokens.shape
    parts = common.split_token_features(tokens.reshape(b * n, -1))
    return (
        torch.from_numpy(parts["dense"].reshape(b, n, -1)),
        torch.from_numpy(parts["content_value"].reshape(b, n)),
        torch.from_numpy(parts["target_value"].reshape(b, n)),
    )


def key(st):
    return common.compact_state_key(st)


def detect_toggle_only(env) -> bool:
    try:
        enc = env.encode_actions()
        return bool(enc.get("actions", [])) and all(int(t) == ACTION_TOGGLE for t in enc.get("action_types", []))
    except Exception:
        return False


def collect_action_vocab(env, deadline, seed=123, walks=900, max_walk=140):
    rng = random.Random(seed)
    vocab = set()
    env.reset()
    try:
        vocab.update(env.valid_actions())
    except Exception:
        pass

    for _ in range(walks):
        if time.time() >= deadline:
            break
        env.reset(seed=rng.randint(0, 10**9))
        prev = None
        length = rng.randint(1, max_walk)
        for _ in range(length):
            if time.time() >= deadline:
                break
            try:
                valid = list(env.valid_actions())
            except Exception:
                break
            vocab.update(valid)
            if prev is not None and len(valid) > 1:
                try:
                    inv = env.inverse_action(prev)
                    valid = [a for a in valid if a != inv] or valid
                except Exception:
                    pass
            a = rng.choice(valid)
            try:
                env.step(a)
            except Exception:
                break
            prev = a
    return sorted(vocab)


def build_reverse_table(env, deadline, max_states=350_000, max_depth=8):
    """Reverse BFS from goal. table[state_key] = (dist_to_goal, action_to_goal)."""
    if detect_toggle_only(env):
        return {}, []

    env.reset()
    goal = common.to_jsonable(env.get_state())
    table = {key(goal): (0, None)}
    samples = []  # (state, action_to_goal)
    q = deque([(goal, 0, None)])

    while q and len(table) < max_states and time.time() < deadline:
        st, dist, prev = q.popleft()
        if dist >= max_depth:
            continue
        try:
            env.set_state(st)
            actions = list(env.valid_actions())
        except Exception:
            continue
        if prev is not None and len(actions) > 1:
            try:
                inv_prev = env.inverse_action(prev)
                actions = [a for a in actions if a != inv_prev] or actions
            except Exception:
                pass
        for a in actions:
            if len(table) >= max_states or time.time() >= deadline:
                break
            try:
                env.set_state(st)
                env.step(a)
                ns = common.to_jsonable(env.get_state())
                nk = key(ns)
                if nk in table:
                    continue
                back = env.inverse_action(a)
                table[nk] = (dist + 1, back)
                samples.append((ns, back))
                q.append((ns, dist + 1, a))
            except Exception:
                continue
    return table, samples


def collect_policy_dataset(env, action_vocab, deadline, seed=42,
                           target_pairs=120_000, max_walk=160,
                           table_samples=None):
    action_to_id = {a: i for i, a in enumerate(action_vocab)}
    rng = random.Random(seed)
    states = []
    labels = []

    if table_samples:
        rng.shuffle(table_samples)
        for st, action in table_samples[:min(len(table_samples), target_pairs // 2)]:
            if action in action_to_id:
                states.append(st)
                labels.append(action_to_id[action])

    # Backward random-walk imitation: after making action a from a known-good path,
    # inverse(a) is a valid step back toward goal.
    while len(states) < target_pairs and time.time() < deadline:
        env.reset(seed=rng.randint(0, 10**9))
        prev = None
        length = rng.randint(1, max_walk)
        for _ in range(length):
            if len(states) >= target_pairs or time.time() >= deadline:
                break
            try:
                valid = list(env.valid_actions())
            except Exception:
                break
            if prev is not None and len(valid) > 1:
                try:
                    inv = env.inverse_action(prev)
                    valid = [a for a in valid if a != inv] or valid
                except Exception:
                    pass
            if not valid:
                break
            a = rng.choice(valid)
            try:
                env.step(a)
                label = env.inverse_action(a)
            except Exception:
                break
            if label in action_to_id:
                states.append(common.to_jsonable(env.get_state()))
                labels.append(action_to_id[label])
            prev = a

    if not states:
        return np.zeros((0, 1, common.TOKEN_FEAT_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)

    idx = list(range(len(states)))
    rng.shuffle(idx)
    idx = idx[:target_pairs]
    states = [states[i] for i in idx]
    labels = np.asarray([labels[i] for i in idx], dtype=np.int64)
    tokens = np.stack([common.encode_tokens(env, st) for st in states]).astype(np.float32)
    return tokens, labels


def train_policy(tokens, labels, num_actions, deadline):
    model = PolicyNet(num_actions=num_actions)
    if len(tokens) == 0:
        return model

    opt = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    batch = 256
    n = len(tokens)
    best_loss = 1e18
    bad_epochs = 0

    for epoch in range(200):
        if time.time() >= deadline:
            break
        perm = np.random.permutation(n)
        total = 0.0
        steps = 0
        model.train()
        for s in range(0, n, batch):
            if time.time() >= deadline:
                break
            sel = perm[s:s + batch]
            dense, cv, tv = to_tensors(tokens[sel])
            y = torch.from_numpy(labels[sel])
            logits = model(dense, cv, tv)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        avg = total / max(1, steps)
        print(f"epoch={epoch} loss={avg:.4f}", flush=True)
        if avg + 1e-4 < best_loss:
            best_loss = avg
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 10 and epoch >= 25:
            break
    model.eval()
    return model


def main():
    start = time.time()
    deadline = start + TIME_LIMIT_DEFAULT - 25
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    random.seed(42)
    np.random.seed(42)

    env = gym.make_env()
    env.reset()

    print("collecting action vocab...", flush=True)
    action_vocab = collect_action_vocab(env, min(deadline, time.time() + 90), walks=1200, max_walk=180)
    if not action_vocab:
        action_vocab = sorted(env.valid_actions())
    print(f"action_vocab={len(action_vocab)}", flush=True)

    print("building reverse table...", flush=True)
    # Depth/state caps are intentionally conservative: solve.py uses it as a bridge, not as the only solver.
    table_minutes = float(os.environ.get("TABLE_MINUTES", "18"))
    table_states = int(os.environ.get("TABLE_MAX_STATES", "1200000"))
    table_depth = int(os.environ.get("TABLE_MAX_DEPTH", "35"))
    table_deadline = min(deadline, time.time() + table_minutes * 60)
    # Bigger table matters only after solve.py can use it as a meet-in-the-middle target.
    # Keep these tunable because hidden memory limits may differ.
    table, table_samples = build_reverse_table(env, table_deadline, max_states=table_states, max_depth=table_depth)
    if table:
        with open(BFS_TABLE_PATH, "wb") as f:
            pickle.dump({"table": table}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"table_states={len(table)} table_samples={len(table_samples)}", flush=True)

    print("collecting policy dataset...", flush=True)
    dataset_minutes = float(os.environ.get("DATASET_MINUTES", "11"))
    target_pairs = int(os.environ.get("TARGET_PAIRS", "180000"))
    tokens, labels = collect_policy_dataset(
        env, action_vocab, min(deadline, time.time() + dataset_minutes * 60),
        seed=43, target_pairs=target_pairs, max_walk=180, table_samples=table_samples,
    )
    print(f"dataset={tokens.shape} labels={len(labels)}", flush=True)

    model = train_policy(tokens, labels, len(action_vocab), deadline)
    torch.save({
        "state_dict": model.state_dict(),
        "action_vocab": action_vocab,
        "num_actions": len(action_vocab),
    }, MODEL_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "action_vocab_size": len(action_vocab),
            "table_states": len(table),
            "dataset_size": int(len(labels)),
            "time_sec": time.time() - start,
        }, f, ensure_ascii=False, indent=2)
    print(f"train finished in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
