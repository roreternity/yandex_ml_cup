"""Hidden-game adaptation: fit V(s) on backward random walks; save model.pt."""

import argparse
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
from model import ValueNet


TIME_LIMIT_DEFAULT = 1 * 60
SAFETY_MARGIN = 20
MODEL_PATH = "model.pt"
META_PATH = "meta.json"
BFS_TABLE_PATH = "bfs_table.pkl"
ACTION_TOGGLE = getattr(gym, "ACTION_TOGGLE", 2)
ACTION_SWAP = getattr(gym, "ACTION_SWAP", 0)
CONTENT_EMPTY = getattr(gym, "CONTENT_EMPTY", 0)


def build_bfs_table(env, deadline, max_states):
    """Build reverse BFS from solved state: key -> (dist, action_to_goal)."""
    if max_states <= 1 or time.time() >= deadline:
        return None

    env.reset()
    solved_state = common.to_jsonable(env.get_state())
    solved_key = common.compact_state_key(solved_state)
    table = {solved_key: (0, None)}
    q = deque([(solved_state, solved_key, None)])
    depth_counts = {}
    expanded = 0
    t0 = time.time()

    print(f"building reverse BFS table: max_states={max_states:,}")
    while q and len(table) < max_states and time.time() < deadline:
        state, key, action_to_parent = q.popleft()
        dist = table[key][0]
        depth_counts[dist] = depth_counts.get(dist, 0) + 1
        expanded += 1

        if expanded % 100_000 == 0:
            elapsed = max(1e-6, time.time() - t0)
            print(
                f"  BFS: states={len(table):,} expanded={expanded:,} "
                f"depth~{dist} speed={len(table)/elapsed:.0f}/s"
            )

        try:
            env.set_state(state)
            actions = env.valid_actions()
        except Exception:
            continue

        # The action that solves this state only returns to the parent and is
        # already visited; skipping it saves a little time on reversible puzzles.
        if action_to_parent is not None and len(actions) > 1:
            actions = [a for a in actions if a != action_to_parent]

        for action in actions:
            try:
                env.set_state(state)
                env.step(action)
                child = common.to_jsonable(env.get_state())
                child_key = common.compact_state_key(child)
            except Exception:
                continue

            if child_key in table:
                continue
            try:
                action_to_goal = env.inverse_action(action)
            except Exception:
                action_to_goal = action
            table[child_key] = (dist + 1, action_to_goal)
            q.append((child, child_key, action_to_goal))

            if len(table) >= max_states or time.time() >= deadline:
                break

    elapsed = time.time() - t0
    max_depth = max(depth_counts) if depth_counts else 0
    print(
        f"BFS done: states={len(table):,}, expanded={expanded:,}, "
        f"max_depth={max_depth}, time={elapsed:.1f}s"
    )
    print(f"BFS depth_counts={dict(sorted(depth_counts.items()))}")
    return {
        "table": table,
        "meta": {
            "states": len(table),
            "expanded": expanded,
            "max_depth": max_depth,
            "depth_counts": depth_counts,
            "time_sec": elapsed,
        },
    }


def detect_puzzle_type(env):
    try:
        env.reset()
        state = common.to_jsonable(env.get_state())
        enc_state = env.encode_state(state)
        enc_actions = env.encode_actions(state=state)
        action_types = set(enc_actions.get("action_types", []))
        content_types = set(enc_state.get("content_types", []))
        if action_types and action_types == {ACTION_TOGGLE}:
            return "toggle"
        if ACTION_SWAP in action_types and CONTENT_EMPTY in content_types:
            return "sliding"
        return "generic"
    except Exception:
        return "generic"


def collect_dataset(env, num_pairs, max_walk, seed):
    pairs = common.backward_walks(
        env, num_walks=max(100, num_pairs // max(1, max_walk // 2)),
        min_len=1, max_len=max_walk, seed=seed,
    )
    random.Random(seed + 1).shuffle(pairs)
    pairs = pairs[:num_pairs]
    if not pairs:
        return (
            np.zeros((0, 1, common.TOKEN_FEAT_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    tokens = np.stack([common.encode_tokens(env, s) for s, _ in pairs])
    labels = np.array([d for _, d in pairs], dtype=np.float32)
    return tokens, labels


def to_tensors(tokens):
    B, N, _ = tokens.shape
    parts = common.split_token_features(tokens.reshape(B * N, -1))
    return (
        torch.from_numpy(parts["dense"].reshape(B, N, -1)),
        torch.from_numpy(parts["content_value"].reshape(B, N)),
        torch.from_numpy(parts["target_value"].reshape(B, N)),
    )


def finetune(model, tokens, labels, deadline, batch_size=256, lr=1e-3, max_epochs=1000):
    n = tokens.shape[0]
    if n == 0:
        return []
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    history = []

    for epoch in range(max_epochs):
        if time.time() >= deadline:
            break
        idx = np.random.permutation(n)
        epoch_loss, steps = 0.0, 0
        for s in range(0, n, batch_size):
            if time.time() >= deadline:
                break
            sel = idx[s:s + batch_size]
            if len(sel) < 2:
                continue
            dense, cv, tv = to_tensors(tokens[sel])
            y = torch.from_numpy(labels[sel])

            pred = model(dense, cv, tv)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1

        avg = epoch_loss / max(1, steps)
        history.append(avg)
        print(f"  epoch {epoch}: loss={avg:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("TRAIN_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--num_pairs", type=int, default=40000)
    parser.add_argument("--max_walk", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bfs_max_states", type=int,
                        default=int(os.environ.get("BFS_MAX_STATES", "1000000")))
    parser.add_argument("--bfs_time_limit", type=float,
                        default=float(os.environ.get("BFS_TIME_LIMIT", "-1")))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN

    env = gym.make_env()
    env_id = getattr(gym, "ENV_ID", "unknown")
    print(f"env_id={env_id}")
    puzzle_type = detect_puzzle_type(env)
    print(f"puzzle_type={puzzle_type}")

    available = max(0.0, deadline - time.time())
    if args.bfs_time_limit >= 0:
        bfs_seconds = min(args.bfs_time_limit, available)
    else:
        bfs_seconds = min(900.0, available * 0.45)
    bfs_payload = None
    if puzzle_type == "toggle":
        print("skipping BFS table: exact GF(2) toggle solver is available")
    elif bfs_seconds >= 1.0 and args.bfs_max_states > 1:
        bfs_deadline = min(deadline, time.time() + bfs_seconds)
        bfs_payload = build_bfs_table(env, bfs_deadline, args.bfs_max_states)
        if bfs_payload is not None:
            with open(BFS_TABLE_PATH, "wb") as f:
                pickle.dump(bfs_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved {BFS_TABLE_PATH}")

    print("collecting data...")
    t0 = time.time()
    tokens, labels = collect_dataset(env, args.num_pairs, args.max_walk, args.seed)
    print(f"dataset: {tokens.shape[0]} pairs, {time.time()-t0:.1f}s")

    model = ValueNet()

    print("fitting V...")
    history = finetune(
        model, tokens, labels, deadline,
        batch_size=args.batch_size, lr=args.lr,
    )

    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"env_id": env_id, "history": history}},
        MODEL_PATH,
    )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "env_id": env_id,
            "puzzle_type": puzzle_type,
            "num_train_samples": int(tokens.shape[0]),
            "bfs": bfs_payload["meta"] if bfs_payload else None,
            "wall_time_sec": time.time() - start,
        }, f, indent=2)

    print(f"train.py done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
