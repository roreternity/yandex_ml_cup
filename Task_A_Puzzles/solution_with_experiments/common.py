"""Shared utilities: state tokenization, canonical keys, backward random walks."""

import json
import random
import struct
from typing import Any, Dict, List, Tuple

import numpy as np


VALUE_VOCAB = 64
CONTENT_TYPES = 4
# layout per cell: pos(3) + content_type_oh(4) + content_value_idx(1) +
#                  target_type_oh(4) + target_value_idx(1) + match/mismatch(2)
TOKEN_FEAT_DIM = 3 + CONTENT_TYPES + 1 + CONTENT_TYPES + 1 + 2
DENSE_DIM = 3 + CONTENT_TYPES + CONTENT_TYPES + 2  # everything except value idxs


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x


def state_key(state) -> str:
    """Compact canonical bytes used as dict key for state equality."""
    return compact_state_key(state)


def compact_state_key(state) -> bytes:
    """Fast deterministic key for JSON-like puzzle states.

    The public and hidden APIs expose numeric arrays, nested lists, and small
    dictionaries. Packing those directly avoids `json.dumps` in hot search
    loops while staying generic across puzzle types.
    """
    if isinstance(state, np.ndarray):
        return _pack_array(state)
    if isinstance(state, np.integer):
        return b"i" + struct.pack("<q", int(state))
    if isinstance(state, np.floating):
        return b"f" + struct.pack("<d", float(state))
    if isinstance(state, bool):
        return b"b" + (b"\x01" if state else b"\x00")
    if isinstance(state, int):
        return b"i" + struct.pack("<q", state)
    if isinstance(state, float):
        return b"f" + struct.pack("<d", state)
    if isinstance(state, dict):
        parts = [b"d", struct.pack("<I", len(state))]
        for key in sorted(state.keys()):
            kb = str(key).encode("utf-8")
            vb = compact_state_key(state[key])
            parts.append(struct.pack("<I", len(kb)))
            parts.append(kb)
            parts.append(struct.pack("<I", len(vb)))
            parts.append(vb)
        return b"".join(parts)
    if isinstance(state, (list, tuple)):
        arr = np.asarray(state)
        if arr.dtype != object:
            return _pack_array(arr)
        parts = [b"l", struct.pack("<I", len(state))]
        for item in state:
            ib = compact_state_key(item)
            parts.append(struct.pack("<I", len(ib)))
            parts.append(ib)
        return b"".join(parts)
    if state is None:
        return b"n"
    rb = repr(state).encode("utf-8")
    return b"r" + struct.pack("<I", len(rb)) + rb


def _pack_array(value) -> bytes:
    arr = np.ascontiguousarray(value)
    shape = struct.pack("<B", arr.ndim) + b"".join(struct.pack("<I", int(s)) for s in arr.shape)

    if arr.dtype.kind in ("b", "?"):
        data = arr.astype(np.uint8, copy=False).tobytes()
        return b"A" + shape + b"u1" + data

    if arr.dtype.kind in ("i", "u"):
        if arr.size == 0:
            data = b""
            dtype_code = b"i1"
        else:
            mn = int(arr.min())
            mx = int(arr.max())
            if 0 <= mn and mx <= 255:
                data = arr.astype(np.uint8, copy=False).tobytes()
                dtype_code = b"u1"
            elif -32768 <= mn and mx <= 32767:
                data = arr.astype("<i2", copy=False).tobytes()
                dtype_code = b"i2"
            else:
                data = arr.astype("<i4", copy=False).tobytes()
                dtype_code = b"i4"
        return b"A" + shape + dtype_code + data

    if arr.dtype.kind == "f":
        data = arr.astype("<f4", copy=False).tobytes()
        return b"A" + shape + b"f4" + data

    # Rare fallback for strings or mixed object-ish arrays.
    js = json.dumps(to_jsonable(arr.tolist()), sort_keys=True).encode("utf-8")
    return b"J" + struct.pack("<I", len(js)) + js


def encode_tokens(env, state=None) -> np.ndarray:
    """(N, TOKEN_FEAT_DIM) float32 features per cell."""
    obs = env.encode_state(state)
    pos = np.asarray(obs["positions"], dtype=np.float32)
    ct = np.asarray(obs["content_types"], dtype=np.int64)
    cv = np.asarray(obs["content_values"], dtype=np.int64)
    tt = np.asarray(obs["target_types"], dtype=np.int64)
    tv = np.asarray(obs["target_values"], dtype=np.int64)

    n = len(ct)
    feat = np.zeros((n, TOKEN_FEAT_DIM), dtype=np.float32)
    feat[:, 0:3] = pos
    for t in range(CONTENT_TYPES):
        feat[:, 3 + t] = (ct == t).astype(np.float32)
    feat[:, 3 + CONTENT_TYPES] = np.clip(cv, 0, VALUE_VOCAB - 1).astype(np.float32)
    base = 3 + CONTENT_TYPES + 1
    for t in range(CONTENT_TYPES):
        feat[:, base + t] = (tt == t).astype(np.float32)
    feat[:, base + CONTENT_TYPES] = np.clip(tv, 0, VALUE_VOCAB - 1).astype(np.float32)
    match = ((ct == tt) & (cv == tv)).astype(np.float32)
    feat[:, -2] = match
    feat[:, -1] = 1.0 - match
    return feat


def split_token_features(tokens: np.ndarray) -> Dict[str, np.ndarray]:
    base = 3 + CONTENT_TYPES + 1
    dense = np.concatenate(
        [tokens[:, :3 + CONTENT_TYPES], tokens[:, base:base + CONTENT_TYPES], tokens[:, -2:]],
        axis=-1,
    ).astype(np.float32)
    return {
        "dense": dense,
        "content_value": tokens[:, 3 + CONTENT_TYPES].astype(np.int64),
        "target_value": tokens[:, base + CONTENT_TYPES].astype(np.int64),
    }


def backward_walks(
    env, num_walks: int, min_len: int, max_len: int, seed: int = 0,
) -> List[Tuple[Any, int]]:
    """Sample (state, depth) pairs by random walks from solved."""
    rng = random.Random(seed)
    pairs: List[Tuple[Any, int]] = []

    for _ in range(num_walks):
        L = rng.randint(min_len, max_len)
        env.reset(seed=rng.randint(0, 10**9))
        for depth in range(1, L + 1):
            a = rng.choice(env.valid_actions())
            env.step(a)
            pairs.append((to_jsonable(env.get_state()), depth))

    return pairs
