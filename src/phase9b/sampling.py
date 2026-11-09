"""Deterministic nested stratified sampling of the official training partition (Phase 9A/9B protocol).
Row identifiers are the official 'id' column. A sample D_N is stored as the ASCENDING array of its official ids; row position p of a
sample always means the p-th id in that ascending array."""
import math, hashlib
import numpy as np
import pandas as pd

SEEDS = (101, 102, 103, 104, 105)
NS = (2000, 4000, 8000, 16000, 32000, 64000)
EXPECTED_ALLOC = {2000: (1361, 639), 4000: (2722, 1278), 8000: (5445, 2555), 16000: (10890, 5110), 32000: (21780, 10220), 64000: (43560, 20440)}  # (label 1, label 0)


def read_ids_labels(path):
    df = pd.read_csv(path, usecols=["id", "label"], keep_default_na=False)
    ids, labels = df["id"].to_numpy(np.int64), df["label"].to_numpy(np.int64)
    assert len(np.unique(ids)) == len(ids), "official ids must be unique"
    return ids, labels


def largest_remainder(N, counts):
    """counts: {label: official count}. Ties in remainders: larger fractional part first, then label 1."""
    tot = sum(counts.values()); q = {k: N * v / tot for k, v in counts.items()}; fl = {k: int(math.floor(x)) for k, x in q.items()}
    for k in sorted(q, key=lambda k: (-(q[k] - fl[k]), -k))[: N - sum(fl.values())]:
        fl[k] += 1
    return fl


def class_orders(labels, seed):
    """Positions (file order) of each label permuted once by default_rng(SeedSequence([seed, label]))."""
    out = {}
    for y in (0, 1):
        pos = np.flatnonzero(labels == y)
        rng = np.random.default_rng(np.random.SeedSequence([seed, y]))
        out[y] = pos[rng.permutation(len(pos))]
    return out


def nested_samples(ids, labels, seed, Ns=NS):
    counts = {int(y): int((labels == y).sum()) for y in (0, 1)}
    orders = class_orders(labels, seed); res = {}
    for N in Ns:
        a = largest_remainder(N, counts)
        pos = np.concatenate([orders[1][: a[1]], orders[0][: a[0]]])
        res[N] = np.sort(ids[pos])
    return res


def verify_nested(samples, Ns=NS):
    for a, b in zip(Ns[:-1], Ns[1:]):
        assert np.isin(samples[a], samples[b]).all(), f"D_{a} is not a subset of D_{b}"
    return True


def content_sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
