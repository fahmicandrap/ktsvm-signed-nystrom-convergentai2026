"""Evaluation-probe and kernel-pair manifest generators (identifiers only; no features, no kernels)."""
import numpy as np
from sampling import largest_remainder, content_sha256

EVAL_SEED = 20260922
EVAL_SIZE = 5000
EVAL_ALLOC = {1: 2753, 0: 2247}
PAIRS_PER_CELL = 200000
PAIR_TAG = 5150


def eval_probe(test_ids, test_labels, seed=EVAL_SEED, alloc=EVAL_ALLOC):
    """Class-stratified probe from the OFFICIAL TEST partition; returned ids sorted ascending. Labels are not returned."""
    picks = []
    for y in (0, 1):
        pos = np.flatnonzero(test_labels == y)
        rng = np.random.default_rng(np.random.SeedSequence([seed, y]))
        picks.append(pos[rng.permutation(len(pos))][: alloc[y]])
    return np.sort(test_ids[np.concatenate(picks)])


def kernel_pairs(seed, N, P=PAIRS_PER_CELL):
    """P ordered pairs of sample POSITIONS (0..N-1), uniform with replacement subject to i != k, SeedSequence([seed, N, 5150]):
    i uniform on 0..N-1, offset uniform on 1..N-1, k = (i + offset) mod N (uniform over k != i)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, N, PAIR_TAG]))
    i = rng.integers(0, N, size=P, dtype=np.int64)
    off = rng.integers(1, N, size=P, dtype=np.int64)
    k = (i + off) % N
    assert (i != k).all() and i.min() >= 0 and k.max() < N
    return np.stack([i, k], axis=1).astype(np.int32)


def pair_id_hash(pairs, sample_ids):
    return content_sha256(np.stack([sample_ids[pairs[:, 0]], sample_ids[pairs[:, 1]]], axis=1).astype(np.int64))
