"""Deterministic nested landmark ordering. L_250 is a prefix of L_1000; duplicates of predictor vectors are kept; no spectrum is inspected."""
import hashlib
import numpy as np

NAMESPACE = "main-nystrom-landmark-v1"
M_VALUES = (250, 1000)


def landmark_order(sample_ids):
    keys = sorted((hashlib.sha256(f"{NAMESPACE}|{int(i)}".encode()).hexdigest(), int(i)) for i in sample_ids)   # 1) hash, 2) official id
    return np.array([k[1] for k in keys], dtype=np.int64)


def landmark_sets(sample_ids, ms=M_VALUES):
    order = landmark_order(sample_ids)[: max(ms)]
    assert len(order) >= max(ms) and all(m <= len(sample_ids) for m in ms)
    return {m: order[:m] for m in ms}, order
