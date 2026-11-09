"""Object/memory audit helper for the matrix-free integrity check (no dense K_hat / H_hat).
numpy arrays are not tracked by the garbage collector, so the inventory walks the objects that hold the production state."""
import numpy as np


def reachable_arrays(*objs):
    seen, out = set(), []
    def walk(o, name):
        if id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            out.append((name, o.shape, int(o.nbytes)))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{name}[{k!r}]")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{name}[{i}]")
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            for k, v in vars(o).items():
                walk(v, f"{name}.{k}")
    for i, o in enumerate(objs):
        walk(o, f"obj{i}")
    return out


def forbidden_dense(arrs, min_dim=400):
    """2-D arrays with both dimensions >= min_dim (a dense N x N or N_class x N_class object would qualify)."""
    return [a for a in arrs if len(a[1]) == 2 and min(a[1]) >= min_dim]
