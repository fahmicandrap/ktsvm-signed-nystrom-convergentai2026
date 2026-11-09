"""Timing, memory, heartbeat, wall-time and fidelity-metric utilities. Same mechanism for every method."""
import os, sys, json, time, threading, resource, datetime, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, HERE + "../g3r2/")
from g3r2_audit_utils import reachable_arrays

utcnow = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def _status_bytes(key):
    for l in open("/proc/self/status"):
        if l.startswith(key + ":"):
            return int(l.split()[1]) * 1024
    return None


def rss_bytes(): return _status_bytes("VmRSS")
def hwm_bytes(): return _status_bytes("VmHWM")


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as fh: fh.write("5")
        return True
    except Exception:
        return False


class WallTimeLimit(Exception):
    pass


class WallClock:
    def __init__(self, start_epoch, limit_s): self.start, self.limit = float(start_epoch), float(limit_s)
    def elapsed(self): return time.time() - self.start
    def check(self):
        if self.elapsed() >= self.limit: raise WallTimeLimit(f"wall-time limit {self.limit:.0f} s reached")


class Heartbeat:
    """Atomic JSON heartbeat (timestamp, cell_id, phase, elapsed wall, twin, HVP count, r_inf, gap_rel, RSS); optional daemon thread."""
    def __init__(self, path, cell_id, clock, interval_s=30.0):
        self.path, self.cell_id, self.clock, self.interval = path, cell_id, clock, interval_s
        self.state = {"phase": None, "twin": None, "hvp_count": None, "r_inf": None, "gap_rel": None}; self._stop = threading.Event(); self._th = None; self.last = None
    def set(self, **kw): self.state.update(kw)
    def write(self):
        rec = {"timestamp_utc": utcnow(), "cell_id": self.cell_id, "elapsed_wall_s": self.clock.elapsed(), "rss_bytes": rss_bytes(), **self.state}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh: json.dump(rec, fh)
        os.replace(tmp, self.path); self.last = rec; return rec
    def start(self):
        def loop():
            while not self._stop.wait(self.interval):
                try: self.write()
                except Exception: pass
        self.write(); self._th = threading.Thread(target=loop, daemon=True); self._th.start()
    def stop(self):
        self._stop.set()
        try: self.write()
        except Exception: pass


class Recorder:
    def __init__(self, hb): self.hb = hb; self.timings = {}; self.memory = {}
    @contextlib.contextmanager
    def phase(self, name):
        reset_peak_rss(); r0 = rss_bytes(); self.hb.set(phase=name); self.hb.write(); t0 = time.perf_counter(); c0 = time.process_time()
        try: yield
        finally:
            self.timings[name] = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}
            self.memory[name] = {"rss_start_bytes": r0, "rss_end_bytes": rss_bytes(), "peak_rss_bytes": hwm_bytes()}


class HVPWrapper:
    """Pass-through around the raw HVP callable: counts calls, enforces the wall-time limit and feeds the heartbeat. Numerically neutral."""
    def __init__(self, fn, hb, clock, twin, gpcg_hb_path):
        self.fn, self.hb, self.clock, self.twin, self.gpcg_hb = fn, hb, clock, twin, gpcg_hb_path; self.count = 0; self._t = time.time()
    def __call__(self, v):
        self.count += 1; self.clock.check()
        if time.time() - self._t > 15.0:
            self._t = time.time(); st = {"twin": self.twin, "hvp_count": self.count}
            try:
                g = json.load(open(self.gpcg_hb)); st.update({"r_inf": g.get("natural_residual_inf"), "gap_rel": g.get("relative_gap")})
            except Exception: pass
            self.hb.set(**st)
        return self.fn(v)


def inventory(*objs, top=5):
    arrs = reachable_arrays(*objs); big = sorted(arrs, key=lambda a: -a[2])[:top]
    return {"n_arrays": len(arrs), "max_array_bytes": big[0][2] if big else 0, "max_array_shape": list(big[0][1]) if big else None, "top": [{"name": a[0], "shape": list(a[1]), "bytes": a[2]} for a in big]}


def function_fidelity(f_exact, f_hat):
    f, h = np.asarray(f_exact, float), np.asarray(f_hat, float); e = h - f; sd = float(np.std(f))
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "nrmse": (float(np.sqrt(np.mean(e ** 2)) / sd) if sd >= 1e-12 else None),
            "pearson": (float(np.corrcoef(f, h)[0, 1]) if (np.var(f) > 0 and np.var(h) > 0) else None), "max_abs_deviation": float(np.max(np.abs(e))),
            "p99_abs_deviation": float(np.percentile(np.abs(e), 99)), "relative_l2": (float(np.linalg.norm(e) / np.linalg.norm(f)) if np.linalg.norm(f) > 0 else None)}


def pair_fidelity(K, Khat):
    e = np.asarray(K, float) - np.asarray(Khat, float)
    return {"rmse_pair": float(np.sqrt(np.mean(e ** 2))), "relative_l2": float(np.linalg.norm(e) / np.linalg.norm(K)), "max_abs_error": float(np.max(np.abs(e))), "p99_abs_error": float(np.percentile(np.abs(e), 99)), "mean_signed_error_K_minus_Khat": float(np.mean(e))}
