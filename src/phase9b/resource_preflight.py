"""Resource preflight (byte-level analytical peak model of Phase 9A) and thread-policy check. No model is run here."""
import os, shutil, time, math
from sampling import largest_remainder

BASELINE_RSS_BYTES = 168841216          # Phase 9A audit: RSS after imports + official training-CSV load (bytes)
FRACTION = 0.60
MIN_FREE_STORAGE_BYTES = 2_000_000_000
RETRY_WAIT_S = 600
TRAIN_COUNTS = {1: 119341, 0: 56000}
D_FEAT = 194
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _sizes(N):
    a = largest_remainder(N, TRAIN_COUNTS); return a[1], a[0]        # (N label 1, N label 0)


def exact_peak_arrays(N, d=D_FEAT):
    n1, n0 = _sizes(N); n1p = N + 1; best = 0
    for ell in (1, -1):
        Nl = n1 if ell == 1 else n0; No = N - Nl
        K = 8 * N * N; Se = 8 * n1p * Nl; So = 8 * n1p * No; M = 8 * n1p * n1p; Ylu = 8 * n1p * No; Y = 8 * n1p * No; L = 8 * n1p * n1p; base_live = K + Se + So + M + Ylu + Y
        st = [K + Se + So + 8 * N * max(Nl, No), K + Se + So + 3 * M, K + Se + So + M + L + 16 * n1p * No, K + Se + So + M + Ylu + L + 16 * n1p * No,
              base_live + 16 * n1p * n1p + 48 * n1p * No, base_live + 16 * n1p * n1p + 16 * n1p * No + 8 * n1p * No + L + 16 * n1p * No, base_live + 6 * 8 * No * No]
        best = max(best, max(st))
    return best + 8 * N * d + 3 * 8 * 1000 * N


def nystrom_peak_arrays(N, m, d=D_FEAT, probe=5000):
    n1, n0 = _sizes(N); r = m; best = 0
    for ell in (1, -1):
        Nl = n1 if ell == 1 else n0; No = N - Nl; X = 8 * N * d; Phi = 8 * N * r; C = 8 * N * m
        st = [X + 3 * C, X + C + Phi + 3 * 8 * (m * m + m * r), X + Phi + 3 * 8 * Nl * r, X + Phi + 8 * (r + 1) * Nl + 3 * 8 * No * r, X + Phi + 8 * (r + 1) * (Nl + No) + 8 * (r + 1) * Nl + 6 * 8 * (r + 1) ** 2]
        best = max(best, max(st))
    return best + 8 * probe * (m + r + d)


def estimated_peak_bytes(method, N, m=None):
    arrays = exact_peak_arrays(N) if method == "EXACT" else nystrom_peak_arrays(N, m)
    return float(BASELINE_RSS_BYTES + arrays)


def required_mem_available(method, N, m=None):
    return estimated_peak_bytes(method, N, m) / FRACTION


def read_mem_available():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def read_swap_total():
    for l in open("/proc/meminfo"):
        if l.startswith("SwapTotal"):
            return int(l.split()[1]) * 1024
    return None


def free_bytes(path):
    return shutil.disk_usage(path).free


def check_once(est, mem_available_fn, disk_fn, work_dirs):
    mem = mem_available_fn(); disks = {d: disk_fn(d) for d in work_dirs}
    ok_mem = est <= FRACTION * mem; ok_disk = min(disks.values()) >= MIN_FREE_STORAGE_BYTES
    return {"mem_available_bytes": mem, "estimated_peak_bytes": est, "limit_bytes": FRACTION * mem, "memory_ok": bool(ok_mem), "free_storage_bytes": disks, "storage_ok": bool(ok_disk), "pass": bool(ok_mem and ok_disk)}


def preflight(method, N, m=None, mem_available_fn=read_mem_available, disk_fn=free_bytes, work_dirs=("/home/claude", "/tmp"), sleep_fn=time.sleep, retry_wait_s=RETRY_WAIT_S):
    """Predeclared rule: est <= 0.60 x CURRENT MemAvailable and >= 2 GB free working storage. On failure wait 10 minutes, retry ONCE, else DEFERRED_RESOURCE."""
    est = estimated_peak_bytes(method, N, m); attempts = [check_once(est, mem_available_fn, disk_fn, work_dirs)]
    if attempts[0]["pass"]:
        return {"status": "PASS", "attempts": attempts}
    sleep_fn(retry_wait_s); attempts.append(check_once(est, mem_available_fn, disk_fn, work_dirs))
    return {"status": "PASS" if attempts[-1]["pass"] else "DEFERRED_RESOURCE", "attempts": attempts, "status_text": None if attempts[-1]["pass"] else "DEFERRED — PREDECLARED RESOURCE PREFLIGHT FAILED"}


def check_thread_policy(info_fn=None):
    """Env vars must equal '1' and every numerical threadpool must report <= 1 thread; otherwise ENVIRONMENT_THREAD_POLICY_FAIL."""
    if info_fn is None:
        from threadpoolctl import threadpool_info as info_fn
    env = {k: os.environ.get(k) for k in THREAD_ENV}; pools = [{"api": p.get("internal_api"), "num_threads": p.get("num_threads")} for p in info_fn()]
    ok = all(v == "1" for v in env.values()) and all((p["num_threads"] or 1) <= 1 for p in pools)
    return {"ok": bool(ok), "env": env, "pools": pools, "status": None if ok else "ENVIRONMENT_THREAD_POLICY_FAIL"}
