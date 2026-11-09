#!/usr/bin/env python3
"""Executes ONE cell attempt (one process). Usage: python cell_worker.py <spec.json>. In Phase 9B this file is tested only on toy data and the archived N=100
validation data; no main-study cell is executed. Methods: EXACT, NYS (signed Nystrom, matrix-free SPD) and TOY_* (orchestration tests only)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time, traceback, hashlib, signal, io, zipfile, platform
T_PROC0 = time.perf_counter()
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"; ROOT = HERE + "../"
for _p in (ROOT + "g1", ROOT + "g2", ROOT + "g2r", ROOT + "g2r3", ROOT + "g3r1", ROOT + "g3r2", HERE):
    sys.path.insert(0, _p)
import g2r_common as G  # noqa: E402
import exact_ktsvm as ek  # noqa: E402
import preprocessing as pp  # noqa: E402
from gpcg_solver import gpcg, dual_metrics  # noqa: E402
from signed_nystrom import signed_nystrom, phi_new  # noqa: E402
from nystrom_ktsvm_matrixfree_spd import TwinOperatorSPD, CholeskyFailure  # noqa: E402
import measurement as M  # noqa: E402
import resource_preflight as RP  # noqa: E402

sha_file = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
sha_arr = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
PRED = G.NUM + ["proto", "service", "state"]


class InfraTermination(Exception): pass
class EnvFail(Exception): pass


class Ctx:
    pass


def verify_implementation():
    man = json.load(open(ROOT + "phase9/phase9_validated_implementation_manifest.json"))
    bad = [k for k, v in man["implementation_files"].items() if sha_file(ROOT + v["path"]) != v["sha256"]]
    return bad


def load_frames(spec):
    P = spec["paths"]
    if spec["mode"] == "main":
        return pp.read_partition(P["train_csv"]), pp.read_partition(P["test_csv"])
    z = zipfile.ZipFile(P["zip"]); rd = lambda n: pd.read_csv(io.BytesIO(z.read(n)), keep_default_na=False)
    return rd(P["train_member"]), rd(P["test_member"])


def warmup():
    """Identical unrecorded warm-up in every process: both numerical paths on the archived N=100 validation data (never main-study ids)."""
    d = G.load_subset(100); X, t, K = d["X"], d["t"], d["K"]
    P = G.build_problem(K, t, 1); H = P["H"]; gpcg(lambda v: H @ v, H.shape[0], budget=300)
    L = np.arange(0, 100, 4)[:25]; ny = signed_nystrom(ek.sigmoid_gram(X, X[L]), L); op = TwinOperatorSPD(ny["Phi"], ny["J"], t, 1); gpcg(op.hvp, op.n, budget=300)


def solve_twin(ctx, name, hvp_raw, n):
    hbp = os.path.join(ctx.out, f"gpcg_heartbeat_{name}.json"); wr = M.HVPWrapper(hvp_raw, ctx.hb, ctx.clock, name, hbp); ctx.wrapper = wr; ctx.current_twin = name
    ctx.hb.set(twin=name, hvp_count=0, r_inf=None, gap_rel=None)
    with ctx.rec.phase(f"T_solve_{name}"):
        res = gpcg(wr, n, c=G.C, budget=ctx.budget, tol_r=1e-6, tol_gap=1e-6, heartbeat_path=hbp, heartbeat_every=ctx.gpcg_hb_every)
    return res


def twin_record(res, val, n, extra=None):
    occ = res["occurrences"]
    rec = {"dual_dimension": int(n), "status": res["status"], "hvp_count": res["hvp_count"], "hvp_budget": res["hvp_budget"], "hvp_by_category": res["hvp_by_category"], "r_inf": val["r_inf"], "gap": val["gap"], "gap_rel": val["gap_rel"],
           "dual_objective_D": val["D"], "bound_violation": val["viol"], "cg_cap_events": occ["phaseB_exit_cap"], "cg_infeasible_steps": occ["cg_infeasible_steps"], "zero_curvature_events": {"descent": occ["zero_curvature_descent"], "nondescent": occ["zero_curvature_nondescent"]},
           "refresh_events": {"gradient": occ["gradient_refreshes"], "verification": occ["verification_refreshes"], "verification_failures": occ["verification_failures"]}, "phase_counts": occ,
           "solver_seconds_reported": res["seconds_wall_clock"], "maintained_gradient_metrics": res["maintained_gradient_metrics"]}
    rec.update(extra or {}); return rec


def validate_from_scratch(hvp, a):
    g = hvp(a) - 1.0; r, gap, gr, D, _ = dual_metrics(a, g, G.C)
    return {"r_inf": r, "gap": gap, "gap_rel": gr, "D": D, "viol": float(np.max(np.maximum(0.0, np.maximum(-a, a - G.C))))}


def save_out(ctx, name, arr):
    p = os.path.join(ctx.out, name); np.save(p, np.asarray(arr)); ctx.outputs[name] = {"sha256_content": sha_arr(np.asarray(arr)), "n": int(len(arr))}


def run_exact(ctx, X, Xp, t):
    with ctx.rec.phase("T_kernel"):
        K = ek.sigmoid_gram(X, X)
    ctx.major["after_kernel"] = M.inventory(X, K); ctx.clock.check()
    for ell, name in ((1, "plus"), (-1, "minus")):
        with ctx.rec.phase(f"T_buildH_{name}"):
            P = G.build_problem(K, t, ell)
        H = P["H"]; n = H.shape[0]; ctx.major[f"after_buildH_{name}"] = M.inventory(X, K, P)
        res = solve_twin(ctx, name, lambda v, H=H: H @ v, n); a = res["alpha"]
        val = validate_from_scratch(lambda v, H=H: H @ v, a); cert = G.certificates(P, a); rho = cert["rho"]
        ctx.twins[name] = twin_record(res, val, n, {"certified_gap_closed_form": cert["duality_gap_closed_form"]})
        with ctx.rec.phase(f"T_probe_{name}"):
            f = np.empty(len(Xp))
            for s in range(0, len(Xp), 500):
                f[s:s + 500] = ek.sigmoid_gram(Xp[s:s + 500], X) @ rho[:-1] + rho[-1]
        save_out(ctx, f"probe_f_{name}.npy", f); ctx.nonfinite += int((~np.isfinite(f)).sum() + (~np.isfinite(a)).sum())
        ctx.major[f"after_solve_{name}"] = M.inventory(X, K, P, res); del P, H; ctx.clock.check()


def run_nystrom(ctx, X, Xp, t, sample_df, sample_ids):
    P = ctx.spec["paths"]; m = int(ctx.spec["m"]); lm = np.load(P["landmarks"])[:m]; assert len(lm) == m and m <= len(X)
    Lpos = np.searchsorted(sample_ids, lm); assert (sample_ids[Lpos] == lm).all(), "landmark ids must belong to the sample"
    with ctx.rec.phase("T_kernel"):
        C = ek.sigmoid_gram(X, X[Lpos])
    ctx.major["after_kernel"] = M.inventory(X, C); ctx.clock.check()
    with ctx.rec.phase("T_factor"):
        ny = signed_nystrom(C, Lpos)
    del C; ctx.major["after_factor"] = M.inventory(X, ny)
    lam = ny["eigenvalues_W"][ny["retained"]]; lp = sample_df.iloc[Lpos][PRED]
    ctx.spectrum = {"m": ny["m"], "retained_rank_r": ny["r"], "n_positive_W": ny["n_plus"], "n_negative_W": ny["n_minus"], "n_zero_W": ny["n_zero"], "tau_W": ny["tau_W"], "min_retained_abs_lambda": float(np.min(np.abs(lam))),
                    "max_retained_abs_lambda": float(np.max(np.abs(lam))), "unique_landmark_predictor_vectors": int(len(lp.drop_duplicates())), "duplicate_landmark_predictor_vectors": int(ny["m"] - len(lp.drop_duplicates())), "Phi_shape": list(ny["Phi"].shape)}
    with ctx.rec.phase("T_probe_setup"):
        phip = phi_new(ek.sigmoid_gram(Xp, X[Lpos]), ny)
    for ell, name in ((1, "plus"), (-1, "minus")):
        with ctx.rec.phase(f"T_operator_{name}"):
            op = TwinOperatorSPD(ny["Phi"], ny["J"], t, ell)
        ctx.major[f"after_operator_{name}"] = M.inventory(X, ny, op); n = op.n
        res = solve_twin(ctx, name, op.hvp, n); a = res["alpha"]; val = validate_from_scratch(op.hvp, a); ub = op.reconstruct(a)
        with ctx.rec.phase(f"T_diagnostics_{name}"):
            dg = op.diagnostics()
        ctx.twins[name] = twin_record(res, val, n, {"cond_B": dg["cond_B"], "cond_A": dg["cond_A"], "lambda_min_A": dg["lambda_min_A"], "Q_asymmetry_relative_assembled_diagnostic": dg["Q_asymmetry_relative"], "reduced_dimension": int(ny["r"] + 1), "finite_reduced_parameters": bool(np.isfinite(ub).all())})
        with ctx.rec.phase(f"T_probe_{name}"):
            f = op.outputs_new(phip, ny["J"], ub)
        save_out(ctx, f"probe_f_{name}.npy", f); ctx.nonfinite += int((~np.isfinite(f)).sum() + (~np.isfinite(a)).sum() + (~np.isfinite(ub)).sum())
        ctx.major[f"after_solve_{name}"] = M.inventory(X, ny, op, res); del op; ctx.clock.check()
    pairs = np.load(P["pairs"]); Kp = np.empty(len(pairs)); Kh = np.empty(len(pairs)); Phi, J = ny["Phi"], ny["J"]
    with ctx.rec.phase("T_pairs"):
        for s in range(0, len(pairs), 5000):
            i, k = pairs[s:s + 5000, 0], pairs[s:s + 5000, 1]
            Kp[s:s + 5000] = np.tanh(ek.GAMMA * np.einsum("ij,ij->i", X[i], X[k]) + ek.THETA); Kh[s:s + 5000] = (Phi[i] * J * Phi[k]).sum(1)
    ctx.pair = M.pair_fidelity(Kp, Kh); ctx.pair["n_pairs"] = int(len(pairs)); ctx.pair["sha256_content_positions"] = sha_arr(pairs)


def toy_problem(n, sleep_s):
    rng = np.random.default_rng(3); A = rng.standard_normal((n, 5)); H = A @ A.T + 1e-3 * np.eye(n)
    def hv(v):
        if sleep_s: time.sleep(sleep_s)
        return H @ v
    return hv


def run_toy(ctx):
    s = ctx.spec; meth = s["method"]
    if s.get("toy_ignore_sigterm"): signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if meth == "TOY_SLEEP":
        with ctx.rec.phase("T_kernel"): time.sleep(s["toy_sleep_s"])
    elif meth == "TOY_ERROR":
        raise RuntimeError("toy software error")
    elif meth == "TOY_CRASH":
        if s["attempt"] <= s.get("toy_crash_attempts", 1): os.kill(os.getpid(), signal.SIGKILL)
    for name in ("plus", "minus"):
        n = s.get("toy_n", 40); res = solve_twin(ctx, name, toy_problem(n, s.get("toy_hvp_sleep_s", 0.0)), n); val = validate_from_scratch(toy_problem(n, 0.0), res["alpha"]); ctx.twins[name] = twin_record(res, val, n)


def decide_status(ctx):
    sts = [ctx.twins[k]["status"] for k in ("plus", "minus") if ctx.twins.get(k)]
    if ctx.nonfinite: return "NUMERICAL_FAILURE", "NaN/Inf in solver or output arrays"
    if any(x == "STEP_UNDERFLOW" for x in sts): return "NUMERICAL_FAILURE", "GPCG STEP_UNDERFLOW"
    if any(x == "BUDGET_EXHAUSTED" for x in sts): return "HVP_BUDGET_EXCEEDED", None
    return ("CONVERGED", None) if len(sts) == 2 and all(x == "CONVERGED" for x in sts) else ("SOFTWARE_ERROR", "twin statuses incomplete")


def assemble(ctx, status, detail, tb):
    T = {k: v["wall_s"] for k, v in ctx.rec.timings.items()}; C = {k: v["cpu_s"] for k, v in ctx.rec.timings.items()}
    def total(names): return sum(T[n] for n in names) if all(n in T for n in names) else None
    ex = ctx.spec["method"] == "EXACT"; op_names = ["T_buildH_plus", "T_buildH_minus"] if ex else ["T_operator_plus", "T_operator_minus"]
    train_names = ["T_kernel"] + ([] if ex else ["T_factor"]) + op_names + ["T_solve_plus", "T_solve_minus"]
    peaks = [v["peak_rss_bytes"] for v in ctx.rec.memory.values() if v["peak_rss_bytes"]]
    timing = {"T_load": T.get("T_load"), "T_prep": T.get("T_prep"), "T_kernel": T.get("T_kernel"), "T_factor": T.get("T_factor"), "T_operator_or_buildH": total(op_names), "T_operator_or_buildH_plus": T.get(op_names[0]), "T_operator_or_buildH_minus": T.get(op_names[1]),
              "T_solve_plus": T.get("T_solve_plus"), "T_solve_minus": T.get("T_solve_minus"), "T_train": total(train_names) if ctx.spec["method"] in ("EXACT", "NYS") else None,
              "T_probe": sum(v for k, v in T.items() if k.startswith("T_probe")) if any(k.startswith("T_probe") for k in T) else None, "T_pairs": T.get("T_pairs"), "T_diagnostics": sum(v for k, v in T.items() if k.startswith("T_diagnostics")) or None,
              "T_total_process": time.perf_counter() - T_PROC0, "cpu_seconds_by_phase": C, "wall_seconds_by_phase": T, "T_warmup_excluded": T.get("WARMUP")}
    return {"schema_version": "phase9b-1", "cell": {"cell_id": ctx.spec["cell_id"], "tag": ctx.spec["tag"], "method": ctx.spec["method"], "N": ctx.spec["N"], "m": ctx.spec.get("m"), "seed": ctx.spec["seed"], "mode": ctx.spec["mode"], "attempt": ctx.spec["attempt"]},
            "status": status, "status_detail": detail, "started_utc": ctx.started, "finished_utc": M.utcnow(),
            "environment": {"thread_policy_ok": ctx.thread.get("ok"), "threads": {"env": ctx.thread.get("env"), "pools": ctx.thread.get("pools")}, "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
            "hashes": ctx.hashes, "timing": timing, "memory": {"peak_rss_bytes_over_phases": max(peaks) if peaks else None, "ru_maxrss_bytes": int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss) * 1024, "baseline_rss_bytes": ctx.baseline_rss,
                                                                 "phases": ctx.rec.memory, "major_arrays": ctx.major}, "twins": {"plus": ctx.twins.get("plus"), "minus": ctx.twins.get("minus")}, "spectrum": ctx.spectrum, "kernel_pair_fidelity": ctx.pair, "outputs": ctx.outputs,
            "final_checkpoint": ctx.hb.last if status == "NOT_COMPLETED_WALL_TIME_LIMIT" else None, "partial_solver_state": ({"twin": ctx.current_twin, "hvp_count": ctx.wrapper.count} if (ctx.wrapper is not None and status == "NOT_COMPLETED_WALL_TIME_LIMIT") else None), "traceback": tb}


def main():
    spec = json.load(open(sys.argv[1])); out = spec["out_dir"]; os.makedirs(out, exist_ok=True)
    clock = M.WallClock(spec["start_epoch"], spec["time_limit_s"]); hb = M.Heartbeat(os.path.join(out, "heartbeat.json"), spec["cell_id"], clock, spec.get("heartbeat_interval_s", 30.0)); hb.start()
    ctx = Ctx(); ctx.spec, ctx.out, ctx.clock, ctx.hb, ctx.rec = spec, out, clock, hb, M.Recorder(hb); ctx.twins, ctx.outputs, ctx.major, ctx.hashes = {}, {}, {}, {}; ctx.spectrum = ctx.pair = ctx.wrapper = None; ctx.current_twin = None; ctx.nonfinite = 0
    ctx.budget = spec.get("budget", 2001015); ctx.gpcg_hb_every = spec.get("gpcg_heartbeat_every", 2000); ctx.started = M.utcnow(); ctx.baseline_rss = M.rss_bytes(); ctx.thread = {}
    def _term(sig, frm):
        reason = open(os.path.join(out, "STOP_REASON")).read().strip() if os.path.exists(os.path.join(out, "STOP_REASON")) else "EXTERNAL"
        raise M.WallTimeLimit("terminated at wall-time limit") if reason == "WALL_TIME" else InfraTermination("external termination signal")
    signal.signal(signal.SIGTERM, _term)
    status, detail, tb = "SOFTWARE_ERROR", None, None
    try:
        info = (lambda: spec["test_fake_threadpools"]) if spec.get("test_fake_threadpools") else None
        ctx.thread = RP.check_thread_policy(info)
        if not ctx.thread["ok"]: raise EnvFail(str(ctx.thread))
        bad = verify_implementation()
        ctx.hashes["implementation_files_match_manifest"] = not bad
        if bad: raise RuntimeError(f"IMPLEMENTATION_HASH_MISMATCH: {bad}")
        with ctx.rec.phase("WARMUP"):
            warmup()
        clock.check()
        meth = spec["method"]
        if meth in ("EXACT", "NYS"):
            with ctx.rec.phase("T_load"):
                train_df, test_df = load_frames(spec)
            P = spec["paths"]
            with ctx.rec.phase("T_prep"):
                ids = np.load(P["sample"]); pids = np.array(json.load(open(P["probe_ids"]))["ids"], dtype=np.int64)
                sample_df = train_df[train_df["id"].isin(ids)].sort_values("id").reset_index(drop=True); assert (sample_df["id"].to_numpy() == ids).all(), "sample ids not found in partition"
                probe_df = test_df[test_df["id"].isin(pids)].sort_values("id").reset_index(drop=True); assert len(probe_df) == len(pids)
                del train_df, test_df
                pre = pp.PreprocessorV2(G.NUM, G.VOCAB).fit_scaler(sample_df); X = pre.transform(sample_df); Xp = pre.transform(probe_df); t = np.where(sample_df["label"].to_numpy() == 1, 1, -1)
            ctx.hashes.update({"X_sha256": sha_arr(X), "probe_X_sha256": sha_arr(Xp), "sample_ids_content_sha256": sha_arr(ids), "probe_ids_content_sha256": sha_arr(pids), "scaler_params_sha256": sha_arr(np.concatenate([pre.scaler.mean_, pre.scaler.scale_])),
                               "n_sample": int(len(ids)), "n_probe": int(len(pids))})
            ctx.major["after_prep"] = M.inventory(X, Xp); clock.check()
            run_exact(ctx, X, Xp, t) if meth == "EXACT" else run_nystrom(ctx, X, Xp, t, sample_df, ids)
        else:
            run_toy(ctx)
        status, detail = decide_status(ctx)
    except M.WallTimeLimit as e:
        status, detail = "NOT_COMPLETED_WALL_TIME_LIMIT", "NOT COMPLETED WITHIN PREDECLARED WALL-TIME LIMIT"
    except InfraTermination as e:
        status, detail = "INFRASTRUCTURE_FAILURE", str(e)
    except EnvFail as e:
        status, detail = "ENVIRONMENT_THREAD_POLICY_FAIL", str(e)
    except CholeskyFailure as e:
        status, detail = "NUMERICAL_FAILURE", str(e)
    except MemoryError as e:
        status, detail = "INFRASTRUCTURE_FAILURE", "MemoryError (resource exhaustion)"
    except Exception as e:
        status, detail, tb = "SOFTWARE_ERROR", f"{type(e).__name__}: {e}", traceback.format_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); hb.stop()
    res = assemble(ctx, status, detail, tb); tmp = os.path.join(out, "result.json.tmp"); json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, os.path.join(out, "result.json"))
    print(status)


if __name__ == "__main__":
    main()
