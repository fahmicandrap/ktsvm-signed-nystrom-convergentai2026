"""GPCG (gradient projection + conjugate gradient) for  min 0.5 a^T H a - 1^T a,  0 <= a <= c.

Implements g2r3_gpcg_specification.md exactly as registered (config hash recorded before any run).
Established method: J. J. Moré and G. Toraldo, SIAM J. Optim. 1(1) 93-113 (1991), DOI 10.1137/0801008.
All numerical constants and safeguards are implementation choices of this project (see the specification).
H is accessed ONLY through hvp(v) = H v, and every use is counted by one central counter.
"""
import json
import time
import numpy as np


class BudgetExhausted(Exception):
    pass


def dual_metrics(a, g, c):
    ag = float(a @ g)
    gap = ag + c * float(np.sum(np.maximum(0.0, -g)))
    D = 0.5 * (float(a.sum()) - ag)
    denom = max(1.0, abs(D))
    r = float(np.max(np.abs(a - np.clip(a - g, 0.0, c))))
    return r, gap, gap / denom, D, denom


class CountedHVP:
    def __init__(self, f, budget):
        self.f, self.budget, self.count = f, int(budget), 0
        self.cats = {}
        self.Lhat = 0.0

    def __call__(self, v, cat):
        if self.count >= self.budget:
            raise BudgetExhausted()
        self.count += 1
        self.cats[cat] = self.cats.get(cat, 0) + 1
        w = self.f(v)
        nv = float(np.linalg.norm(v))
        if nv > 0.0:
            self.Lhat = max(self.Lhat, float(np.linalg.norm(w)) / nv)
        return w


def gpcg(hvp_raw, n, c=1.0, budget=2001015, tol_r=1e-6, tol_gap=1e-6, sigma=1e-4, beta=0.5, s_first=1.0, s_max=1e6, s_min=1e-30,
         eta1=0.1, tau_cg=1e-7, cg_cap_factor=2, eps_curv=1e-13, refresh_every=1000, heartbeat_path=None, heartbeat_every=50000):
    hv = CountedHVP(hvp_raw, budget)
    a = np.zeros(n)
    g = -np.ones(n)                       # H*0 - 1, exact, no HVP
    st = {"s_last": None, "acc": 0, "next_beat": heartbeat_every}
    occ = dict(outer_iterations=0, phaseA_exit_same_face=0, phaseA_exit_stagnation=0, phaseB_exit_gradient=0, phaseB_exit_cap=0, cg_infeasible_steps=0,
               zero_curvature_descent=0, zero_curvature_nondescent=0, gradient_refreshes=0, verification_refreshes=0, verification_failures=0)
    hist = []
    t0 = time.perf_counter()

    def beat(force=False):
        if force or hv.count >= st["next_beat"]:
            r, gap, gr, D, dn = dual_metrics(a, g, c)
            rec = {"hvp": hv.count, "outer_iterations": occ["outer_iterations"], "objective_q": -D, "natural_residual_inf": r, "relative_gap": gr, "seconds": time.perf_counter() - t0}
            hist.append(rec)
            while st["next_beat"] <= hv.count:
                st["next_beat"] += heartbeat_every
            if heartbeat_path:
                with open(heartbeat_path, "w") as fh:
                    json.dump(rec, fh)

    def maybe_refresh():
        nonlocal g
        if refresh_every and st["acc"] % refresh_every == 0:
            g = hv(a, "gradient_refresh") - 1.0
            occ["gradient_refreshes"] += 1

    def converged():
        nonlocal g
        r, gap, gr, D, dn = dual_metrics(a, g, c)
        if r <= tol_r and gr <= tol_gap:
            gf = hv(a, "verification_refresh") - 1.0     # from-scratch gradient (counted)
            occ["verification_refreshes"] += 1
            g = gf
            r2, gap2, gr2, D2, dn2 = dual_metrics(a, g, c)
            if r2 <= tol_r and gr2 <= tol_gap:
                return True
            occ["verification_failures"] += 1
        return False

    def armijo_search(direction_pad, t):
        """Projected search x(t) = P(a + t p) with backtracking; returns accepted (x, Hd, t) or None on underflow."""
        while True:
            x = np.clip(a + t * direction_pad, 0.0, c)
            d = x - a
            Hd = hv(d, "projected_search")
            gd = float(g @ d)
            dq = gd + 0.5 * float(d @ Hd)
            if dq <= sigma * gd:
                return x, Hd, t
            t *= beta
            if t < s_min:
                return None

    def phase_A():
        nonlocal a, g
        prev_dec = []
        lo_prev, up_prev = (a == 0.0), (a == c)
        while True:
            if converged():
                return "converged"
            s = s_first if st["s_last"] is None else min(2.0 * st["s_last"], s_max)
            while True:
                x = np.clip(a - s * g, 0.0, c)
                d = x - a
                Hd = hv(d, "phaseA_trial")
                gd = float(g @ d)
                dq = gd + 0.5 * float(d @ Hd)
                if dq <= sigma * gd:
                    break
                s *= beta
                if s < s_min:
                    return "underflow"
            a, g = x, g + Hd
            st["s_last"] = s
            st["acc"] += 1
            maybe_refresh()
            beat()
            dec = -dq
            lo, up = (a == 0.0), (a == c)
            same = bool(np.array_equal(lo, lo_prev) and np.array_equal(up, up_prev))
            lo_prev, up_prev = lo, up
            stagn = bool(prev_dec) and dec <= eta1 * max(prev_dec)
            prev_dec.append(dec)
            if same:
                occ["phaseA_exit_same_face"] += 1
                return "same_face"
            if stagn:
                occ["phaseA_exit_stagnation"] += 1
                return "stagnation"

    def phase_B():
        nonlocal a, g
        F = np.where((a > 0.0) & (a < c))[0]
        if len(F) == 0:
            return "empty_face"
        cap = cg_cap_factor * len(F)
        r = -g[F]
        p = r.copy()
        rr = float(r @ r)
        it = 0
        while it < cap:
            if converged():
                return "converged"
            if float(np.max(np.abs(g[F]))) <= tau_cg:
                occ["phaseB_exit_gradient"] += 1
                return "face_solved"
            pp = np.zeros(n)
            pp[F] = p
            w = hv(pp, "cg")
            curv = float(p @ w[F])
            if curv <= eps_curv * hv.Lhat * float(p @ p):                       # zero / negative curvature rule
                if float(g @ pp) < 0.0:
                    ti = np.full(len(F), np.inf)
                    pos, neg = p > 0, p < 0
                    ti[pos] = (c - a[F][pos]) / p[pos]
                    ti[neg] = (0.0 - a[F][neg]) / p[neg]
                    res = armijo_search(pp, float(np.min(ti)))
                    if res is None:
                        return "underflow"
                    a, Hd, _ = res
                    g = g + Hd
                    st["acc"] += 1
                    occ["zero_curvature_descent"] += 1
                    maybe_refresh()
                    beat()
                    return "zero_curvature"
                occ["zero_curvature_nondescent"] += 1
                return "zero_curvature_nondescent"
            t = rr / curv
            x = a + t * pp
            if float(x.min()) >= 0.0 and float(x.max()) <= c:
                a = x
                g = g + t * w
                st["acc"] += 1
                maybe_refresh()
                beat()
                rn = -g[F]
                rrn = float(rn @ rn)
                p = rn + (rrn / rr) * p
                r, rr = rn, rrn
                it += 1
            else:
                occ["cg_infeasible_steps"] += 1
                res = armijo_search(pp, t)
                if res is None:
                    return "underflow"
                a, Hd, _ = res
                g = g + Hd
                st["acc"] += 1
                maybe_refresh()
                beat()
                return "infeasible_step"
        occ["phaseB_exit_cap"] += 1
        return "cap"

    status = "RUNNING"
    try:
        while True:
            occ["outer_iterations"] += 1
            ra = phase_A()
            if ra == "converged":
                status = "CONVERGED"
                break
            if ra == "underflow":
                status = "STEP_UNDERFLOW"
                break
            rb = phase_B()
            if rb == "converged":
                status = "CONVERGED"
                break
            if rb == "underflow":
                status = "STEP_UNDERFLOW"
                break
    except BudgetExhausted:
        status = "BUDGET_EXHAUSTED"
    beat(force=True)
    seconds = time.perf_counter() - t0
    maintained = dual_metrics(a, g, c)
    g_fresh = hvp_raw(a) - 1.0                      # validation-only: NOT counted in the optimization budget
    fresh = dual_metrics(a, g_fresh, c)
    return {"alpha": a, "status": status, "hvp_count": hv.count, "hvp_budget": int(budget), "hvp_by_category": dict(hv.cats), "occurrences": occ, "seconds_wall_clock": seconds,
            "maintained_gradient_metrics": {"r_inf": maintained[0], "gap": maintained[1], "gap_rel": maintained[2], "D": maintained[3]},
            "final_from_scratch_metrics_validation_only": {"r_inf": fresh[0], "gap": fresh[1], "gap_rel": fresh[2], "D": fresh[3], "denominator": fresh[4]},
            "Lhat_final": hv.Lhat, "history": hist[-20:], "history_length": len(hist)}
