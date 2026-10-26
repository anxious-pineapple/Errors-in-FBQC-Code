"""Threshold vs the XX/ZZ outcome bias δ at twice the notebook's time extent (Hexagon §14b).

The hexagon notebook's §13/§14 run at n_cycles = x_max (2·Lx timesteps). At δ = 1 that makes the
time direction the shortest way for errors to wrap (Lx/2 flips vs 2Lx/3 in space), and the δ = 1
crossings did not converge. Here every circuit has n_cycles = 2·x_max (4·Lx timesteps), so space
sets the distance at every δ.

δ biases the fusion OUTCOMES (X_aX_b at 2 (1 - δ) p, Z_aZ_b at 2 δ p), as in the notebook's §11. Only
regular fusion is run: an H-fusion code measures the same operators with records relabelled, so under
this noise model it is the same decoding problem. (CODES can still take h_fusion_kinds; the structure
then swaps checks/logicals onto the H-fused records and the noise mask follows the X_aX_b outcome into
the 2nd record of every H-fusion. An earlier run of an all-H code here used a *slot* mask by mistake;
its rows were removed.)

Everything else is the notebook's own code, executed from the .ipynb exactly as dense_delta_sweep.py
does: grids (TH_DELTAS, TH_COARSE_PS, TH_FINE_SPAN), shots, seeds, slot_split, reweight_matching,
the erasure decoder and find_crossing. The two-pass search (with dense_delta_sweep's per-pair
fallback) is unchanged.

Outputs (in bias_threshold/):
  hexagon_{meas,erasure}_threshold_vs_delta_time2x.csv          crossings per (code, δ)
  hexagon_{meas,erasure}_threshold_vs_delta_time2x_curves.jsonl  every sampled curve
Re-running skips (code, δ) rows already in the CSV.

Usage (from Code/):  python bias_threshold/long_time_delta_sweep.py [--validate]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dense_delta_sweep as D

TIME_MULT = 2
CODES = {"regular fusion (time 2x)": ()}
N_WORKERS = 4


def make_structures(ns):
    """The notebook's bias_structure / get_outcome_matching, rebuilt with n_cycles = TIME_MULT * x_max
    and a choice of code. Built per (x_max, code), cached until clear() is called."""
    cache = {}

    def structure(x_max, code):
        key = (x_max, code)
        if key in cache:
            return cache[key]
        t0 = time.time()
        kinds = set(CODES[code])
        net = ns["HexLattice"](x_max, x_max)
        st = ns["Spacetime"](net, n_cycles=TIME_MULT * x_max)
        circ, rec, nm = ns["build"](st, 0.0, h_fusion_kinds=kinds)
        ch, _ = ns["hexagon_check_rows"](st, rec, nm)
        lg = ns["closed_form_logicals"](net, st, rec, nm)
        if kinds:
            perm = ns["xx_slot_swap"](rec, nm, h_fusion_kinds=kinds)
            ch, lg = ch[:, perm], lg[:, perm]
        # every check and logical must be deterministic in the noiseless circuit. With the reference sample
        # skipped (as everywhere in the notebook; computing it is a full tableau simulation, hours at
        # Lx = 24) a deterministic parity reads as a constant 0 and a non-deterministic one as random.
        S = circ.compile_sampler(seed=99, skip_reference_sample=True).sample(64).astype(np.uint8)
        assert not ((S @ ch.T) % 2).any(), f"{code} Lx={3 * x_max}: a check is not deterministic"
        assert not ((S @ lg.T) % 2).any(), f"{code} Lx={3 * x_max}: a logical is not deterministic"
        matcher, outcome_edge, _ = ns["build_outcome_matching"](st, ch, lg, 0.010, h_fusion_kinds=kinds)
        m_er, oe_er, w_er = ns["build_outcome_matching"](st, ch, lg, ns["p_meas_eps_h"], h_fusion_kinds=kinds)
        is_xx = np.zeros(nm, dtype=bool)                                   # where each X_aX_b outcome is:
        is_xx[[izz if kind in kinds else ixx                               # 2nd record of an H-fusion,
               for (_, _, kind, _, _), (ixx, izz) in rec.items()]] = True  # 1st of a plain fusion
        raw = circ.compile_sampler(seed=ns["SEED_BIAS"], skip_reference_sample=True).sample(
            ns["SHOTS_BIAS_MEAS"]).astype(np.uint8)
        cache[key] = dict(
            meas=dict(checks=ch, logs=lg, nmeas=nm, matcher=matcher, outcome_edge=outcome_edge, is_xx=is_xx, raw=raw),
            erasure=dict(checks=ch, logs=lg, matcher=m_er, outcome_edge=oe_er, weight=w_er))
        print(f"  [{code} Lx={3 * x_max} n_cycles={TIME_MULT * x_max}: outcomes={nm}, checks and logicals "
              f"deterministic, {matcher.num_edges} edges ({time.time() - t0:.0f}s)]", flush=True)
        return cache[key]

    def meas_sweep(x_max, code, delta, ps):
        """§13's bias_meas_p_sweep on the long-time structure."""
        S = structure(x_max, code)["meas"]
        rates = []
        for p in ps:
            q_xx, q_zz = ns["slot_split"](p, delta)
            rng = np.random.default_rng((ns["SEED_BIAS"], x_max, int(round(p * 1e6)), int(round(delta * 1e4)), 7))
            q = np.where(S["is_xx"], q_xx, q_zz).astype(np.float32)
            outcomes = S["raw"].copy()
            for s0 in range(0, outcomes.shape[0], 2000):
                blk = outcomes[s0:s0 + 2000]
                blk ^= (rng.random(blk.shape, dtype=np.float32) < q).astype(np.uint8)
            ns["reweight_matching"](S, q_xx, q_zz, 0.0)
            rates.append(ns["_decode_rate"](S, outcomes, N_WORKERS))
        return np.array(rates)

    def erasure_sweep(x_max, code, delta, ps):
        """§14's bias_erasure_p_sweep on the long-time structure."""
        B = structure(x_max, code)
        S, raw = B["erasure"], B["meas"]["raw"][:ns["shots_by_L_h"][3 * x_max]]
        rates = []
        for p in ps:
            rng = np.random.default_rng((ns["SEED_BIAS"], x_max, int(round(p * 1e6)), int(round(delta * 1e4)), 11))
            q = np.where(B["meas"]["is_xx"], *ns["slot_split"](p, delta)).astype(np.float32)
            erased = rng.random(raw.shape, dtype=np.float32) < q
            outcomes = np.where(erased, rng.integers(0, 2, size=raw.shape, dtype=np.uint8), raw)
            dets = ns["rows_to_parities_packed"](outcomes, S["checks"])
            obs = ns["rows_to_parities_packed"](outcomes, S["logs"])
            pred = ns["decode_with_erasure_parallel"](dets, S["matcher"], S["outcome_edge"], S["weight"],
                                                      len(S["logs"]), erased, n_workers=N_WORKERS)
            rates.append(np.any(pred != obs, axis=1).sum() / raw.shape[0])
        return np.array(rates)

    def clear():
        """Free the cached structures. The notebook's rows_to_parities_packed and reweight_matching cache
        by id(), and freed arrays' ids get reused, so their caches must be cleared with them."""
        cache.clear()
        ns["rows_to_parities_packed"].__defaults__[0].clear()
        ns["_reweight_state"].clear()

    return structure, meas_sweep, erasure_sweep, clear


def fast_point_colour_edge(ns):
    """Same result as the notebook's point_colour_edge (the tuple of the unique `colour` edge at p), from a
    lookup table built once per lattice instead of a scan over every edge on every call."""
    tables = {}

    def point_colour_edge(net, p, colour):
        if id(net) not in tables:
            t = {}
            for edge, c in net.edge_colour.items():
                for q in edge:
                    assert (q, c) not in t
                    t[(q, c)] = tuple(edge)
            tables[id(net)] = (net, t)          # keep net alive so its id can't be reused
        return tables[id(net)][1][(p, colour)]

    return point_colour_edge


def validate(ns):
    """Every code in CODES builds and verifies at Lx = 12, and gives sensible rates at one point."""
    structure, meas_sweep, erasure_sweep, _ = make_structures(ns)
    for code in CODES:
        for d in (0.0, 1.0):
            a = meas_sweep(4, code, d, [0.012])[0]
            print(f"  {code}: meas p=1.2%, δ={d:.2f} -> {a * 100:.2f}%", flush=True)


def main():
    os.chdir(CODE_DIR)
    ns = D.load_notebook("hexagon")
    slow = ns["point_colour_edge"]
    ns["point_colour_edge"] = fast_point_colour_edge(ns)
    if "--check-fast-edge" in sys.argv:          # the lookup must give exactly the notebook's edges/logicals
        for x in (2, 4):
            net = ns["HexLattice"](x, x)
            assert all(slow(net, p, c) == ns["point_colour_edge"](net, p, c)
                       for p in net.points for c in ("red", "green", "blue"))
            st = ns["Spacetime"](net, n_cycles=TIME_MULT * x)
            _, rec, nm = ns["build"](st)
            fast_lg = ns["closed_form_logicals"](net, st, rec, nm)
            ns["point_colour_edge"] = slow
            slow_lg = ns["closed_form_logicals"](net, st, rec, nm)
            ns["point_colour_edge"] = fast_point_colour_edge(ns)
            assert np.array_equal(fast_lg, slow_lg)
            print(f"  fast point_colour_edge: identical edges and logicals at Lx={3 * x}", flush=True)
        return
    if "--validate" in sys.argv:
        validate(ns)
        return
    structure, meas_sweep, erasure_sweep, clear = make_structures(ns)
    sweeps = {"meas": meas_sweep, "erasure": erasure_sweep}
    t_all = time.time()
    for code in CODES:
        for noise in ("meas", "erasure"):
            dst = os.path.join(HERE, f"hexagon_{noise}_threshold_vs_delta_time2x.csv")
            logp = os.path.join(HERE, f"hexagon_{noise}_threshold_vs_delta_time2x_curves.jsonl")
            allrows = {}
            if os.path.exists(dst):
                for c in CODES:
                    allrows[c] = D.read_csv(dst, c)
            rows = allrows.setdefault(code, {})
            sweep = lambda x, d, ps, _s=sweeps[noise], _c=code: _s(x, _c, d, ps)
            for d in ns["TH_DELTAS"][noise]:
                d = float(d)
                if d in rows:
                    continue
                t0, log = time.time(), []
                res = D.two_pass(sweep, ns["TH_COARSE_PS"][noise], ns["TH_FINE_SPAN"][noise], ns["find_crossing"],
                                 d, log=log)
                r = rows.setdefault(d, dict(circuit=code, delta=d))
                for k, col in enumerate(("12x18", "18x24")):
                    x, how = res.get(k, (None, "no crossing"))
                    r[f"cross_{col}"] = "" if x is None else f"{x:.6g}"
                    r[f"source_{col}"] = how
                write_all(dst, allrows)
                with open(logp, "a") as f:
                    for entry in log:
                        f.write(json.dumps(dict(entry, circuit=code)) + "\n")
                fmt = lambda c: f"{float(r[c]) * 100:.3f}%" if r.get(c) else "--"
                print(f"[{code}] {noise} δ={d:.4f}: 12x18 {fmt('cross_12x18')} ({r['source_12x18']})  "
                      f"18x24 {fmt('cross_18x24')} ({r['source_18x24']})  ({time.time() - t0:.0f}s, "
                      f"total {(time.time() - t_all) / 3600:.2f} h)", flush=True)
        clear()             # free this code's samples (Lx = 24 is ~0.74 GB) before the next code
    print("done", flush=True)


def write_all(path, allrows):
    cols = ["circuit", "delta", "cross_12x18", "cross_18x24", "source_12x18", "source_18x24"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for code in CODES:
            for d in sorted(allrows.get(code, {})):
                f.write(",".join(str(allrows[code][d].get(c, "")) for c in cols) + "\n")


if __name__ == "__main__":
    main()
