"""Step 2: synthetic design dataset from the electrothermal model.

Latin hypercube sampling over the heater design space, evaluated with
physics.evaluate (report eqs. 18-24). SYNTHETIC data, not measurements.

Usage:  .venv/bin/python src/generate.py [--n 5000] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from physics import BASELINE, T_ACT, evaluate, geometry_valid

ROOT = Path(__file__).resolve().parents[1]

# name: (low, high, SI scale). Columns in the CSV are in the display units.
RANGES = {
    "d_w_mm": (0.10, 0.40, 1e-3),
    "p_mm":   (2.0, 6.0, 1e-3),
    "a_mm":   (10.0, 25.0, 1e-3),
    "b_mm":   (20.0, 30.0, 1e-3),   # sampled so the zigzag (a) can fit
    "t_mm":   (1.0, 3.0, 1e-3),
    "V":      (3.0, 12.0, 1.0),
    "h":      (5.0, 15.0, 1.0),
}
L_H = BASELINE["l_h"]  # active hinge length held at 30 mm (Table 5.5)

# Assumed ceiling for the epoxy next to the wire. This is a project assumption,
# not a datasheet value: replace it with the TGA/DSC limit of the chosen resin.
T_WIRE_MAX_C = 200.0


def sample(n_target: int, seed: int) -> pd.DataFrame:
    """LHS, discarding geometrically invalid designs until n_target remain.

    Rejection only removes the a > b - 2 mm corner; the other constraints
    always hold over these ranges. The kept set is close to, but no longer
    exactly, a Latin hypercube.
    """
    rng_seed = seed
    keep = []
    n_kept = 0
    while n_kept < n_target:
        sampler = qmc.LatinHypercube(d=len(RANGES), seed=rng_seed)
        u = sampler.random(int(n_target * 1.6))
        lo = np.array([r[0] for r in RANGES.values()])
        hi = np.array([r[1] for r in RANGES.values()])
        df = pd.DataFrame(qmc.scale(u, lo, hi), columns=list(RANGES))
        si = {k: df[k].to_numpy() * RANGES[k][2] for k in RANGES}
        ok = geometry_valid(si["d_w_mm"], si["p_mm"], si["a_mm"], si["t_mm"], si["b_mm"])
        keep.append(df[ok])
        n_kept += int(ok.sum())
        rng_seed += 1
    return pd.concat(keep, ignore_index=True).iloc[:n_target]


def label(df: pd.DataFrame) -> pd.DataFrame:
    si = {k: df[k].to_numpy() * RANGES[k][2] for k in RANGES}
    args = dict(d_w=si["d_w_mm"], p=si["p_mm"], a=si["a_mm"], t=si["t_mm"],
                b=si["b_mm"], l_h=L_H, V=si["V"], h=si["h"])
    lab = evaluate(**args, environment="lab")
    orb = evaluate(**args, environment="orbit")
    out = df.copy()
    out["L_w_mm"] = lab["L_w"] * 1e3
    out["R_w_ohm"] = lab["R_w"]
    out["I_A"] = lab["I"]
    out["P_W"] = lab["P"]
    out["tau_s"] = lab["tau"]
    out["T_ss_C"] = lab["T_ss"]
    out["reachable"] = lab["reachable"]
    out["t_act_s"] = lab["t_act"]
    out["t_cool_s"] = lab["t_cool"]
    out["E_cyc_J"] = lab["E_cyc"]
    out["Bi"] = lab["Bi"]
    out["Fo"] = lab["Fo"]
    out["dT_int_K"] = lab["dT_int"]
    out["dT_wire_K"] = lab["dT_wire"]
    out["T_wire_peak_C"] = T_ACT + lab["dT_wire"]   # wire surface at activation
    # Validity flags (rows are kept; filter at training time)
    out["lumped_valid"] = (lab["Bi"] < 0.1) & (lab["Fo"] >= 1.0)   # eq. (23)
    out["wire_safe"] = out["T_wire_peak_C"] <= T_WIRE_MAX_C
    out["train_ok"] = out["reachable"] & out["lumped_valid"] & out["wire_safe"]
    out["t_act_orbit_s"] = orb["t_act"]
    out["t_cool_orbit_s"] = orb["t_cool"]
    out["E_cyc_orbit_J"] = orb["E_cyc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_v1.csv"))
    args = ap.parse_args()

    df = label(sample(args.n, args.seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6g")

    r = df[df.reachable]
    print(f"wrote {len(df)} rows -> {args.out}")
    print(f"reachable (lab): {df.reachable.mean():.1%}")
    print(f"lumped_valid: {df.lumped_valid.mean():.1%}  wire_safe: {df.wire_safe.mean():.1%}"
          f"  train_ok: {df.train_ok.sum()} rows ({df.train_ok.mean():.1%})")
    r = df[df.train_ok]
    print("\ntrain_ok subset:")
    print(r[["P_W", "t_act_s", "t_cool_s", "E_cyc_J", "dT_wire_K", "Bi", "T_ss_C"]]
          .describe(percentiles=[.05, .5, .95]).round(3).to_string())
    r = df[df.reachable]
    print("\nall reachable rows:")
    print(r[["P_W", "t_act_s", "t_cool_s", "E_cyc_J", "dT_wire_K", "Bi", "T_ss_C"]]
          .describe(percentiles=[.05, .5, .95]).round(3).to_string())


if __name__ == "__main__":
    main()
