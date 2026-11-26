"""ANSYS run plan for the multi-fidelity surrogate (replaces step 5).

Writes data/ansys_run_plan.csv: 25 designs in priority order.
  run 1  = baseline (Table 5.5)
  run 2  = stiffness-constrained optimum D (step 4b)
  runs 3-25 = valid designs (train_ok, P <= 10 W) picked by greedy farthest-point
              sampling in normalised input space, so ANY prefix of the list
              (e.g. the first 15) covers the design space as evenly as possible.
Each row carries everything ANSYS needs (geometry, wire heat generation,
convection) plus the lumped predictions used later for Delta-learning, and a
suggested end time and time step.

Also writes data/ansys_results_template.csv to fill in.
Usage: .venv/bin/python src/ansys_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_model import lumped_band  # noqa: E402
from physics import BASELINE, evaluate, heater  # noqa: E402
from train import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_RUNS, N_ESSENTIAL, SEED = 25, 15, 42
P_MAX_PLAN = 10.0   # W; keep runs near the optimiser's region (P <= 5 W) plus margin
L_H = BASELINE["l_h"]


def describe(d):
    """Everything ANSYS needs for one design (d: dict in mm / V / W m^-2 K^-1)."""
    si = dict(d_w=d["d_w_mm"] * 1e-3, p=d["p_mm"] * 1e-3, a=d["a_mm"] * 1e-3,
              t=d["t_mm"] * 1e-3, l_h=L_H, V=d["V"], h=d["h"])
    L_w, R_w, I, P = heater(si["d_w"], si["p"], si["a"], L_H, si["V"])
    A_w = np.pi * si["d_w"] ** 2 / 4
    full = evaluate(**si, b=d["b_mm"] * 1e-3)
    band = lumped_band(**si)
    t_ref = float(band["t_act"])
    return dict(
        d, L_h_mm=L_H * 1e3, s_mm=d["p_mm"] / 2, n_runs=round(2 * L_H / (d["p_mm"] * 1e-3)),
        L_w_mm=L_w * 1e3, R_w_ohm=R_w, I_A=I, P_W=P,
        q_gen_W_m3=P / (L_w * A_w),                       # ANSYS internal heat generation
        t_act_lumped_full_s=float(full["t_act"]),          # whole hinge, eq. 22
        t_act_lumped_band_s=t_ref,                         # heated band only
        dT_wire_eq24_K=float(band["dT_wire"]),
        end_time_s=round(3 * t_ref, 1),
        max_step_s=round(max(t_ref / 100, 0.05), 3),
    )


def main():
    df = pd.read_csv(ROOT / "data" / "synthetic_v1.csv")
    pool = df[df.train_ok & (df.P_W <= P_MAX_PLAN)][FEATURES].reset_index(drop=True)
    opt = pd.read_csv(ROOT / "results" / "optimisation.csv", index_col=0)
    D = opt.loc[[i for i in opt.index if i.startswith("D:")][0]]
    fixed = [
        dict(d_w_mm=0.20, p_mm=3.0, a_mm=20.0, b_mm=25.0, t_mm=1.60, V=7.5, h=10.0),
        dict(d_w_mm=round(D.d_w_mm, 3), p_mm=round(D.p_mm, 3), a_mm=round(D.a_mm, 2),
             b_mm=round(D.b_mm, 2), t_mm=round(D.t_mm, 3), V=round(D.V, 3), h=10.0),
    ]
    # greedy farthest-point ordering in [0,1]-scaled input space
    lo, hi = pool.min(), pool.max()
    Z = ((pool - lo) / (hi - lo)).to_numpy()
    chosen = [((pd.DataFrame(fixed) - lo) / (hi - lo)).to_numpy()]
    picked = []
    dmin = np.min(np.linalg.norm(Z[:, None, :] - chosen[0][None, :, :], axis=2), axis=1)
    for _ in range(N_RUNS - len(fixed)):
        i = int(np.argmax(dmin))
        picked.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[i], axis=1))
    rows = fixed + [{k: round(float(pool.loc[i, k]), 3) for k in FEATURES} for i in picked]
    plan = pd.DataFrame([describe(r) for r in rows])
    plan.insert(0, "run_id", [f"R{i + 1:02d}" for i in range(len(plan))])
    plan.insert(1, "priority", ["essential" if i < N_ESSENTIAL else "optional"
                                for i in range(len(plan))])
    plan.insert(2, "label", ["baseline", "optimum D"] + [""] * (len(plan) - 2))
    out = ROOT / "data" / "ansys_run_plan.csv"
    plan.to_csv(out, index=False, float_format="%.5g")

    tmpl = plan[["run_id"]].assign(t_act_band_min_s="", t_act_mean_s="", dT_hot_K="",
                                   T_max_at_act_C="", t_act_hinge_min_s="", notes="")
    tmpl.to_csv(ROOT / "data" / "ansys_results_template.csv", index=False)
    cols = ["run_id", "priority", "label", "d_w_mm", "p_mm", "a_mm", "b_mm", "t_mm", "V", "h",
            "P_W", "q_gen_W_m3", "t_act_lumped_band_s", "end_time_s", "max_step_s"]
    pd.set_option("display.width", 220)
    print(plan[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
