"""Bring ANSYS Transient Thermal results into the multi-fidelity surrogate.

Input : data/ansys_results.csv  (copy of data/ansys_results_template.csv, filled in;
        blank cells are fine — a run without t_act_band_min_s is skipped)
Plan  : data/ansys_run_plan.csv (from src/ansys_plan.py)

What it does
1. Three-way comparison per run: ANSYS vs the report's lumped equations vs the
   2D finite-difference model (two checks):
     - whole-hinge mean temperature reaching 90 degC  <->  eq. (22), full hinge
     - coldest point of the zigzag band reaching 90 degC  <->  lumped band / 2D FD
2. Leave-one-out error of three predictors of the ANSYS band time:
     lumped alone | lumped x constant ANSYS ratio | Delta-XGBoost (lumped + ML correction)
3. Corrected predictions for the baseline (R01) and optimum D (R02).
Writes results/ansys_validation.csv and results/ansys_validation.md.

Usage:
  .venv/bin/python src/ansys_ingest.py            # real ANSYS data
  .venv/bin/python src/ansys_ingest.py --demo     # dry run: 2D FD results stand in for ANSYS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_model import simulate  # noqa: E402
from physics import BASELINE  # noqa: E402
from train import FEATURES, PHYS, add_physics_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
L_H = BASELINE["l_h"]


def fd_for(row):
    return simulate(d_w=row.d_w_mm * 1e-3, p=row.p_mm * 1e-3, a=row.a_mm * 1e-3,
                    t=row.t_mm * 1e-3, l_h=L_H, V=row.V, h=row.h)


def mape(y, yhat):
    return float(np.mean(np.abs(yhat - y) / y) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    plan = pd.read_csv(ROOT / "data" / "ansys_run_plan.csv")
    fd = pd.DataFrame([fd_for(r) for r in plan.itertuples()])
    plan["t_act_fd_cold_s"] = fd.t_act_cold
    plan["t_act_fd_mean_s"] = fd.t_act_mean

    if args.demo:
        res = plan[["run_id"]].assign(t_act_band_min_s=plan.t_act_fd_cold_s,
                                      t_act_mean_s=np.nan)
        tag = "_demo"
        print("DEMO MODE: 2D FD results stand in for ANSYS. Nothing here is ANSYS data.\n")
    else:
        path = ROOT / "data" / "ansys_results.csv"
        if not path.exists():
            sys.exit("Fill in data/ansys_results.csv (copy the template) first.")
        res = pd.read_csv(path)
        tag = ""
    df = plan.merge(res, on="run_id", how="inner")
    df = df[pd.to_numeric(df.t_act_band_min_s, errors="coerce").notna()].copy()
    df["t_act_band_min_s"] = df.t_act_band_min_s.astype(float)
    n = len(df)
    if n == 0:
        sys.exit("No runs with t_act_band_min_s filled in yet.")

    # 1. three-way comparison
    df["ratio_ansys_to_lumped_band"] = df.t_act_band_min_s / df.t_act_lumped_band_s
    df["ratio_ansys_to_fd"] = df.t_act_band_min_s / df.t_act_fd_cold_s
    if "t_act_mean_s" in df and pd.to_numeric(df.t_act_mean_s, errors="coerce").notna().any():
        df["ratio_mean_to_eq22_full"] = (pd.to_numeric(df.t_act_mean_s, errors="coerce")
                                         / df.t_act_lumped_full_s)

    # 2. leave-one-out comparison of predictors
    y, lf = df.t_act_band_min_s.to_numpy(), df.t_act_lumped_band_s.to_numpy()
    X = add_physics_features(df[FEATURES].copy())[FEATURES + PHYS].assign(lf_log=np.log(lf))
    loo = {"lumped equations alone": lf.copy(), "lumped x constant ANSYS ratio": np.empty(n),
           "Delta-XGBoost (lumped + ML correction)": np.empty(n)}
    for i in range(n):
        tr = np.arange(n) != i
        loo["lumped x constant ANSYS ratio"][i] = lf[i] * np.median(y[tr] / lf[tr])
        if n >= 5:
            m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                             min_child_weight=1, random_state=42, n_jobs=1)
            m.fit(X[tr], np.log(y[tr] / lf[tr]))
            loo["Delta-XGBoost (lumped + ML correction)"][i] = lf[i] * np.exp(m.predict(X[~tr]))[0]
        else:
            loo["Delta-XGBoost (lumped + ML correction)"][i] = np.nan
    scores = {k: (mape(y, v) if np.isfinite(v).all() else np.nan) for k, v in loo.items()}

    out = df[["run_id", "label", "P_W", "t_act_lumped_band_s", "t_act_fd_cold_s",
              "t_act_band_min_s", "ratio_ansys_to_lumped_band", "ratio_ansys_to_fd"]
             + (["ratio_mean_to_eq22_full"] if "ratio_mean_to_eq22_full" in df else [])]
    out.to_csv(ROOT / "results" / f"ansys_validation{tag}.csv", index=False, float_format="%.4g")

    lines = [f"# ANSYS validation{' (DEMO: FD stands in for ANSYS)' if tag else ''}", "",
             f"Runs used: {n}", "",
             "| Predictor of ANSYS band activation time | Leave-one-out MAPE |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} % |" if np.isfinite(v) else f"| {k} | needs >= 5 runs |"
              for k, v in scores.items()]
    lines += ["", "| Ratio | Median | Min | Max |", "|---|---|---|---|"]
    for c in [c for c in out.columns if c.startswith("ratio")]:
        s = out[c].dropna()
        lines.append(f"| {c} | {s.median():.3f} | {s.min():.3f} | {s.max():.3f} |")
    for lab in ("baseline", "optimum D"):
        r = df[df.label == lab]
        if len(r):
            r = r.iloc[0]
            lines.append(f"\n{lab}: ANSYS {r.t_act_band_min_s:.1f} s · 2D FD "
                         f"{r.t_act_fd_cold_s:.1f} s · lumped band {r.t_act_lumped_band_s:.1f} s")
    b, d = df[df.label == "baseline"], df[df.label == "optimum D"]
    if len(b) and len(d):
        lines.append(f"\nSpeed-up of D over baseline in ANSYS: "
                     f"{b.t_act_band_min_s.iloc[0] / d.t_act_band_min_s.iloc[0]:.2f}x "
                     f"(lumped: 1.45x, 2D FD: 1.42x)")
    text = "\n".join(lines)
    (ROOT / "results" / f"ansys_validation{tag}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
