"""Step 4: heater design optimisation on the ML surrogate, verified by physics.

Problem (lab conditions, h fixed at 10 W/m^2K as in Table 5.5):
    minimise   t_act   (time to reach 90 degC)
    subject to dT_wire < 30 K, P <= 5 W, geometry valid,
               design feasible (reachable, lumped model valid Fo>=1 & Bi<0.1,
               wire peak <= 200 degC)
    over       d_w, p, a, b, t, V   (within the training ranges)

Three runs:
  A. Optuna TPE (500 trials) on the XGBoost surrogate  <- the ML result
  B. Same search on the exact physics model             <- ground-truth check
  C. NSGA-II on the surrogate: Pareto front t_act vs dT_wire (speed vs hotspot;
     hotspot cap lifted to the 200 degC wire-safe limit, P <= 5 W kept)
Every surrogate optimum is re-evaluated with physics.py before being reported.

Usage: .venv/bin/python src/optimise.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import RANGES  # noqa: E402
from physics import BASELINE, K_PERP, RHO_C, evaluate, geometry_valid  # noqa: E402
from train import FEATURE_SETS, add_physics_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MOD, RES, FIG = ROOT / "models", ROOT / "results", ROOT / "figures"
optuna.logging.set_verbosity(optuna.logging.WARNING)

H_FIXED = 10.0          # environment, not a design choice
DT_WIRE_MAX = 30.0      # K
P_MAX = 5.0             # W
P_FEAS_MIN = 0.9        # required classifier probability of train_ok
N_TRIALS = 500
N_TRIALS_PARETO = 3000
SEED = 42
DESIGN = ["d_w_mm", "p_mm", "a_mm", "b_mm", "t_mm", "V"]
FEATS = FEATURE_SETS["physics"]


def load(name, cls):
    m = cls()
    m.load_model(MOD / name)
    return m


REG = {k: load(f"{k}_xgb_physics.json", XGBRegressor)
       for k in ("t_act_s", "dT_wire_K", "E_cyc_J")}
CLF = load("feasibility_xgb_physics.json", XGBClassifier)


def surrogate(designs: pd.DataFrame) -> pd.DataFrame:
    X = add_physics_features(designs.assign(h=H_FIXED))
    out = X.copy()
    for k, m in REG.items():                       # all fitted in log space
        out[k] = np.exp(m.predict(X[FEATS]))
    out["p_feasible"] = CLF.predict_proba(X[FEATS])[:, 1]
    # Fo from the predicted t_act, eq. (23)
    L = np.maximum(X.p_mm / 4, X.t_mm / 2) * 1e-3
    out["Fo"] = (K_PERP / RHO_C) * out.t_act_s / L**2
    return out


def physics(designs: pd.DataFrame) -> pd.DataFrame:
    r = evaluate(d_w=designs.d_w_mm * 1e-3, p=designs.p_mm * 1e-3,
                 a=designs.a_mm * 1e-3, t=designs.t_mm * 1e-3,
                 b=designs.b_mm * 1e-3, l_h=BASELINE["l_h"], V=designs.V,
                 h=H_FIXED)
    out = designs.copy()
    out["P_W"] = r["P"]; out["t_act_s"] = r["t_act"]; out["dT_wire_K"] = r["dT_wire"]
    out["E_cyc_J"] = r["E_cyc"]; out["t_cool_s"] = r["t_cool"]
    out["Bi"] = r["Bi"]; out["Fo"] = r["Fo"]; out["T_ss_C"] = r["T_ss"]
    out["I_A"] = r["I"]; out["R_w_ohm"] = r["R_w"]
    out["reachable"] = r["reachable"]
    return out


def suggest(trial) -> pd.DataFrame:
    d = {k: trial.suggest_float(k, *RANGES[k][:2]) for k in DESIGN}
    return pd.DataFrame([d])


def geom_ok(d):
    return bool(geometry_valid(d.d_w_mm * 1e-3, d.p_mm * 1e-3, d.a_mm * 1e-3,
                               d.t_mm * 1e-3, d.b_mm * 1e-3).iloc[0])


def violations(row, use_classifier, dt_max=DT_WIRE_MAX):
    """Constraint values (<= 0 means satisfied), for Optuna's constraints_func."""
    v = [row.dT_wire_K - dt_max, row.P_W_feat if "P_W_feat" in row else row.P_W]
    v[1] = v[1] - P_MAX
    v.append(1.0 - row.Fo)                                    # Fo >= 1
    if use_classifier:
        v.append(P_FEAS_MIN - row.p_feasible)
    else:
        v.append(row.Bi - 0.1)
        v.append(0.0 if row.reachable else 1.0)
    return v


def make_objective(mode, multi=False):
    def objective(trial):
        d = suggest(trial)
        if not geom_ok(d):
            trial.set_user_attr("c", [1.0] * 5)
            return (1e4, 1e4) if multi else 1e4
        row = (surrogate(d) if mode == "surrogate" else physics(d)).iloc[0]
        if mode == "physics" and not row.reachable:
            trial.set_user_attr("c", [1.0] * 5)
            return (1e4, 1e4) if multi else 1e4
        trial.set_user_attr("c", violations(row, mode == "surrogate",
                                            dt_max=110.0 if multi else DT_WIRE_MAX))
        return (row.t_act_s, row.dT_wire_K) if multi else row.t_act_s
    return objective


def run(mode, sampler, multi=False, n_trials=N_TRIALS):
    study = optuna.create_study(
        directions=["minimize", "minimize"] if multi else None,
        direction=None if multi else "minimize", sampler=sampler)
    study.optimize(make_objective(mode, multi), n_trials=n_trials)
    return study


def feasible_trials(study):
    return [t for t in study.trials
            if t.values is not None and all(c <= 0 for c in t.user_attrs["c"])]


def main():
    cons = lambda t: t.user_attrs["c"]  # noqa: E731
    base = pd.DataFrame([{k: BASELINE[k.replace("_mm", "")] * 1e3 if k.endswith("_mm")
                          else BASELINE[k] for k in DESIGN}])
    rows = {"Baseline (Table 5.5)": physics(base).iloc[0]}

    # A: surrogate
    sA = run("surrogate", optuna.samplers.TPESampler(seed=SEED, constraints_func=cons))
    bestA = min(feasible_trials(sA), key=lambda t: t.value)
    dA = pd.DataFrame([bestA.params])
    predA = surrogate(dA).iloc[0]
    rows["A: surrogate optimum (physics-checked)"] = physics(dA).iloc[0]

    # B: physics ground truth
    sB = run("physics", optuna.samplers.TPESampler(seed=SEED, constraints_func=cons))
    bestB = min(feasible_trials(sB), key=lambda t: t.value)
    rows["B: physics optimum (ground truth)"] = physics(pd.DataFrame([bestB.params])).iloc[0]


    check = dict(surrogate_pred_t_act=float(predA.t_act_s),
                 physics_t_act=float(rows["A: surrogate optimum (physics-checked)"].t_act_s),
                 surrogate_pred_dT_wire=float(predA.dT_wire_K),
                 physics_dT_wire=float(rows["A: surrogate optimum (physics-checked)"].dT_wire_K),
                 n_trials=N_TRIALS,
                 feasible_trials_surrogate=len(feasible_trials(sA)),
                 feasible_trials_physics=len(feasible_trials(sB)))
    (RES / "optimisation_check.json").write_text(json.dumps(check, indent=2))

    # C: Pareto t_act vs dT_wire on the surrogate, verified by physics
    sC = run("surrogate", optuna.samplers.NSGAIISampler(seed=SEED, constraints_func=cons),
             multi=True, n_trials=N_TRIALS_PARETO)
    # front from every feasible surrogate design evaluated (run C + run A)
    feas = feasible_trials(sC)
    params = [t.params for t in feas] + [t.params for t in feasible_trials(sA)]
    sur = surrogate(pd.DataFrame(params))
    pts = sur[["t_act_s", "dT_wire_K"]].to_numpy()
    nd = [i for i, p in enumerate(pts)
          if not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))]
    front = (physics(pd.DataFrame([params[i] for i in nd]))
             .round(6).drop_duplicates(subset=DESIGN).sort_values("t_act_s"))
    rows["C: fastest Pareto design (physics-checked)"] = front.iloc[0]
    cols = DESIGN + ["P_W", "I_A", "R_w_ohm", "t_act_s", "dT_wire_K", "E_cyc_J",
                     "t_cool_s", "Bi", "Fo", "T_ss_C"]
    table = pd.DataFrame(rows).T[cols].astype(float)
    table.to_csv(RES / "optimisation.csv", float_format="%.4g")
    front.to_csv(RES / "pareto_front.csv", index=False, float_format="%.4g")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color="0.75", label="feasible trials (surrogate)")
    ax.plot(front.t_act_s, front.dT_wire_K, "o", color="#2a6fdb", ms=4,
            label="Pareto designs (picked by surrogate,\nplotted with exact physics)")
    b = rows["Baseline (Table 5.5)"]
    ax.scatter([b.t_act_s], [b.dT_wire_K], marker="*", s=180, color="#d1495b",
               zorder=5, label="baseline design")
    ax.set_xlabel("heating time to 90 °C, t_act [s]")
    bB = rows["B: physics optimum (ground truth)"]
    ax.scatter([bB.t_act_s], [bB.dT_wire_K], marker="D", s=50, color="#2e9e5b",
               zorder=5, label="exact-physics optimum B (ΔT < 30 K)")
    ax.axhline(DT_WIRE_MAX, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_ylabel("wire hotspot above mean, ΔT_wire [K]")
    ax.set_title(f"Speed vs hotspot trade-off (P ≤ {P_MAX:.0f} W, lab, synthetic model)",
                 fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "pareto_tact_hotspot.png", dpi=200)

    pd.set_option("display.width", 200)
    print(table.round(3).T.to_string())
    print("\nsurrogate check:", json.dumps(check, indent=1))
    print(f"\nPareto front: {len(front)} designs, "
          f"t_act {front.t_act_s.min():.1f}-{front.t_act_s.max():.1f} s, "
          f"dT_wire {front.dT_wire_K.min():.1f}-{front.dT_wire_K.max():.1f} K")
    print(front[DESIGN + ["P_W", "t_act_s", "dT_wire_K", "E_cyc_J"]].round(2).to_string())


if __name__ == "__main__":
    main()
