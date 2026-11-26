"""Step 6b: multi-fidelity surrogate (lumped model + 2D finite-difference model).

1. Run the 2D FD model (fd_model.py) on N_HF designs drawn from the valid
   (train_ok) rows of data/synthetic_v1.csv -> data/fd_highfidelity.csv.
2. Compare, on a fixed held-out set of 100 FD designs, three predictors of the
   high-fidelity outputs (coldest-point activation time t_act_cold and the
   wire hotspot dT_hot):
     LF only   - the lumped-band equations (eqs. 18-24), no ML
     HF-only   - XGBoost trained only on n FD runs
     Delta     - XGBoost trained on n FD runs to correct the lumped
                 prediction: target log(HF / LF), lumped output as a feature
   for n = 25, 50, 100, 200, 300 training runs (5 random subsets each).
3. Write results/multifidelity.csv and figures/multifidelity_learning_curve.png.

Usage: .venv/bin/python src/multifidelity.py
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_model import lumped_band, simulate  # noqa: E402
from physics import BASELINE  # noqa: E402
from train import FEATURES, PHYS, add_physics_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
N_HF, N_TEST, SEED = 400, 100, 42
SIZES, REPEATS = [25, 50, 100, 200, 300], 5
L_H = BASELINE["l_h"]


def run_one(row):
    args = dict(d_w=row["d_w_mm"] * 1e-3, p=row["p_mm"] * 1e-3, a=row["a_mm"] * 1e-3,
                t=row["t_mm"] * 1e-3, l_h=L_H, V=row["V"], h=row["h"])
    hf = simulate(**args)
    lf = lumped_band(**args)
    return dict(row, t_act_cold_hf=hf["t_act_cold"], t_act_mean_hf=hf["t_act_mean"],
                dT_hot_hf=hf["dT_hot"], T_peak_hf=hf["T_peak"],
                t_act_lf=float(lf["t_act"]), dT_wire_lf=float(lf["dT_wire"]),
                b_band_mm=hf["b_band"] * 1e3)


def generate():
    df = pd.read_csv(DATA / "synthetic_v1.csv")
    pool_rows = df[df.train_ok][FEATURES].sample(N_HF, random_state=SEED)
    with Pool(4) as pool:
        out = pool.map(run_one, pool_rows.to_dict("records"))
    hf = pd.DataFrame(out)
    hf.to_csv(DATA / "fd_highfidelity.csv", index=False, float_format="%.6g")
    return hf


def model():
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.9, min_child_weight=1,
                        n_jobs=1, random_state=SEED)


def mape(y, yhat):
    return float(np.mean(np.abs(yhat - y) / y) * 100)


def main():
    path = DATA / "fd_highfidelity.csv"
    hf = pd.read_csv(path) if path.exists() else generate()
    hf = hf.dropna(subset=["t_act_cold_hf", "dT_hot_hf"]).reset_index(drop=True)
    X = add_physics_features(hf[FEATURES].copy())
    feats = FEATURES + PHYS
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(hf))
    te, pool_idx = perm[:N_TEST], perm[N_TEST:]

    targets = {"t_act_cold": ("t_act_cold_hf", "t_act_lf"),
               "dT_hot": ("dT_hot_hf", "dT_wire_lf")}
    rows = []
    for name, (hcol, lcol) in targets.items():
        y, lf = hf[hcol].to_numpy(), hf[lcol].to_numpy()
        rows.append(dict(target=name, method="LF only (lumped)", n_hf=0,
                         MAPE=mape(y[te], lf[te]), MAPE_sd=0.0))
        Xd = X[feats].assign(lf_log=np.log(lf))
        for n in SIZES:
            for method in ("HF-only XGBoost", "Delta (lumped + XGBoost correction)"):
                errs = []
                for r in range(REPEATS):
                    tr = np.random.default_rng(SEED + r).choice(pool_idx, n, replace=False)
                    if method.startswith("HF"):
                        m = model().fit(X[feats].iloc[tr], np.log(y[tr]))
                        pred = np.exp(m.predict(X[feats].iloc[te]))
                    else:
                        m = model().fit(Xd.iloc[tr], np.log(y[tr] / lf[tr]))
                        pred = lf[te] * np.exp(m.predict(Xd.iloc[te]))
                    errs.append(mape(y[te], pred))
                rows.append(dict(target=name, method=method, n_hf=n,
                                 MAPE=float(np.mean(errs)), MAPE_sd=float(np.std(errs))))
            print(name, n, [f"{r['method'][:5]} {r['MAPE']:.2f}%" for r in rows[-2:]], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "multifidelity.csv", index=False, float_format="%.4g")

    # learning-curve figure
    titles = {"t_act_cold": "Time for the coldest point to reach 90 °C",
              "dT_hot": "Wire hotspot above mean temperature"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"HF-only XGBoost": dict(color="0.55", marker="o"),
              "Delta (lumped + XGBoost correction)": dict(color="#2a6fdb", marker="s")}
    for ax, name in zip(axes, targets):
        r = res[res.target == name]
        lf_err = float(r[r.n_hf == 0].MAPE.iloc[0])
        ax.axhline(lf_err, color="#d1495b", ls="--", lw=1.2,
                   label=f"lumped equations alone ({lf_err:.1f} %)")
        for method, st in styles.items():
            q = r[r.method == method]
            ax.errorbar(q.n_hf, q.MAPE, yerr=q.MAPE_sd, capsize=3, lw=1.6, ms=5,
                        label=method, **st)
        ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("high-fidelity (2D FD) training runs")
        ax.set_ylabel("error on 100 held-out FD designs, MAPE [%]")
        ax.set_title(titles[name], fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Multi-fidelity learning: correcting the lumped model needs far fewer "
                 "expensive runs", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "multifidelity_learning_curve.png", dpi=200)

    # model-form comparison on all FD designs
    d = hf.assign(r_time=hf.t_act_cold_hf / hf.t_act_lf, r_hot=hf.dT_hot_hf / hf.dT_wire_lf,
                  r_mean=hf.t_act_mean_hf / hf.t_act_lf)
    summ = d[["r_mean", "r_time", "r_hot"]].describe(percentiles=[.05, .5, .95]).round(3)
    summ.to_csv(RES / "fd_vs_lumped_ratios.csv")
    print("\nFD / lumped ratios over", len(d), "designs:\n", summ.to_string())
    print(res.round(2).to_string())


if __name__ == "__main__":
    main()
