"""Step 3: train ML surrogates of the electrothermal model.

- Feasibility classifier: predicts train_ok (reachable & lumped-valid & wire-safe)
  from the 7 design inputs, trained on all rows.
- Regressors (Random Forest vs XGBoost): t_act, dT_wire, t_cool, E_cyc,
  trained on train_ok rows only. Skewed targets are fitted in log space.
- SHAP (TreeExplainer on XGBoost) for every regression target.

Outputs: models/, figures/, results/metrics.csv
Usage:   .venv/bin/python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_v1.csv"
FIG, MOD, RES = ROOT / "figures", ROOT / "models", ROOT / "results"
SEED = 42

FEATURES = ["d_w_mm", "p_mm", "a_mm", "b_mm", "t_mm", "V", "h"]
# Physics-informed features: computed from the 7 inputs only, with eqs. (18)-(21).
# They encode the "cheap" part of the physics so the ML learns eqs. (22)-(24).
PHYS = ["P_W_feat", "G_mW_K", "C_h_J_K", "power_ratio"]
FEATURE_SETS = {"raw": FEATURES, "physics": FEATURES + PHYS}
LABELS = {"d_w_mm": "wire diameter", "p_mm": "pitch", "a_mm": "zigzag width",
          "b_mm": "hinge width", "t_mm": "thickness", "V": "voltage",
          "h": "convection h", "P_W_feat": "heater power P",
          "G_mW_K": "heat-loss conductance G", "C_h_J_K": "heat capacity C_h",
          "power_ratio": "power ratio P/(G·ΔT)"}


def add_physics_features(df):
    """Eqs. (18)-(21) from the design inputs (no outputs used)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from physics import BASELINE, RHO_C, T0, T_ACT, heater
    l_h = BASELINE["l_h"]
    _, _, _, P = heater(df.d_w_mm * 1e-3, df.p_mm * 1e-3, df.a_mm * 1e-3, l_h, df.V)
    G = df.h * 2 * df.b_mm * 1e-3 * l_h
    out = df.copy()
    out["P_W_feat"] = P
    out["G_mW_K"] = G * 1e3
    out["C_h_J_K"] = RHO_C * df.b_mm * 1e-3 * df.t_mm * 1e-3 * l_h
    out["power_ratio"] = P / (G * (T_ACT - T0))
    return out
# target column: (pretty name, unit, fit in log space?)
TARGETS = {
    "t_act_s":   ("Heating time to 90 °C", "s", True),
    "dT_wire_K": ("Wire hotspot ΔT", "K", True),
    "t_cool_s":  ("Cooling time to 60 °C", "s", True),
    "E_cyc_J":   ("Energy per cycle", "J", True),
}


def rf_reg():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=1,
                                 n_jobs=-1, random_state=SEED)


def xgb_reg():
    return XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6,
                        subsample=0.8, colsample_bytree=0.9, min_child_weight=2,
                        n_jobs=-1, random_state=SEED)


def metrics(y, yhat):
    return dict(R2=r2_score(y, yhat),
                RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
                MAE=mean_absolute_error(y, yhat),
                MAPE_pct=float(np.mean(np.abs((yhat - y) / y)) * 100))


def train_classifier(df, rows, fs="physics"):
    X, y = df[FEATURE_SETS[fs]], df["train_ok"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
    clf = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5,
                        subsample=0.8, n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    rows.append(dict(target="train_ok (classifier)", model="XGBoost", features=fs,
                     accuracy=accuracy_score(yte, p > 0.5),
                     F1=f1_score(yte, p > 0.5), ROC_AUC=roc_auc_score(yte, p),
                     n_train=len(Xtr), n_test=len(Xte)))
    clf.save_model(MOD / f"feasibility_xgb_{fs}.json")
    print(f"classifier[{fs}]  acc={rows[-1]['accuracy']:.3f}  AUC={rows[-1]['ROC_AUC']:.4f}")


def parity_plot(ax, y, yhat, title, unit, r2):
    lo, hi = min(y.min(), yhat.min()), max(y.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=1)
    ax.scatter(y, yhat, s=6, alpha=0.5, color="#2a6fdb", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=10)
    ax.set_xlabel(f"physics model [{unit}]"); ax.set_ylabel(f"XGBoost [{unit}]")


def main():
    for d in (FIG, MOD, RES):
        d.mkdir(exist_ok=True)
    df = add_physics_features(pd.read_csv(DATA))
    rows = []
    for fs in FEATURE_SETS:
        train_classifier(df, rows, fs)

    ok = df[df.train_ok].reset_index(drop=True)
    idx_tr, idx_te = train_test_split(np.arange(len(ok)), test_size=0.2,
                                      random_state=SEED)
    (RES / "split_indices.json").write_text(
        json.dumps(dict(train=idx_tr.tolist(), test=idx_te.tolist())))
    kf = KFold(5, shuffle=True, random_state=SEED)

    for fs, feats in FEATURE_SETS.items():
        X = ok[feats]
        Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
        fig_par, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4))
        for ax, (col, (name, unit, logt)) in zip(axes, TARGETS.items()):
            y = ok[col].to_numpy()
            f = np.log if logt else (lambda v: v)
            finv = np.exp if logt else (lambda v: v)
            ytr, yte = f(y[idx_tr]), y[idx_te]

            for mname, make in (("RandomForest", rf_reg), ("XGBoost", xgb_reg)):
                cv = cross_val_score(make(), Xtr, ytr, cv=kf, scoring="r2", n_jobs=1)
                m = make().fit(Xtr, ytr)
                yhat = finv(m.predict(Xte))
                r = dict(target=col, model=mname, features=fs, **metrics(yte, yhat),
                         CV_R2_mean=cv.mean(), CV_R2_std=cv.std(),
                         n_train=len(Xtr), n_test=len(Xte))
                rows.append(r)
                print(f"[{fs:7s}] {col:10s} {mname:12s} R2={r['R2']:.4f} "
                      f"RMSE={r['RMSE']:.3g} MAPE={r['MAPE_pct']:.2f}%  "
                      f"CV R2(log)={cv.mean():.4f}±{cv.std():.4f}", flush=True)
                if mname == "RandomForest":
                    joblib.dump(m, MOD / f"{col}_rf_{fs}.joblib", compress=3)
                else:
                    m.save_model(MOD / f"{col}_xgb_{fs}.json")
                    parity_plot(ax, yte, yhat, name, unit, r["R2"])
                    xgb_model = m

            # SHAP on the test set (log-fitted targets are explained in log space)
            sv = shap.TreeExplainer(xgb_model)(Xte)
            sv.feature_names = [LABELS[c] for c in feats]
            for kind, plot in (("beeswarm", shap.plots.beeswarm), ("bar", shap.plots.bar)):
                plt.figure()
                plot(sv, show=False, max_display=len(feats))
                plt.title(f"SHAP ({fs} features): {name}"
                          + (" [log]" if logt and kind == "beeswarm" else ""), fontsize=10)
                plt.tight_layout()
                plt.savefig(FIG / f"shap_{kind}_{col}_{fs}.png", dpi=200)
                plt.close("all")

        fig_par.suptitle(f"XGBoost surrogate vs physics model, held-out test set "
                         f"({fs} features, synthetic data)", fontsize=11)
        fig_par.tight_layout()
        fig_par.savefig(FIG / f"parity_xgb_{fs}.png", dpi=200)
        plt.close(fig_par)

    pd.DataFrame(rows).to_csv(RES / "metrics.csv", index=False, float_format="%.6g")
    print(f"\nwrote {RES/'metrics.csv'}, figures/ and models/")


if __name__ == "__main__":
    main()
