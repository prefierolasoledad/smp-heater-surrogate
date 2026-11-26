# ML Surrogate for the Zigzag Nichrome SMP Heater: Plan

**What this is:** an ML surrogate trained on synthetic data from a physics model calibrated to the literature, namely the report's equations (18) to (24) and Table 5.5. It is **not** experimental data. The surrogate will be retrained once bench or ANSYS data exist.

*Last updated: 2026-11-26*

## Status

| Step | Status | Output |
|---|---|---|
| 0. Setup | ✅ Done | `.venv/`, `requirements.txt`, `.gitignore`, `README.md` |
| 1. Physics model | ✅ Done | `src/physics.py`; `tests/test_physics.py` passes 20/20 |
| 2. Dataset | ✅ Done | `data/synthetic_v1.csv`, `data/README.md` |
| 3. Surrogate | ✅ Done | `src/train.py`, `results/metrics.csv`, figures, `models/` |
| 4. Optimisation | ✅ Done | `src/optimise.py`, `results/optimisation.csv`, `results/pareto_front.csv` |
| 4b. Torque / stiffness constraint | ⬜ To do (recommended) | |
| 5. Literature check | ⏳ Blocked: needs the digitised curves | |
| 6. Multi-fidelity upgrade | ⬜ Optional | |
| 7. Report section and slides | ⬜ To do | |

## 0. Setup ✅

- **Runs locally** on this machine (4 cores, 11 GB RAM, Python 3.14). Colab is not needed and remains an option only for sharing.
- **Environment:** `.venv/`. Install with `.venv/bin/pip install -r requirements.txt`.
- **Packages:** numpy, pandas, scipy, scikit-learn, xgboost, shap, optuna, matplotlib, pytest and jupyter.
- **Code layout:** plain Python scripts in `src/`, run from the command line. These replace the notebooks planned originally. Notebooks can import `src/` later if needed for the presentation.
- **Run times:** `generate.py` about 3 s, `train.py` about 8 min, `optimise.py` about 2 min.
- **Git:** `.gitignore` excludes `.venv/` (about 1.1 GB) and `models/` (about 250 MB, which `train.py` regenerates). Nothing has been committed or pushed yet.
- **Colab** (if needed later): either mount Google Drive and `%cd` into the folder, or `git clone` the repo.

## 1. Physics model ✅ (`src/physics.py`)

- **Equations:** report eqs. (18)–(24), vectorised, for both lab (convection) and orbit (linearised radiation).
- **Extra outputs:** steady-state temperature with no cut-off (T_ss) and current density J.
- **Validation:** reproduces every Stage-1 value in Table 5.6:
  - lab: L_w 401.1 mm, R_w 14.04 Ω, P 4.01 W, τ 138 s, t_act 38.5 s, t_cool 85.4 s, Bi 0.040, Fo 6.96, ΔT_int 4.7 K, ΔT_wire 16.0 K
  - orbit: G 11.4 mW/K, τ 181.6 s, t_act 37.2 s, t_cool 112.4 s, E_cyc 149 J
- **Unreachable designs:** where P ≤ G(T_act − T0), the design is flagged `reachable = False` and its time-based outputs are NaN.
- **Geometry check** (`geometry_valid`), three rules:
  - the zigzag fits the hinge: a ≤ b − 2 mm
  - adjacent wire runs don't touch: s/2 = p/4 > d_w
  - the hinge is thicker than the wire channel: t > 2·d_w
- **Finding:** at baseline, T_ss ≈ 292 °C, so constant power without a cut-off overheats the epoxy.

## 2. Dataset ✅ (`src/generate.py` → `data/synthetic_v1.csv`)

- **Sampling:** 7-dimensional Latin hypercube, N = 5,000, seed 42. Designs failing the geometry check are rejected and resampled. Hinge length is fixed at 30 mm.

| Input | Range |
|---|---|
| d_w | 0.10–0.40 mm |
| p | 2–6 mm |
| a | 10–25 mm |
| b (added so a can fit) | 20–30 mm |
| t | 1–3 mm |
| V | 3–12 V |
| h | 5–15 W/m²K |

- **Outputs:** 32 columns, covering lab and orbit outputs plus the validity flags. Documented in `data/README.md`.
- **Validity flags** (all rows kept):
  - `reachable` 94.9 %
  - `lumped_valid` (Bi < 0.1, Fo ≥ 1) 63.8 %
  - `wire_safe` (wire peak ≤ 200 °C) 72.1 %
  - **`train_ok` (all three): 3,037 rows (60.7 %)**
- **Finding:** the suggested ranges include unrealistic designs, up to about 100 W and wire peaks over 1,000 °C. The flags exclude them.
- **Open item:** the 200 °C wire limit is an assumption. Replace it with the resin's datasheet limit (`T_WIRE_MAX_C` in `generate.py`) once the epoxy is chosen, then rerun steps 2–4.

## 3. Surrogate ✅ (`src/train.py`)

**Changes from the original plan:**
- The classifier predicts `train_ok` (feasible design), not just `reachable`.
- Two feature sets are compared: raw inputs only (7), and inputs plus 4 physics features. The physics features are P, G, C_h and the power ratio P/(G·ΔT), all computed from the inputs with eqs. (18)–(21).
- All four regression targets are fitted in log space.

**Setup:** an 80/20 split on the `train_ok` rows, 5-fold cross-validation, Random Forest compared with XGBoost, and SHAP explanations for every target.

**Results (XGBoost, test-set R²):**

| Target | Raw inputs | With physics features |
|---|---|---|
| t_act | 0.771 | **0.987** |
| ΔT_wire | 0.960 | **0.985** |
| t_cool | 0.998 | **0.999** |
| E_cyc | 0.793 | **0.991** |
| Classifier | AUC 0.993 | AUC 0.997, accuracy 97 % |

**Findings:**
- **The original guess of "R² ≈ 0.99 for free" was wrong for raw inputs.** About 3 % of designs only just reach 90 °C, and t_act follows ln(1 − 1/ratio), which rises extremely steeply. These designs caused 90 % of the error. The physics features fix this, which makes a good viva point.
- **The SHAP rankings match the physics.** Wire diameter and voltage dominate heating time, and cooling time ignores the electrical inputs.
- **Caveat:** high R² shows that the surrogate reproduces the physics model, not that the physics model is correct.

## 4. Optimisation ✅ (`src/optimise.py`)

**Problem:**
- Minimise t_act.
- Subject to: ΔT_wire < 30 K, P ≤ 5 W, valid geometry, feasibility probability ≥ 0.9 and Fo ≥ 1.
- h is fixed at 10 W/m²K.

**Runs:**
- **A:** TPE on the surrogate, 500 trials.
- **B:** TPE on the exact physics, as a reference.
- **C:** NSGA-II on the surrogate, 3,000 trials, giving the speed-versus-hotspot Pareto front. This was changed from t_act versus E_cyc, because those two fall together and give no real trade-off.

Every design selected with the surrogate was re-checked with `physics.py`.

| Design | t_act | ΔT_wire | E_cyc |
|---|---|---|---|
| Baseline | 38.5 s | 16.0 K | 154 J |
| C: fastest surrogate Pareto design (d_w 0.40, p 2.04, a 19.2, b 21.5, t 1.01 mm, 5.0 V) | **16.1 s** | **6.6 K** | 80 J |
| B: exact-physics optimum | 15.0 s | 28.2 K | 74 J |

**Findings:**
- **Accurate at the optimum.** At design A the surrogate's error is under 1 % for t_act and under 2 % for ΔT_wire.
- **Faster and cooler than the baseline.** Design C is 2.4× faster, with a lower hotspot and half the energy per cycle.
- **Close to the true optimum.** Design C is within 8 % of the exact-physics best.
- **What limits speed:** power and heat capacity. Every fast design uses the full 5 W budget with the thinnest, narrowest hinge allowed.
- **Cut-off needed.** The optimised designs reach T_ss ≈ 410–435 °C without a cut-off, so a thermostat or timer is mandatory.

## 4b. Torque / stiffness constraint ⬜ (recommended next)

**Why:** the optimum uses t = 1.0 mm. Fold strain is fine (5.2 % against 8.4 % at baseline, eq. 28), but k_θ scales with t³. At 1.0 mm it falls to about 24 % of the baseline, and the neat hinge already fails the torque margin (Table 5.6).

**Tasks:**
- Add eqs. (25)–(27) and (31)–(34) to `physics.py`.
- Test them against the Stage-1 values in Table 5.6: k_θ,lock 0.427 N·m/rad, τ_rec,0 13.4 mN·m, TM −0.46, f1 1.20 Hz.
- Add a constraint to the optimiser, either "k_θ,lock ≥ baseline" or, for the Stage-2 CFR-SMPC case, "TM ≥ 1.0 and f1 ≥ 1.5 Hz".
- Rerun step 4.

## 5. Literature check ⏳ (blocked: needs the curves)

- **Needed from you:** digitise 1–3 recovery-angle-versus-time curves with WebPlotDigitizer and save them as `data/digitized/<paper>_<fig>.csv`, with columns `t_s` and `theta_deg`. Candidates: Lan et al. [7], [9], Zhou et al. [10] and Iorio et al. [14].
- **To build:** `src/literature_fit.py`. It fits the bi-exponential law of eq. (30) with `scipy.optimize.curve_fit`, reports R², τ_I and τ_II, and overlays the fit on the data.
- **Viva caveat:** these papers used filler-heated or fibre-reinforced composites. Fitting their data checks the *shape* of eq. (30). It does not calibrate neat-epoxy parameters.

## 6. Multi-fidelity upgrade ⬜ (optional; strengthens "why ML?")

- **(a)** A 2D transient finite-difference conduction model of the wire cross-section, or ANSYS Transient Thermal runs, to capture non-lumped hotspots and gradients. Train the surrogate on this data, or use it to correct the lumped model.
- **(b)** Retrain the surrogate on measured T(t) data from the bench tests in §5.6 of the report.

## 7. Report section and slides ⬜

- **Report section:** a new section, for example 5.7 "ML Surrogate and Design Optimisation", covering method, the data card, the results tables and figures, and the synthetic-data caveat. Also fill in the Chapter 6 placeholders with the ML results, without presenting them as experimental data.
- **Slides:** 6–8 slides covering the concept, the dataset, the accuracy comparison (raw versus physics features), SHAP, the optimum against the baseline, the Pareto chart and the limitations.
- **Fix before submission:** the report's unverified references [10], [13], [15] and [19], which are still marked "[Verify…]".

## Folder layout (current)

```
README.md  PLAN.md  requirements.txt  .gitignore
src/       physics.py  generate.py  train.py  optimise.py
tests/     test_physics.py
data/      synthetic_v1.csv  README.md         (digitized/ for step 5)
results/   metrics.csv  optimisation.csv  optimisation_check.json
           pareto_front.csv  split_indices.json  README.md
figures/   parity_*, shap_beeswarm_*, shap_bar_*, pareto_tact_hotspot.png
models/    trained models (git-ignored; regenerate with train.py)
```

## Order of remaining work

1. 4b: add the torque and stiffness constraint and rerun the optimisation. *(Claude)*
2. 5: digitise the curves *(you)*, then fit eq. (30) *(Claude)*.
3. 7: write the report section and slides.
4. 6: optional multi-fidelity upgrade, if time allows.
5. Set the resin's real wire temperature limit, then rerun steps 2–4.
