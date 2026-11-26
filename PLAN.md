# ML Surrogate for the Zigzag Nichrome SMP Heater: Plan

**What this is:** an ML surrogate trained on synthetic data from a physics model calibrated to the literature, namely the report's equations (18) to (24) and Table 5.5. It is **not** experimental data. The surrogate will be retrained once bench or ANSYS data exist.

*Last updated: 2026-11-26*

## Status

| Step | Status | Output |
|---|---|---|
| 0. Setup | ✅ Done | `.venv/`, `requirements.txt`, `.gitignore`, `README.md` |
| 1. Physics model | ✅ Done | `src/physics.py`; `tests/test_physics.py`: 33 pass, 1 expected failure |
| 2. Dataset | ✅ Done | `data/synthetic_v1.csv`, `data/README.md` |
| 3. Surrogate | ✅ Done | `src/train.py`, `results/metrics.csv`, figures, `models/` |
| 4. Optimisation | ✅ Done | `src/optimise.py`, `results/optimisation.csv`, `results/pareto_front.csv` |
| 4b. Torque / stiffness constraint | ✅ Done | `physics.mechanics()`, `results/pareto_front_stiffness.csv`, `figures/pareto_tact_stiffness.png` |
| 5. ANSYS validation (replaces the literature check) | ⏳ Prepared, waiting for ANSYS runs (about 1 week) | `ANSYS_RUNS.md`, `data/ansys_run_plan.csv`, `data/ansys_results_template.csv`, `src/ansys_plan.py`, `src/ansys_ingest.py` |
| 6. Multi-fidelity upgrade | ✅ Done (2D FD; ANSYS can slot in later) | `src/fd_model.py`, `src/multifidelity.py`, `data/fd_highfidelity.csv`, `figures/multifidelity_learning_curve.png` |
| 7. Report section and slides | ✅ Drafted (includes step 6) | Report section 5.7.1–5.7.7 (Claude Doc) and a 12-slide deck (Slides artifact); links in chat |

## 0. Setup ✅

- **Runs locally** on this machine (4 cores, 11 GB RAM, Python 3.14). Colab is not needed and remains an option only for sharing.
- **Environment:** `.venv/`. Install with `.venv/bin/pip install -r requirements.txt`.
- **Packages:** numpy, pandas, scipy, scikit-learn, xgboost, shap, optuna, matplotlib, pytest and jupyter.
- **Code layout:** plain Python scripts in `src/`, run from the command line. These replace the notebooks planned originally. Notebooks can import `src/` later if needed for the presentation.
- **Run times:** `generate.py` about 3 s, `train.py` about 8 min, `optimise.py` about 4 min.
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

## 4b. Torque / stiffness constraint ✅

**What was built:**
- `physics.mechanics()` implements eqs. (25)–(28), (31), (33) and (34).
- It reproduces Table 5.6: Stage-1 in full (0.427 N·m/rad, 13.4 mN·m, TM −0.46, 1.20 Hz, 8.38 %), and Stage-2's k_θ,lock (10.87) and f1 (6.06).
- **Convention:** glassy limits for the locked hinge, rubbery limits for recovery, no η_s.

**Inconsistency found in the report:**
- The Stage-2 τ_rec,0 of 141.4 mN·m doesn't follow the Stage-1 rule, which gives 935.6 mN·m. 141.4 is close to η_s = 0.15 (140.3), but Stage-1 doesn't use η_s.
- This is recorded as an `xfail` test.
- **Action for you:** fix Table 5.6 and the "10.5×" claim in the report (details in `results/README.md`).

**Optimiser additions:**
- New constraints: k_θ,lock ≥ baseline and ε_max ≤ 10 %. The mechanics are evaluated exactly.
- New runs: A_mech and B_mech (TPE), plus NSGA-II speed-against-stiffness front D.
- Every reported design is audited with the exact physics.

**Results:**

| Design | t_act | k_θ,lock | Note |
|---|---|---|---|
| Baseline | 38.5 s | 0.427 | |
| C: thermal-only optimum | 16.1 s | 0.092 | Too floppy, about 22 % of the baseline's stiffness |
| **D: best with k ≥ baseline** (b 20.1, t 1.79 mm) | **26.6 s** | **0.483** | 1.45× faster and 13 % stiffer, ΔT_wire 16.4 K, ε 9.4 % |

- **Why D looks like this:** stiffness scales with b·t³ but heat capacity scales with b·t, so the hinge becomes narrower and thicker. The fold-strain limit caps t at 1.91 mm.
- **TPE fell short:** the Pareto search beat both constrained TPE runs (30.1 s and 29.4 s).
- **Neat hinge still can't deploy:** TM ≈ −0.4 and f1 ≈ 1.2–1.3 Hz for every design, which supports Stage-2 reinforcement.

## 5. ANSYS validation ⏳ (prepared; replaces the literature check)

**Why the change:** the cited papers aren't available, and ANSYS runs on our own hinge are a stronger check than fitting other groups' composites. The literature fit below is dropped unless the papers turn up.

**Prepared:**
- **`src/ansys_plan.py` → `data/ansys_run_plan.csv`:** 25 designs, all at P ≤ 10 W. R01 is the baseline and R02 is optimum D; the rest are ordered by farthest-point spacing, so any first-N subset covers the design space evenly. R01–R15 are marked essential. Each row gives geometry, power, heat generation, end time and time step.
- **`ANSYS_RUNS.md`:** the model set-up (geometry parameters and Design Points, materials, loads, mesh check) and exactly what to record.
- **`src/ansys_ingest.py`:** a three-way comparison (ANSYS against the lumped equations against the 2D FD model), leave-one-out errors and the ANSYS speed-up of D. It has been tested with `--demo`, where FD stands in for ANSYS.

**Runs needed (from the FD learning curve):**

| ANSYS runs | Heating-time error with the ML correction |
|---|---|
| 5 | about 6 % (equations alone: 12 %) |
| 15 | about 3 % |
| 25 | about 2.7 % |

**Original plan (dropped):**

- **Needed from you:** digitise 1–3 recovery-angle-versus-time curves with WebPlotDigitizer and save them as `data/digitized/<paper>_<fig>.csv`, with columns `t_s` and `theta_deg`. Candidates: Lan et al. [7], [9], Zhou et al. [10] and Iorio et al. [14].
- **To build:** `src/literature_fit.py`. It fits the bi-exponential law of eq. (30) with `scipy.optimize.curve_fit`, reports R², τ_I and τ_II, and overlays the fit on the data.
- **Viva caveat:** these papers used filler-heated or fibre-reinforced composites. Fitting their data checks the *shape* of eq. (30). It does not calibrate neat-epoxy parameters.

## 6. Multi-fidelity upgrade ✅

**Built:**
- `src/fd_model.py`: a 2D transient finite-difference cell of the heated band, verified against the lumped model (within 1–3 %) and grid-converged (under 0.3 %).
- `src/multifidelity.py`: runs 400 FD designs and builds learning curves.

**Results:**
- Full-band activation takes a median 14 % longer than the lumped t_act (range 4–50 %).
- Eq. (24) overestimates the wire hotspot by about 1.5× on average.
- Δ-learning (lumped model plus an XGBoost correction) reaches 2.7 % error from 25 FD runs. HF-only XGBoost is at 9.3 % even after 300 runs.

Details are in `results/README.md`. The original plan for this step:

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
           pareto_front.csv  pareto_front_stiffness.csv  split_indices.json  README.md
figures/   parity_*, shap_beeswarm_*, shap_bar_*, pareto_tact_hotspot.png, pareto_tact_stiffness.png
models/    trained models (git-ignored; regenerate with train.py)
```

## Order of remaining work

1. 5: run R01–R15 in ANSYS following `ANSYS_RUNS.md` and fill in `data/ansys_results.csv` *(you)*; then run `src/ansys_ingest.py` and add section 5.7.8 and one slide *(Claude)*.
2. Pick one η_s convention and fix the Table 5.6 Stage-2 torque and the "10.5×" claim in the report *(you)*.
3. Set the resin's real wire temperature limit, then rerun steps 2–4 (about 15 min) *(you choose, Claude reruns)*.
4. Before submission *(you)*:
    - verify references [22]–[28] and the existing "[Verify…]" references [10], [13], [15] and [19]
    - share the doc and deck links if others need them.
5. Later, after experiments: bench-test the optimised design against the baseline, swap ANSYS runs in for the 2D FD model, and retrain.
