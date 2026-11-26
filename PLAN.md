# ML Surrogate for the Zigzag Nichrome SMP Heater: Plan

**What this is:** an ML surrogate trained on synthetic data from a physics model calibrated to the literature, namely the report's equations (18) to (24) and Table 5.5. It is **not** experimental data. The surrogate will be retrained once bench or ANSYS data exist.

## 0. Setup (Colab and local)

**Decision:** run locally on this machine (4 cores, 11 GB RAM, Python 3.14). The virtual environment is `.venv/`. Install with `.venv/bin/pip install -r requirements.txt` and test with `.venv/bin/python -m pytest tests`. Colab stays an option for sharing only.

- **Code lives in one place:** `src/` as plain Python modules, plus notebooks that import them.
- **Running on Colab** (pick one):
  1. Upload this folder to Google Drive. In Colab, run `from google.colab import drive; drive.mount('/content/drive')`, then `%cd` into the folder. This is the simplest option and keeps outputs in Drive.
  2. Push the folder to a GitHub repo. In Colab, `!git clone` it. This is the cleanest option for version history.
  3. Use the Google Colab extension for VS Code to run the local `.ipynb` on a Colab kernel. Note that it still can't see local files unless they are in Drive or git.
- **Requirements:** `requirements.txt` lists numpy, pandas, scipy, scikit-learn, xgboost, shap, optuna and matplotlib.
- **Compute:** the whole pipeline (5,000 samples plus XGBoost) runs on a laptop CPU in under a minute. Colab is for convenience and sharing, not for speed.

## 1. Physics model (`src/physics.py`)

Code these as vectorised NumPy functions and unit-test them against Table 5.6. Using the baseline inputs, the code must reproduce Lw = 401.1 mm, Rw = 14.04 Ω, P = 4.01 W, τh = 138 s, t_act = 38.5 s, t_cool = 85.4 s and ΔT_wire = 16.0 K. A hand check already matches these.

| Eq. | Output |
|---|---|
| (18) | Lw, Rw, I, P |
| (19)–(21) | Ch, G (convection or radiation), τh |
| (22) | t_act, t_cool, E_cyc |
| (23) | Bi, Fo |
| (24) | ΔT_int, ΔT_wire |
| extra | T_ss = T0 + P/G (steady state with no cut-off), current density J = I/Aw |

**Points to handle explicitly:**
- **Unreachable activation temperature.** If P ≤ G(T_act − T0), the hinge never reaches 90 °C and t_act is undefined. Store `reachable = False` for these cases. Don't drop them silently.
- **Runaway steady state.** At baseline, T_ss is about 292 °C, so constant power without a cut-off will overheat the epoxy. Report this. It also argues for the feedback control listed in Future Scope 3.
- **Geometric validity.** Require a ≤ b − 2 mm (the zigzag must fit inside the hinge width), p > 2·dw and t > dw (room for the channel). Either reject samples that break these or sample b too.

## 2. Dataset (`src/generate.py` → `data/synthetic_v1.csv`)

Use Latin hypercube sampling (`scipy.stats.qmc.LatinHypercube`) with N = 5,000 and a fixed seed.

| Input | Range | Note |
|---|---|---|
| dw | 0.10–0.40 mm | Common nichrome gauges |
| p | 2–6 mm | |
| a | 10–25 mm | Needs b ≥ a + 2 mm, so sample b over 20–30 mm or tie it to a |
| t | 1–3 mm | |
| V | 3–12 V | |
| h | 5–15 W/m²K | Natural convection |
| lh | fixed 30 mm (or 20–40 mm) | |

- **Outputs:** t_act, t_cool, ΔT_wire, E_cyc, Bi, P, reachable.
- **Data card:** save one in `data/README.md` covering the source equations, ranges, seed and the "synthetic" disclaimer.

## 3. Surrogate (`notebooks/02_train.ipynb`)

- Use an 80/20 split with 5-fold CV on the training set.
- **Classifier:** predicts `reachable`.
- **Regressors:** predict t_act on reachable rows, plus ΔT_wire, t_cool and E_cyc. Use log(t_act) as the target because its distribution is heavily skewed.
- **Models:** compare a Random Forest with XGBoost. Report R², RMSE and MAE on the test set, with parity plots.
- **SHAP:** produce a beeswarm plot and bar importance for t_act and ΔT_wire. These are the slide figures.
- **Honesty check:** the target is a closed-form function, so R² will be about 0.99+. Say plainly that this proves the surrogate *can* learn the mapping, not that the physics is right (see step 6).

## 4. Optimisation (`notebooks/03_optimise.ipynb`)

Run Optuna (TPE, about 500 trials) on the surrogate.

- **Objective:** minimise t_act.
- **Constraints:** ΔT_wire < 30 K, P ≤ 5 W, `reachable`, plus the geometry checks.
- **Verify the result:** rerun the best design through `physics.py`, which is the ground truth, and report both values.
- **Compare with baseline:** put the optimum next to the baseline (dw = 0.20 mm, p = 3 mm, 7.5 V) in a table.
- **Pareto front:** optionally run NSGA-II for t_act against E_cyc against ΔT_wire.

## 5. Literature check (`notebooks/04_literature_fit.ipynb`)

- Digitise curves with WebPlotDigitizer and save them to `data/digitized/<paper>_<fig>.csv`.
- Fit the bi-exponential form of eq. (30) (`scipy.optimize.curve_fit`), then report R² and the fitted τI and τII.
- **Caveat for the viva:** Zhou et al. [10] used a filler-heated composite and Lan et al. [7], [9] used fibre-reinforced SMPC. Fitting their data checks the *shape* of eq. (30). It does not calibrate our neat-epoxy parameters.

## 6. Recommended upgrade: make the ML do real work

Expect the examiner to ask: *"Why train ML on a formula you can just evaluate?"* Have two answers ready:

- **(a) Multi-fidelity.** Run a cheap 2D transient finite-difference conduction model of the wire cross-section (or a few dozen ANSYS Transient Thermal runs once available) to capture non-lumped effects, such as hotspots and gradients that eq. (24) only bounds. Then train the surrogate on, or correct it with, that data. This is where a surrogate pays off.
- **(b) Retraining path.** The same pipeline swaps in measured T(t) data from the bench tests in §5.6.

## 7. Folder layout

```
PLAN.md
requirements.txt
src/physics.py  src/generate.py
tests/test_physics.py         # Table 5.6 regression test
notebooks/01_dataset.ipynb 02_train.ipynb 03_optimise.ipynb 04_literature_fit.ipynb
data/  figures/  models/
```

## 8. Order of work

1. Write `physics.py` and the Table 5.6 test, then pass the test.
2. Generate the dataset and the data card.
3. Train the surrogate, then produce metrics and SHAP figures.
4. Run Optuna and compare against the baseline.
5. Digitise and fit the literature curves.
6. (Optional) Build the FD or ANSYS multi-fidelity data.
7. Assemble the report section and slides.
