# Step 3 results: ML surrogate (synthetic data)

Run `.venv/bin/python src/train.py`. It takes about 8 minutes on 4 CPU cores. Full numbers are in `metrics.csv`.

**Setup**
- **Training data:** 3,037 rows where `train_ok` is true, split 80/20 with seed 42. The split indices are in `split_indices.json`.
- **Targets:** fitted in log space and scored in real units.
- **Cross-validation:** CV R² is the 5-fold result on the training set, in log space.

## Test-set R², XGBoost

| Target | Raw inputs (7) | + physics features (11) |
|---|---|---|
| Heating time t_act | 0.771 | **0.987** (MAPE 2.8 %) |
| Wire hotspot ΔT | 0.960 | **0.985** (MAPE 5.1 %) |
| Cooling time t_cool | 0.998 | **0.999** (MAPE 0.9 %) |
| Energy per cycle | 0.793 | **0.991** (MAPE 1.0 %) |
| Feasibility classifier (`train_ok`) | AUC 0.993 | **AUC 0.997**, accuracy 97 % |

XGBoost beat the Random Forest on almost every target. Details are in `metrics.csv`.

## Findings

1. **Why the raw-input model scores low.** About 3 % of the test designs have just enough power to reach 90 °C (P ≈ 1.0–1.2 × the heat loss). For these, t_act follows −τ·ln(1 − 1/ratio) and rises extremely steeply. They account for 90 % of the squared error. Leaving them out, the raw-input R² is 0.93.
2. **Physics-informed features fix it.** The extra features are heater power P, loss conductance G, heat capacity C_h and the power ratio P/(G·ΔT). All four come from the design inputs alone, using eqs. (18)–(21), so no outputs leak into the inputs. The model then only has to learn eqs. (22)–(24). Combining physics with ML this way is a recognised hybrid approach and a good viva point.
3. **The SHAP results match the physics** (`figures/shap_beeswarm_t_act_s_raw.png`):
   - Wire diameter and voltage dominate heating time, because both raise the power.
   - A thicker hinge heats more slowly, because it has more heat capacity.
   - A larger pitch heats faster: the wire is shorter, so resistance is lower and power is higher at a fixed voltage.
   - A wider zigzag heats more slowly, because the longer wire means lower power.
   - Convection h matters least, which is expected while P is much larger than G·ΔT.
4. **Cooling time ignores the electrical inputs.** The model gives voltage and wire diameter roughly zero importance for cooling time, as eq. (22) predicts. This is a useful check that the model hasn't learned spurious patterns.

## Caveat (state this in the report and viva)

These scores show that the surrogate reproduces the *physics model*. They do not validate the physics model itself. That needs ANSYS runs or bench data (PLAN.md, step 6).

## Figures (`figures/`)

- `parity_xgb_{raw,physics}.png`: predicted against true values, on log axes
- `shap_beeswarm_<target>_{raw,physics}.png` and `shap_bar_<target>_{raw,physics}.png`

---

# Step 4 results: design optimisation

Run `.venv/bin/python src/optimise.py`. It takes about 2 minutes.

**Problem**
- **Objective:** minimise the time to reach 90 °C.
- **Constraints:**
  - wire hotspot ΔT < 30 K
  - P ≤ 5 W
  - valid geometry
  - feasibility probability ≥ 0.9 from the classifier
  - Fo ≥ 1, so the lumped model stays valid
- **Design variables:** d_w, p, a, b, t and V, all within the training ranges.
- **Fixed:** h = 10 W/m²K, which describes the environment rather than the design.

**Outputs:** `optimisation.csv`, `optimisation_check.json`, `pareto_front.csv` and `../figures/pareto_tact_hotspot.png`.

| Design | d_w (mm) | p (mm) | a (mm) | b (mm) | t (mm) | V | P (W) | t_act (s) | ΔT_wire (K) | E_cyc (J) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (Table 5.5) | 0.20 | 3.00 | 20.0 | 25.0 | 1.60 | 7.50 | 4.01 | 38.5 | 16.0 | 154 |
| A: surrogate, Optuna TPE (500 trials) | 0.28 | 2.37 | 15.1 | 20.7 | 1.11 | 5.80 | 4.95 | 17.1 | 14.8 | 85 |
| C: fastest surrogate Pareto design | 0.40 | 2.04 | 19.2 | 21.5 | 1.01 | 5.02 | 4.97 | **16.1** | **6.6** | 80 |
| B: exact-physics search (reference) | 0.25 | 3.62 | 16.8 | 20.2 | 1.00 | 5.66 | 4.97 | 15.0 | 28.2 | 74 |

All values in the table come from the exact physics model (`physics.py`), including the rows the surrogate selected.

## Findings

1. **The surrogate is accurate at its own optimum.** For design A it predicted t_act = 17.25 s and ΔT = 15.1 K. The exact model gives 17.11 s and 14.8 K, an error of under 1 % and 2 %.
2. **About 2.4× faster than the baseline.** Design C heats in 16.1 s instead of 38.5 s, cuts the hotspot from 16.0 K to 6.6 K and uses half the energy per cycle (80 J instead of 154 J).
3. **Close to the true optimum.** The surrogate's best design is within 8 % of the exact-physics best (16.1 s against 15.0 s), and its hotspot is far lower.
4. **Power limits speed, not the hotspot.** Every fast design uses the full power budget (P ≈ 5 W) with the thinnest, narrowest hinge allowed (t ≈ 1.0 mm, b ≈ 20–21 mm), which minimises heat capacity. Every Pareto design has ΔT_wire < 15 K, well inside the 30 K limit. A thick wire at low voltage and a tight pitch spread the heat over a longer wire, which keeps the hotspot low.
5. **Every fast design needs a cut-off.** Their steady-state temperature without one is about 410–435 °C, compared with 292 °C for the baseline. A thermostat or timed cut-off is therefore essential (Future Scope 3).

## Caveats

- **The optimum sits on the edges of the search space** (t = 1 mm and the lower limit of b). A wider range would push it further. The fold itself is fine: by eq. (28), a thinner hinge *lowers* fold strain (5.2 % at t = 1.0 mm against 8.4 % at the 1.6 mm baseline). The real cost is mechanical. Stiffness and recovery torque scale with t³, so going from 1.6 mm to 1.0 mm cuts them to about 24 % of the baseline (eq. 27), and the neat hinge already fails the torque margin (Table 5.6). Step 4b adds the stiffness constraint; see below.
- **These are optima of the synthetic lumped model**, not measured performance. Test design C on the bench (or in ANSYS) next to the baseline.

---

# Step 4b results: mechanical constraint (stiffness and strain)

**New code**
- `physics.mechanics()` implements report eqs. (25)–(28), (31), (33) and (34).
- It reproduces Table 5.6 for the Stage-1 neat SMP hinge: k_θ,lock 0.427 N·m/rad, τ_rec,0 13.4 mN·m, TM −0.46, f1 1.20 Hz, ε_max 8.38 %.
- It also reproduces Stage-2 k_θ,lock (10.87 N·m/rad) and f1 (6.06 Hz).
- `tests/test_physics.py` now has 30 passing tests and 1 expected failure (see below).

**Moduli convention that reproduces Table 5.6:** the locked hinge uses the fully glassy limits (E_g, E_f) and recovery uses the fully rubbery limits (E_r, E_f,act), with no η_s. Using E_m(60 °C) = 1.449 GPa instead of E_g would lower k_θ,lock by 3.5 %.

## ⚠ Inconsistency found in report Table 5.6

The Stage-2 (CFR-SMPC) initial recovery torque of **141.4 mN·m cannot be reproduced** with the rule that reproduces Stage-1. That rule (rubbery moduli, no η_s) gives **935.6 mN·m** for Stage-2. The closest match is 140.3 mN·m, using η_s = 0.15, but Stage-1's 13.4 mN·m is computed without η_s.

The two stages need one consistent convention, and fixing it changes the Stage-2 figures in the table:
- **Without η_s:** τ_rec,0 = 935.6 mN·m and TM = +36.4. The 10.5× claim becomes about 70×.
- **With η_s = 0.15 for both stages:** Stage-1 drops to 2.0 mN·m (TM −0.92) and Stage-2 is about 140 mN·m (TM about +4.6).

Either way the conclusion holds (the neat hinge fails and the CFR hinge passes), but Table 5.6 and the "10.5×" text should be corrected. The test is marked `xfail` until this is resolved.

## Optimisation with mechanical constraints (neat hinge)

**Constraints added:**
- locked stiffness k_θ,lock ≥ baseline (0.427 N·m/rad); for the neat hinge this also keeps τ_rec,0 ≥ baseline
- fold strain ε_max ≤ 10 % (eq. 28)

**Evaluation:** the mechanics are closed-form, so they are evaluated exactly, not through the surrogate.

| Design | b × t (mm) | t_act (s) | ΔT_wire (K) | E_cyc (J) | k_θ,lock (N·m/rad) | τ_rec,0 (mN·m) | f1 (Hz) | ε_max | Meets thermal / mech |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 25.0 × 1.60 | 38.5 | 16.0 | 154 | 0.427 | 13.4 | 1.20 | 8.4 % | ✓ / ✓ |
| C: thermal-only optimum (step 4) | 21.5 × 1.01 | 16.1 | 6.6 | 80 | 0.092 | 2.9 | 0.56 | 5.3 % | ✓ / ✗ |
| A_mech: surrogate TPE | 24.6 × 1.62 | 30.1 | 20.7 | 149 | 0.437 | 13.7 | 1.22 | 8.5 % | ✓ / ✓ |
| B_mech: exact-physics TPE | 23.7 × 1.66 | 29.4 | 28.8 | 147 | 0.456 | 14.3 | 1.24 | 8.7 % | ✓ / ✓ |
| **D: best design on the speed–stiffness front** | **20.1 × 1.79** | **26.6** | **16.4** | **132** | **0.483** | **15.2** | **1.28** | **9.4 %** | **✓ / ✓** |

Design D's heater: d_w = 0.23 mm, p = 2.39 mm, a = 15.8 mm, V = 7.25 V, P = 4.97 W.

## Findings

1. **The thermal-only optimum is mechanically unusable.** Design C heats 2.4× faster, but its stiffness and torque are only about 22 % of the baseline's (k_θ,lock 0.092 against 0.427).
2. **With stiffness held at baseline or better, the honest gain is 1.45× faster.** Design D heats in 26.6 s against 38.5 s. It is also 13 % stiffer, has a similar hotspot (16.4 K) and uses 14 % less energy. This is the result to quote.
3. **How the trade-off works.** Stiffness scales with b·t³, while heat capacity, and so heating time, scales with b·t. The optimiser therefore makes the hinge narrower (b → 20 mm) and slightly thicker (t → 1.8 mm), gaining stiffness from t³ for little extra heat capacity. The 10 % fold-strain limit caps t at 1.91 mm. Design D sits at ε = 9.4 %, close to that limit.
4. **The Pareto search beat both constrained TPE runs.** NSGA-II found 26.6 s against 30.1 s (surrogate TPE) and 29.4 s (exact-physics TPE). TPE with 500 trials hasn't converged under the extra constraints. Design D was re-checked against every constraint with the exact physics.
5. **The surrogate is still accurate:** 30.21 s predicted against 30.08 s exact at A_mech, an error of 0.4 %.
6. **Deployment is still out of reach for neat epoxy.** All neat designs keep TM ≈ −0.4 (recovery torque about 15 mN·m against 25 mN·m drag) and f1 ≈ 1.2–1.3 Hz, below the 1.5–2.0 Hz ADCS limit. No heater design fixes this, which supports the report's case for Stage-2 carbon-fibre reinforcement.

**Figure:** `../figures/pareto_tact_stiffness.png` (speed against stiffness, with the baseline, design C, A_mech and design D marked).

---

# Step 6 results: multi-fidelity surrogate (lumped + 2D finite-difference)

Run `.venv/bin/python src/multifidelity.py`. It takes about 1 minute; the 400 FD runs are cached in `data/fd_highfidelity.csv`.

## High-fidelity model (`src/fd_model.py`)

The FD model is a 2D transient conduction solver for one repeating cell of the heated band.

**Geometry**
- **Plane:** x along the hinge, z through the thickness.
- **Wire:** one wire run at mid-thickness, modelled as nichrome cells. Quarter-cell symmetry is used.
- **Boundaries:** convection h on the surface; the other faces are symmetry planes.
- **Heating:** the wire carries its true line power, P/L_w.

**Numerical method:** backward Euler on a cell-centred grid with up to 80 cells per side.

**Verification** (`tests/test_physics.py`: 33 pass, 1 expected failure):
- The cell's mean temperature reaches 90 °C within 1–3 % of the lumped model applied to the same band. The gap comes mostly from the wire's own heat capacity, which the lumped model leaves out.
- Grid and time-step convergence: refining both 4× changes t_act by less than 0.3 % and ΔT_hot by less than 0.5 %.

**Scope:** like eqs. (19)–(22) applied to the band, the cell ignores the unheated side margins (b − a) and any variation along the wire run.

## Where the lumped equations are wrong (400 designs)

| FD / lumped ratio | Median | 5–95 % range | Meaning |
|---|---|---|---|
| Mean-temperature activation time | 1.007 | 0.99–1.06 | Energy balance consistent |
| **Coldest-point activation time** | **1.14** | **1.04–1.31** (max 1.50) | The whole band takes 4–50 % longer to pass 90 °C than the lumped t_act |
| **Wire hotspot ΔT** | **0.68** | **0.47–0.90** (range 0.33–1.16) | Eq. (24) overestimates the hotspot by about 1.5× on average, and underestimates it for some thin-wire designs |

## Multi-fidelity learning (100 held-out FD designs, mean ± sd over 5 subsets)

| FD training runs | Coldest-point time: HF-only XGBoost | Coldest-point time: Δ-learning | Hotspot: HF-only XGBoost | Hotspot: Δ-learning |
|---|---|---|---|---|
| 0 (lumped equations alone) | 12.4 % | — | 51.3 % | — |
| 25 | 37.6 ± 5.5 % | **2.7 ± 0.3 %** | 47.2 ± 8.9 % | **10.3 ± 1.7 %** |
| 50 | 23.6 ± 4.9 % | **2.2 ± 0.2 %** | 34.9 ± 2.8 % | **7.3 ± 0.7 %** |
| 100 | 17.4 ± 1.2 % | **1.8 ± 0.1 %** | 24.6 ± 0.6 % | **5.1 ± 0.4 %** |
| 300 | 9.3 ± 0.2 % | **1.0 ± 0.0 %** | 15.4 ± 0.4 % | **3.1 ± 0.1 %** |

Δ-learning is an XGBoost model that predicts the correction log(HF/LF), with the lumped prediction supplied as an input.

**Figure:** `../figures/multifidelity_learning_curve.png`.

## Findings

1. **Δ-learning is 4–14× more accurate than learning from the high-fidelity data alone, at every training size.** With only 25 FD runs it reaches 2.7 % error on heating time. HF-only XGBoost needs more than 300 runs to get below 10 %, and it does worse than the bare lumped equations until about 200 runs.
2. **This is the answer to "why ML?"** The physics equations carry most of the structure, and ML learns only what they miss. That keeps the number of expensive simulations small: ANSYS runs replace the FD model in exactly the same pipeline, and about 25–50 of them should be enough.
3. **The lumped model is optimistic about activation, but the step 4b speed-up survives.** Across the 400 designs, full-band activation takes a median 14 % longer than the lumped-band t_act. The penalty is largest for wide pitch with a thin hinge, where heat has the furthest to travel between runs. For the two designs that matter, the FD coldest point reaches 90 °C in 32.2 s (baseline) and 22.7 s (design D): **1.42× faster**, against 1.45× in the lumped model. These FD times are not directly comparable with the step 4 values (38.5 s, 26.6 s), which also heat the unheated side margins (b − a).
4. **Eq. (24) is conservative for most designs.** It overestimates the hotspot by about 1.5× on average, so the 30 K hotspot constraint used in step 4 was stricter than necessary.

## Caveats

- **The FD model is a higher-fidelity model, not reality.** It is 2D, uses constant properties and leaves out the side margins and contact resistance. ANSYS 3D or bench data remain the validation targets.
- **The step 4 optimisation still uses the lumped surrogate.** Re-running it with the Δ-corrected model is a possible extension.
