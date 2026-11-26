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

- **The optimum sits on the edges of the search space** (t = 1 mm and the lower limit of b). A wider range would push it further. The fold itself is fine: by eq. (28), a thinner hinge *lowers* fold strain (5.2 % at t = 1.0 mm against 8.4 % at the 1.6 mm baseline). The real cost is mechanical. Stiffness and recovery torque scale with t³, so going from 1.6 mm to 1.0 mm cuts them to about 24 % of the baseline (eq. 27), and the neat hinge already fails the torque margin (Table 5.6). The optimiser covers only the thermal side, so a torque or stiffness constraint is needed before using this design for deployment.
- **These are optima of the synthetic lumped model**, not measured performance. Test design C on the bench (or in ANSYS) next to the baseline.
