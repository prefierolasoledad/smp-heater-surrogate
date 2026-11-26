# smp-heater-surrogate

This project trains an ML model to predict how a shape-memory epoxy actuator heats up, and uses it to optimise the actuator's heater design. The heater is a zigzag nichrome wire. Because no lab or ANSYS data exists yet, the model is trained on **synthetic data generated from a physics model calibrated to published literature**.

It is the ML part of the B.Tech minor project *Design and Electrothermal Characterization of a Shape-Memory Epoxy Actuator with a Zigzag Nichrome Heater* (ARP-455, University School of Automation and Robotics, GGSIPU).

> **About the data:** every number in this repository comes from the lumped electrothermal model in the project report (equations 18–24, Table 5.5), not from measurements. The ML results show that a surrogate model can learn and optimise that physics model. They do not show that the physics model matches reality; that needs ANSYS or bench data, and the pipeline is built to be retrained on such data.

## Results at a glance

| Item | Result |
|---|---|
| Physics model | Reproduces report Table 5.6, thermal and mechanical (35 tests pass; the report's Stage-2 torque inconsistency is resolved with η_s = 0.15 for both stages) |
| Dataset | 5,000 Latin hypercube designs; 3,037 pass the validity checks |
| Surrogate (XGBoost, held-out test set) | Heating time R² 0.987, wire hotspot 0.985, cooling time 0.999, energy 0.991 |
| Physics-informed features | Raise heating-time R² from 0.77 (raw inputs only) to 0.99 |
| Thermal-only optimum (P ≤ 5 W, hotspot < 30 K) | 16.1 s against 38.5 s for the baseline, but only 22 % of the baseline stiffness |
| **Optimum with stiffness ≥ baseline** | **26.6 s against 38.5 s (1.45× faster)**, 13 % stiffer, 14 % less energy, 16.4 K hotspot |
| Multi-fidelity (2D FD + lumped) | Δ-learning reaches 2.7 % error from just 25 high-fidelity runs; HF-only ML needs more than 300 runs to get below 10 % |

Details are in [results/README.md](results/README.md).

<p align="center">
  <img src="figures/parity_xgb_physics.png" width="100%" alt="Surrogate predictions against physics-model values"><br>
  <img src="figures/shap_beeswarm_t_act_s_raw.png" width="48%" alt="SHAP importance for heating time">
  <img src="figures/pareto_tact_stiffness.png" width="44%" alt="Heating speed against hinge stiffness trade-off">
</p>

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Physics model | `src/physics.py` | Report eqs. (18)–(24): wire resistance and power, lumped heating and cooling, Biot and Fourier checks, wire hotspot, for lab (convection) and orbit (radiation). Eqs. (25)–(34): matrix softening, hinge stiffness, recovery torque, torque margin, deployed frequency, fold strain |
| 2. Dataset | `src/generate.py` | Latin hypercube over 7 design inputs → `data/synthetic_v1.csv`, with validity flags |
| 3. Surrogate | `src/train.py` | Feasibility classifier, plus Random Forest and XGBoost regressors on raw and physics-informed features; SHAP explanations |
| 4. Optimisation | `src/optimise.py` | Optuna TPE on the surrogate, checked against the exact physics; NSGA-II Pareto fronts for speed against hotspot and speed against stiffness; optional stiffness and fold-strain constraints |
| 5. Literature check | *planned* | Fit the recovery law (eq. 30) to curves digitised from the cited papers |
| 6. Multi-fidelity | `src/fd_model.py`, `src/multifidelity.py` | 2D transient finite-difference model of the heated band; Δ-learning on top of the lumped model, compared with HF-only ML |

### Design inputs and ranges

| Input | Range |
|---|---|
| Wire diameter | 0.10–0.40 mm |
| Zigzag pitch | 2–6 mm |
| Zigzag width | 10–25 mm |
| Hinge width | 20–30 mm |
| Hinge thickness | 1–3 mm |
| Supply voltage | 3–12 V |
| Convection coefficient h | 5–15 W/m²K |

Hinge length is fixed at 30 mm.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest tests      # physics model must reproduce Table 5.6
.venv/bin/python src/generate.py      # ~3 s   -> data/synthetic_v1.csv
.venv/bin/python src/train.py         # ~8 min -> models/, figures/, results/metrics.csv
.venv/bin/python src/optimise.py      # ~4 min -> results/optimisation.csv, pareto_front*.csv
.venv/bin/python src/multifidelity.py # ~1 min -> data/fd_highfidelity.csv, results/multifidelity.csv
```

The commands above were tested with Python 3.14 on a 4-core laptop CPU; no GPU is needed. Trained models are not committed (about 250 MB), so run `train.py` before `optimise.py`.

## Repository layout

```
src/        physics.py · generate.py · train.py · optimise.py
tests/      test_physics.py       regression test against report Table 5.6
data/       synthetic_v1.csv + README.md (data card: sources, ranges, flags, limitations)
results/    metrics.csv · optimisation.csv · pareto_front.csv · README.md (findings)
figures/    parity plots, SHAP beeswarm/bar plots, Pareto chart
PLAN.md     full project plan and remaining steps
```

## Known limitations

- **Physics model.** It uses lumped capacitance, constant nichrome resistivity, linear convection and a straight-wire conduction estimate for the hotspot. The surrogate inherits all of these assumptions.
- **Wire temperature limit.** The 200 °C wire-safe limit is an assumption. Replace it with the TGA/DSC limit of the actual resin (`T_WIRE_MAX_C` in `src/generate.py`).
- **Heater cut-off.** Optimised designs have steady-state temperatures of about 410–435 °C, so they need a thermostat or timed cut-off.
- **Neat epoxy can't deploy a panel.** Even with stiffness held at the baseline, every neat-epoxy design has a negative torque margin (TM ≈ −0.9 with η_s = 0.15) and f1 ≈ 1.2–1.3 Hz. No heater design fixes this; it needs the Stage-2 carbon-fibre hinge.
- **η_s = 0.15 is assumed.** It is the lower end of the report's range, applied to both stages (this corrects Table 5.6). Measure it with a load-cell recovery test.

## Next steps

1. Fit eq. (30) to recovery curves digitised from the cited papers (step 5).
2. Replace the 2D FD high-fidelity runs with ANSYS Transient Thermal runs in the same multi-fidelity pipeline; about 25–50 runs should be enough.
3. Retrain the surrogate on measured bench data once tests begin.

## Authors

Karan Bhatt and Nizaam Khan, B.Tech Automation & Robotics, USAR, GGSIPU. Supervised by Prof. Pushp Kr Baghel.

All equations and parameter values are cited in the project report; key sources include Incropera et al. (heat transfer), Liu et al. 2006 (SMP constitutive model), Lan et al. 2009 (SMPC hinges) and Momeni & Ni 2020 (laws of 4D printing).
