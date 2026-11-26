# Data card: `synthetic_v1.csv`

> **Synthetic data.** Every row is computed from the lumped electrothermal model in the project report, equations (18) to (24), using literature-based parameters (Table 5.5). No row is a measurement. The data supports an *ML surrogate of a literature-calibrated physics model*. It will be replaced or supplemented once ANSYS or bench data exist.

## Generation

- **Script:** `src/generate.py`. Reproduce the file with `.venv/bin/python src/generate.py --n 5000 --seed 42`.
- **Sampling:** a 7-dimensional Latin hypercube (`scipy.stats.qmc`, seed 42). Designs that fail the geometry check are rejected and resampled until 5,000 remain. The only rule that binds is that the zigzag must fit inside the hinge (a ≤ b − 2 mm). Because of this rejection, the kept set is close to a Latin hypercube but not exact.
- **Validation:** the model reproduces Table 5.6 (Stage-1 neat SMP) exactly; see `tests/test_physics.py`.

## Inputs (design variables)

| Column | Range | Meaning |
|---|---|---|
| `d_w_mm` | 0.10–0.40 | Nichrome wire diameter |
| `p_mm` | 2–6 | Zigzag pitch (run spacing s = p/2) |
| `a_mm` | 10–25 | Zigzag peak-to-peak width |
| `b_mm` | 20–30 | Hinge width |
| `t_mm` | 1–3 | Hinge thickness |
| `V` | 3–12 | DC supply voltage (V) |
| `h` | 5–15 | Natural-convection coefficient (W/m²K) |

**Fixed values:** hinge length l_h = 30 mm, nichrome resistivity ρ_nc = 1.10 µΩ·m, epoxy ρc = 1.725 MJ/m³K, k⊥ = 0.20 W/m·K, T0 = 25 °C, T_act = 90 °C, T_lock = 60 °C, orbit ε = 0.85 and T_ref = 340 K.

## Outputs (lab convection unless the column name says `orbit`)

| Column | Eq. | Meaning |
|---|---|---|
| `L_w_mm`, `R_w_ohm`, `I_A`, `P_W` | 18 | Wire length, resistance, current, power |
| `tau_s`, `T_ss_C` | 21 | Thermal time constant; steady-state temperature with no cut-off |
| `t_act_s`, `t_cool_s`, `E_cyc_J` | 22 | Time to reach 90 °C, time to cool to 60 °C, energy per heating cycle |
| `Bi`, `Fo` | 23 | Biot and Fourier numbers |
| `dT_int_K`, `dT_wire_K` | 24 | Through-thickness temperature gradient; wire hotspot above the mean temperature |
| `T_wire_peak_C` | 24 | 90 °C + `dT_wire_K` |
| `t_act_orbit_s`, `t_cool_orbit_s`, `E_cyc_orbit_J` | 21–22 | The same quantities with radiation-only heat loss (orbit) |

## Flags (all rows are kept; filter by flag when training)

| Flag | Rule | Share of rows |
|---|---|---|
| `reachable` | P > G·(T_act − T0) | 94.9 % |
| `lumped_valid` | Bi < 0.1 and Fo ≥ 1, which eq. (23) requires for the uniform-temperature model to hold | 63.8 % |
| `wire_safe` | `T_wire_peak_C` ≤ 200 °C | 72.1 % |
| `train_ok` | All three flags true | **60.7 % (3,037 rows)** |

When reachable is false, `t_act_s`, `E_cyc_J`, `Fo` and `dT_int_K` are NaN.

## Known limitations

- **The 200 °C wire ceiling is an assumption**, not a datasheet value. Replace it with the TGA/DSC limit of the actual resin and regenerate the data (`T_WIRE_MAX_C` in `generate.py`).
- **The suggested ranges include extreme designs.** At the high-V, thick-wire corner, power reaches about 100 W and the wire peak exceeds 1,000 °C. These designs are not credible, and the flags exclude them.
- **T_ss is the temperature with no cut-off.** In most designs it is far above 90 °C, so constant-voltage heating needs a timer or thermostat cut-off in practice.
- **The model's simplifications carry into the data:** lumped capacitance, constant ρ_nc (drift is under 0.7 % over 65 K), linear convection, and a straight-wire conduction estimate for the hotspot. An ML model trained on this data inherits all of them.
