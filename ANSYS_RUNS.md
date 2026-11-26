# ANSYS Runs for the Multi-Fidelity Surrogate

**Goal:** run the designs in `data/ansys_run_plan.csv` in ANSYS Transient Thermal and record a few numbers per run. These runs replace the 2D finite-difference model as the high-fidelity data. They also check the report's lumped equations directly.

**How many runs:**
- **Minimum:** runs R01–R15, marked `essential`. The heating-time error falls from about 12 % (equations alone) to about 3 % at 15 runs.
- **Ideal:** all 25 runs.
- **If time runs short:** R01 and R02 alone (the baseline and optimum D) already test the main claim, "1.45× faster".
- **Order:** run them in plan order. Any first-N subset is spread evenly across the design space.

## 1. Geometry (one model, driven by parameters)

Model only the hinge: a block of **b × t × l_h**, with l_h = 30 mm, holding a zigzag wire at mid-thickness (Fig. 4.1).

| Parameter | Column in plan | Notes |
|---|---|---|
| Hinge width b | `b_mm` | Across the zigzag |
| Thickness t | `t_mm` | Wire centre at t/2 |
| Wire diameter d_w | `d_w_mm` | Round wire body (nichrome) |
| Zigzag width a | `a_mm` | Long runs are a long, centred across b |
| Run spacing s = p/2 | `s_mm` | Number of runs: `n_runs` |

- **Make these ANSYS parameters** in SpaceClaim or DesignModeler (or SolidWorks with the parameters linked). Then each run is one **Design Point** in the Workbench Parameter Set, and you don't redraw anything.
- **Split the epoxy into three bodies:** the **band** (width a, centred, holding the zigzag) and two side **margins**. Use *Share Topology* so heat flows freely between them. The split exists only so you can read results over the band separately.
- **Simplification allowed:** model the zigzag as straight runs joined by short end links. The heat input is set from total power (below), so small geometry details don't matter.

## 2. Materials

| Body | Density (kg/m³) | Specific heat (J/kg·K) | Conductivity (W/m·K) |
|---|---|---|---|
| Epoxy (band and margins) | 1150 | 1500 | 0.20 |
| Nichrome wire | 8400 | 450 | 11.3 |

These are the values used in every other model in the project. Keep them identical, so that any difference you find comes from the geometry and physics, not the inputs.

## 3. Loads and boundary conditions

| Item | Setting |
|---|---|
| Initial temperature | 25 °C everywhere |
| Heating | **Internal Heat Generation** on the wire body = `P_W` / (wire volume from your CAD). Read the wire volume from the body's Properties in Mechanical; don't use the plan's `q_gen_W_m3`, which assumes the nominal length. |
| Convection | Film coefficient `h` from the plan, ambient 25 °C, on the **top and bottom faces only**. Leave the side and end faces adiabatic, to match the report's model (A_s = 2·b·l_h). |
| Analysis | Transient Thermal, end time `end_time_s`, maximum step `max_step_s`, auto time stepping on |

**Mesh:** at least 3 elements across the wire diameter and at least 6 through the thickness; inflation round the wire helps. Run R01 twice, once with the mesh refined about 2×. If t_act changes by more than 2 %, keep the finer mesh for all runs.

## 4. What to record (per run, into `data/ansys_results.csv`)

Copy `data/ansys_results_template.csv` to `data/ansys_results.csv` and fill in one row per run. A blank cell is fine for anything you can't get.

| Column | How to get it in Mechanical | Needed? |
|---|---|---|
| `t_act_band_min_s` | Temperature result scoped to the **band** body → Tabular Data → **Minimum** column. Record the time when it first reaches 90 °C (interpolate between steps). | **Yes** |
| `t_act_mean_s` | Temperature result scoped to **all epoxy** bodies → **Average** column → time it reaches 90 °C | Strongly recommended (checks eq. 22 directly) |
| `T_max_at_act_C` | **Maximum** temperature, all bodies, at time `t_act_band_min_s` | Recommended (hotspot check) |
| `t_act_hinge_min_s` | **Minimum** over **all epoxy**, including margins, reaching 90 °C. Leave blank if it never gets there before the end time. | Optional |
| `dT_hot_K` | Peak epoxy temperature minus epoxy average, at time `t_act_mean_s` | Optional |
| `notes` | Mesh size, anything odd | Optional |

**Screenshots worth saving for the report:** the temperature contour for R01 and R02 at `t_act_band_min_s`, and a min/average/max against time chart for R01.

## 5. Then run

```bash
.venv/bin/python src/ansys_ingest.py
```

This writes `results/ansys_validation.md` and `.csv`, containing:
- ANSYS against the lumped equations against the 2D FD model, run by run
- leave-one-out error of three predictors: equations alone, a constant correction, and the ML correction
- the ANSYS speed-up of optimum D over the baseline

To dry-run the whole pipeline without ANSYS data, use `src/ansys_ingest.py --demo`, which uses 2D FD results in place of ANSYS. That output is labelled DEMO.
