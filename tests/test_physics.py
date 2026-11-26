"""Regression test: the code must reproduce Table 5.6 (Stage-1 neat SMP)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from physics import BASELINE, evaluate, geometry_valid  # noqa: E402

lab = evaluate(**BASELINE, environment="lab")
orb = evaluate(**BASELINE, environment="orbit")


@pytest.mark.parametrize("key,expected,tol", [
    ("L_w", 401.1e-3, 0.1e-3),
    ("R_w", 14.04, 0.01),
    ("I", 0.53, 0.005),
    ("P", 4.01, 0.01),
    ("C_h", 2.07, 0.005),
    ("G", 15.0e-3, 0.05e-3),
    ("tau", 138.0, 0.5),
    ("t_act", 38.5, 0.1),
    ("t_cool", 85.4, 0.2),
    ("Bi", 0.040, 0.001),
    ("Fo", 6.96, 0.05),
    ("dT_int", 4.7, 0.05),
    ("dT_wire", 16.0, 0.1),
])
def test_lab_table_5_6(key, expected, tol):
    assert float(lab[key]) == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("key,expected,tol", [
    ("G", 11.4e-3, 0.05e-3),
    ("tau", 181.6, 1.0),
    ("t_act", 37.2, 0.2),
    ("t_cool", 112.4, 0.5),
    ("E_cyc", 149.0, 1.0),
])
def test_orbit_table_5_6(key, expected, tol):
    assert float(orb[key]) == pytest.approx(expected, abs=tol)


def test_unreachable_flagged():
    r = evaluate(**{**BASELINE, "V": 3.0})   # ~0.64 W < G*(Tact-T0) = 0.975 W
    assert not bool(r["reachable"]) and np.isnan(r["t_act"])


def test_geometry():
    assert geometry_valid(BASELINE["d_w"], BASELINE["p"], BASELINE["a"],
                          BASELINE["t"], BASELINE["b"])
    assert not geometry_valid(0.2e-3, 3e-3, 25e-3, 1.6e-3, 25e-3)  # zigzag too wide


# --- Mechanics, eqs. (25)-(28), (31), (33), (34) -------------------------
from physics import J_P, frozen_fraction, matrix_modulus, mechanics  # noqa: E402

B, T, LH = BASELINE["b"], BASELINE["t"], BASELINE["l_h"]
s1 = mechanics(B, T, LH)                 # Stage-1 neat SMP
s2 = mechanics(B, T, LH, t_f=0.20e-3)    # Stage-2 CFR-SMPC


def test_eq25_transition():
    assert float(frozen_fraction(70.0)) == pytest.approx(0.5)
    assert float(matrix_modulus(40.0)) == pytest.approx(1.5e9, rel=1e-3)
    assert float(matrix_modulus(100.0)) == pytest.approx(15e6, rel=1e-2)


def test_panel_inertia():
    assert J_P == pytest.approx(7.50e-3)


@pytest.mark.parametrize("key,expected,tol", [
    ("k_lock", 0.427, 0.001),
    ("tau_rec0", 13.4e-3, 0.05e-3),
    ("TM", -0.46, 0.01),
    ("f1", 1.20, 0.01),
    ("eps_max", 0.0838, 0.0001),
])
def test_stage1_mechanics_table_5_6(key, expected, tol):
    assert float(s1[key]) == pytest.approx(expected, abs=tol)


@pytest.mark.parametrize("key,expected,tol", [
    ("k_lock", 10.87, 0.01),
    ("f1", 6.06, 0.01),
    ("eps_max", 0.0838, 0.0001),
])
def test_stage2_mechanics_table_5_6(key, expected, tol):
    assert float(s2[key]) == pytest.approx(expected, abs=tol)


@pytest.mark.xfail(strict=True, reason=(
    "Table 5.6 lists 141.4 mN.m for Stage-2, but the Stage-1 rule that "
    "reproduces 13.4 mN.m (E_r, E_f,act, no eta_s) gives 935.6 mN.m. "
    "141.4 is close to eta_s = 0.15 (140.3), which Stage-1 does not use. "
    "Inconsistency in the report; check the supplementary study."))
def test_stage2_recovery_torque_table_5_6():
    assert float(s2["tau_rec0"]) == pytest.approx(141.4e-3, abs=0.5e-3)


# --- 2D finite-difference model (step 6) --------------------------------
from fd_model import lumped_band, simulate  # noqa: E402

FD_ARGS = dict(d_w=BASELINE["d_w"], p=BASELINE["p"], a=BASELINE["a"], t=BASELINE["t"],
               l_h=BASELINE["l_h"], V=BASELINE["V"], h=BASELINE["h"])


def test_fd_mean_matches_lumped_band():
    """Same energy balance: the FD volume-mean must track the lumped band
    model (difference = wire heat capacity + discretisation, < 3 %)."""
    fd = simulate(**FD_ARGS)
    lb = lumped_band(**FD_ARGS)
    assert fd["t_act_mean"] == pytest.approx(float(lb["t_act"]), rel=0.03)


def test_fd_coldest_point_lags_mean():
    fd = simulate(**FD_ARGS)
    assert fd["t_act_cold"] > fd["t_act_mean"]


def test_fd_time_step_converged():
    a = simulate(**FD_ARGS, n_steps_per_tau=200)["t_act_cold"]
    b = simulate(**FD_ARGS, n_steps_per_tau=400)["t_act_cold"]
    assert a == pytest.approx(b, rel=0.005)
