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
