"""Electrothermal sub-model of the zigzag-nichrome SMP hinge.

Implements report equations (18)-(24) (Section 5.5.7 A) in vectorised NumPy.
All inputs are SI units; every function accepts scalars or equal-length arrays.

Baseline (Table 5.5) reproduces Table 5.6, Stage-1 neat SMP, and is
checked in tests/test_physics.py.
"""
from __future__ import annotations

import numpy as np

SIGMA = 5.67e-8  # Stefan-Boltzmann, W/(m^2 K^4)

# Table 5.5 baseline, Stage-1 neat DGEBA hinge (SI units)
BASELINE = dict(
    d_w=0.20e-3,     # wire diameter, m
    p=3.0e-3,        # zigzag pitch, m
    a=20.0e-3,       # zigzag peak-to-peak width, m
    t=1.60e-3,       # hinge thickness, m
    b=25.0e-3,       # hinge width, m
    l_h=30.0e-3,     # active hinge length, m
    V=7.5,           # applied voltage, V
    h=10.0,          # natural-convection coefficient, W/(m^2 K)
)

# Material / boundary constants (Stage-1 neat epoxy, Table 5.5 / 5.6)
RHO_NC = 1.10e-6      # nichrome 80/20 resistivity, ohm*m  [12]
RHO_C = 1150 * 1500   # epoxy volumetric heat capacity, J/(m^3 K) -> Ch = 2.07 J/K
K_PERP = 0.20         # epoxy through-thickness conductivity, W/(m K)
EPS_TH = 0.85         # surface emissivity (orbit)
T_REF = 340.0         # radiation linearisation temperature, K
T0, T_ACT, T_LOCK = 25.0, 90.0, 60.0  # degC


def heater(d_w, p, a, l_h, V):
    """Eq. (18): zigzag wire length, resistance, current, power."""
    L_w = (2 * l_h / p) * np.sqrt(a**2 + p**2 / 4)
    A_w = np.pi * d_w**2 / 4
    R_w = RHO_NC * L_w / A_w
    I = V / R_w
    P = V**2 / R_w
    return L_w, R_w, I, P


def evaluate(d_w, p, a, t, b, l_h, V, h, environment="lab"):
    """Full electrothermal evaluation, eqs. (18)-(24).

    environment: "lab" (convection, G = h*As) or "orbit" (linearised radiation).
    Returns a dict of NumPy arrays. Where the hinge cannot reach T_ACT
    (P <= G*(T_ACT-T0)), reachable=False and the time-based outputs are NaN.
    """
    d_w, p, a, t, b, l_h, V, h = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (d_w, p, a, t, b, l_h, V, h)]
    )
    L_w, R_w, I, P = heater(d_w, p, a, l_h, V)

    # Eq. (19)-(21): lumped capacitance and conductance
    V_h = b * t * l_h
    C_h = RHO_C * V_h
    A_s = 2 * b * l_h
    if environment == "lab":
        G = h * A_s
    elif environment == "orbit":
        G = 4 * EPS_TH * SIGMA * A_s * T_REF**3
    else:
        raise ValueError("environment must be 'lab' or 'orbit'")
    tau = C_h / G
    T_ss = T0 + P / G                 # steady state with no cut-off

    # Eq. (22): heating time, cooling time, energy per cycle
    arg = 1 - G * (T_ACT - T0) / P
    reachable = arg > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_act = np.where(reachable, -tau * np.log(np.where(reachable, arg, 1.0)), np.nan)
    t_cool = tau * np.log((T_ACT - T0) / (T_LOCK - T0))
    E_cyc = P * t_act

    # Eq. (23): Biot and Fourier numbers
    s = p / 2
    alpha = K_PERP / RHO_C
    L_cond = np.maximum(s / 2, t / 2)
    Bi = (G / A_s) * (t / 2) / K_PERP
    Fo = alpha * t_act / L_cond**2

    # Eq. (24): through-thickness gradient and wire hotspot
    T_dot = (T_ACT - T0) / t_act
    dT_int = T_dot * L_cond**2 / (2 * alpha)
    r_w = d_w / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        dT_wire = (P / L_w) / (2 * np.pi * K_PERP) * np.log((s / 2) / r_w)

    return dict(
        L_w=L_w, R_w=R_w, I=I, P=P, C_h=C_h, G=G, tau=tau, T_ss=T_ss,
        reachable=reachable, t_act=t_act, t_cool=t_cool, E_cyc=E_cyc,
        Bi=Bi, Fo=Fo, dT_int=dT_int, dT_wire=dT_wire,
        J=I / (np.pi * d_w**2 / 4),   # current density, A/m^2
    )


def geometry_valid(d_w, p, a, t, b, margin=2.0e-3):
    """Physical feasibility of a design.

    The zigzag must fit inside the hinge width, adjacent runs must not touch
    (the ln((s/2)/r_w) term in eq. (24) needs s/2 > r_w), and the hinge must
    be thicker than the wire.
    """
    s = p / 2
    return (a <= b - margin) & (s / 2 > d_w) & (t > 2 * d_w)
