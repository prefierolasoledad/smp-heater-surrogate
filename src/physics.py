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


# ---------------------------------------------------------------------------
# Mechanics: report eqs. (25)-(28), (31), (33), (34)  (Section 5.5.7 B-C)
# ---------------------------------------------------------------------------
T_G, W_G = 70.0, 3.0            # degC, K
E_G, E_R = 1.5e9, 15.0e6        # glassy / rubbery matrix modulus, Pa
E_F, E_F_ACT = 65.0e9, 1.8e9    # CF ply modulus locked / microbuckled, Pa
EPS_CRIT = 0.10                 # outer-fibre/surface strain limit at the fold
THETA_0 = np.pi                 # stowed angle, rad (180 deg)
M_P, L_P = 0.25, 0.300          # 3U CubeSat panel mass (kg) and length (m)
J_P = M_P * L_P**2 / 3          # eq. (31): 7.5e-3 kg m^2
TAU_DRAG = 25e-3                # harness/MLI drag torque, N m
# Strain-energy storage efficiency of eq. (32), assumed: lower (conservative)
# end of the report's 0.15-0.35 range. Measure it on the bench (load cell).
ETA_S = 0.15


def frozen_fraction(T):
    """Eq. (25): frozen volume fraction of the DGEBA matrix."""
    return 1.0 / (1.0 + np.exp((np.asarray(T, dtype=float) - T_G) / W_G))


def matrix_modulus(T):
    """Eq. (25): E_m(T) = phi_f E_g + (1 - phi_f) E_r."""
    phi = frozen_fraction(T)
    return phi * E_G + (1 - phi) * E_R


def mechanics(b, t, l_h, t_f=0.0, eta_s=ETA_S):
    """Hinge stiffness, recovery torque, torque margin and deployed frequency.

    t_f = 0 gives the Stage-1 neat SMP hinge; t_f > 0 the Stage-2 symmetric
    CFR-SMPC sandwich (core t_c = t - 2 t_f), eqs. (26)-(27).

    Moduli convention (the one that reproduces Table 5.6): the locked hinge
    uses the fully glassy limits (E_g, E_f) and recovery uses the fully rubbery
    limits (E_r, E_f,act). Using E_m(60 degC) = 1.449 GPa instead of E_g would
    lower k_lock by 3.5 %. eta_s (eq. 32) defaults to ETA_S = 0.15 for BOTH
    stages; this is the corrected Table 5.6 convention (the original table
    omitted it for Stage-1 only, which made the two stages inconsistent).
    """
    b, t, l_h, t_f = (np.asarray(x, dtype=float) for x in (b, t, l_h, t_f))
    t_c = t - 2 * t_f
    I_core = b * t_c**3 / 12
    I_plies = b * (t**3 - t_c**3) / 12
    k_lock = (E_F * I_plies + E_G * I_core) / l_h              # eq. (27), cold
    k_rec = (E_F_ACT * I_plies + E_R * I_core) / l_h           # eq. (27), hot
    tau_rec0 = eta_s * k_rec * THETA_0                         # peak torque at 180 deg
    return dict(
        k_lock=k_lock, k_rec=k_rec, tau_rec0=tau_rec0,
        TM=tau_rec0 / TAU_DRAG - 1,                            # eq. (33)
        f1=np.sqrt(k_lock / J_P) / (2 * np.pi),                # eq. (34)
        eps_max=THETA_0 * t / (2 * l_h),                       # eq. (28)
        l_h_min=THETA_0 * t / (2 * EPS_CRIT),                  # eq. (28)
    )
