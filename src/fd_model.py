"""Step 6a: 2D transient finite-difference model of the heated hinge band.

Higher-fidelity counterpart of the lumped model (eqs. 19-24). It resolves
conduction in the plane that cuts across the wire runs:

    x : along the hinge length (runs repeat every s = p/2)
    z : through the thickness

One repeating cell holds one wire run at mid-thickness. By symmetry only a
quarter cell is solved: 0 <= x <= s/2, 0 <= z <= t/2, wire centre at the
origin. Faces x = 0, x = s/2 and z = 0 are symmetry (adiabatic) planes; the
face z = t/2 loses heat by convection h to T0 (laboratory case).

Energy: the wire carries its true line power q' = P / L_w (W per metre of
wire), the same quantity eq. (24) uses. Like eqs. (19)-(22) applied to the
heated band, the cell assumes the zigzag band is uniform along the run
direction; the unheated side margins (b - a) are outside this model. The
matching "lumped band" reference is therefore eq. (21)-(22) with the hinge
width replaced by b_band = s * L_w / l_h (~ a), which gives identical total
power, heat capacity and surface area per cell.

Discretisation: cell-centred uniform grid, nichrome cells inside the wire
radius (own rho*c and k), harmonic-mean face conductances, backward-Euler
time stepping with a pre-factorised sparse matrix (unconditionally stable).

Outputs per design:
  t_act_mean  time the cell's volume-mean temperature reaches T_act
  t_act_cold  time the COLDEST point reaches T_act (whole band activated)
  dT_hot      peak epoxy temperature minus mean, at t_act_mean (cf. eq. 24)
  T_peak      hottest temperature anywhere (wire) at t_act_cold
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from physics import K_PERP, RHO_C, T0, T_ACT, heater

RHO_C_NC = 8400 * 450      # nichrome volumetric heat capacity, J/(m^3 K)
K_NC = 11.3                # nichrome thermal conductivity, W/(m K)


def build(s, t, r_w, n_max=80):
    Lx, Lz = s / 2, t / 2
    hg = max(r_w / 3, max(Lx, Lz) / n_max)
    nx, nz = max(4, int(round(Lx / hg))), max(4, int(round(Lz / hg)))
    dx, dz = Lx / nx, Lz / nz
    xc = (np.arange(nx) + 0.5) * dx
    zc = (np.arange(nz) + 0.5) * dz
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    wire = X**2 + Z**2 <= r_w**2
    if not wire.any():
        wire[0, 0] = True
    k = np.where(wire, K_NC, K_PERP)
    rc = np.where(wire, RHO_C_NC, RHO_C)
    return dict(nx=nx, nz=nz, dx=dx, dz=dz, wire=wire, k=k, rc=rc)


def assemble(g, h):
    nx, nz, dx, dz, k = g["nx"], g["nz"], g["dx"], g["dz"], g["k"]
    idx = np.arange(nx * nz).reshape(nx, nz)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * nz)
    # x-faces (per unit depth y): conductance = k_face * dz / dx
    kx = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    gx = (kx * dz / dx).ravel()
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    # z-faces
    kz = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    gz = (kz * dx / dz).ravel()
    c, d = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    for p_, q_, gg in ((a, b, gx), (c, d, gz)):
        rows += [p_, q_]; cols += [q_, p_]; vals += [-gg, -gg]
        np.add.at(diag, p_, gg); np.add.at(diag, q_, gg)
    # convection on top face z = t/2: series of half-cell conduction and h
    top = idx[:, -1]
    g_top = dx / (1.0 / h + (dz / 2) / k[:, -1])
    np.add.at(diag, top, g_top)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * nz, nx * nz)) + sp.diags(diag)
    bconv = np.zeros(nx * nz)
    bconv[top] = g_top * T0
    return K, bconv


def simulate(d_w, p, a, t, l_h, V, h, n_steps_per_tau=200, t_max_factor=6.0):
    """Run one design (SI units). Returns a dict of scalars."""
    s, r_w = p / 2, d_w / 2
    L_w, R_w, I, P = heater(d_w, p, a, l_h, V)
    q_line = P / L_w                       # W per m of wire
    g = build(s, t, r_w)
    K, bconv = assemble(g, h)
    vol = g["dx"] * g["dz"]                # cell volume per unit depth
    C = (g["rc"] * vol).ravel()
    src = np.zeros_like(C)
    w = g["wire"].ravel()
    src[w] = (q_line / 4) / w.sum()        # quarter of the line power
    epoxy = ~w

    # lumped-band reference for time scales
    b_band = s * L_w / l_h
    G_band = h * 2 * b_band * l_h
    C_band = RHO_C * b_band * t * l_h
    tau = C_band / G_band
    dt = tau / n_steps_per_tau
    t_end = t_max_factor * tau

    A = sp.diags(C / dt) + K
    lu = spla.splu(A.tocsc())
    T = np.full_like(C, T0)
    Ctot = C.sum()
    tm = tc = None
    dT_hot = T_peak = np.nan
    prev_mean = prev_min = T0
    time = 0.0
    while time < t_end and (tm is None or tc is None):
        T = lu.solve(C / dt * T + src + bconv)
        time += dt
        mean = (C * T).sum() / Ctot
        tmin = T[epoxy].min()
        if tm is None and mean >= T_ACT:
            tm = time - dt * (mean - T_ACT) / (mean - prev_mean)
            dT_hot = T[epoxy].max() - mean
        if tc is None and tmin >= T_ACT:
            tc = time - dt * (tmin - T_ACT) / (tmin - prev_min)
            T_peak = T.max()
        prev_mean, prev_min = mean, tmin
    return dict(t_act_mean=tm if tm is not None else np.nan,
                t_act_cold=tc if tc is not None else np.nan,
                dT_hot=dT_hot, T_peak=T_peak, b_band=b_band,
                nx=g["nx"], nz=g["nz"], n_wire=int(w.sum()))


def lumped_band(d_w, p, a, t, l_h, V, h):
    """Eqs. (18)-(24) applied to the heated band (b -> b_band)."""
    from physics import evaluate
    s = p / 2
    L_w = heater(d_w, p, a, l_h, V)[0]
    return evaluate(d_w=d_w, p=p, a=a, t=t, b=s * L_w / l_h, l_h=l_h, V=V, h=h)
