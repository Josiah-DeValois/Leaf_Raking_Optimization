#!/usr/bin/env python3
"""
Animate 4 leaf-clearing strategies as heatmaps in a 2x2 grid.
- 1 frame = 1 minute of work
- Playback = 0.5 s per frame (2 fps)
- Saves MP4 (falls back to GIF if ffmpeg unavailable)
- No external dependencies beyond numpy/matplotlib/pandas (pandas only used to echo a table if desired)
"""

import os
import itertools
from math import radians
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize

# =========================
# CONFIG
# =========================
FPS = 2                     # 0.5 seconds per frame
SECONDS_PER_FRAME = 60      # "1 frame = 1 minute" of simulated work
SAVE_OUTPUT = True
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "heatmaps_2x2.mp4"  # fallback to .gif if ffmpeg not present

PANEL_TITLES = [
    "BF-centers",
    "Forward-to-Backward",
    "Micro-piles (soft)",
    f"Optimization (discrete K)"
]

# =========================
# Mock data & yard model (from your original script)
# =========================

# Raking cumulative times at 10-ft splits (seconds)
raking_splits = {
    10: 7, 20: 9, 30: 13, 40: 17, 50: 21,
    60: 28, 70: 35, 80: 42, 90: 49, 100: 51
}

# Bagging times by fullness fraction (mm:ss -> seconds)
bagging_times = {
    0.25: "1:39",
    0.50: "1:45",
    0.75: "1:45",
    1.00: "2:00",
}

# Yard and density model
L, W = 60.0, 40.0
tree = (15.0, 20.0)
trunk_radius = 1.5  # ft (3 ft diameter exclusion)
phi_deg = 90.0
axis_ratio = 1.5
sigma = 10.0
rho0, A_amp, p_pow = 0.03, 0.28, 2.0  # baseline density + tree bump

# Discretization (ft). Use 1.0 for full fidelity; 2.0/5.0 if you want it faster.
s = 1.0

# Heuristic/optimization search parameters
candidate_spacing = 10.0   # pile candidate grid spacing (ft)
K_max = 5                  # brute-force max piles over candidate set
bag_capacity_lb = 35.0     # C
tau_turn_per_100 = 1.0     # s per 100 sqft (small constant, ignored in animation)

# =========================
# Helpers
# =========================
def mmss_to_seconds(t: str) -> int:
    t = t.strip()
    if ":" in t:
        mm, ss = t.split(":")
        return int(mm)*60 + int(ss)
    return int(float(t))

def fit_power_time(distances, times):
    """Fit T(d) ~ k * d^beta using log-log linear regression."""
    d = np.array(distances, dtype=float)
    T = np.array(times, dtype=float)
    mask = (d > 0) & (T > 0)
    d, T = d[mask], T[mask]
    x = np.log(d); y = np.log(T)
    A = np.vstack([np.ones_like(x), x]).T
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_k, beta = theta
    k = np.exp(ln_k)
    return k, beta

def fit_bag_time(fullness_to_time):
    """Fit t(f) ~ s0 + s1*f + s2*f^2 with s2>=0. Returns (s0,s1,s2) and a function."""
    fracs = np.array(sorted(fullness_to_time.keys()), dtype=float)
    secs  = np.array([mmss_to_seconds(fullness_to_time[f]) for f in fracs], dtype=float)
    X = np.vstack([np.ones_like(fracs), fracs, fracs**2]).T
    coef, *_ = np.linalg.lstsq(X, secs, rcond=None)
    s0, s1, s2 = coef
    if s2 < 0:
        s2 = 0.0
        X2 = X[:, :2]
        s0, s1 = np.linalg.lstsq(X2, secs, rcond=None)[0]
    def t_of_fraction(f):
        return s0 + s1*f + s2*(f**2)
    return (s0, s1, s2), t_of_fraction

def euclid(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(axis=2))

# =========================
# 1) Calibrate raking & bagging from mock data
# =========================
distances = sorted(raking_splits.keys())
times = [raking_splits[d] for d in distances]
alpha, beta = fit_power_time(distances, times)  # sec/(lb * ft^beta)

(b0_hat, b1_hat, b2_hat), t_frac = fit_bag_time(bagging_times)
b0 = t_frac(0.0)  # setup-like (f=0)
bag_capacity_lb = float(bag_capacity_lb)
# Approximate marginal stuffing rate (sec/lb) near mid fullness
b1 = (t_frac(0.8) - t_frac(0.4)) / 0.4 / bag_capacity_lb
# Spread the per-bag setup across mass as a per-lb overhead:
bag_overhead_per_lb = b1 + (b0 / bag_capacity_lb)

# =========================
# 2) Build yard grid and density
# =========================
nx, ny = int(L/s), int(W/s)
xs = np.linspace(s/2, L - s/2, nx)
ys = np.linspace(s/2, W - s/2, ny)
X, Y = np.meshgrid(xs, ys)
cells = np.stack([X.ravel(), Y.ravel()], axis=1)
Acell = s * s

phi = radians(phi_deg)
sigma_par = axis_ratio * sigma
sigma_perp = sigma

dx = X - tree[0]; dy = Y - tree[1]
u =  dx*np.cos(phi) + dy*np.sin(phi)
v = -dx*np.sin(phi) + dy*np.cos(phi)
R_aniso = np.sqrt((u/sigma_par)**2 + (v/sigma_perp)**2)
rho = rho0 + A_amp * np.exp(-(R_aniso**p_pow))
R_circ = np.sqrt(dx**2 + dy**2)
rho = np.where(R_circ < trunk_radius, rho0, rho)
m_cell = rho * Acell
masses = m_cell.ravel()
M_total = float(masses.sum())

# =========================
# 3) Strategies -> pile centers (compatible but simplified where needed)
# =========================
def strategy_bf_centers():
    """Balanced field centers placed across the yard widthwise."""
    k = max(1, int(round(M_total/(2*bag_capacity_lb))))
    x_bounds = np.linspace(0, L, k+1)
    centers = np.array([[(x_bounds[j]+x_bounds[j+1])/2.0, W/2.0] for j in range(k)])
    return centers

def strategy_f2b_centers():
    """Forward-to-Backward: piles along the front edge (y=0), binned by x mass."""
    sx = np.arange(candidate_spacing/2, L, candidate_spacing)  # candidate x positions
    xi = cells[:, 0]; wi = masses
    order = np.argsort(xi)
    bins = []; cur_mass = 0.0; cur_items = []
    for idx in order:
        if cur_mass + wi[idx] <= bag_capacity_lb:
            cur_mass += wi[idx]; cur_items.append(idx)
        else:
            bins.append(cur_items); cur_mass = wi[idx]; cur_items = [idx]
    if cur_items: bins.append(cur_items)
    piles_idx = []
    for b in bins:
        xbar = float(np.average(xi[b], weights=wi[b]))
        p = int(np.argmin(np.abs(sx - xbar)))
        piles_idx.append(p)
    centers = np.stack([sx[piles_idx], np.zeros(len(piles_idx))], axis=1) if piles_idx else np.zeros((0,2))
    return centers

def strategy_micro_centers():
    """Micro-piles via weighted seeding + local updates and splitting heavy clusters."""
    rng = np.random.default_rng(42)
    k0 = max(1, int(round(M_total/(2*bag_capacity_lb))))
    probs = masses / (M_total + 1e-12)
    idx0 = rng.choice(len(masses), size=k0, replace=False, p=probs)
    centers = cells[idx0].copy()
    for _ in range(8):
        D2 = ((cells[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        labels = np.argmin(D2, axis=1)
        m_c = np.array([masses[labels==j].sum() for j in range(centers.shape[0])], dtype=float)
        to_split = np.where(m_c > 2*bag_capacity_lb)[0]
        new_centers = []
        for j in range(centers.shape[0]):
            mask = labels == j
            if not np.any(mask):
                new_centers.append(centers[j]); continue
            pts = cells[mask]; w = masses[mask]
            if j in to_split:
                a = pts[np.argmax(((pts-pts.mean(0))**2).sum(1))]
                b = pts[np.argmax(((pts-a)**2).sum(1))]
                ca, cb = a.copy(), b.copy()
                for _ in range(5):
                    da = ((pts-ca)**2).sum(1); db = ((pts-cb)**2).sum(1)
                    lab = (da <= db)
                    wa = w[lab].sum() + 1e-9; wb = w[~lab].sum() + 1e-9
                    ca = np.array([np.sum(pts[lab,0]*w[lab])/wa, np.sum(pts[lab,1]*w[lab])/wa])
                    cb = np.array([np.sum(pts[~lab,0]*w[~lab])/wb, np.sum(pts[~lab,1]*w[~lab])/wb])
                new_centers.append(ca); new_centers.append(cb)
            else:
                cx = np.average(pts[:, 0], weights=w); cy = np.average(pts[:, 1], weights=w)
                new_centers.append([cx, cy])
        centers = np.array(new_centers)
    return centers

def strategy_opt_discrete_centers():
    """Brute-force discrete optimization over a candidate grid (K<=K_max)."""
    px = np.arange(candidate_spacing/2, L, candidate_spacing)
    py = np.arange(candidate_spacing/2, W, candidate_spacing)
    Psites = np.array(list(itertools.product(px, py)))
    D = np.sqrt(((cells[:, None, :] - Psites[None, :, :])**2).sum(axis=2))
    best_total = None; best_combo = None
    for K in range(1, K_max+1):
        for combo in itertools.combinations(range(len(Psites)), K):
            Csub = D[:, combo]
            idx = np.argmin(Csub, axis=1)
            dmin = Csub[np.arange(Csub.shape[0]), idx]
            # Raking cost term only (bagging per-lb is same sum over mass, independent of centers)
            rake_s = float(np.sum(alpha * masses * (dmin ** beta)))
            if (best_total is None) or (rake_s < best_total):
                best_total = rake_s; best_combo = combo
    centers = Psites[list(best_combo)] if best_combo is not None else np.zeros((1,2))
    return centers

# =========================
# 4) Convert centers -> per-cell work times & a completion schedule
# =========================
def per_cell_times_for_centers(centers: np.ndarray):
    """
    Given pile centers, assign each cell to nearest center and compute per-cell time:
      t_cell = alpha * m_i * d_i^beta + m_i * bag_overhead_per_lb
    Returns:
      order_idx: indices of cells sorted by t_cell (shortest first)
      t_cell: per-cell times (seconds)
      cumsum: cumulative sum of t_cell in sorted order (seconds)
      dmin: nearest distance (for debugging/overlay if needed)
    """
    if centers.size == 0:
        centers = np.array([[L/2, W/2]])
    D = euclid(cells, centers)
    dmin = D.min(axis=1)
    # per-cell work time (rake + avg bagging overhead)
    t_cell = alpha * masses * (dmin ** beta) + masses * bag_overhead_per_lb
    # Sort shortest job first (SPT)
    order_idx = np.argsort(t_cell)
    sorted_times = t_cell[order_idx]
    cumsum = np.cumsum(sorted_times)
    return order_idx, t_cell, cumsum, dmin

def remaining_density_at_time(order_idx, t_cell, cumsum, t_sec):
    """
    Compute remaining mass per cell at time t_sec under SPT schedule,
    with partial progress on the current cell.
    """
    n = len(t_cell)
    # number of fully-cleared cells
    k = int(np.searchsorted(cumsum, t_sec, side="right"))
    remaining = masses.copy()

    if k > 0:
        cleared = order_idx[:k]
        remaining[cleared] = 0.0

    if k < n:
        curr = order_idx[k] if k > 0 else order_idx[0]
        start = 0.0 if k == 0 else cumsum[k-1]
        frac = np.clip((t_sec - start) / (t_cell[curr] + 1e-12), 0.0, 1.0)
        remaining[curr] = masses[curr] * (1.0 - frac)

    # Convert to density
    rho_rem = (remaining / Acell).reshape(ny, nx)
    return rho_rem

# =========================
# 5) Build strategies & schedules
# =========================
centers_list = [
    strategy_bf_centers(),
    strategy_f2b_centers(),
    strategy_micro_centers(),
    strategy_opt_discrete_centers(),
]

schedules = []   # list of dicts with per-strategy precomputes
for centers in centers_list:
    order_idx, t_cell, cumsum, dmin = per_cell_times_for_centers(centers)
    schedules.append({
        "centers": centers,
        "order_idx": order_idx,
        "t_cell": t_cell,
        "cumsum": cumsum
    })

# Determine total simulated minutes (slowest panel completion)
total_seconds = max(sch["cumsum"][-1] for sch in schedules)
total_minutes = int(np.ceil(total_seconds / 60.0))
n_frames = total_minutes + 1  # include minute 0

# =========================
# 6) Animate
# =========================
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
axes = axes.ravel()
vmin, vmax = 0.0, float(rho.max())  # fixed color scale across time/panels

# Initialize images & colorbars
ims = []
cbs = []
for i, ax in enumerate(axes):
    init_rho = remaining_density_at_time(
        schedules[i]["order_idx"], schedules[i]["t_cell"], schedules[i]["cumsum"], 0.0
    )
    im = ax.imshow(init_rho, origin="lower", extent=[0, L, 0, W],
                   aspect="auto", norm=Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="lb/ft$^2$")
    cbs.append(cb)
    ims.append(im)
    # overlay pile centers
    centers = centers_list[i]
    if centers.size:
        ax.scatter(centers[:, 0], centers[:, 1], s=35, marker="x")
    ax.set_title(PANEL_TITLES[i], fontsize=10)
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")

suptitle = fig.suptitle("Minute 0", fontsize=14)

def update(frame_idx):
    # Sim time in seconds
    t_sec = frame_idx * SECONDS_PER_FRAME
    for i in range(4):
        sch = schedules[i]
        rho_rem = remaining_density_at_time(sch["order_idx"], sch["t_cell"], sch["cumsum"], t_sec)
        ims[i].set_data(rho_rem)
    suptitle.set_text(f"Minute {frame_idx} / {total_minutes}")
    return (*ims, suptitle)

anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000 / FPS,  # playback speed (0.5 s per frame with FPS=2)
    blit=False,
    repeat=True
)

# Show interactively (comment out if running headless)
plt.show(block=False)

# Save output
if SAVE_OUTPUT:
    try:
        writer = FFMpegWriter(fps=FPS)
        anim.save(str(OUT_PATH), writer=writer, dpi=150)
        print(f"Saved animation to {OUT_PATH}")
    except Exception as e:
        gif_path = OUT_DIR / "heatmaps_2x2.gif"
        print("FFmpeg not available or failed; saving GIF instead...", e)
        anim.save(str(gif_path), writer=PillowWriter(fps=FPS))
        print(f"Saved animation to {gif_path}")

plt.show()


