#!/usr/bin/env python3
"""
Four animated strategies in a 2x2 grid with:
  • Column-aware spillage for the back→front (active strip) method
  • Pile spreading for pile strategies (cap = RHO_CAP)
  • No bagging until raking is complete (piles/band visibly grow)

Playback: 1 frame = 1 minute; 0.5 s per frame (2 FPS).
Saves MP4; falls back to GIF if ffmpeg isn't available.
"""

import itertools
from math import radians, sqrt, pi
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize

# =========================
# CONFIG
# =========================
FPS = 2                      # playback speed: 0.5 s per frame
SECONDS_PER_FRAME = 60       # 1 frame = 1 minute of simulated work
SAVE_OUTPUT = True
SHOW_WINDOW = True           # set False if you only want files saved

OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "heatmaps_2x2.mp4"

# Visuals: add orange/red for high densities
COLORS = ["#edf8e9", "#a1d99b", "#31a354", "#fed976", "#fd8d3c", "#e31a1c"]
CMAP = LinearSegmentedColormap.from_list("leaf_hot", COLORS, N=256)

# Physical-ish cap on leaf surface density (lb/ft^2)
RHO_CAP = 3.0

# Outside-in ray discretization
ANGLE_BINS = 24

# Front-sweep strip thickness (feet)
FRONT_SWEEP_STRIP_FT = 2.0

# =========================
# Mock data & yard model (from original)
# =========================
raking_splits = {10:7, 20:9, 30:13, 40:17, 50:21, 60:28, 70:35, 80:42, 90:49, 100:51}
bagging_times = {0.25:"1:39", 0.50:"1:45", 0.75:"1:45", 1.00:"2:00"}

L, W = 60.0, 40.0
tree = (15.0, 20.0)
trunk_radius = 1.5
phi_deg = 90.0
axis_ratio = 1.5
sigma = 10.0
rho0, A_amp, p_pow = 0.03, 0.28, 2.0

# grid (ft)
s = 1.0
candidate_spacing = 10.0
K_max = 5
bag_capacity_lb = 35.0  # bagging deferred; kept for completeness

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
    fracs = np.array(sorted(fullness_to_time.keys()), dtype=float)
    secs  = np.array([mmss_to_seconds(fullness_to_time[f]) for f in fracs], dtype=float)
    X = np.vstack([np.ones_like(fracs), fracs, fracs**2]).T
    coef, *_ = np.linalg.lstsq(X, secs, rcond=None)
    s0, s1, s2 = coef
    if s2 < 0:
        s2 = 0.0
        X2 = X[:, :2]
        s0, s1 = np.linalg.lstsq(X2, secs, rcond=None)[0]
    def t_of_fraction(f): return s0 + s1*f + s2*(f**2)
    return (s0, s1, s2), t_of_fraction

def euclid(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(axis=2))

def nearest_index(grid_coords, value):
    return int(np.argmin(np.abs(grid_coords - value)))

# =========================
# 1) Calibrate
# =========================
distances = sorted(raking_splits.keys())
times = [raking_splits[d] for d in distances]
alpha, beta = fit_power_time(distances, times)

(b0_hat, b1_hat, b2_hat), t_frac = fit_bag_time(bagging_times)
b0 = t_frac(0.0)
bag_capacity_lb = float(bag_capacity_lb)

# =========================
# 2) Build grid & leaf distribution
# =========================
nx, ny = int(L/s), int(W/s)
xs = np.linspace(s/2, L - s/2, nx)  # x: 0→L
ys = np.linspace(s/2, W - s/2, ny)  # y: 0 (front) → W (back)
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
rho_init = rho0 + A_amp * np.exp(-(R_aniso**p_pow))
R_circ = np.sqrt(dx**2 + dy**2)
rho_init = np.where(R_circ < trunk_radius, rho0, rho_init)

mass_grid_init = rho_init * Acell
masses = mass_grid_init.ravel()
M_total = float(masses.sum())

# =========================
# 3) Pile centers
# =========================
def centers_bf():
    k = max(1, int(round(M_total/(2*bag_capacity_lb))))
    x_bounds = np.linspace(0, L, k+1)
    centers = np.array([[(x_bounds[j]+x_bounds[j+1])/2.0, W/2.0] for j in range(k)])
    return centers

def centers_micro():
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
                    da = ((pts-ca)**2).sum(1); db=((pts-cb)**2).sum(1)
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

def centers_opt_discrete():
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
            rake_s = float(np.sum(alpha * masses * (dmin ** beta)))
            if (best_total is None) or (rake_s < best_total):
                best_total = rake_s; best_combo = combo
    centers = Psites[list(best_combo)] if best_combo is not None else np.array([[L/2, W/2]])
    return centers

# =========================
# 4) Baselines for calibration
# =========================
def baseline_rake_time_to_centers(centers):
    if centers.size == 0:
        centers = np.array([[L/2, W/2]])
    D = euclid(cells, centers)
    dmin = D.min(axis=1)
    return float(np.sum(alpha * masses * (dmin ** beta)))

def baseline_rake_time_to_front():
    dfront = Y.ravel()
    return float(np.sum(alpha * masses * (dfront ** beta)))

# =========================
# 5) Outside-in (calibrated) + pile deposit with cap
# =========================
def radial_arrival_times_calibrated(centers, angle_bins=24):
    if centers.size == 0:
        centers = np.array([[L/2, W/2]])
    D = euclid(cells, centers)
    pile_id = np.argmin(D, axis=1)
    r_to_pile = D[np.arange(len(cells)), pile_id]

    bin_w = 2*np.pi / angle_bins
    t_arrive = np.full(len(cells), np.inf, dtype=float)

    for p in range(centers.shape[0]):
        cx, cy = centers[p]
        sel = (pile_id == p)
        if not np.any(sel): continue
        idxs = np.where(sel)[0]
        pts = cells[sel]
        vx = pts[:,0] - cx
        vy = pts[:,1] - cy
        theta = np.arctan2(vy, vx)
        bins = np.clip(np.floor((theta + np.pi)/bin_w).astype(int), 0, angle_bins-1)

        for b in range(angle_bins):
            ray_mask = (bins == b)
            if not np.any(ray_mask): continue
            ray_idxs = idxs[ray_mask]
            r = r_to_pile[ray_idxs]
            order = np.argsort(-r)  # outside → inside
            ray_idxs = ray_idxs[order]
            r = r[order]
            m = masses[ray_idxs]
            r_ext = np.concatenate([r, np.array([0.0])])
            M_cum = np.cumsum(m[::-1])[::-1]
            dt = alpha * M_cum * ((r_ext[:-1] - r_ext[1:]) ** beta)
            T = np.cumsum(dt)
            t_arrive[ray_idxs] = T

    # calibrate to baseline
    raw_total = float(np.nanmax(t_arrive[np.isfinite(t_arrive)])) if np.isfinite(t_arrive).any() else 0.0
    target_total = baseline_rake_time_to_centers(centers)
    scale = (target_total / raw_total) if raw_total > 0 else 1.0
    t_arrive *= scale

    per_pile = []
    for p in range(centers.shape[0]):
        mask = (pile_id == p)
        per_pile.append({"t": t_arrive[mask], "m": masses[mask]})
    return t_arrive, pile_id, per_pile

def deposit_piles_disk(centers, per_pile, t_sec, nx, ny, xs, ys, rho_cap, acell):
    acc = np.zeros((ny, nx), dtype=float)  # density (lb/ft^2)
    if centers.size == 0:
        return acc
    Xc, Yc = np.meshgrid(xs, ys)
    for p, (cx, cy) in enumerate(centers):
        tp = per_pile[p]["t"]; mp = per_pile[p]["m"]
        M = float(mp[tp <= t_sec].sum())  # arrived mass
        if M <= 0: 
            continue
        r = sqrt(M / (pi * rho_cap))  # ft
        mask = (Xc - cx)**2 + (Yc - cy)**2 <= r**2
        area = float(mask.sum()) * acell
        if area <= 0: 
            continue
        dens = min(rho_cap, M / area)
        acc[mask] += dens
    return acc

# =========================
# 6) Front-sweep timing and column-aware spillage
# =========================
def front_sweep_band_mass_time(strip_ft=2.0):
    rows_per = max(1, int(round(strip_ft / s)))
    # raw cumulative time per pass using total mass above the current threshold
    n_steps = int(np.ceil(ny / rows_per))
    T_raw = []
    raw_total = 0.0
    # mass per row (total, not per column)
    mass_per_row_init = mass_grid_init.sum(axis=1)  # ny
    for k in range(n_steps):
        # back threshold y_k corresponds to row index start_of_slab
        start = max(0, ny - (k+1)*rows_per)
        M_above = float(mass_per_row_init[start:].sum())
        dt_k = alpha * M_above * (rows_per * s) ** beta
        raw_total += dt_k
        T_raw.append(raw_total)
    T_raw = np.array(T_raw, dtype=float)
    # calibrate to push-to-front baseline
    T_target = baseline_rake_time_to_front()
    scale = (T_target / T_raw[-1]) if len(T_raw) and T_raw[-1] > 0 else 1.0
    T_steps = T_raw * scale
    return rows_per, mass_per_row_init, T_steps

def band_snapshot_with_spillage_columns(t_sec, rows_per, mass_per_row_init, T_steps, rho_cap):
    """
    Column-aware spillage for the active back strip method.
    - Only the active back strip is moved each pass.
    - Band mass is computed per column and deposited starting at the first
      fully-cleared row, filling each cell to rho_cap before spilling downward
      (toward the back) into the next row in that same column.
    """
    # Progress in strips
    k = int(np.searchsorted(T_steps, t_sec, side="right"))
    k = min(k, len(T_steps))
    adv_rows = k * rows_per

    # Fractional progress in current strip
    if k == len(T_steps):
        frac = 1.0
    else:
        prev_t = 0.0 if k == 0 else T_steps[k-1]
        dt_k = T_steps[k] - prev_t
        frac = 0.0 if dt_k <= 1e-12 else np.clip((t_sec - prev_t) / dt_k, 0.0, 1.0)

    # Base density (start from initial; remove swept mass)
    rho = rho_init.copy()

    # Fully swept rows (from the back): zero them out
    if adv_rows > 0:
        rho[ny - adv_rows: , :] = 0.0

    # Partially swept next strip: remove a fraction
    next_start = max(0, ny - (adv_rows + rows_per))
    next_end   = max(-1, ny - adv_rows - 1)
    if next_start <= next_end and frac > 0:
        rho[next_start:next_end+1, :] *= (1.0 - frac)

    # Compute band mass per column (from fully swept + fractional strip)
    M_full_cols = mass_grid_init[ny - adv_rows: ny, :].sum(axis=0) if adv_rows > 0 else np.zeros(nx)
    M_next_cols = mass_grid_init[next_start:next_end+1, :].sum(axis=0) if next_start <= next_end else np.zeros(nx)
    M_band_cols = M_full_cols + frac * M_next_cols  # shape (nx,)

    # Deposit starting at the first cleared row (just behind the partial strip)
    lead_row = ny - adv_rows
    if lead_row >= ny:
        lead_row = ny - 1  # happens only before any progress; place at back edge

    # Per-cell capacity in pounds at the cap
    K_cell = rho_cap * Acell

    remaining = M_band_cols.copy()
    # Fill rows r = lead_row .. ny-1 (backwards toward the yard's back)
    for r in range(lead_row, ny):
        if np.all(remaining <= 1e-12):
            break
        # capacity of each cell in this row (subtract any existing density, though it should be ~0 in cleared region)
        cap_row = np.maximum(0.0, K_cell - rho[r, :] * Acell)
        place = np.minimum(remaining, cap_row)
        rho[r, :] += place / Acell
        remaining -= place

    return rho

# Convenience wrapper to keep your earlier call sites clean
def density_snapshot_front_sweep_spread(t_sec):
    return band_snapshot_with_spillage_columns(t_sec, rows_per, mass_per_row_init, T_steps, RHO_CAP)

# =========================
# 7) Build strategies & schedules
# =========================
centers_list = [
    centers_bf(),
    np.empty((0,2)),  # front-sweep placeholder
    centers_micro(),
    centers_opt_discrete(),
]

PANEL_TITLES = [
    "BF-centers (outside-in, spread)",
    "Front-sweep: back→front (column-aware spillage)",
    "Micro-piles (outside-in, spread)",
    "Optimization (discrete K≤5, spread)",
]

# Outside-in schedules (panels 0,2,3)
radial_data = {}
for idx in [0, 2, 3]:
    t_arrive, pile_of_cell, per_pile = radial_arrival_times_calibrated(centers_list[idx], angle_bins=ANGLE_BINS)
    radial_data[idx] = {
        "centers": centers_list[idx],
        "t_arrive": t_arrive,
        "pile_of_cell": pile_of_cell,
        "per_pile": per_pile,
    }

# Front-sweep precompute
rows_per, mass_per_row_init, T_steps = front_sweep_band_mass_time(FRONT_SWEEP_STRIP_FT)

# Determine total simulated horizon
def total_seconds_outside_in(sched):
    ta = sched["t_arrive"]
    return float(np.nanmax(ta[np.isfinite(ta)])) if np.isfinite(ta).any() else 0.0

sec_candidates = [
    total_seconds_outside_in(radial_data[0]),
    float(T_steps[-1]) if len(T_steps) else 0.0,
    total_seconds_outside_in(radial_data[2]),
    total_seconds_outside_in(radial_data[3]),
]
total_seconds = max(sec_candidates) if sec_candidates else 0.0
total_minutes = int(np.ceil(total_seconds / 60.0))
n_frames = max(1, total_minutes + 1)

# =========================
# 8) Density snapshots per panel (pile strategies also have spreading)
# =========================
def density_snapshot_outside_in_spread(panel_idx, t_sec):
    info = radial_data[panel_idx]
    centers = info["centers"]
    t_arrive = info["t_arrive"]

    remain = rho_init.copy().ravel()
    arrived = (t_arrive <= t_sec)
    remain[arrived] = 0.0
    remain = remain.reshape(ny, nx)

    acc = deposit_piles_disk(centers, info["per_pile"], t_sec, nx, ny, xs, ys, RHO_CAP, Acell)
    return remain + acc

# =========================
# 9) Animate
# =========================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
axes = axes.ravel()

# panel-wise vmax so background is visible and piles/band hit orange/red
vmax_panels = [max(float(rho_init.max()), RHO_CAP)] * 4

ims = []
for i, ax in enumerate(axes):
    if i == 1:
        img0 = density_snapshot_front_sweep_spread(0.0)
    else:
        img0 = density_snapshot_outside_in_spread(i, 0.0)
    im = ax.imshow(
        img0, origin="lower", extent=[0, L, 0, W], aspect="auto",
        cmap=CMAP, norm=Normalize(vmin=0.0, vmax=vmax_panels[i])
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="lb/ft$^2$")
    if i != 1 and centers_list[i].size:
        ax.scatter(centers_list[i][:, 0], centers_list[i][:, 1], s=30, c="#2b8cbe", marker="x", linewidths=1.5)
    ax.set_title(PANEL_TITLES[i], fontsize=10)
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ims.append(im)

suptitle = fig.suptitle("Minute 0", fontsize=14)

def update(frame_idx):
    t_sec = frame_idx * SECONDS_PER_FRAME
    ims[0].set_data(density_snapshot_outside_in_spread(0, t_sec))
    ims[1].set_data(density_snapshot_front_sweep_spread(t_sec))   # column-aware spillage
    ims[2].set_data(density_snapshot_outside_in_spread(2, t_sec))
    ims[3].set_data(density_snapshot_outside_in_spread(3, t_sec))
    suptitle.set_text(f"Minute {frame_idx} / {total_minutes}")
    return (*ims, suptitle)

anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False, repeat=True)

# Optional interactive view
if SHOW_WINDOW:
    plt.show(block=False)

# Save MP4 (fallback to GIF if ffmpeg missing)
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

if SHOW_WINDOW:
    plt.show()
else:
    plt.close(fig)
