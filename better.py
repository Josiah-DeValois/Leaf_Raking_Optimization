#!/usr/bin/env python3
"""
2x2 animated comparison with interactive front-sweep controls:

• Panels:
  0: BF-centers (outside-in to piles; disks spread; walk-between-piles during bagging)
  1: Front-sweep (active strip from back→front; column-aware spillage; bag from front)
  2: Micro-piles (outside-in)
  3: Discrete Opt (outside-in)

• Control panel (right side of window):
  - [ ] Front: use efficiency      -> toggles alpha_sweep=alpha/eta and beta_sweep=beta-Δβ
  - [ ] Front: skip rescale        -> if checked, do NOT rescale to the baseline total
  - η (1.0–3.0)                    -> bulk-push efficiency factor
  - Δβ (0.0–0.5)                   -> exponent reduction for distance in front-sweep
  - Strip (1–6 ft)                 -> sweep band thickness
  - Apply                          -> recompute schedule, totals, and update on-figure stats

Playback: 1 frame = 1 minute; 0.5 s per frame (2 FPS).
Saves MP4 if ffmpeg present; falls back to GIF.
"""

import itertools
from math import radians, sqrt, pi, ceil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import CheckButtons, Slider, Button

# =========================
# CONFIG
# =========================
FPS = 2
SECONDS_PER_FRAME = 60
SAVE_OUTPUT = True
SHOW_WINDOW = True

OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "heatmaps_2x2.mp4"
OUT_GIF_MINIMAL = OUT_DIR / "heatmaps_2x2_minimal.gif"
EXPORT_FIGSIZE = (10.5, 8.0)

# Colormap
COLORS = ["#edf8e9", "#a1d99b", "#31a354", "#fed976", "#fd8d3c", "#e31a1c"]
CMAP = LinearSegmentedColormap.from_list("leaf_hot", COLORS, N=256)

# Cap on surface density (lb/ft^2)
RHO_CAP = 3.0

# Outside-in ray discretization
ANGLE_BINS = 24

# Bagging parameters
BAG_CAPACITY_LB = 35.0

# Walking between piles (outside-in panels only)
WALK_SPEED_FTPS = 3.5
PILE_ORDER_METHOD = "left_to_right"  # or "nn" (nearest-neighbor)

# =========================
# Mock data & yard model (original)
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

# Grid
s = 1.0
candidate_spacing = 10.0
K_max = 5

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
    d = np.array(distances, dtype=float); T = np.array(times, dtype=float)
    mask = (d > 0) & (T > 0); d, T = d[mask], T[mask]
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

# =========================
# 1) Calibrate
# =========================
distances = sorted(raking_splits.keys())
times = [raking_splits[d] for d in distances]
alpha, beta = fit_power_time(distances, times)

(b0_hat, b1_hat, b2_hat), t_frac = fit_bag_time(bagging_times)
b0 = t_frac(0.0)                                   # sec/bag setup
b1 = (t_frac(0.8) - t_frac(0.4)) / (0.4 * BAG_CAPACITY_LB)  # sec/lb stuffing

# =========================
# 2) Grid & distribution
# =========================
nx, ny = int(L/s), int(W/s)
xs = np.linspace(s/2, L - s/2, nx)  # x: 0→L
ys = np.linspace(s/2, W - s/2, ny)  # y: 0(front)→W(back)
X, Y = np.meshgrid(xs, ys)
cells = np.stack([X.ravel(), Y.ravel()], axis=1)
Acell = s*s

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
# 3) Pile centers (3 strategies)
# =========================
def centers_bf():
    k = max(1, int(round(M_total/(2*BAG_CAPACITY_LB))))
    x_bounds = np.linspace(0, L, k+1)
    return np.array([[(x_bounds[j]+x_bounds[j+1])/2.0, W/2.0] for j in range(k)])

def centers_micro():
    rng = np.random.default_rng(42)
    k0 = max(1, int(round(M_total/(2*BAG_CAPACITY_LB))))
    probs = masses / (M_total + 1e-12)
    idx0 = rng.choice(len(masses), size=k0, replace=False, p=probs)
    centers = cells[idx0].copy()
    for _ in range(8):
        D2 = ((cells[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        labels = np.argmin(D2, axis=1)
        m_c = np.array([masses[labels==j].sum() for j in range(centers.shape[0])], dtype=float)
        to_split = np.where(m_c > 2*BAG_CAPACITY_LB)[0]
        new_centers=[]
        for j in range(centers.shape[0]):
            mask = labels==j
            if not np.any(mask): new_centers.append(centers[j]); continue
            pts = cells[mask]; w = masses[mask]
            if j in to_split:
                a = pts[np.argmax(((pts-pts.mean(0))**2).sum(1))]
                b = pts[np.argmax(((pts-a)**2).sum(1))]
                ca, cb = a.copy(), b.copy()
                for _ in range(5):
                    da = ((pts-ca)**2).sum(1); db=((pts-cb)**2).sum(1)
                    lab = (da<=db)
                    wa = w[lab].sum()+1e-9; wb = w[~lab].sum()+1e-9
                    ca = np.array([np.sum(pts[lab,0]*w[lab])/wa, np.sum(pts[lab,1]*w[lab])/wa])
                    cb = np.array([np.sum(pts[~lab,0]*w[~lab])/wb, np.sum(pts[~lab,1]*w[~lab])/wb])
                new_centers.append(ca); new_centers.append(cb)
            else:
                cx = np.average(pts[:,0], weights=w); cy = np.average(pts[:,1], weights=w)
                new_centers.append([cx,cy])
        centers = np.array(new_centers)
    return centers

def centers_opt_discrete():
    px = np.arange(candidate_spacing/2, L, candidate_spacing)
    py = np.arange(candidate_spacing/2, W, candidate_spacing)
    Psites = np.array(list(itertools.product(px, py)))
    D = np.sqrt(((cells[:, None, :] - Psites[None, :, :])**2).sum(axis=2))
    best_total=None; best_combo=None
    for K in range(1, K_max+1):
        for combo in itertools.combinations(range(len(Psites)), K):
            Csub = D[:, combo]
            idx = np.argmin(Csub, axis=1)
            dmin = Csub[np.arange(Csub.shape[0]), idx]
            rake_s = float(np.sum(alpha*masses*(dmin**beta)))
            if (best_total is None) or (rake_s<best_total):
                best_total=rake_s; best_combo=combo
    return Psites[list(best_combo)] if best_combo is not None else np.array([[L/2,W/2]])

# =========================
# 4) Baselines (no “turns” overhead)
# =========================
def baseline_rake_time_to_centers(centers):
    if centers.size==0: centers=np.array([[L/2,W/2]])
    D = euclid(cells, centers)
    dmin = D.min(axis=1)
    return float(np.sum(alpha*masses*(dmin**beta)))

def baseline_rake_time_to_front(alpha_eff=None, beta_eff=None):
    a = alpha if alpha_eff is None else alpha_eff
    b = beta  if beta_eff  is None else beta_eff
    dfront = Y.ravel()
    return float(np.sum(a*masses*(dfront**b)))

# =========================
# 5) Outside-in raking (calibrated) + pile deposit
# =========================
def radial_arrival_times_calibrated(centers, angle_bins=24):
    if centers.size==0: centers=np.array([[L/2,W/2]])
    D = euclid(cells, centers)
    pile_id = np.argmin(D, axis=1)
    r_to_pile = D[np.arange(len(cells)), pile_id]

    bin_w = 2*np.pi/angle_bins
    t_arrive = np.full(len(cells), np.inf, dtype=float)

    for p in range(centers.shape[0]):
        cx, cy = centers[p]
        sel = (pile_id==p)
        if not np.any(sel): continue
        idxs = np.where(sel)[0]
        pts = cells[sel]
        vx = pts[:,0]-cx; vy = pts[:,1]-cy
        theta = np.arctan2(vy, vx)
        bins = np.clip(np.floor((theta+np.pi)/bin_w).astype(int), 0, angle_bins-1)
        for b in range(angle_bins):
            ray_mask = (bins==b)
            if not np.any(ray_mask): continue
            ray_idxs = idxs[ray_mask]
            r = r_to_pile[ray_idxs]
            order = np.argsort(-r)
            ray_idxs = ray_idxs[order]; r = r[order]
            m = masses[ray_idxs]
            r_ext = np.concatenate([r, np.array([0.0])])
            M_cum = np.cumsum(m[::-1])[::-1]
            dt = alpha * M_cum * ((r_ext[:-1]-r_ext[1:])**beta)
            T = np.cumsum(dt)
            t_arrive[ray_idxs] = T

    raw_total = float(np.nanmax(t_arrive[np.isfinite(t_arrive)])) if np.isfinite(t_arrive).any() else 0.0
    target_total = baseline_rake_time_to_centers(centers)
    scale = (target_total/raw_total) if raw_total>0 else 1.0
    t_arrive *= scale

    per_pile=[]
    for p in range(centers.shape[0]):
        mask = (pile_id==p); mp = masses[mask]
        per_pile.append({"t": t_arrive[mask], "m": mp, "M_total": float(mp.sum())})
    return t_arrive, pile_id, per_pile

def deposit_pile_disks_from_masses(centers, M_list, nx, ny, xs, ys, rho_cap, acell):
    acc = np.zeros((ny, nx), dtype=float)
    if centers.size==0: return acc
    Xc, Yc = np.meshgrid(xs, ys)
    for p, (cx,cy) in enumerate(centers):
        M = float(M_list[p])
        if M<=0: continue
        r = sqrt(M/(pi*rho_cap))
        mask = (Xc-cx)**2 + (Yc-cy)**2 <= r**2
        area = float(mask.sum())*acell
        if area<=0: continue
        dens = min(rho_cap, M/area)
        acc[mask] += dens
    return acc

def deposit_piles_disk_from_arrivals(centers, per_pile, t_sec, nx, ny, xs, ys, rho_cap, acell):
    Ms=[]
    for p in range(len(per_pile)):
        tp = per_pile[p]["t"]; mp = per_pile[p]["m"]
        Ms.append(float(mp[tp<=t_sec].sum()))
    return deposit_pile_disks_from_masses(centers, Ms, nx, ny, xs, ys, rho_cap, acell)

# =========================
# 6) Front-sweep timing + column-aware spillage
# =========================
def front_sweep_band_time(strip_ft, use_eff, eta, delta_beta, skip_rescale):
    """Return (rows_per, T_steps, alpha_eff, beta_eff)."""
    rows_per = max(1, int(round(strip_ft/s)))
    # Effective physics for front-sweep:
    alpha_eff = alpha/(eta if use_eff else 1.0)
    beta_eff  = max(0.5, beta - (delta_beta if use_eff else 0.0))  # keep >0
    # Raw pass timeline
    n_steps = int(np.ceil(ny/rows_per))
    T_raw = []; raw_total=0.0
    mass_per_row_init = mass_grid_init.sum(axis=1)  # (ny,)
    for k in range(n_steps):
        start = max(0, ny - (k+1)*rows_per)
        M_above = float(mass_per_row_init[start:].sum())
        dt_k = alpha_eff * M_above * (rows_per*s)**beta_eff
        raw_total += dt_k; T_raw.append(raw_total)
    T_raw = np.array(T_raw, dtype=float)
    # Optional rescale to the original baseline (with original alpha,beta)
    if (not skip_rescale) and len(T_raw)>0 and T_raw[-1]>0:
        T_target = baseline_rake_time_to_front(alpha, beta)  # baseline parity
        scale = T_target / T_raw[-1]
        T_steps = T_raw * scale
    else:
        T_steps = T_raw
    return rows_per, T_steps, alpha_eff, beta_eff

def band_snapshot_with_spillage_columns(t_sec, rows_per, T_steps, rho_cap):
    """Column-aware spillage for the active back-strip method."""
    k = int(np.searchsorted(T_steps, t_sec, side="right"))
    k = min(k, len(T_steps))
    adv_rows = k * rows_per

    # Fraction inside current pass
    if k == len(T_steps):
        frac = 1.0
    else:
        prev_t = 0.0 if k==0 else T_steps[k-1]
        dt_k = T_steps[k] - prev_t
        frac = 0.0 if dt_k<=1e-12 else np.clip((t_sec - prev_t)/dt_k, 0.0, 1.0)

    rho = rho_init.copy()
    if adv_rows>0:
        rho[ny-adv_rows:, :] = 0.0

    next_start = max(0, ny-(adv_rows+rows_per))
    next_end   = max(-1, ny-adv_rows-1)
    if next_start<=next_end and frac>0:
        rho[next_start:next_end+1,:] *= (1.0 - frac)

    # Mass per column entering band: fully-swept + fractional next strip
    M_full_cols = mass_grid_init[ny-adv_rows:ny,:].sum(axis=0) if adv_rows>0 else np.zeros(nx)
    M_next_cols = mass_grid_init[next_start:next_end+1,:].sum(axis=0) if next_start<=next_end else np.zeros(nx)
    M_band_cols = M_full_cols + frac*M_next_cols

    lead_row = min(max(0, ny-adv_rows), ny-1)
    K_cell = rho_cap * Acell
    remaining = M_band_cols.copy()
    for r in range(lead_row, ny):
        if np.all(remaining<=1e-12): break
        cap_row = np.maximum(0.0, K_cell - rho[r,:]*Acell)
        place = np.minimum(remaining, cap_row)
        rho[r,:] += place / Acell
        remaining -= place

    return rho

# =========================
# 7) Bagging utilities (outside-in + front band)
# =========================
def bag_mass_removed(M_total, t, C, b0, b1):
    if M_total<=1e-12 or t<=0: return 0.0
    M_rem=M_total; t_rem=t; removed=0.0
    while M_rem>1e-12 and t_rem>1e-12:
        cap=min(C, M_rem)
        if t_rem<=b0: break
        t_rem-=b0
        fill_mass=min(cap, t_rem/b1)
        removed+=fill_mass; M_rem-=fill_mass; t_rem-=fill_mass*b1
        if fill_mass<cap: break
    return removed

def compute_pile_order(centers, method="left_to_right"):
    n=centers.shape[0]
    if n==0: return np.array([], dtype=int)
    if method=="nn":
        left = int(np.argmin(centers[:,0])); order=[left]; used={left}; cur=left
        for _ in range(n-1):
            remaining=[i for i in range(n) if i not in used]
            d=np.linalg.norm(centers[remaining]-centers[cur], axis=1)
            nxt = remaining[int(np.argmin(d))]
            order.append(nxt); used.add(nxt); cur=nxt
        return np.array(order, dtype=int)
    return np.argsort(centers[:,0])

def walk_times_from_order(centers, order, v_walk_ftps):
    if centers.size==0: return np.array([])
    times=np.zeros(len(order))
    for j in range(1, len(order)):
        a=centers[order[j-1]]; b=centers[order[j]]
        dist=float(np.linalg.norm(a-b))
        times[j]=dist/max(1e-6, v_walk_ftps)
    return times

# =========================
# 8) Build strategies (static outside-in; front-sweep is dynamic)
# =========================
centers_list = [
    centers_bf(),
    np.empty((0,2)),
    centers_micro(),
    centers_opt_discrete(),
]
PANEL_TITLES = [
    "BF-centers (outside-in) — walking between piles",
    "Front-sweep (active strip) — column-aware spillage",
    "Micro-piles (outside-in) — walking between piles",
    "Optimization (discrete K≤5) — walking between piles",
]

# Precompute outside-in arrival schedules and bagging plans (static)
radial_data = {}
rake_total_secs = [0.0]*4
bag_total_secs  = [0.0]*4

for idx in [0,2,3]:
    centers = centers_list[idx]
    t_arrive, pile_of_cell, per_pile = radial_arrival_times_calibrated(centers, ANGLE_BINS)
    rake_total_secs[idx] = float(np.nanmax(t_arrive[np.isfinite(t_arrive)])) if np.isfinite(t_arrive).any() else 0.0
    pile_totals = np.array([pp["M_total"] for pp in per_pile]) if len(per_pile) else np.array([])
    order = compute_pile_order(centers, PILE_ORDER_METHOD)
    walk_times = walk_times_from_order(centers, order, WALK_SPEED_FTPS)
    if pile_totals.size:
        n_bags = np.ceil(pile_totals/BAG_CAPACITY_LB).astype(int)
        bag_times = n_bags*b0 + pile_totals*b1
        bag_total_secs[idx] = float(bag_times[order].sum() + walk_times.sum())
    else:
        bag_times = np.array([]); bag_total_secs[idx]=0.0
    radial_data[idx] = {
        "centers": centers,
        "t_arrive": t_arrive,
        "per_pile": per_pile,
        "pile_totals": pile_totals,
        "order": order,
        "bag_times": bag_times,
        "walk_times": walk_times,
    }

# =========================
# 9) Front-sweep state (dynamic; controlled by UI)
# =========================
# Defaults for UI:
FRONT_USE_EFF = True
FRONT_SKIP_RESCALE = True
FRONT_ETA = 1.5
FRONT_DELTA_BETA = 0.2
FRONT_STRIP_FT = 2.0

# These will be computed by recompute_front_sweep()
rows_per = 1
T_steps = np.array([0.0])
rho_final_band = None
mass_final_band = None
M_band_total = 0.0
bag_total_band = 0.0

def recompute_front_sweep():
    global rows_per, T_steps, rho_final_band, mass_final_band, M_band_total, bag_total_band
    rows_per, T_steps, a_eff, b_eff = front_sweep_band_time(
        FRONT_STRIP_FT, FRONT_USE_EFF, FRONT_ETA, FRONT_DELTA_BETA, FRONT_SKIP_RESCALE
    )
    rake_total_secs[1] = float(T_steps[-1]) if len(T_steps) else 0.0
    # Final band at end of raking
    rho_final = band_snapshot_with_spillage_columns(rake_total_secs[1], rows_per, T_steps, RHO_CAP)
    rho_final_band = rho_final
    mass_final_band = rho_final_band * Acell
    M_band_total = float(mass_final_band.sum())
    n_bags_band = int(ceil(M_band_total / BAG_CAPACITY_LB))
    bag_total_band = n_bags_band*b0 + b1*M_band_total
    bag_total_secs[1] = bag_total_band

recompute_front_sweep()

# =========================
# 10) Snapshot functions (animation)
# =========================
def density_snapshot_outside_in(panel_idx, t_sec):
    info = radial_data[panel_idx]; centers = info["centers"]
    T_rake = rake_total_secs[panel_idx]
    if t_sec <= T_rake or centers.size==0:
        remain = rho_init.copy().ravel()
        arrived = (info["t_arrive"]<=t_sec)
        remain[arrived]=0.0
        remain = remain.reshape(ny, nx)
        acc = deposit_piles_disk_from_arrivals(centers, info["per_pile"], t_sec, nx, ny, xs, ys, RHO_CAP, Acell)
        return remain + acc
    # Bagging (with walking)
    tau = t_sec - T_rake
    totals = info["pile_totals"]; Ms_rem = totals.copy()
    order = info["order"]; bag_times = info["bag_times"]; walk_times = info["walk_times"]
    t_left = tau
    for j, p in enumerate(order):
        wj = walk_times[j] if j < len(walk_times) else 0.0
        if t_left <= wj + 1e-12: break  # still walking
        t_left -= wj
        Mj = Ms_rem[p]; Tj = bag_times[p]
        if t_left >= Tj - 1e-12:
            Ms_rem[p] = 0.0; t_left -= Tj
        else:
            removed = bag_mass_removed(Mj, t_left, BAG_CAPACITY_LB, b0, b1)
            Ms_rem[p] = max(0.0, Mj - removed); t_left = 0.0; break
    acc = deposit_pile_disks_from_masses(centers, Ms_rem, nx, ny, xs, ys, RHO_CAP, Acell)
    return acc

# Front band: bag from front-first
band_order = np.argsort(Y.ravel())
band_cum = None  # computed on first call after recompute

def density_snapshot_front_sweep(t_sec):
    global band_cum
    T_rake = rake_total_secs[1]
    if t_sec <= T_rake:
        return band_snapshot_with_spillage_columns(t_sec, rows_per, T_steps, RHO_CAP)
    # Bagging
    tau = t_sec - T_rake
    if mass_final_band is None:
        return band_snapshot_with_spillage_columns(T_rake, rows_per, T_steps, RHO_CAP)
    removed = bag_mass_removed(M_band_total, tau, BAG_CAPACITY_LB, b0, b1)
    removed = min(removed, M_band_total)
    m0 = mass_final_band.ravel().copy()
    if band_cum is None:
        m_sorted = m0[band_order]
        band_cum = np.cumsum(m_sorted)
    idx = np.searchsorted(band_cum, removed, side="right")
    out = m0.copy()
    if idx>0: out[band_order[:idx]] = 0.0
    if idx < len(band_order):
        prev_cum = 0.0 if idx==0 else band_cum[idx-1]
        need = removed - prev_cum
        cid = band_order[idx]
        out[cid] = max(0.0, out[cid]-need)
    return out.reshape(ny, nx)/Acell

# =========================
# 11) Timeline, figure, and UI
# =========================
def panel_totals_seconds():
    return [rake_total_secs[i] + (bag_total_secs[i] if i!=1 else bag_total_band) for i in range(4)]

def minutes(x): return x/60.0

# Choose a horizon big enough for all settings (baseline is longest)
BASE_TOTAL = max(panel_totals_seconds())
total_minutes = int(np.ceil(BASE_TOTAL/60.0))
n_frames = max(1, total_minutes+1)

# Figure: main 2x2 + control panel axes
fig = plt.figure(figsize=(13.5, 8.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1,1,0.55], wspace=0.25, hspace=0.18)
ax00 = fig.add_subplot(gs[0,0]); ax01 = fig.add_subplot(gs[0,1])
ax10 = fig.add_subplot(gs[1,0]); ax11 = fig.add_subplot(gs[1,1])
axes = [ax00, ax01, ax10, ax11]

# Control panel axes (on the right column)
ax_ctrl = fig.add_subplot(gs[:,2])
ax_ctrl.axis("off")

# Place widgets inside ax_ctrl using add_axes with relative coords
# Make small sub-axes
def subax(x, y, w, h): return fig.add_axes([ax_ctrl.get_position().x0 + x*0.3,
                                            ax_ctrl.get_position().y0 + y*0.9,
                                            w*0.3, h*0.9])

# Checkboxes
rax = subax(0.05, 0.74, 0.9, 0.2)
check = CheckButtons(rax, ["Front: use efficiency", "Front: skip rescale"],
                     [FRONT_USE_EFF, FRONT_SKIP_RESCALE])

# Sliders
sax_eta  = subax(0.05, 0.60, 0.9, 0.06)
s_eta = Slider(sax_eta, "η (efficiency)", 1.0, 3.0, valinit=FRONT_ETA, valstep=0.05)

sax_db   = subax(0.05, 0.52, 0.9, 0.06)
s_db = Slider(sax_db, "Δβ", 0.0, 0.5, valinit=FRONT_DELTA_BETA, valstep=0.01)

sax_strip = subax(0.05, 0.44, 0.9, 0.06)
s_strip = Slider(sax_strip, "Strip (ft)", 1.0, 6.0, valinit=FRONT_STRIP_FT, valstep=0.5)

# Apply button
bax = subax(0.05, 0.33, 0.9, 0.08)
btn_apply = Button(bax, "Apply", color="#dddddd", hovercolor="#cccccc")

# Stats text box
stats_text = ax_ctrl.text(0.0, 0.28, "", transform=ax_ctrl.transAxes, fontsize=10, family="monospace", va="top")

# Minimal snapshot helper shared by both figures
def panel_snapshots(t_sec):
    return [
        density_snapshot_outside_in(0, t_sec),
        density_snapshot_front_sweep(t_sec),
        density_snapshot_outside_in(2, t_sec),
        density_snapshot_outside_in(3, t_sec),
    ]

# Initialize panels
vmax_panels = [max(float(rho_init.max()), RHO_CAP)]*4
initial_frames = panel_snapshots(0.0)
ims = []
titles = [
    "BF-centers (outside-in)",
    "Front-sweep (strip + spillage)",
    "Micro-piles (outside-in)",
    "Optimization (discrete K≤5)",
]
for i, ax in enumerate(axes):
    data0 = initial_frames[i]
    im = ax.imshow(data0, origin="lower", extent=[0,L,0,W], aspect="auto",
                   cmap=CMAP, norm=Normalize(vmin=0.0, vmax=vmax_panels[i]))
    fig.colorbar(im, ax=ax, shrink=0.85, label="lb/ft$^2$")
    if i!=1 and centers_list[i].size:
        ax.scatter(centers_list[i][:,0], centers_list[i][:,1], s=28, c="#2b8cbe", marker="x", linewidths=1.4)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    ims.append(im)

main_title = fig.suptitle("Comparing leaf raking strategies", fontsize=15)
time_text = fig.text(0.98, 0.965, f"Minute 0 / {total_minutes}", ha="right", va="top", fontsize=12)

def refresh_stats():
    totals = panel_totals_seconds()
    lines = []
    names = ["BF-centers", "Front-sweep", "Micro-piles", "Opt (disc)"]
    for i, nm in enumerate(names):
        rmin = minutes(rake_total_secs[i])
        bmin = minutes((bag_total_secs[i] if i!=1 else bag_total_band))
        tmin = minutes(totals[i])
        lines.append(f"{nm:12}: rake={rmin:6.1f}  bag={bmin:6.1f}  total={tmin:6.1f} min")
    stats_text.set_text("\n".join(lines))
refresh_stats()

def update(frame_idx):
    # seconds from start
    t_sec = frame_idx * SECONDS_PER_FRAME
    frames = panel_snapshots(t_sec)
    for im_obj, data in zip(ims, frames):
        im_obj.set_data(data)
    time_text.set_text(f"Minute {frame_idx} / {total_minutes}")
    return (*ims, time_text)

# Minimal export figure (2x2 only)
export_fig, export_axes = plt.subplots(2, 2, figsize=EXPORT_FIGSIZE, constrained_layout=True)
export_axes = export_axes.ravel()
export_ims = []
export_titles = ["BF-centers", "Front sweep", "Micro-piles", "Optimization"]
for i, ax in enumerate(export_axes):
    data0 = initial_frames[i]
    im = ax.imshow(data0, origin="lower", extent=[0, L, 0, W], aspect="auto",
                   cmap=CMAP, norm=Normalize(vmin=0.0, vmax=vmax_panels[i]))
    if i != 1 and centers_list[i].size:
        ax.scatter(centers_list[i][:, 0], centers_list[i][:, 1], s=28, c="#2b8cbe", marker="x", linewidths=1.4)
    ax.set_title(export_titles[i], fontsize=11)
    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    export_ims.append(im)

export_cbar = export_fig.colorbar(export_ims[0], ax=export_axes, fraction=0.035, pad=0.04)
export_cbar.set_label("lb/ft$^2$")
export_main_title = export_fig.suptitle("Comparing leaf raking strategies", fontsize=15)
export_time_text = export_fig.text(0.98, 0.965, f"Minute 0 / {total_minutes}", ha="right", va="top", fontsize=12)

def update_export(frame_idx):
    t_sec = frame_idx * SECONDS_PER_FRAME
    frames = panel_snapshots(t_sec)
    for im_obj, data in zip(export_ims, frames):
        im_obj.set_data(data)
    export_time_text.set_text(f"Minute {frame_idx} / {total_minutes}")
    return (*export_ims, export_time_text)

anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/FPS, blit=False, repeat=True)
export_anim = FuncAnimation(export_fig, update_export, frames=n_frames, interval=1000/FPS, blit=False, repeat=True)
EXPORT_FIG_ACTIVE = True

# === UI callbacks ===
def on_apply(event):
    global FRONT_USE_EFF, FRONT_SKIP_RESCALE, FRONT_ETA, FRONT_DELTA_BETA, FRONT_STRIP_FT, band_cum
    vals = check.get_status()
    FRONT_USE_EFF      = bool(vals[0])
    FRONT_SKIP_RESCALE = bool(vals[1])
    FRONT_ETA          = float(s_eta.val)
    FRONT_DELTA_BETA   = float(s_db.val)
    FRONT_STRIP_FT     = float(s_strip.val)
    band_cum = None  # reset prefix sums for band bagging
    recompute_front_sweep()
    refresh_stats()
    frames0 = panel_snapshots(0.0)
    for im_obj, data in zip(ims, frames0):
        im_obj.set_data(data)
    time_text.set_text(f"Minute 0 / {total_minutes}")
    if EXPORT_FIG_ACTIVE:
        for im_obj, data in zip(export_ims, frames0):
            im_obj.set_data(data)
        export_time_text.set_text(f"Minute 0 / {total_minutes}")
        export_fig.canvas.draw_idle()
    fig.canvas.draw_idle()

btn_apply.on_clicked(on_apply)

# Save file (MP4 preferred; GIF fallback)
if SAVE_OUTPUT:
    try:
        writer = FFMpegWriter(fps=FPS)
        anim.save(str(OUT_PATH), writer=writer, dpi=150)
        print(f"Saved animation to {OUT_PATH}")
    except Exception as e:
        gif_path = OUT_DIR / "heatmaps_2x2.gif"
        print("FFmpeg not available or failed; saving GIF instead...", e)
        anim.save(str(gif_path), writer=PillowWriter(fps=FPS), dpi=150)
        print(f"Saved animation to {gif_path}")
    try:
        export_anim.save(str(OUT_GIF_MINIMAL), writer=PillowWriter(fps=FPS), dpi=150)
        print(f"Saved minimal GIF to {OUT_GIF_MINIMAL}")
    except Exception as e:
        print("Failed to save minimal GIF:", e)

# Do not show the export-only figure in interactive mode
plt.close(export_fig)
EXPORT_FIG_ACTIVE = False

if SHOW_WINDOW:
    plt.show()
else:
    plt.close(fig)
