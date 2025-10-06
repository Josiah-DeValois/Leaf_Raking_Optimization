from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Dict, List

import numpy as np

from .config import GridCache
from .math_utils import euclid


@dataclass
class RadialRakeSchedule:
    centers: np.ndarray
    arrival_times: np.ndarray
    pile_ids: np.ndarray
    per_pile: List[Dict[str, np.ndarray]]
    total_seconds: float

    @property
    def pile_totals(self) -> np.ndarray:
        return np.array([entry["M_total"] for entry in self.per_pile], dtype=float) if self.per_pile else np.array([])


def baseline_rake_time_to_centers(alpha: float, beta: float, centers: np.ndarray, grid: GridCache,
                                  yard_length: float, yard_width: float) -> float:
    if centers.size == 0:
        centers = np.array([[yard_length / 2, yard_width / 2]])
    D = euclid(grid.cell_centers, centers)
    dmin = D.min(axis=1)
    return float(np.sum(alpha * grid.mass_vector * (dmin ** beta)))


def baseline_rake_time_to_front(alpha: float, beta: float, grid: GridCache) -> float:
    dfront = grid.mesh_y.ravel()
    return float(np.sum(alpha * grid.mass_vector * (dfront ** beta)))


def radial_arrival_schedule(alpha: float, beta: float, centers: np.ndarray, grid: GridCache,
                             angle_bins: int, yard_length: float, yard_width: float) -> RadialRakeSchedule:
    if centers.size == 0:
        centers = np.array([[yard_length / 2, yard_width / 2]])
    D = euclid(grid.cell_centers, centers)
    pile_id = np.argmin(D, axis=1)
    r_to_pile = D[np.arange(len(grid.cell_centers)), pile_id]

    bin_w = 2 * np.pi / angle_bins
    t_arrive = np.full(len(grid.cell_centers), np.inf, dtype=float)

    masses = grid.mass_vector
    cells = grid.cell_centers

    for p in range(centers.shape[0]):
        cx, cy = centers[p]
        sel = pile_id == p
        if not np.any(sel):
            continue
        idxs = np.where(sel)[0]
        pts = cells[sel]
        vx = pts[:, 0] - cx
        vy = pts[:, 1] - cy
        theta = np.arctan2(vy, vx)
        bins = np.clip(np.floor((theta + np.pi) / bin_w).astype(int), 0, angle_bins - 1)
        for b in range(angle_bins):
            ray_mask = bins == b
            if not np.any(ray_mask):
                continue
            ray_idxs = idxs[ray_mask]
            r = r_to_pile[ray_idxs]
            order = np.argsort(-r)
            ray_idxs = ray_idxs[order]
            r = r[order]
            m = masses[ray_idxs]
            r_ext = np.concatenate([r, np.array([0.0])])
            M_cum = np.cumsum(m[::-1])[::-1]
            dt = alpha * M_cum * ((r_ext[:-1] - r_ext[1:]) ** beta)
            T = np.cumsum(dt)
            t_arrive[ray_idxs] = T

    finite_mask = np.isfinite(t_arrive)
    raw_total = float(np.nanmax(t_arrive[finite_mask])) if finite_mask.any() else 0.0
    target_total = baseline_rake_time_to_centers(alpha, beta, centers, grid, yard_length, yard_width)
    scale = (target_total / raw_total) if raw_total > 0 else 1.0
    t_arrive *= scale

    per_pile: List[Dict[str, np.ndarray]] = []
    for p in range(centers.shape[0]):
        mask = pile_id == p
        mp = grid.mass_vector[mask]
        per_pile.append({"t": t_arrive[mask], "m": mp, "M_total": float(mp.sum())})

    total_seconds = float(np.nanmax(t_arrive[finite_mask])) if finite_mask.any() else 0.0
    return RadialRakeSchedule(centers=centers,
                              arrival_times=t_arrive,
                              pile_ids=pile_id,
                              per_pile=per_pile,
                              total_seconds=total_seconds)


def deposit_pile_disks_from_masses(centers: np.ndarray, masses: List[float], grid: GridCache,
                                   rho_cap: float) -> np.ndarray:
    acc = np.zeros_like(grid.mass_grid, dtype=float)
    if centers.size == 0:
        return acc
    xs = grid.xs
    ys = grid.ys
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    for p, (cx, cy) in enumerate(centers):
        M = float(masses[p])
        if M <= 0:
            continue
        r = sqrt(M / (pi * rho_cap))
        mask = (mesh_x - cx) ** 2 + (mesh_y - cy) ** 2 <= r ** 2
        area = float(mask.sum()) * grid.cell_area
        if area <= 0:
            continue
        dens = min(rho_cap, M / area)
        acc[mask] += dens
    return acc


def deposit_from_arrivals(centers: np.ndarray, per_pile, t_sec: float, grid: GridCache,
                          rho_cap: float) -> np.ndarray:
    collected = []
    for p in range(len(per_pile)):
        tp = per_pile[p]["t"]
        mp = per_pile[p]["m"]
        collected.append(float(mp[tp <= t_sec].sum()))
    return deposit_pile_disks_from_masses(centers, collected, grid, rho_cap)

