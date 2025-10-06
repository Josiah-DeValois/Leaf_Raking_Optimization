from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .config import CalibrationResult, GridCache, SimulationConfig
from .math_utils import euclid


@dataclass
class PileCentersResult:
    centers: np.ndarray
    metadata: Dict[str, float]


def centers_balanced(config: SimulationConfig, grid: GridCache, calibration: CalibrationResult) -> PileCentersResult:
    total_mass = grid.total_mass
    bag_cap = config.bag_capacity_lb
    k = max(1, int(round(total_mass / (2 * bag_cap))))
    x_bounds = np.linspace(0, config.yard.length_ft, k + 1)
    centers = np.array([[(x_bounds[j] + x_bounds[j + 1]) / 2.0, config.yard.width_ft / 2.0]
                        for j in range(k)])
    return PileCentersResult(centers=centers, metadata={"k": float(k)})


def centers_micro(config: SimulationConfig, grid: GridCache, calibration: CalibrationResult, seed: int = 42) -> PileCentersResult:
    rng = np.random.default_rng(seed)
    bag_cap = config.bag_capacity_lb
    masses = grid.mass_vector
    total_mass = grid.total_mass
    k0 = max(1, int(round(total_mass / (2 * bag_cap))))
    probs = masses / (total_mass + 1e-12)
    idx0 = rng.choice(len(masses), size=k0, replace=False, p=probs)
    centers = grid.cell_centers[idx0].copy()
    for _ in range(8):
        D2 = euclid(grid.cell_centers, centers) ** 2
        labels = np.argmin(D2, axis=1)
        m_c = np.array([masses[labels == j].sum() for j in range(centers.shape[0])], dtype=float)
        to_split = np.where(m_c > 2 * bag_cap)[0]
        new_centers = []
        for j in range(centers.shape[0]):
            mask = labels == j
            if not np.any(mask):
                new_centers.append(centers[j])
                continue
            pts = grid.cell_centers[mask]
            w = masses[mask]
            if j in to_split:
                a = pts[np.argmax(((pts - pts.mean(0)) ** 2).sum(1))]
                b = pts[np.argmax(((pts - a) ** 2).sum(1))]
                ca, cb = a.copy(), b.copy()
                for _ in range(5):
                    da = ((pts - ca) ** 2).sum(1)
                    db = ((pts - cb) ** 2).sum(1)
                    lab = da <= db
                    wa = w[lab].sum() + 1e-9
                    wb = w[~lab].sum() + 1e-9
                    ca = np.array([np.sum(pts[lab, 0] * w[lab]) / wa,
                                   np.sum(pts[lab, 1] * w[lab]) / wa])
                    cb = np.array([np.sum(pts[~lab, 0] * w[~lab]) / wb,
                                   np.sum(pts[~lab, 1] * w[~lab]) / wb])
                new_centers.append(ca)
                new_centers.append(cb)
            else:
                cx = np.average(pts[:, 0], weights=w)
                cy = np.average(pts[:, 1], weights=w)
                new_centers.append([cx, cy])
        centers = np.array(new_centers)
    return PileCentersResult(centers=centers, metadata={"k": float(len(centers))})


def centers_optimal_discrete(config: SimulationConfig, grid: GridCache,
                              calibration: CalibrationResult) -> PileCentersResult:
    candidate_spacing = config.candidate_spacing_ft
    px = np.arange(candidate_spacing / 2, config.yard.length_ft, candidate_spacing)
    py = np.arange(candidate_spacing / 2, config.yard.width_ft, candidate_spacing)
    Psites = np.array(list(itertools.product(px, py)))
    if Psites.size == 0:
        center = np.array([[config.yard.length_ft / 2, config.yard.width_ft / 2]])
        return PileCentersResult(centers=center, metadata={"k": 1.0})
    D = euclid(grid.cell_centers, Psites)
    best_total = None
    best_combo = None
    masses = grid.mass_vector
    alpha = calibration.alpha
    beta = calibration.beta
    for K in range(1, config.max_candidates + 1):
        for combo in itertools.combinations(range(len(Psites)), K):
            Csub = D[:, combo]
            idx = np.argmin(Csub, axis=1)
            dmin = Csub[np.arange(Csub.shape[0]), idx]
            rake_s = float(np.sum(alpha * masses * (dmin ** beta)))
            if (best_total is None) or (rake_s < best_total):
                best_total = rake_s
                best_combo = combo
    centers = Psites[list(best_combo)] if best_combo is not None else np.array([[config.yard.length_ft / 2,
                                                                                config.yard.width_ft / 2]])
    return PileCentersResult(centers=centers, metadata={"k": float(len(centers))})


PileStrategy = Callable[[SimulationConfig, GridCache, CalibrationResult], PileCentersResult]

PILE_STRATEGIES: Dict[str, PileStrategy] = {
    "balanced": centers_balanced,
    "micro": centers_micro,
    "discrete_opt": centers_optimal_discrete,
}

