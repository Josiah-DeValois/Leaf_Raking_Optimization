from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import CalibrationResult, GridCache, SimulationConfig
from ..core.timeline import baseline_rake_time_to_front


@dataclass
class FrontSweepSchedule:
    rows_per_pass: int
    mass_per_row: np.ndarray
    time_steps: np.ndarray
    total_seconds: float


def compute_front_sweep_schedule(config: SimulationConfig, calibration: CalibrationResult,
                                 grid: GridCache, strip_ft: float) -> FrontSweepSchedule:
    spacing_ft = config.grid.spacing_ft
    rows_per = max(1, int(round(strip_ft / spacing_ft)))
    ny = grid.mass_grid.shape[0]
    n_steps = int(np.ceil(ny / rows_per))
    mass_per_row = grid.mass_grid.sum(axis=1)
    T_raw = []
    raw_total = 0.0
    for k in range(n_steps):
        start = max(0, ny - (k + 1) * rows_per)
        M_above = float(mass_per_row[start:].sum())
        dt_k = calibration.alpha * M_above * (rows_per * spacing_ft) ** calibration.beta
        raw_total += dt_k
        T_raw.append(raw_total)
    T_raw = np.array(T_raw, dtype=float)
    if len(T_raw) and T_raw[-1] > 0:
        target = baseline_rake_time_to_front(calibration.alpha, calibration.beta, grid)
        scale = target / T_raw[-1]
        T_steps = T_raw * scale
    else:
        T_steps = T_raw
    total_seconds = float(T_steps[-1]) if len(T_steps) else 0.0
    return FrontSweepSchedule(rows_per_pass=rows_per,
                              mass_per_row=mass_per_row,
                              time_steps=T_steps,
                              total_seconds=total_seconds)


def band_snapshot_with_spillage_columns(grid: GridCache, schedule: FrontSweepSchedule,
                                        rho_cap: float, t_sec: float) -> np.ndarray:
    rows_per = schedule.rows_per_pass
    T_steps = schedule.time_steps
    mass_per_row_init = schedule.mass_per_row
    rho_init = grid.mass_grid / grid.cell_area
    ny, nx = rho_init.shape
    k = int(np.searchsorted(T_steps, t_sec, side="right"))
    k = min(k, len(T_steps))
    adv_rows = k * rows_per

    if k == len(T_steps):
        frac = 1.0
    else:
        prev_t = 0.0 if k == 0 else T_steps[k - 1]
        dt_k = T_steps[k] - prev_t
        frac = 0.0 if dt_k <= 1e-12 else np.clip((t_sec - prev_t) / dt_k, 0.0, 1.0)

    rho = rho_init.copy()
    if adv_rows > 0:
        rho[ny - adv_rows:, :] = 0.0

    next_start = max(0, ny - (adv_rows + rows_per))
    next_end = max(-1, ny - adv_rows - 1)
    if next_start <= next_end and frac > 0:
        rho[next_start:next_end + 1, :] *= (1.0 - frac)

    mass_grid_init = grid.mass_grid
    M_full_cols = mass_grid_init[ny - adv_rows:ny, :].sum(axis=0) if adv_rows > 0 else np.zeros(nx)
    M_next_cols = mass_grid_init[next_start:next_end + 1, :].sum(axis=0) if next_start <= next_end else np.zeros(nx)
    M_band_cols = M_full_cols + frac * M_next_cols

    lead_row = ny - adv_rows
    lead_row = min(max(0, lead_row), ny - 1)

    Acell = grid.cell_area
    remaining = M_band_cols.copy()
    for r in range(lead_row, ny):
        if np.all(remaining <= 1e-12):
            break
        cap_row = np.maximum(0.0, rho_cap * Acell - rho[r, :] * Acell)
        place = np.minimum(remaining, cap_row)
        rho[r, :] += place / Acell
        remaining -= place

    return rho
