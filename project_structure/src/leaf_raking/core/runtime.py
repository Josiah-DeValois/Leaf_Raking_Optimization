from __future__ import annotations

from .calibration import run_calibration
from .config import SimulationConfig, SimulationState
from .distribution import build_mass_grid


def build_state(config: SimulationConfig) -> SimulationState:
    calibration = run_calibration(config.calibration, config.bag_capacity_lb)
    grid = build_mass_grid(config)
    return SimulationState(config=config, calibration=calibration, grid=grid)

