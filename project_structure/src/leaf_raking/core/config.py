from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class YardGeometry:
    length_ft: float = 60.0
    width_ft: float = 40.0
    tree_center: Tuple[float, float] = (15.0, 20.0)
    trunk_radius_ft: float = 1.5


@dataclass
class LeafDistributionParams:
    phi_deg: float = 90.0
    axis_ratio: float = 1.5
    sigma: float = 10.0
    rho0: float = 0.03
    amplitude: float = 0.28
    power: float = 2.0


@dataclass
class GridSpec:
    spacing_ft: float = 1.0


@dataclass
class CalibrationData:
    raking_splits_sec: Dict[int, int] = field(default_factory=lambda: {
        10: 7,
        20: 9,
        30: 13,
        40: 17,
        50: 21,
        60: 28,
        70: 35,
        80: 42,
        90: 49,
        100: 51,
    })
    bagging_times: Dict[float, str] = field(default_factory=lambda: {
        0.25: "1:39",
        0.50: "1:45",
        0.75: "1:45",
        1.00: "2:00",
    })


@dataclass
class OutputConfig:
    fps: int = 2
    seconds_per_frame: int = 60
    save_output: bool = True
    show_window: bool = True
    output_dir: Path = Path("./outputs")
    mp4_name: str = "heatmaps_2x2.mp4"
    fallback_gif_name: str = "heatmaps_2x2_minimal.gif"

    def ensure_dir(self) -> Path:
        self.output_dir.mkdir(exist_ok=True)
        return self.output_dir


@dataclass
class SimulationConfig:
    yard: YardGeometry = field(default_factory=YardGeometry)
    distribution: LeafDistributionParams = field(default_factory=LeafDistributionParams)
    grid: GridSpec = field(default_factory=GridSpec)
    calibration: CalibrationData = field(default_factory=CalibrationData)
    output: OutputConfig = field(default_factory=OutputConfig)
    bag_capacity_lb: float = 35.0
    rho_cap_lb_ft2: float = 3.0
    angle_bins: int = 24
    candidate_spacing_ft: float = 10.0
    max_candidates: int = 5
    walk_speed_ft_s: float = 3.5
    pile_order_method: str = "left_to_right"


@dataclass
class BaggingCurve:
    setup_seconds: float
    stuffing_rate_sec_per_lb: float
    curve_fn: Callable[[float], float]


@dataclass
class CalibrationResult:
    alpha: float
    beta: float
    bag_curve: BaggingCurve


@dataclass
class GridCache:
    xs: np.ndarray
    ys: np.ndarray
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    cell_centers: np.ndarray
    cell_area: float
    mass_grid: np.ndarray
    mass_vector: np.ndarray
    total_mass: float


@dataclass
class SimulationState:
    config: SimulationConfig
    calibration: CalibrationResult
    grid: GridCache

