from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import BaggingCurve, SimulationConfig


@dataclass
class BaggingPlan:
    pile_order: np.ndarray
    bag_durations: np.ndarray
    walk_durations: np.ndarray

    @property
    def total_seconds(self) -> float:
        return float(self.bag_durations.sum() + self.walk_durations.sum())

    @property
    def cumulative_bag_seconds(self) -> np.ndarray:
        return np.cumsum(self.bag_durations)


def bag_mass_removed(total_mass: float, time_seconds: float, capacity_lb: float,
                     setup_seconds: float, stuffing_rate_sec_per_lb: float) -> float:
    if total_mass <= 1e-12 or time_seconds <= 0:
        return 0.0
    remaining_mass = total_mass
    remaining_time = time_seconds
    removed = 0.0
    while remaining_mass > 1e-12 and remaining_time > 1e-12:
        cap = min(capacity_lb, remaining_mass)
        if remaining_time <= setup_seconds:
            break
        remaining_time -= setup_seconds
        fill_mass = min(cap, remaining_time / stuffing_rate_sec_per_lb)
        removed += fill_mass
        remaining_mass -= fill_mass
        remaining_time -= fill_mass * stuffing_rate_sec_per_lb
        if fill_mass < cap:
            break
    return removed


def compute_pile_order(centers: np.ndarray, method: Literal["left_to_right", "nn"] = "left_to_right") -> np.ndarray:
    n = centers.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if method == "nn":
        left = int(np.argmin(centers[:, 0]))
        order = [left]
        used = {left}
        cur = left
        for _ in range(n - 1):
            remaining = [i for i in range(n) if i not in used]
            d = np.linalg.norm(centers[remaining] - centers[cur], axis=1)
            nxt = remaining[int(np.argmin(d))]
            order.append(nxt)
            used.add(nxt)
            cur = nxt
        return np.array(order, dtype=int)
    return np.argsort(centers[:, 0])


def walk_times_from_order(centers: np.ndarray, order: np.ndarray, speed_ft_s: float) -> np.ndarray:
    if centers.size == 0:
        return np.array([])
    times = np.zeros(len(order))
    for j in range(1, len(order)):
        a = centers[order[j - 1]]
        b = centers[order[j]]
        dist = float(np.linalg.norm(a - b))
        times[j] = dist / max(1e-6, speed_ft_s)
    return times


def build_bagging_plan(config: SimulationConfig, bag_curve: BaggingCurve, centers: np.ndarray,
                       pile_masses: np.ndarray) -> BaggingPlan:
    if centers.size == 0 or pile_masses.size == 0:
        order = compute_pile_order(centers, config.pile_order_method)
        walk_times = walk_times_from_order(centers, order, config.walk_speed_ft_s)
        return BaggingPlan(pile_order=order, bag_durations=np.zeros_like(order, dtype=float), walk_durations=walk_times)

    order = compute_pile_order(centers, config.pile_order_method)
    walk_times = walk_times_from_order(centers, order, config.walk_speed_ft_s)
    bag_capacity = config.bag_capacity_lb
    setup = bag_curve.setup_seconds
    stuffing = bag_curve.stuffing_rate_sec_per_lb

    bag_times = np.zeros(len(order), dtype=float)
    for idx, pile_idx in enumerate(order):
        mass = float(pile_masses[pile_idx])
        if mass <= 0:
            bag_times[idx] = 0.0
            continue
        n_bags = int(np.ceil(mass / bag_capacity))
        bag_times[idx] = n_bags * setup + mass * stuffing
    return BaggingPlan(pile_order=order, bag_durations=bag_times, walk_durations=walk_times)

