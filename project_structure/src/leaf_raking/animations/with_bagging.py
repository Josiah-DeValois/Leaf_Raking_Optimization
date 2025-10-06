#!/usr/bin/env python3
"""Structured reimplementation of the original with_bagging animation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize

from leaf_raking.core.bagging import BaggingPlan, bag_mass_removed, build_bagging_plan
from leaf_raking.core.config import SimulationConfig
from leaf_raking.core.runtime import build_state
from leaf_raking.core.piles import PILE_STRATEGIES, PileCentersResult
from leaf_raking.core.timeline import (RadialRakeSchedule, deposit_from_arrivals,
                                       deposit_pile_disks_from_masses, radial_arrival_schedule)
from leaf_raking.strategies.front_sweep import band_snapshot_with_spillage_columns, compute_front_sweep_schedule

# =========================
# CONFIG
# =========================
FPS = 2
SECONDS_PER_FRAME = 60
SAVE_OUTPUT = True
SHOW_WINDOW = True
FRONT_SWEEP_STRIP_FT = 2.0

COLORS = ["#edf8e9", "#a1d99b", "#31a354", "#fed976", "#fd8d3c", "#e31a1c"]
CMAP = LinearSegmentedColormap.from_list("leaf_hot", COLORS, N=256)
PANEL_TITLES = [
    "BF-centers (outside-in, bag after rake)",
    "Front-sweep (active strip, bag from front)",
    "Micro-piles (outside-in, bag after rake)",
    "Optimization (discrete K≤5, bag after rake)",
]


def build_outside_in_panels(config: SimulationConfig):
    state = build_state(config)
    bag_curve = state.calibration.bag_curve
    rho_cap = config.rho_cap_lb_ft2

    rho_init = state.grid.mass_grid / state.grid.cell_area
    ny, nx = rho_init.shape

    radial_data = {}
    rake_total_secs = [0.0] * 4
    bag_total_secs = [0.0] * 4

    strategy_map = {0: "balanced", 2: "micro", 3: "discrete_opt"}
    for panel_idx, strategy_key in strategy_map.items():
        centers: PileCentersResult = PILE_STRATEGIES[strategy_key](config, state.grid, state.calibration)
        schedule: RadialRakeSchedule = radial_arrival_schedule(
            state.calibration.alpha,
            state.calibration.beta,
            centers.centers,
            state.grid,
            config.angle_bins,
            config.yard.length_ft,
            config.yard.width_ft,
        )
        bag_plan: BaggingPlan = build_bagging_plan(config, bag_curve, schedule.centers, schedule.pile_totals)
        radial_data[panel_idx] = {
            "centers": schedule.centers,
            "schedule": schedule,
            "bag_plan": bag_plan,
        }
        rake_total_secs[panel_idx] = schedule.total_seconds
        bag_total_secs[panel_idx] = float(bag_plan.bag_durations.sum())

    front_schedule = compute_front_sweep_schedule(config, state.calibration, state.grid, FRONT_SWEEP_STRIP_FT)
    rake_total_secs[1] = front_schedule.total_seconds

    rho_final_band = band_snapshot_with_spillage_columns(state.grid, front_schedule, rho_cap, rake_total_secs[1])
    mass_final_band = rho_final_band * state.grid.cell_area
    M_band_total = float(mass_final_band.sum())
    bag_curve_setup = bag_curve.setup_seconds
    bag_curve_rate = bag_curve.stuffing_rate_sec_per_lb
    n_bags_band = int(np.ceil(M_band_total / config.bag_capacity_lb))
    bag_total_band = n_bags_band * bag_curve_setup + M_band_total * bag_curve_rate
    bag_total_secs[1] = bag_total_band

    band_order = np.argsort(state.grid.mesh_y.ravel())
    band_mass_sorted = mass_final_band.ravel()[band_order]
    band_cumulative = np.cumsum(band_mass_sorted)

    totals = [rake_total_secs[i] + bag_total_secs[i] for i in range(4)]

    return {
        "state": state,
        "rho_init": rho_init,
        "radial_data": radial_data,
        "front_schedule": front_schedule,
        "rho_final_band": rho_final_band,
        "mass_final_band": mass_final_band,
        "M_band_total": M_band_total,
        "band_order": band_order,
        "band_cumulative": band_cumulative,
        "rake_total_secs": rake_total_secs,
        "bag_total_secs": bag_total_secs,
        "panel_total_secs": totals,
    }


def build_animation():
    config = SimulationConfig()
    output_dir = config.output.ensure_dir()
    assets = build_outside_in_panels(config)

    state = assets["state"]
    bag_curve = state.calibration.bag_curve
    rho_init = assets["rho_init"]
    radial_data = assets["radial_data"]
    front_schedule = assets["front_schedule"]
    rho_final_band = assets["rho_final_band"]
    mass_final_band = assets["mass_final_band"]
    M_band_total = assets["M_band_total"]
    band_order = assets["band_order"]
    band_cumulative = assets["band_cumulative"]
    rake_total_secs = assets["rake_total_secs"]
    bag_total_secs = assets["bag_total_secs"]
    panel_total_secs = assets["panel_total_secs"]

    ny, nx = rho_init.shape
    xs = state.grid.xs
    ys = state.grid.ys
    rho_cap = state.config.rho_cap_lb_ft2

    total_seconds = max(panel_total_secs)
    total_minutes = int(np.ceil(total_seconds / 60.0))
    n_frames = max(1, total_minutes + 1)

    def density_snapshot_outside_in(panel_idx: int, t_sec: float) -> np.ndarray:
        data = radial_data[panel_idx]
        schedule: RadialRakeSchedule = data["schedule"]
        centers = data["centers"]
        T_rake = rake_total_secs[panel_idx]
        if t_sec <= T_rake or centers.size == 0:
            remain = rho_init.copy().ravel()
            arrived = schedule.arrival_times <= t_sec
            remain[arrived] = 0.0
            remain = remain.reshape(ny, nx)
            acc = deposit_from_arrivals(centers, schedule.per_pile, t_sec, state.grid, rho_cap)
            return remain + acc
        tau = t_sec - T_rake
        totals = schedule.pile_totals
        Ms_rem = totals.copy()
        bag_plan: BaggingPlan = data["bag_plan"]
        order = bag_plan.pile_order
        bag_times = bag_plan.bag_durations
        for j, pile_idx in enumerate(order):
            if tau <= 1e-12:
                break
            Tj = bag_times[j]
            if tau >= Tj - 1e-12:
                Ms_rem[pile_idx] = 0.0
                tau -= Tj
            else:
                removed = bag_mass_removed(totals[pile_idx], tau, state.config.bag_capacity_lb,
                                           bag_curve.setup_seconds, bag_curve.stuffing_rate_sec_per_lb)
                Ms_rem[pile_idx] = max(0.0, totals[pile_idx] - removed)
                tau = 0.0
                break
        acc = deposit_pile_disks_from_masses(centers, Ms_rem, state.grid, rho_cap)
        return acc

    def density_snapshot_front_sweep(t_sec: float) -> np.ndarray:
        T_rake = front_schedule.total_seconds
        if t_sec <= T_rake:
            return band_snapshot_with_spillage_columns(state.grid, front_schedule, rho_cap, t_sec)
        tau = t_sec - T_rake
        removed = bag_mass_removed(M_band_total, tau, state.config.bag_capacity_lb,
                                   bag_curve.setup_seconds, bag_curve.stuffing_rate_sec_per_lb)
        removed = min(removed, M_band_total)
        out = mass_final_band.ravel().copy()
        idx = np.searchsorted(band_cumulative, removed, side="right")
        if idx > 0:
            out[band_order[:idx]] = 0.0
        if idx < len(band_order):
            prev_cum = 0.0 if idx == 0 else band_cumulative[idx - 1]
            need = removed - prev_cum
            if need > 0:
                out[band_order[idx]] = max(0.0, out[band_order[idx]] - need)
        return out.reshape(ny, nx) / state.grid.cell_area

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    axes = axes.ravel()
    vmax_panels = [max(float(rho_init.max()), rho_cap)] * 4
    ims = []

    initial_images = [
        density_snapshot_outside_in(0, 0.0),
        density_snapshot_front_sweep(0.0),
        density_snapshot_outside_in(2, 0.0),
        density_snapshot_outside_in(3, 0.0),
    ]

    for i, ax in enumerate(axes):
        im = ax.imshow(
            initial_images[i], origin="lower", extent=[0, state.config.yard.length_ft, 0, state.config.yard.width_ft],
            aspect="auto", cmap=CMAP, norm=Normalize(vmin=0.0, vmax=vmax_panels[i])
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="lb/ft$^2$")
        if i != 1:
            centers = radial_data.get(i, {}).get("centers")
            if centers is not None and centers.size:
                ax.scatter(centers[:, 0], centers[:, 1], s=28, c="#2b8cbe", marker="x", linewidths=1.4)
        ax.set_title(PANEL_TITLES[i], fontsize=10)
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")
        ims.append(im)

    suptitle = fig.suptitle("Minute 0", fontsize=14)

    def update(frame_idx):
        t_sec = frame_idx * SECONDS_PER_FRAME
        ims[0].set_data(density_snapshot_outside_in(0, t_sec))
        ims[1].set_data(density_snapshot_front_sweep(t_sec))
        ims[2].set_data(density_snapshot_outside_in(2, t_sec))
        ims[3].set_data(density_snapshot_outside_in(3, t_sec))
        suptitle.set_text(f"Minute {frame_idx} / {total_minutes}")
        return (*ims, suptitle)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False, repeat=True)

    if SHOW_WINDOW:
        plt.show(block=False)

    if SAVE_OUTPUT:
        out_path = output_dir / config.output.mp4_name
        try:
            writer = FFMpegWriter(fps=FPS)
            anim.save(str(out_path), writer=writer, dpi=150)
            print(f"Saved animation to {out_path}")
        except Exception as exc:
            gif_path = output_dir / config.output.fallback_gif_name
            print("FFmpeg unavailable or failed; saving GIF", exc)
            anim.save(str(gif_path), writer=PillowWriter(fps=FPS))
            print(f"Saved animation to {gif_path}")

    if SHOW_WINDOW:
        plt.show()
    else:
        plt.close(fig)


def main():
    build_animation()


if __name__ == "__main__":
    main()
