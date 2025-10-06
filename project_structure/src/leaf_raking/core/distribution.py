from __future__ import annotations

import numpy as np

from .config import GridCache, GridSpec, LeafDistributionParams, SimulationConfig, YardGeometry


def build_grid(yard: YardGeometry, grid: GridSpec):
    s = grid.spacing_ft
    nx = int(yard.length_ft / s)
    ny = int(yard.width_ft / s)
    xs = np.linspace(s / 2, yard.length_ft - s / 2, nx)
    ys = np.linspace(s / 2, yard.width_ft - s / 2, ny)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    cell_centers = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
    cell_area = s * s
    return xs, ys, mesh_x, mesh_y, cell_centers, cell_area


def leaf_density(mesh_x, mesh_y, params: LeafDistributionParams, yard: YardGeometry):
    from math import radians

    phi = radians(params.phi_deg)
    sigma_par = params.axis_ratio * params.sigma
    sigma_perp = params.sigma

    dx = mesh_x - yard.tree_center[0]
    dy = mesh_y - yard.tree_center[1]
    u = dx * np.cos(phi) + dy * np.sin(phi)
    v = -dx * np.sin(phi) + dy * np.cos(phi)
    R_aniso = np.sqrt((u / sigma_par) ** 2 + (v / sigma_perp) ** 2)
    rho = params.rho0 + params.amplitude * np.exp(-(R_aniso ** params.power))
    R_circ = np.sqrt(dx ** 2 + dy ** 2)
    rho = np.where(R_circ < yard.trunk_radius_ft, params.rho0, rho)
    return rho


def build_mass_grid(config: SimulationConfig) -> GridCache:
    xs, ys, mesh_x, mesh_y, cells, cell_area = build_grid(config.yard, config.grid)
    rho = leaf_density(mesh_x, mesh_y, config.distribution, config.yard)
    mass_grid = rho * cell_area
    mass_vector = mass_grid.ravel()
    total_mass = float(mass_vector.sum())
    return GridCache(xs=xs,
                     ys=ys,
                     mesh_x=mesh_x,
                     mesh_y=mesh_y,
                     cell_centers=cells,
                     cell_area=cell_area,
                     mass_grid=mass_grid,
                     mass_vector=mass_vector,
                     total_mass=total_mass)

