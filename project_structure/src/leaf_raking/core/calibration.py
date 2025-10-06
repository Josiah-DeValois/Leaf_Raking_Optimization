from __future__ import annotations

import numpy as np

from .config import BaggingCurve, CalibrationData, CalibrationResult


def mmss_to_seconds(t: str) -> int:
    t = t.strip()
    if ":" in t:
        mm, ss = t.split(":")
        return int(mm) * 60 + int(ss)
    return int(float(t))


def fit_power_time(distances, times):
    d = np.array(distances, dtype=float)
    T = np.array(times, dtype=float)
    mask = (d > 0) & (T > 0)
    d, T = d[mask], T[mask]
    x = np.log(d)
    y = np.log(T)
    A = np.vstack([np.ones_like(x), x]).T
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ln_k, beta = theta
    k = np.exp(ln_k)
    return float(k), float(beta)


def fit_bag_time(fullness_to_time):
    fracs = np.array(sorted(fullness_to_time.keys()), dtype=float)
    secs = np.array([mmss_to_seconds(fullness_to_time[f]) for f in fracs], dtype=float)
    X = np.vstack([np.ones_like(fracs), fracs, fracs ** 2]).T
    coef, *_ = np.linalg.lstsq(X, secs, rcond=None)
    s0, s1, s2 = coef
    if s2 < 0:
        s2 = 0.0
        X2 = X[:, :2]
        s0, s1 = np.linalg.lstsq(X2, secs, rcond=None)[0]
    def t_of_fraction(f):
        return s0 + s1 * f + s2 * (f ** 2)
    return (float(s0), float(s1), float(s2)), t_of_fraction


def run_calibration(cal_data: CalibrationData, bag_capacity_lb: float) -> CalibrationResult:
    distances = sorted(cal_data.raking_splits_sec.keys())
    times = [cal_data.raking_splits_sec[d] for d in distances]
    alpha, beta = fit_power_time(distances, times)

    (_, _, _), curve_fn = fit_bag_time(cal_data.bagging_times)
    setup = curve_fn(0.0)
    stuffing_rate = (curve_fn(0.8) - curve_fn(0.4)) / (0.4 * bag_capacity_lb)
    bag_curve = BaggingCurve(setup_seconds=float(setup),
                             stuffing_rate_sec_per_lb=float(stuffing_rate),
                             curve_fn=curve_fn)
    return CalibrationResult(alpha=float(alpha), beta=float(beta), bag_curve=bag_curve)

