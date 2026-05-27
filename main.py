import os
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares


MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.npz"


def _safe_norm(v, axis=None):
    return np.sqrt(np.sum(v * v, axis=axis))


def _mad(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    med = np.median(x)
    return np.median(np.abs(x - med))


def _load_model(num_anchor):
    def scalar_from_model(model, key, default):
        if key not in model:
            return default
        arr = np.asarray(model[key], dtype=float).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[0])

    if os.path.exists(MODEL_PATH):
        model = np.load(MODEL_PATH, allow_pickle=True)
        return {
            "a": np.asarray(model["a"], dtype=float).reshape(-1),
            "b": np.asarray(model["b"], dtype=float).reshape(-1),
            "history": np.asarray(model["history"], dtype=float).reshape(-1),
            "alpha": scalar_from_model(model, "alpha", 2.5),
            "sigma_min": scalar_from_model(model, "sigma_min", 1.0),
            "sigma_max": scalar_from_model(model, "sigma_max", 25.0),
        }

    return {
        "a": np.ones(num_anchor, dtype=float),
        "b": np.zeros(num_anchor, dtype=float),
        "history": np.ones(num_anchor, dtype=float),
        "alpha": 2.5,
        "sigma_min": 1.0,
        "sigma_max": 25.0,
    }


def _preprocess_distance(d):
    d = np.asarray(d, dtype=float).reshape(-1)

    valid = np.isfinite(d) & (d > 0)
    if np.any(valid):
        fill_value = np.nanmedian(d[valid])
    else:
        fill_value = 1.0

    d = np.where(valid, d, fill_value)

    upper = np.percentile(d, 95) + 3.0 * (_mad(d) + 1e-9)
    upper = max(upper, 1.0)

    return np.clip(d, 0.1, upper)


def _calibrate_distance(d, model):
    d = _preprocess_distance(d)

    d_corr = model["a"] * d + model["b"]
    d_corr = np.where(np.isfinite(d_corr) & (d_corr > 0), d_corr, d)

    upper = np.percentile(d_corr, 95) + 3.0 * (_mad(d_corr) + 1e-9)
    upper = max(upper, 1.0)

    return np.clip(d_corr, 0.1, upper)


def _bounds_from_bs(bs):
    bs_min = np.min(bs, axis=1)
    bs_max = np.max(bs, axis=1)

    diag = np.linalg.norm(bs_max - bs_min)
    margin = max(10.0, 0.7 * diag)

    lower = bs_min - margin
    upper = bs_max + margin

    return lower, upper


def _residual_at(x, bs, d):
    x = np.asarray(x, dtype=float).reshape(2)
    pred = _safe_norm(bs - x.reshape(2, 1), axis=0)
    return pred - d


def _weighted_solve(bs, d, weights=None, x0=None, active_mask=None, max_nfev=120):
    bs = np.asarray(bs, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    n_anchor = d.size

    if weights is None:
        weights = np.ones(n_anchor, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)

    if active_mask is None:
        active_mask = np.ones(n_anchor, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)

    valid = (
        active_mask
        & np.isfinite(d)
        & (d > 0)
        & np.isfinite(weights)
        & (weights > 1e-8)
    )

    if np.sum(valid) < 3:
        valid = np.isfinite(d) & (d > 0)

    if np.sum(valid) < 3:
        return np.mean(bs, axis=1)

    if x0 is None:
        x0 = np.average(bs[:, valid], axis=1)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(2)

    lower, upper = _bounds_from_bs(bs)
    x0 = np.clip(x0, lower, upper)

    bs_v = bs[:, valid]
    d_v = d[valid]
    w_v = weights[valid]

    sqrt_w = np.sqrt(np.clip(w_v, 1e-8, None))

    def fun(x):
        r = _residual_at(x, bs_v, d_v)
        return sqrt_w * r

    res = least_squares(
        fun,
        x0,
        bounds=(lower, upper),
        loss="linear",
        max_nfev=max_nfev,
    )

    return res.x


def _calibrated_ls_solution(bs, d_corr):
    weights = np.ones(d_corr.size, dtype=float)
    x0 = np.mean(bs, axis=1)
    return _weighted_solve(bs, d_corr, weights=weights, x0=x0)


def _mad_score_from_residual(r):
    r = np.asarray(r, dtype=float).reshape(-1)

    center = np.median(r)
    scale = 1.4826 * _mad(r) + 1e-9

    z = np.abs(r - center) / scale
    s_mad = 1.0 / (1.0 + z)

    return np.clip(s_mad, 0.05, 1.0), z, scale


def _mcc_weight(r, scale, model):
    alpha = model["alpha"]
    sigma_min = model["sigma_min"]
    sigma_max = model["sigma_max"]

    sigma = np.clip(alpha * scale, sigma_min, sigma_max)
    w = np.exp(-(r ** 2) / (2.0 * sigma ** 2))

    return np.clip(w, 0.03, 1.0)


def _loo_self_diagnosis_weight(bs, d_corr, x_base, history):
    n_anchor = d_corr.size
    influence = np.zeros(n_anchor, dtype=float)

    for i in range(n_anchor):
        active = np.ones(n_anchor, dtype=bool)
        active[i] = False

        x_minus_i = _weighted_solve(
            bs,
            d_corr,
            weights=history,
            x0=x_base,
            active_mask=active,
            max_nfev=80,
        )

        influence[i] = np.linalg.norm(x_base - x_minus_i)

    r_base = _residual_at(x_base, bs, d_corr)
    _, residual_z, _ = _mad_score_from_residual(r_base)

    infl_center = np.median(influence)
    infl_scale = 1.4826 * _mad(influence) + 1e-9
    infl_z = np.abs(influence - infl_center) / infl_scale

    residual_factor = 0.32 + 0.68 * np.clip(residual_z / 3.2, 0.0, 1.0)
    instability = infl_z * residual_factor

    s_loo = 1.0 / (1.0 + 0.90 * instability)

    return np.clip(s_loo, 0.05, 1.0)


def _position_score(x, bs, d_corr):
    r = np.abs(_residual_at(x, bs, d_corr))

    med = np.median(r)
    p75 = np.percentile(r, 75)
    p90 = np.percentile(r, 90)

    return med + 0.30 * p75 + 0.15 * p90


def your_algorithm(d_u, p_bs):
    bs = np.asarray(p_bs, dtype=float)
    d_u = np.asarray(d_u, dtype=float).reshape(-1)

    n_anchor = d_u.size
    model = _load_model(n_anchor)

    d_corr = _calibrate_distance(d_u, model)

    history = np.asarray(model["history"], dtype=float).reshape(-1)
    history = np.clip(history, 0.05, 1.0)

    x_base = _calibrated_ls_solution(bs, d_corr)

    s_loo = _loo_self_diagnosis_weight(bs, d_corr, x_base, history)

    r_base = _residual_at(x_base, bs, d_corr)
    s_mad, _, scale = _mad_score_from_residual(r_base)
    w_mcc = _mcc_weight(r_base, scale, model)

    weights = history * s_loo * s_mad * w_mcc
    weights = np.clip(weights, 0.03, 1.0)

    x_loo = _weighted_solve(
        bs,
        d_corr,
        weights=weights,
        x0=x_base,
        max_nfev=120,
    )

    r_loo = _residual_at(x_loo, bs, d_corr)
    s_mad2, _, scale2 = _mad_score_from_residual(r_loo)
    w_mcc2 = _mcc_weight(r_loo, scale2, model)

    weights2 = history * s_loo * s_mad2 * w_mcc2
    weights2 = np.clip(weights2, 0.03, 1.0)

    x_loo2 = _weighted_solve(
        bs,
        d_corr,
        weights=weights2,
        x0=x_loo,
        max_nfev=120,
    )

    score_base = _position_score(x_base, bs, d_corr)
    score_loo = _position_score(x_loo2, bs, d_corr)

    move = np.linalg.norm(x_loo2 - x_base)

    if score_loo <= score_base:
        return x_loo2

    if score_loo <= 1.05 * score_base and move < 25.0:
        return 0.75 * x_loo2 + 0.25 * x_base

    if move < 10.0:
        return 0.50 * x_loo2 + 0.50 * x_base

    return x_base


def main():
    data = sio.loadmat(MAT_PATH, squeeze_me=False)

    if "BS_positions" in data:
        BS_positions = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        BS_positions = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("MAT file must contain BS_positions or p_bs")

    d_hat = np.asarray(data["d_hat"], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    return p_hat


if __name__ == "__main__":
    main()