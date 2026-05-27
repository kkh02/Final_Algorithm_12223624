import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares


EPS = 1e-9
MODEL_PATH = "model.npz"


def _safe_norm(x, axis=None):
    return np.sqrt(np.sum(x * x, axis=axis))


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 1.0

    m = np.median(x)
    return np.median(np.abs(x - m)) + EPS


def _robust_linear_fit(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 5:
        return 1.0, 0.0

    x_med = np.median(x)
    y_med = np.median(y)

    x_scale = _mad(x)
    y_scale = _mad(y)

    x_n = (x - x_med) / x_scale
    y_n = (y - y_med) / y_scale

    def fun(theta):
        a, b = theta
        return a * x_n + b - y_n

    try:
        res = least_squares(
            fun,
            x0=np.array([1.0, 0.0], dtype=float),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=200,
        )

        a_n, b_n = res.x

        a = a_n * y_scale / x_scale
        b = y_med + y_scale * b_n - a * x_med

        if not np.isfinite(a) or not np.isfinite(b):
            return 1.0, 0.0

        return float(a), float(b)

    except Exception:
        try:
            a, b = np.polyfit(x, y, 1)
            return float(a), float(b)
        except Exception:
            return 1.0, 0.0


def _true_distances(p, bs):
    num_anchor = bs.shape[1]
    num_user = p.shape[1]

    d_true = np.zeros((num_anchor, num_user), dtype=float)

    for i in range(num_anchor):
        diff = p - bs[:, i].reshape(2, 1)
        d_true[i, :] = _safe_norm(diff, axis=0)

    return d_true


def train_model(mat_path="DH_FR1.mat"):
    data = sio.loadmat(mat_path, squeeze_me=False)

    if "BS_positions" in data:
        bs = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        bs = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("MAT file must contain BS_positions or p_bs.")

    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    num_anchor = d_hat.shape[0]
    d_true = _true_distances(p, bs)

    a = np.ones(num_anchor, dtype=float)
    b = np.zeros(num_anchor, dtype=float)
    residual_mad = np.ones(num_anchor, dtype=float)
    residual_mae = np.ones(num_anchor, dtype=float)

    for i in range(num_anchor):
        ai, bi = _robust_linear_fit(d_hat[i, :], d_true[i, :])

        a[i] = ai
        b[i] = bi

        d_corr = ai * d_hat[i, :] + bi
        e = d_corr - d_true[i, :]

        residual_mad[i] = _mad(e)

        finite_e = e[np.isfinite(e)]
        if finite_e.size > 0:
            residual_mae[i] = np.nanmedian(np.abs(finite_e)) + EPS
        else:
            residual_mae[i] = 1.0

    robust_error = 1.4826 * residual_mad + 0.3 * residual_mae + EPS
    raw_history = 1.0 / robust_error

    if np.max(raw_history) > 0:
        history = raw_history / np.max(raw_history)
    else:
        history = np.ones(num_anchor, dtype=float)

    history = np.clip(history, 0.15, 1.0)

    bs_min = np.min(bs, axis=1)
    bs_max = np.max(bs, axis=1)
    diag = _safe_norm(bs_max - bs_min) + EPS

    top_k = min(12, num_anchor)
    cluster_eps = 0.06 * diag
    alpha = 2.5
    sigma_min = 0.02 * diag
    sigma_max = 0.35 * diag

    np.savez(
        MODEL_PATH,
        a=a,
        b=b,
        history=history,
        residual_mad=residual_mad,
        residual_mae=residual_mae,
        top_k=np.array([top_k], dtype=float),
        cluster_eps=np.array([cluster_eps], dtype=float),
        alpha=np.array([alpha], dtype=float),
        sigma_min=np.array([sigma_min], dtype=float),
        sigma_max=np.array([sigma_max], dtype=float),
    )

    print("Saved model to", MODEL_PATH)


def main():
    train_model("DH_FR1.mat")


if __name__ == "__main__":
    main()