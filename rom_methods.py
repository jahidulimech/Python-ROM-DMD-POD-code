"""POD, exact DMD, shock tracking, and validation methods."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd


@dataclass
class DMDModel:
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    amplitudes: np.ndarray
    singular_values: np.ndarray
    rank: int
    dt: float
    elapsed_s: float


def choose_rank(singular_values: np.ndarray, target: float) -> int:
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(energy, target) + 1)


def fit_dmd(q: np.ndarray, dt: float, rank: int) -> DMDModel:
    """Fit exact DMD on a uniformly sampled scalar/vector snapshot state."""
    if q.ndim != 2 or q.shape[1] < 3:
        raise ValueError(f"DMD expects (degrees_of_freedom, >=3 snapshots); got {q.shape}")
    t0 = time.perf_counter()
    mean = q.mean(axis=1, keepdims=True)
    y = q - mean
    x1, x2 = y[:, :-1], y[:, 1:]
    u, s, vh = np.linalg.svd(x1, full_matrices=False)
    max_rank = int(np.count_nonzero(s > s[0] * 1e-12))
    rank = min(int(rank), max_rank)
    if rank < 1:
        raise ValueError("Numerically zero snapshot matrix after mean removal")
    ur, sr, vhr = u[:, :rank], s[:rank], vh[:rank]
    atilde = ur.conj().T @ x2 @ vhr.conj().T @ np.diag(1.0 / sr)
    eigenvalues, w = np.linalg.eig(atilde)
    modes = x2 @ vhr.conj().T @ np.diag(1.0 / sr) @ w
    amplitudes = np.linalg.lstsq(modes, y[:, 0], rcond=None)[0]
    return DMDModel(mean, modes, eigenvalues, amplitudes, s, rank, dt, time.perf_counter() - t0)


def dmd_reconstruct(model: DMDModel, n_times: int) -> np.ndarray:
    """Reconstruct sequential samples from DMD amplitudes at the first snapshot."""
    powers = np.arange(n_times, dtype=float)
    dynamics = model.amplitudes[:, None] * model.eigenvalues[:, None] ** powers[None, :]
    q = model.mean + np.real(model.modes @ dynamics)
    return q


def pod_reconstruct(q: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Energy-optimal projection of all snapshots, using a temporal-mean offset."""
    mean = q.mean(axis=1, keepdims=True)
    y = q - mean
    u, singular_values, _ = np.linalg.svd(y, full_matrices=False)
    rank = min(rank, u.shape[1])
    return mean + u[:, :rank] @ (u[:, :rank].T @ y), singular_values


def field_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    """Metrics over finite common support; fields can be n-space by n-time."""
    valid = np.isfinite(reference) & np.isfinite(estimate)
    if not np.any(valid):
        return {key: np.nan for key in ["relative_frobenius", "l2", "nrmse", "max_abs", "rmse"]}
    ref, err = reference[valid], (estimate - reference)[valid]
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.max(ref) - np.min(ref))
    return {
        "relative_frobenius": float(np.linalg.norm(err) / np.linalg.norm(ref)),
        "l2": float(np.linalg.norm(err)),
        "nrmse": float(rmse / scale) if scale > 0 else np.nan,
        "max_abs": float(np.max(np.abs(err))),
        "rmse": rmse,
    }


def time_metrics(reference: np.ndarray, estimate: np.ndarray) -> pd.DataFrame:
    rows = []
    for j in range(reference.shape[1]):
        rows.append({"time_index": j, **field_metrics(reference[:, j], estimate[:, j])})
    return pd.DataFrame(rows)


def spatial_metrics(reference: np.ndarray, estimate: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(reference.shape[0]):
        rows.append({"space_index": i, **field_metrics(reference[i, :], estimate[i, :])})
    return pd.DataFrame(rows)


def _peak_position(x: np.ndarray, indicator: np.ndarray) -> float:
    index = int(np.argmax(indicator))
    if index == 0 or index == x.size - 1:
        return float(x[index])
    a, b, c = indicator[index - 1:index + 2]
    denom = a - 2.0 * b + c
    offset = 0.0 if abs(denom) < np.finfo(float).eps else 0.5 * (a - c) / denom
    return float(x[index] + np.clip(offset, -0.5, 0.5) * (x[index + 1] - x[index]))


def shock_candidates(x: np.ndarray, p: np.ndarray, rho: np.ndarray | None = None) -> dict[str, float]:
    """Evaluate transparent gradient-based candidates on one 1D snapshot."""
    dpdx = np.gradient(p, x)
    result = {
        "p_positive": _peak_position(x, np.maximum(dpdx, 0.0)),
        "p_absolute": _peak_position(x, np.abs(dpdx)),
    }
    if rho is not None:
        drdx = np.gradient(rho, x)
        result["rho_positive"] = _peak_position(x, np.maximum(drdx, 0.0))
        result["rho_absolute"] = _peak_position(x, np.abs(drdx))
    return result


def track_shock(x: np.ndarray, p: np.ndarray, rho: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    candidates = shock_candidates(x, p, rho)
    # A left-to-right pressure jump is the shock signature for the verified p_L > p_R case.
    return candidates["p_positive"], candidates


def attach_snapshots(x: np.ndarray, q: np.ndarray, shock_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate q(x,t) to the common physically supported xi=x-xs(t) grid."""
    shifted_left, shifted_right = x[0] - shock_x, x[-1] - shock_x
    xi_min, xi_max = float(np.max(shifted_left)), float(np.min(shifted_right))
    dx = float(np.median(np.diff(x)))
    n_xi = int(np.floor((xi_max - xi_min) / dx)) + 1
    if n_xi < 10:
        raise ValueError("Shock-attached common support has too few cells")
    xi = xi_min + dx * np.arange(n_xi)
    attached = np.column_stack([np.interp(xi + shock_x[j], x, q[:, j]) for j in range(q.shape[1])])
    return xi, attached


def to_lab_grid(x: np.ndarray, xi: np.ndarray, attached: np.ndarray, shock_x: np.ndarray) -> np.ndarray:
    """Map a shock-attached representation to the lab grid; unsupported edges are NaN."""
    return np.column_stack([
        np.interp(x - shock_x[j], xi, attached[:, j], left=np.nan, right=np.nan)
        for j in range(attached.shape[1])
    ])


def shock_error(reference: np.ndarray, estimate: np.ndarray) -> dict:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[valid] - reference[valid]
    return {
        "n_valid": int(valid.sum()),
        "mae_m": float(np.mean(np.abs(error))),
        "max_abs_m": float(np.max(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
    }


def eigenvalue_table(model: DMDModel) -> pd.DataFrame:
    lam = model.eigenvalues
    omega = np.log(lam) / model.dt
    return pd.DataFrame({
        "mode": np.arange(1, model.rank + 1),
        "eigenvalue_real": lam.real, "eigenvalue_imag": lam.imag,
        "modulus": np.abs(lam), "growth_rate_per_s": omega.real,
        "frequency_hz": omega.imag / (2 * np.pi),
        "amplitude_modulus": np.abs(model.amplitudes),
    })
