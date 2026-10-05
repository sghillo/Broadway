"""Biomechanical features from keypoint sequences: normalisation, smoothing, joint angles."""
from __future__ import annotations

import numpy as np

from .keypoints import KP

# Angle at the middle joint, formed by (proximal, joint, distal).
JOINT_ANGLES: dict[str, tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def has_confidence(seq: np.ndarray) -> bool:
    """Guess whether the last channel is confidence (x, y, conf) rather than depth (X, Y, Z)."""
    if seq.shape[-1] != 3:
        return False
    c = seq[..., 2]
    return bool(np.nanmin(c) >= 0 and np.nanmax(c) <= 1.0)


def coords(seq: np.ndarray, has_conf: bool | None = None) -> np.ndarray:
    """Return spatial coordinates, dropping a confidence channel from 2D (x, y, conf) input.

    Pass ``has_conf`` explicitly when known; ``None`` falls back to `has_confidence`.
    """
    if has_conf is None:
        has_conf = has_confidence(seq)
    return seq[..., :2] if has_conf else seq


def normalise(seq: np.ndarray, has_conf: bool | None = None) -> np.ndarray:
    """Translate to mid-hip origin and scale by mean torso length -> camera/athlete-size invariant."""
    xyz = coords(seq, has_conf).astype(float)
    mid_hip = (xyz[:, KP["left_hip"]] + xyz[:, KP["right_hip"]]) / 2
    mid_sh = (xyz[:, KP["left_shoulder"]] + xyz[:, KP["right_shoulder"]]) / 2
    torso = np.nanmean(np.linalg.norm(mid_sh - mid_hip, axis=-1))
    if not np.isfinite(torso) or torso == 0:
        torso = 1.0
    return (xyz - mid_hip[:, None, :]) / torso


def smooth(signal: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along time (axis 0). No-op for short sequences."""
    from scipy.signal import savgol_filter

    if signal.shape[0] < window:
        return signal
    return savgol_filter(signal, window_length=window, polyorder=poly, axis=0)


def joint_angles(seq: np.ndarray, joints: list[str] | None = None,
                 has_conf: bool | None = None) -> tuple[np.ndarray, list[str]]:
    """Compute included joint angles in degrees. Works for 2D or 3D. Returns (T, A), names."""
    xyz = coords(seq, has_conf).astype(float)
    names = joints or list(JOINT_ANGLES)
    out = np.empty((xyz.shape[0], len(names)))
    for k, name in enumerate(names):
        a, b, c = (KP[n] for n in JOINT_ANGLES[name])
        v1 = xyz[:, a] - xyz[:, b]
        v2 = xyz[:, c] - xyz[:, b]
        cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
        out[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out, names


def angular_velocity(angles: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(angles, 1.0 / fps, axis=0)
