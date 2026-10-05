"""Compare an athlete's movement against a reference (e.g. coach demo or best-ever rep)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtw import DTWResult, dtw, warp
from .features import joint_angles, normalise, smooth


@dataclass
class JointReport:
    joint: str
    rmse_deg: float
    max_dev_deg: float
    phase_of_max_pct: float   # 0-100 % through the reference movement


@dataclass
class MovementComparison:
    similarity: float           # 0-1, 1 = identical after alignment
    dtw: DTWResult
    joints: list[JointReport]

    def to_markdown(self, top: int = 5) -> str:
        rows = sorted(self.joints, key=lambda r: r.rmse_deg, reverse=True)[:top]
        lines = [f"**Movement similarity:** {self.similarity:.0%} (DTW normalised cost {self.dtw.normalised:.2f})",
                 "", "| Joint | RMSE (°) | Max dev (°) | At phase |", "|---|---|---|---|"]
        lines += [f"| {r.joint} | {r.rmse_deg:.1f} | {r.max_dev_deg:.1f} | {r.phase_of_max_pct:.0f}% |"
                  for r in rows]
        return "\n".join(lines)


def compare_movements(reference: np.ndarray, athlete: np.ndarray, window_pct: float = 0.2,
                      smoothing: bool = True, joints: list[str] | None = None,
                      has_conf: bool | None = None) -> MovementComparison:
    """Align athlete to reference with DTW on joint-angle trajectories and report per-joint error.

    Both inputs are keypoint sequences (T, J, 2|3) in COCO order. Set ``has_conf=True`` for
    YOLO output (x, y, conf) and ``False`` for 3D (X, Y, Z); ``None`` guesses.
    """
    ref_ang, names = joint_angles(normalise(reference, has_conf), joints, has_conf=False)
    ath_ang, _ = joint_angles(normalise(athlete, has_conf), names, has_conf=False)
    if smoothing:
        ref_ang, ath_ang = smooth(ref_ang), smooth(ath_ang)

    window = max(1, int(window_pct * max(len(ref_ang), len(ath_ang))))
    # Scale degrees to radians so the DTW cost is in a sane numeric range.
    result = dtw(np.radians(ref_ang), np.radians(ath_ang), window=window)
    aligned = warp(ath_ang, result.path, len(ref_ang))
    diff = aligned - ref_ang

    reports = []
    for k, name in enumerate(names):
        col = np.abs(diff[:, k])
        idx = int(np.nanargmax(col))
        reports.append(JointReport(name, float(np.sqrt(np.nanmean(diff[:, k] ** 2))),
                                   float(col[idx]), 100.0 * idx / max(1, len(ref_ang) - 1)))
    # Map mean RMSE onto (0, 1]: 0° -> 1.0, 30° -> ~0.37.
    mean_rmse = float(np.mean([r.rmse_deg for r in reports]))
    return MovementComparison(similarity=float(np.exp(-mean_rmse / 30.0)), dtw=result, joints=reports)
