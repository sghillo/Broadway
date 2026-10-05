"""2D keypoint extraction (Ultralytics YOLO-pose) and 3D reconstruction by multi-view triangulation.

Arrays use the shape conventions:
  2D sequence: (T, J, 3)  -> x, y, confidence per joint per frame
  3D sequence: (T, J, 3)  -> X, Y, Z (world units of the camera calibration)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(COCO_KEYPOINTS)}


def iter_frames(video_path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (frame_index, BGR frame, fps) from a video file."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame, fps
            idx += 1
    finally:
        cap.release()


class PoseExtractor:
    """Single-subject 2D pose tracking with YOLO-pose.

    When several people are detected, the one whose box centre is closest to the
    previous frame's subject is kept (simple, robust for single-athlete footage).
    """

    def __init__(self, weights: str = "yolo11n-pose.pt", conf: float = 0.25,
                 device: str | None = None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def extract(self, video_path: str | Path, stride: int = 1) -> tuple[np.ndarray, float]:
        frames: list[np.ndarray] = []
        prev_centre: np.ndarray | None = None
        fps = 30.0
        for _, frame, fps in iter_frames(video_path, stride):
            res = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
            kps = res.keypoints
            if kps is None or len(kps) == 0:
                frames.append(np.full((len(COCO_KEYPOINTS), 3), np.nan))
                continue
            xy = kps.xy.cpu().numpy()                     # (N, 17, 2)
            cf = kps.conf.cpu().numpy() if kps.conf is not None else np.ones(xy.shape[:2])
            centres = np.nanmean(xy, axis=1)
            pick = 0 if prev_centre is None else int(np.argmin(np.linalg.norm(centres - prev_centre, axis=1)))
            prev_centre = centres[pick]
            frames.append(np.concatenate([xy[pick], cf[pick][:, None]], axis=1))
        return np.stack(frames), fps / stride


def clean_sequence(seq: np.ndarray, min_conf: float = 0.3, has_conf: bool = True) -> np.ndarray:
    """Mask low-confidence joints and linearly interpolate gaps over time (per joint, per axis).

    ``seq`` is (T, J, 3) x/y/conf when ``has_conf`` else (T, J, D) coordinates with NaN gaps.
    """
    seq = seq.astype(float).copy()
    coords = seq[..., :2] if has_conf else seq  # a view: edits write through to seq
    if has_conf:
        coords[seq[..., 2] < min_conf] = np.nan
    t = np.arange(seq.shape[0])
    for j in range(coords.shape[1]):
        for a in range(coords.shape[2]):
            col = coords[:, j, a]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                col[~ok] = np.interp(t[~ok], t[ok], col[ok])
    return seq


def triangulate_dlt(projections: list[np.ndarray], points_2d: list[np.ndarray]) -> np.ndarray:
    """Triangulate one 3D point from >=2 calibrated views with the linear DLT method.

    projections: list of 3x4 camera projection matrices P = K [R | t]
    points_2d:   list of (x, y) pixel observations, one per view
    """
    rows = []
    for P, (x, y) in zip(projections, points_2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return X[:3] / X[3]


def triangulate_sequence(projections: list[np.ndarray], sequences_2d: list[np.ndarray],
                         min_conf: float = 0.3) -> np.ndarray:
    """Lift synchronised 2D sequences from multiple cameras to a (T, J, 3) 3D sequence."""
    T, J = sequences_2d[0].shape[:2]
    out = np.full((T, J, 3), np.nan)
    for t in range(T):
        for j in range(J):
            views = [(P, s[t, j, :2]) for P, s in zip(projections, sequences_2d)
                     if not np.isnan(s[t, j, :2]).any()
                     and (s.shape[-1] < 3 or np.isnan(s[t, j, 2]) or s[t, j, 2] >= min_conf)]
            if len(views) >= 2:
                out[t, j] = triangulate_dlt([v[0] for v in views], [v[1] for v in views])
    return out
