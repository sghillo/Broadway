import numpy as np
import pytest

from ssve.vision import (KP, clean_sequence, compare_movements, dtw, joint_angles,
                         triangulate_dlt, triangulate_sequence)


def squat(frames: int, depth_deg: float = 90.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Synthetic 2D (x, y, conf) squat: knee/hip flex from straight to `depth_deg` and back."""
    rng = np.random.default_rng(seed)
    seq = np.zeros((frames, 17, 3))
    seq[..., 2] = 0.9
    phase = np.sin(np.linspace(0, np.pi, frames))
    for t, p in enumerate(phase):
        flex = np.radians(depth_deg * p)
        hip = np.array([0.0, 0.0])
        knee = hip + np.array([np.sin(flex), 1.0 * np.cos(flex)]) * 0.5
        ankle = knee + np.array([-np.sin(flex), np.cos(flex)]) * 0.5
        shoulder = hip + np.array([-0.3 * np.sin(flex), -1.0])
        for side, dx in (("left", -0.1), ("right", 0.1)):
            off = np.array([dx, 0])
            seq[t, KP[f"{side}_hip"], :2] = hip + off
            seq[t, KP[f"{side}_knee"], :2] = knee + off
            seq[t, KP[f"{side}_ankle"], :2] = ankle + off
            seq[t, KP[f"{side}_shoulder"], :2] = shoulder + off
            seq[t, KP[f"{side}_elbow"], :2] = shoulder + off + np.array([0.3, 0.3])
            seq[t, KP[f"{side}_wrist"], :2] = shoulder + off + np.array([0.6, 0.3])
    seq[..., :2] += rng.normal(0, noise, seq[..., :2].shape)
    return seq * np.array([200, 200, 1]) + np.array([320, 240, 0])  # to pixel-ish space


def test_dtw_identical_sequences_zero_cost():
    x = np.sin(np.linspace(0, 6, 50))[:, None]
    r = dtw(x, x)
    assert r.distance == pytest.approx(0.0)
    assert (r.path[:, 0] == r.path[:, 1]).all()


def test_dtw_aligns_time_stretched_signal():
    x = np.sin(np.linspace(0, 2 * np.pi, 40))[:, None]
    y = np.sin(np.linspace(0, 2 * np.pi, 80))[:, None]
    assert dtw(x, y).normalised < 0.05
    assert dtw(x, -y).normalised > 0.3


def test_dtw_path_is_monotonic_and_bounded():
    r = dtw(np.random.rand(30, 3), np.random.rand(45, 3), window=5)
    assert tuple(r.path[0]) == (0, 0) and tuple(r.path[-1]) == (29, 44)
    assert (np.diff(r.path, axis=0) >= 0).all()


def test_joint_angles_straight_leg_is_180():
    angles, names = joint_angles(squat(5, depth_deg=0)[:1], has_conf=True)
    assert angles[0, names.index("left_knee")] == pytest.approx(180.0, abs=1e-6)


def test_compare_same_movement_different_tempo_is_similar():
    ref = squat(60)
    slow = squat(110, noise=0.002, seed=1)
    shallow = squat(60, depth_deg=40)
    same = compare_movements(ref, slow, has_conf=True)
    diff = compare_movements(ref, shallow, has_conf=True)
    assert same.similarity > 0.85
    assert diff.similarity < same.similarity
    worst = max(diff.joints, key=lambda j: j.rmse_deg)
    assert worst.joint.endswith(("knee", "hip"))
    assert 30 <= worst.phase_of_max_pct <= 70  # deepest point of the squat
    assert "| Joint |" in diff.to_markdown()


def test_clean_sequence_interpolates_low_confidence():
    seq = squat(10)
    seq[4, KP["left_knee"], 2] = 0.05
    original = seq[4, KP["left_knee"], :2].copy()
    seq[4, KP["left_knee"], :2] = [0, 0]  # garbage detection
    cleaned = clean_sequence(seq)
    assert np.allclose(cleaned[4, KP["left_knee"], :2], original, atol=15)
    assert not np.allclose(cleaned[4, KP["left_knee"], :2], [0, 0])


def _camera(R: np.ndarray, t: np.ndarray, f: float = 800.0) -> np.ndarray:
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]])
    return K @ np.hstack([R, t[:, None]])


def _project(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def test_triangulation_recovers_3d_point():
    P1 = _camera(np.eye(3), np.array([0.0, 0.0, 5.0]))
    th = np.radians(30)
    R2 = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    P2 = _camera(R2, np.array([0.0, 0.0, 5.0]))
    X = np.array([0.2, -0.4, 0.3])
    assert np.allclose(triangulate_dlt([P1, P2], [_project(P1, X), _project(P2, X)]), X, atol=1e-6)

    world = np.random.default_rng(0).normal(0, 0.5, (4, 17, 3))
    seqs = []
    for P in (P1, P2):
        s = np.ones((4, 17, 3))
        s[..., :2] = [[_project(P, world[t, j]) for j in range(17)] for t in range(4)]
        seqs.append(s)
    assert np.allclose(triangulate_sequence([P1, P2], seqs), world, atol=1e-6)
