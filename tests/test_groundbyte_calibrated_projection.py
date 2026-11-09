"""Regression tests for reusing calibrated homographies across Transplan branches.

No GUI or full Transplan dependency installation is required for these tests.
"""
import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homography_calibration import (metadata_path, read_metadata, save_homography,
                                    undistort_points, distort_points)

TRACK_MODULE = importlib.import_module('Trackers.GroundByteTrack.track')
K = np.array([[240.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1.0]])
H = np.array([[1.2, 0.05, 15.0], [-0.02, 1.15, 27.0], [0.00002, 0.00003, 1.0]])
CALIBRATION = {
    'model': 'pinhole', 'K': K.tolist(),
    'D': [-0.25, 0.032, 0.001, -0.0008, 0.002], 'image_size': [320, 240],
}


def _perspective(points):
    return cv2.perspectiveTransform(
        np.asarray(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def _write_h(path, calibrated):
    if calibrated:
        save_homography(path, H, np.ones((4, 1), np.uint8), CALIBRATION)
    else:
        save_homography(path, H, np.ones((4, 1), np.uint8), None)


@pytest.mark.parametrize('calibrated', [False, True])
def test_tracking_projection_matches_homography_coordinate_convention(tmp_path, monkeypatch, calibrated):
    hpath = tmp_path / 'homography.npy'
    _write_h(hpath, calibrated)
    # Confirm the metadata lives in .npy even after removing its sidecar.
    sidecar = Path(metadata_path(hpath))
    sidecar.unlink(missing_ok=True)
    assert (read_metadata(hpath) is not None) == calibrated

    # Off-axis points make the effect of lens correction measurable.
    data = pd.DataFrame([
        (0, 20.0, 90.0, 45.0, 220.0, 0.8, 2),
        (1, 240.0, 110.0, 275.0, 190.0, 0.75, 2),
        (2, 10.0, 50.0, 55.0, 210.0, 0.05, 2),  # filtered before projection
    ], columns=['fn', 'x1', 'y1', 'x2', 'y2', 'score', 'class'])
    pkl = tmp_path / 'detections.pkl'
    data.to_pickle(pkl)

    class FakeCapture:
        def isOpened(self): return True
        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 30.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240}[prop]
        def release(self): pass

    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: FakeCapture())
    seen = []
    class SpyTracker:
        def __init__(self, config, fps): pass
        def update(self, frame_dets, fn):
            for d in frame_dets:
                seen.append((fn, d.xy.copy()))
            return []
    monkeypatch.setattr(TRACK_MODULE, 'GroundByteTracker', SpyTracker)

    args = SimpleNamespace(HomographyNPY=str(hpath), GPMetersPerPixel=0.06,
                           BackprojectionMethod='Homography', DetectionPkl=str(pkl),
                           TrackingPth=str(tmp_path / 'tracks.txt'), Video='placeholder.mp4',
                           ContactPoint='BottomPoint', ForNFrames=None)
    TRACK_MODULE.track(args)
    assert [fn for fn, _ in seen] == [0, 1]
    pts = np.array([[(20+45)/2, 220], [(240+275)/2, 190]], np.float64)
    expected = _perspective(undistort_points(pts, CALIBRATION) if calibrated else pts) * 0.06
    np.testing.assert_allclose(np.array([xy for _, xy in seen]), expected, atol=1e-8)


def test_homography_downstream_matches_tracker_and_inverts_raw_coords(tmp_path):
    """Exercise the actual downstream functions without optional Transplan packages."""
    source = (ROOT / 'Homography.py').read_text()
    parsed = ast.parse(source)
    names = {'reproj_point', 'reproj_point_homography', 'project_point_homography',
             'reproject_df'}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in names]
    assert {node.name for node in functions} == names
    ns = {
        'np': np, 'read_metadata': read_metadata, 'undistort_points': undistort_points,
        'distort_points': distort_points, 'tqdm': lambda x, total=None: x,
        'get_contact_point': lambda row, args: ((row['x1'] + row['x2']) / 2, row['y2']),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / 'Homography.py'), 'exec'), ns)
    raw = np.array([33.0, 220.0])
    expected = _perspective(undistort_points([raw], CALIBRATION))[0]
    out = ns['reproj_point_homography'](*raw, H, calibration=CALIBRATION)
    np.testing.assert_allclose(out[:2], expected, atol=1e-8)
    recovered = ns['project_point_homography'](*expected, H, calibration=CALIBRATION)
    np.testing.assert_allclose(recovered[:2], raw, atol=1e-3)

    hpath = tmp_path / 'homography.npy'
    _write_h(hpath, True)
    df = pd.DataFrame([{'fn': 0, 'x1': 20.0, 'y1': 90.0, 'x2': 46.0, 'y2': 220.0,
                        'score': 0.8, 'class': 2}])
    output = tmp_path / 'reprojected.txt'
    ns['reproject_df'](SimpleNamespace(HomographyNPY=str(hpath)), df, output, 'Homography')
    cells = list(map(float, output.read_text().split(',')[-4:]))
    np.testing.assert_allclose(cells[:2], raw, atol=1e-8)
    np.testing.assert_allclose(cells[2:], expected, atol=1e-8)


def test_bad_calibration_resolution_fails_fast(tmp_path, monkeypatch):
    hpath = tmp_path / 'calibrated.npy'
    _write_h(hpath, True)
    data = pd.DataFrame(columns=['fn', 'x1', 'y1', 'x2', 'y2', 'score'])
    pkl = tmp_path / 'dets.pkl'
    data.to_pickle(pkl)
    class FakeCapture:
        def isOpened(self): return True
        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_COUNT: 1, cv2.CAP_PROP_FPS: 30.0,
                    cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[prop]
        def release(self): pass
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: FakeCapture())
    args = SimpleNamespace(HomographyNPY=str(hpath), GPMetersPerPixel=0.06,
                           BackprojectionMethod='Homography', DetectionPkl=str(pkl),
                           TrackingPth=str(tmp_path / 'tracks.txt'), Video='placeholder.mp4')
    with pytest.raises(ValueError, match='Calibration image_size'):
        TRACK_MODULE.track(args)
