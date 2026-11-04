"""Headless tests for calibrated homographies (raw CSV + image/point consistency)."""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "homographygui"))
from homography_calibration import (check_image_size, load_calibration_json,
    metadata_path, read_metadata, undistort_image, undistort_points, distort_points)
from backend import backend
from homography_visualization import save_homography_visualizations


K = np.array([[240.0, 0, 160.0], [0, 250.0, 120.0], [0, 0, 1.0]])
POINTS = np.array([
    [20, 20], [300, 15], [305, 220], [15, 230], [160, 18],
    [310, 120], [165, 225], [10, 120], [120, 90], [200, 130],
], dtype=np.float64)
H_EXPECTED = np.array([[0.88, 0.055, 18.0], [-0.026, 0.9, 27.0], [0.00006, 0.0001, 1.]])


def _transform(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _distort(pixels, model, coeffs):
    normalized = np.column_stack(((pixels[:, 0] - K[0, 2]) / K[0, 0],
                                  (pixels[:, 1] - K[1, 2]) / K[1, 1]))
    if model == "fisheye":
        return cv2.fisheye.distortPoints(normalized.reshape(-1, 1, 2), K, coeffs).reshape(-1, 2)
    xyz = np.column_stack((normalized, np.ones(len(normalized))))
    projected, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, coeffs)
    return projected.reshape(-1, 2)


@pytest.mark.parametrize("model,coeffs", [
    ("pinhole", [-0.28, 0.035, 0.001, -0.0008, 0.003]),
    ("fisheye", [-0.12, 0.016, -0.003, 0.001]),
])
def test_fit_and_reproject_using_calibrated_pixels(tmp_path, model, coeffs):
    source = tmp_path / "camera.png"
    target = tmp_path / "ortho.png"
    cv2.imwrite(str(source), np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(str(target), np.zeros((400, 450, 3), np.uint8))
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(json.dumps({"model": model, "camera_matrix": K.tolist(),
                                     "dist_coeffs": coeffs, "image_size": [320, 240]}))
    calibration = load_calibration_json(str(intrinsics))
    raw_points = _distort(POINTS, model, np.asarray(coeffs))
    assert np.max(np.linalg.norm(raw_points - POINTS, axis=1)) > 0.25
    ground_points = _transform(POINTS, H_EXPECTED)
    b = backend()
    b.set_npy_file_path(str(tmp_path / "homography.npy"))
    b.set_txt_file_path(str(tmp_path / "homography.txt"))
    b.set_calibration_file(str(intrinsics))
    b.setFileName1(str(source))
    b.setFileName2(str(target))
    preview = b.doHomography(raw_points.tolist(), ground_points.tolist())
    assert preview.shape == (400, 450, 3)
    saved = np.load(tmp_path / "homography.npy", allow_pickle=True)
    assert saved[0].shape == (3, 3)
    assert len(saved) == 3  # entry [0] stays backward-compatible
    assert np.allclose(saved[0].astype(float) / saved[0][2, 2], H_EXPECTED, atol=2e-3)
    assert read_metadata(str(tmp_path / "homography.npy")) == calibration
    recovered_raw = distort_points(undistort_points(raw_points, calibration), calibration)
    assert np.max(np.linalg.norm(recovered_raw - raw_points, axis=1)) < 0.01
    recovered = _transform(undistort_points(raw_points, calibration), saved[0].astype(float))
    assert np.max(np.linalg.norm(recovered - ground_points, axis=1)) < 0.01
    # Copying only .npy still preserves the correction (no detached metadata).
    copied = tmp_path / "propagated.npy"
    shutil.copy2(tmp_path / "homography.npy", copied)
    assert read_metadata(str(copied)) == calibration
    # Visualized warped image uses the same undistortion as reprojection.
    figure_path, overlay_path = save_homography_visualizations(
        str(source), str(target), saved[0].astype(float), str(tmp_path / "preview.png"),
        dpi=120, calibration=read_metadata(str(copied))
    )
    assert cv2.imread(overlay_path).shape == (400, 450, 3)
    assert Path(figure_path).exists()
    assert undistort_image(cv2.imread(str(source)), calibration).shape == (240, 320, 3)


def test_backwards_compatibility_and_mixed_file_guard(tmp_path):
    source = tmp_path / 'camera.png'
    target = tmp_path / 'ortho.png'
    cv2.imwrite(str(source), np.zeros((200, 300, 3), np.uint8))
    cv2.imwrite(str(target), np.zeros((200, 300, 3), np.uint8))
    b = backend()
    filename = str(tmp_path / 'legacy.npy')
    b.set_npy_file_path(filename)
    b.set_txt_file_path(str(tmp_path / 'legacy.txt'))
    b.setFileName1(str(source))
    b.setFileName2(str(target))
    points = [[20, 20], [200, 20], [200, 150], [20, 150]]
    b.doHomography(points, points)
    assert len(np.load(filename, allow_pickle=True)) == 2
    assert read_metadata(filename) is None
    Path(metadata_path(filename)).write_text(json.dumps({
        'schema_version': 1, 'matrix_source_domain': 'undistorted_street_pixels_P_equals_K',
        'correspondences_domain': 'raw_street_pixels',
        'calibration': {'model':'pinhole','K':K.tolist(),'D':[0,0,0,0],'image_size': None},
    }))
    with pytest.raises(ValueError, match="legacy .npy"):
        read_metadata(filename)


def test_json_multi_camera_and_reject_unsupported_models(tmp_path):
    filename = tmp_path / 'cameras.json'
    filename.write_text(json.dumps({'cam1': {'model': 'pinhole', 'intrinsic_matrix': K.tolist(),
                                             'distortion_coefficients':[0,0,0,0], 'image_size':[320, 240]},
                                    'cam2': {'model':'scaramuzza', 'K': K.tolist(), 'D': [0,0,0,0]}}))
    assert load_calibration_json(filename, camera='cam1')['image_size'] == [320, 240]
    with pytest.raises(ValueError, match="not an OpenCV"):
        load_calibration_json(filename, camera='cam2')
    with pytest.raises(ValueError, match="differs"):
        check_image_size(load_calibration_json(filename, camera='cam1'), (720,1280,3))
    with pytest.raises(ValueError, match="Camera"):
        load_calibration_json(filename, camera='does_not_exist')
