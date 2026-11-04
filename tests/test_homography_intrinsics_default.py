"""Integration tests: reuse Transplan's per-video calibration naming."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homography_calibration import (load_calibration_json, resolve_homography_calibration,
                                    check_image_size)


def _per_camera_json():
    # Transplan's DSM path accesses intrinsic_dict[cam]['intrinsic_matrix'].
    return {"sc1": {
        "model": "pinhole",
        "intrinsic_matrix": [[500, 0, 320, 0], [0, 510, 240, 0], [0, 0, 1, 0]],
        "distortion_coefficients": [-0.21, 0.04, 0.001, -0.002, 0.002],
        "image_size": [640, 480],
    }}


def _args(tmp_path, **overrides):
    values = dict(Dataset=str(tmp_path), Video="/mnt/example/cam1.mp4",
                  HomographyIntrinsics=None, HomographyCalibrationCamera=None,
                  MetaData={"camera": "sc1"},
                  INTRINSICS_PATH=str(tmp_path / "cam1.intrinsic_calibration.json"))
    values.update(overrides)
    return SimpleNamespace(**values)


def test_per_video_intrinsic_calibration_is_default(tmp_path):
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(_per_camera_json()))
    resolved, camera = resolve_homography_calibration(_args(tmp_path))
    assert resolved == str(path)
    assert camera == "sc1"
    calibration = load_calibration_json(resolved, camera)
    assert np.asarray(calibration["K"]).shape == (3, 3)
    assert calibration["K"][0] == [500, 0, 320]
    assert len(calibration["D"]) == 5
    check_image_size(calibration, (480, 640, 3))


def test_explicit_intrinsics_override_takes_priority(tmp_path):
    default = tmp_path / "cam1.intrinsic_calibration.json"
    default.write_text(json.dumps(_per_camera_json()))
    manual = tmp_path / "other.json"
    manual.write_text(json.dumps({"K": np.eye(3).tolist(), "D": [0, 0, 0, 0]}))
    actual, camera = resolve_homography_calibration(_args(tmp_path, HomographyIntrinsics=str(manual)))
    assert actual == str(manual)
    assert camera is None


def test_missing_per_video_file_keeps_legacy_or_embedded_mode(tmp_path):
    assert resolve_homography_calibration(_args(tmp_path)) == (None, None)
    # Fallback naming also works if args.INTRINSICS_PATH has not been populated.
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(_per_camera_json()))
    resolved, camera = resolve_homography_calibration(_args(tmp_path, INTRINSICS_PATH=None))
    assert resolved == str(path)
    assert camera == "sc1"


def test_single_nested_entry_without_camera_metadata(tmp_path):
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(_per_camera_json()))
    resolved, camera = resolve_homography_calibration(_args(tmp_path, MetaData={}))
    assert resolved == str(path) and camera is None
    assert load_calibration_json(resolved)["K"][0][0] == 500


def test_multi_camera_json_requires_resolvable_camera(tmp_path):
    data = _per_camera_json()
    data["sc2"] = data["sc1"].copy()
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Specify --HomographyCalibrationCamera"):
        resolve_homography_calibration(_args(tmp_path, MetaData={}))
    resolved, camera = resolve_homography_calibration(_args(tmp_path))
    assert camera == "sc1"
    resolved, camera = resolve_homography_calibration(_args(
        tmp_path, HomographyCalibrationCamera="sc2"))
    assert camera == "sc2"


def test_missing_distortion_coefficients_are_not_silently_assumed(tmp_path):
    data = _per_camera_json()
    del data["sc1"]["distortion_coefficients"]
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="distortion coefficients"):
        resolve_homography_calibration(_args(tmp_path))


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="--HomographyIntrinsics"):
        resolve_homography_calibration(_args(tmp_path, HomographyIntrinsics="nonexistent.json"))


def test_intrinsic_projection_rejects_nonzero_translation(tmp_path):
    data = _per_camera_json()
    data["sc1"]["intrinsic_matrix"][0][-1] = 55
    path = tmp_path / "cam1.intrinsic_calibration.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="3x4"):
        resolve_homography_calibration(_args(tmp_path))
