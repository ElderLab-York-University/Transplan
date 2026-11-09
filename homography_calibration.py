"""Optional lens correction for pixel-to-orthophoto homographies.

Input correspondence CSVs and detection/track contact points are always RAW
street-image pixels. A calibrated homography maps UNDISTORTED street-image
pixels (using the original camera matrix K as P) to orthophoto pixels.

A portable, embedded calibration snapshot travels with the .npy matrix so
Transplan's reprojection and visualization stages cannot unknowingly apply the
matrix in the wrong coordinate domain.
"""
import json
import os

import cv2
import numpy as np


SIDECAR_SUFFIX = ".calibration.json"


def metadata_path(homography_npy):
    return str(homography_npy) + SIDECAR_SUFFIX


# Transplan's existing per-video calibration JSON often stores an OpenCV K as
# `intrinsic_matrix`, sometimes as a 3x4 camera projection matrix [K | 0].
_MATRIX_KEYS = ("K", "camera_matrix", "cameraMatrix", "intrinsic_matrix", "intrinsics")


def _read_matrix(data):
    for key in _MATRIX_KEYS:
        if key in data:
            val = data[key]
            if isinstance(val, dict) and "data" in val:
                val = val["data"]
            result = np.asarray(val, dtype=np.float64)
            if result.size == 9:
                return result.reshape(3, 3)
            if result.shape == (3, 4) and np.allclose(result[:, 3], 0):
                return result[:, :3]
            if (result.shape == (4, 4) and np.allclose(result[:3, 3], 0)
                    and np.allclose(result[3], [0, 0, 0, 1])):
                return result[:3, :3]
            raise ValueError("{} must be a 3x3 K or a 3x4 [K|0] matrix; got shape {}".format(
                key, result.shape))
    raise ValueError("Calibration needs an intrinsic_matrix / camera_matrix / K")

def _read_distortion(data):
    for key in ("D", "dist_coeffs", "distortion_coefficients", "dist_coefficients",
                "distCoeffs", "distortion", "distortion_coeffs"):
        if key in data:
            val = data[key]
            if isinstance(val, dict) and "data" in val:
                val = val["data"]
            return np.asarray(val, dtype=np.float64).reshape(-1)
    raise ValueError("Calibration needs distortion coefficients: D/dist_coeffs/distortion_coefficients")


def calibration_from_dict(data, camera=None):
    """Parse a single camera or Transplan's {camera_name: calibration} JSON.

    A per-camera file with exactly one entry works without camera metadata.
    Files containing multiple entries need an explicit or metadata camera key.
    """
    if not isinstance(data, dict):
        raise ValueError("Calibration must be a JSON object")
    direct_calibration = any(key in data for key in _MATRIX_KEYS)
    if not direct_calibration:
        if camera is not None and camera in data:
            data = data[camera]
        elif camera is not None:
            raise ValueError("Camera {!r} not found in calibration JSON; available keys: {}".format(
                camera, list(data)))
        else:
            candidates = [val for val in data.values()
                          if isinstance(val, dict) and any(key in val for key in _MATRIX_KEYS)]
            if len(candidates) != 1:
                raise ValueError("Calibration contains {} camera entries. Specify "
                                 "--HomographyCalibrationCamera or provide MetaData['camera'].".format(
                                     len(candidates)))
            data = candidates[0]
    if not isinstance(data, dict):
        raise ValueError("Selected camera calibration must be a JSON object")
    model_string = str(data.get("model", data.get("distortion_model", data.get("camera_model", "pinhole")))).lower()
    if "fisheye" in model_string and "scaramuzza" not in model_string:
        model = "fisheye"
    elif any(keyword in model_string for keyword in ("scaramuzza", "ocam", "mei", "omnidir", "omnidirectional")):
        raise ValueError("{} is not an OpenCV pinhole/fisheye calibration; convert/rectify first".format(model_string))
    elif model_string in ("pinhole", "opencv", "opencv_pinhole", "brown", "brown_conrady", "radtan", "perspective"):
        model = "pinhole"
    else:
        raise ValueError("Unsupported camera distortion model: {}".format(model_string))
    K = _read_matrix(data)
    D = _read_distortion(data)
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Calibration contains nonfinite values")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2]) < 1e-8:
        raise ValueError("Invalid camera intrinsics (focal lengths/K)")
    if model == "fisheye" and len(D) != 4:
        raise ValueError("OpenCV fisheye model needs exactly 4 distortion coefficients")
    if model == "pinhole" and len(D) not in (4, 5, 8, 12, 14):
        raise ValueError("OpenCV pinhole model needs 4, 5, 8, 12, or 14 distortion coefficients")
    size = data.get("image_size", data.get("imageSize", data.get("resolution")))
    if size is not None:
        if isinstance(size, dict):
            size = [size["width"], size["height"]]
        size = [int(x) for x in size]
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("image_size must be [width, height]")
    return {"model": model, "K": K.tolist(), "D": D.tolist(), "image_size": size}


def load_calibration_json(path, camera=None):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return calibration_from_dict(data, camera)


def resolve_homography_calibration(args):
    """Return (path, camera) for the GUI, using Transplan's existing paths.

    Priority: --HomographyIntrinsics > args.INTRINSICS_PATH > legacy/embedded.
    The default path is computed the same way as Utils.get_intrinsics_path
    when args.INTRINSICS_PATH has not yet been populated.

    Missing default file is optional; an explicit missing file or an existing
    malformed calibration is not silently ignored.
    """
    explicit = getattr(args, "HomographyIntrinsics", None)
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError("--HomographyIntrinsics file does not exist: {}".format(path))
        source = "--HomographyIntrinsics"
    else:
        path = getattr(args, "INTRINSICS_PATH", None)
        if not path:
            dataset, video = getattr(args, "Dataset", None), getattr(args, "Video", None)
            if dataset and video:
                stem = os.path.splitext(os.path.basename(video))[0]
                path = os.path.join(dataset, stem + ".intrinsic_calibration.json")
        if not path or not os.path.isfile(path):
            return None, None
        path = os.path.abspath(path)
        source = "Transplan's per-video intrinsic_calibration.json"

    camera = getattr(args, "HomographyCalibrationCamera", None)
    if camera is None:
        metadata = getattr(args, "MetaData", None)
        candidate = metadata.get("camera") if isinstance(metadata, dict) else None
        # Only adopt metadata's camera key when it actually exists in the JSON.
        # Direct/single-camera JSONs don't require a camera key.
        with open(path, "r", encoding="utf-8") as f:
            contents = json.load(f)
        if candidate is not None and isinstance(contents, dict) and candidate in contents:
            camera = candidate

    try:
        load_calibration_json(path, camera=camera)
    except ValueError as e:
        raise ValueError("Cannot use {} ({}): {}. Undistortion requires "
                         "both intrinsic_matrix (K) and lens distortion coefficients "
                         "for the corresponding raw street-view video.".format(source, path, e)) from e
    return path, camera


def check_image_size(calibration, image_shape):
    expected = calibration.get("image_size")
    if expected is not None:
        height, width = image_shape[:2]
        if [width, height] != list(expected):
            raise ValueError("Calibration image_size {} differs from street image {}x{}. "
                             "Use matching calibration or rescale K explicitly.".format(expected, width, height))


def undistort_points(points, calibration):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if calibration is None:
        return pts.reshape(-1, 2).copy()
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    if calibration["model"] == "fisheye":
        result = cv2.fisheye.undistortPoints(pts, K, D, P=K)
    else:
        if hasattr(cv2, "undistortPointsIter"):
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-12)
            result = cv2.undistortPointsIter(pts, K, D, None, K, criteria)
        else:
            result = cv2.undistortPoints(pts, K, D, P=K)
    return result.reshape(-1, 2)


def distort_points(points, calibration):
    """Convert undistorted K-frame pixel coordinates back to RAW camera pixels."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if calibration is None:
        return pts.copy()
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    normalized = np.column_stack(((pts[:, 0] - K[0, 2]) / K[0, 0],
                                  (pts[:, 1] - K[1, 2]) / K[1, 1]))
    if calibration["model"] == "fisheye":
        return cv2.fisheye.distortPoints(normalized.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    xyz = np.column_stack((normalized, np.ones(len(normalized))))
    raw, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, D)
    return raw.reshape(-1, 2)


def undistort_image(image, calibration):
    if calibration is None:
        return image
    check_image_size(calibration, image.shape)
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    if calibration["model"] == "fisheye":
        return cv2.fisheye.undistortImage(image, K, D, Knew=K, new_size=(image.shape[1], image.shape[0]))
    return cv2.undistort(image, K, D, None, K)


def _payload(calibration):
    return {
        "schema_version": 1,
        "matrix_source_domain": "undistorted_street_pixels_P_equals_K",
        "correspondences_domain": "raw_street_pixels",
        "calibration": calibration,
    }


def save_homography(homography_npy, matrix, mask, calibration):
    """Save compatible matrix [0], optional embedded metadata [2], and readable sidecar.

    Embedding ensures copying only the .npy between videos retains calibration.
    """
    if calibration is None:
        data = np.empty(2, dtype=object)
        data[0], data[1] = matrix, mask
    else:
        data = np.empty(3, dtype=object)
        data[0], data[1], data[2] = matrix, mask, _payload(calibration)
    np.save(homography_npy, data)
    sidecar = metadata_path(homography_npy)
    if calibration is None:
        if os.path.exists(sidecar):
            os.remove(sidecar)
    else:
        with open(sidecar, "w", encoding="utf-8") as file:
            json.dump(data[2], file, indent=2)
            file.write("\n")


def _validated_payload(metadata, location):
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or metadata.get("matrix_source_domain") != "undistorted_street_pixels_P_equals_K"
            or metadata.get("correspondences_domain") != "raw_street_pixels"):
        raise ValueError("Unsupported calibrated homography metadata: {}".format(location))
    return calibration_from_dict(metadata["calibration"])


def read_metadata(homography_npy):
    """Read embedded calibration, validate any sidecar, or return None for legacy H."""
    matrix_file = str(homography_npy)
    sidecar = metadata_path(matrix_file)
    embedded = None
    if os.path.exists(matrix_file):
        contents = np.load(matrix_file, allow_pickle=True)
        if len(contents) >= 3:
            embedded = contents[2]
            _validated_payload(embedded, matrix_file)
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as file:
            external = json.load(file)
        _validated_payload(external, sidecar)
        if embedded is None:
            raise ValueError("A legacy .npy is paired with calibrated metadata {}; possible mismatched files".format(sidecar))
        if embedded != external:
            raise ValueError("Homography .npy embedded calibration conflicts with {}".format(sidecar))
    if embedded is None:
        return None
    return _validated_payload(embedded, matrix_file)
