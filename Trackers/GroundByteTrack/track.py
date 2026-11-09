"""Transplan adapter for GroundByteTrack (same TXT/PKL schema as ByteTrack)."""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from homography_calibration import check_image_size, read_metadata, undistort_points

from .core import GroundByteTrackConfig, GroundByteTracker, GroundDetection

TRACK_COLUMNS = ["fn", "id", "score", "x1", "y1", "x2", "y2", "class"]


def _meters_per_pixel(args):
    explicit = getattr(args, "GPMetersPerPixel", None)
    if explicit is not None:
        if not np.isfinite(explicit) or explicit <= 0:
            raise ValueError("--GPMetersPerPixel must be positive and finite")
        return float(explicit)
    if getattr(args, "TopView", None) == "GoogleMap":
        center = getattr(args, "MetaData", {}).get("center")
        if center is None:
            raise ValueError("GoogleMap tracking requires metadata.center")
        lat = float(center[0])
        zoom = int(getattr(args, "GPMapZoom", 19))
        return 156543.03392 * np.cos(np.deg2rad(lat)) / 2 ** zoom
    raise ValueError(
        "GroundByteTrack needs --GPMetersPerPixel for orthophoto/custom top views "
        "(e.g. 180 metre crop / 3000 pixels = 0.06). "
        "For GoogleMap, set --TopView GoogleMap to use metadata.center and zoom."
    )


def _load_homography(path):
    # Transplan's homography GUI saves cv2.findHomography's (H, inlier_mask)
    # tuple to .npy; older NumPy versions saved this as an object array.
    # Loading legacy object arrays requires trusted local pickle contents.
    arr = np.load(path, allow_pickle=True)
    if arr.shape == (3, 3):
        h = arr
    elif arr.ndim >= 1 and len(arr):
        h = arr[0]
    else:
        raise ValueError(f"Invalid homography data: {path}")
    h = np.asarray(h, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError(f"Invalid 3x3 homography: {path}")
    return h


def _contact_points(frame_dets, args):
    mode = getattr(args, "ContactPoint", None) or "BottomPoint"
    x = (frame_dets["x1"].to_numpy(dtype=float) + frame_dets["x2"].to_numpy(dtype=float)) / 2
    if mode == "BottomPoint":
        y = frame_dets["y2"].to_numpy(dtype=float)
    elif mode == "Center":
        y = (frame_dets["y1"].to_numpy(dtype=float) + frame_dets["y2"].to_numpy(dtype=float)) / 2
    elif mode in ("BottomSeg", "LineSeg"):
        from Homography import get_contact_point
        return np.asarray([get_contact_point(row, args)
                           for _, row in frame_dets.iterrows()], dtype=float).reshape(-1, 2)
    else:
        raise ValueError(f"GroundByteTrack unsupported ContactPoint: {mode}")
    return np.column_stack((x, y))


def _project(points, h, scale):
    if not len(points):
        return np.zeros((0, 2), dtype=float)
    src = np.column_stack((points, np.ones(len(points))))
    proj = src @ h.T
    denom = proj[:, 2]
    result = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.isfinite(proj).all(axis=1) & (np.abs(denom) > 1e-9)
    result[valid] = (proj[valid, :2] / denom[valid, None]) * scale
    return result


def track(args, *unused):
    """Read existing detections, track on ground plane, save compatible TXT.

    --Track uses existing Transplan's Track.store_df_pickle() afterwards.
    No video decoding is necessary; the video is used only for frame count/FPS.
    """
    import cv2

    method = getattr(args, "BackprojectionMethod", None) or "Homography"
    if method != "Homography":
        raise NotImplementedError(
            "GroundByteTrack currently supports --BackprojectionMethod Homography only. "
            "DSM tracking requires a ground-metric coordinate adapter."
        )
    h = _load_homography(args.HomographyNPY)
    # The updated homography GUI stores the same camera calibration it used
    # to fit H directly in the .npy. Legacy H files return None here.
    calibration = read_metadata(args.HomographyNPY)
    if calibration is None:
        print("GroundByteTrack: legacy homography (raw camera coordinates; no lens correction)")
    else:
        print("GroundByteTrack: calibrated homography (undistort raw contact points with P=K)")
    scale = _meters_per_pixel(args)
    config = GroundByteTrackConfig(
        high_thresh=getattr(args, "GPHighThresh", 0.5),
        low_thresh=getattr(args, "GPLowThresh", 0.1),
        new_track_thresh=getattr(args, "GPNewTrackThresh", 0.6),
        match_distance_m=getattr(args, "GPMatchDistance", 3.0),
        second_match_distance_m=getattr(args, "GPSecondMatchDistance", 2.0),
        track_buffer=getattr(args, "GPTrackBuffer", 90),
        min_hits=getattr(args, "GPMinHits", 1),
        class_aware=not getattr(args, "GPClassAgnostic", False),
    )
    detections = pd.read_pickle(args.DetectionPkl)
    required = {"fn", "x1", "y1", "x2", "y2", "score"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"DetectionPkl missing columns: {sorted(missing)}")

    cap = cv2.VideoCapture(args.Video)
    if not cap.isOpened():
        raise OSError(f"Cannot read video for frame count/FPS: {args.Video}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"Invalid video FPS: {fps}")
    if frames <= 0:
        raise ValueError(f"Video has no frames: {args.Video}")
    if calibration is not None:
        check_image_size(calibration, (video_height, video_width))
    limit = getattr(args, "ForNFrames", None)
    if limit is not None:
        frames = min(frames, int(limit))
    if frames < 0:
        raise ValueError("ForNFrames must be nonnegative")

    tracker = GroundByteTracker(config, fps=fps)
    if len(detections):
        frame_values = pd.to_numeric(detections.fn, errors="raise").to_numpy()
        if not np.isfinite(frame_values).all() or not np.all(frame_values == frame_values.astype(int)):
            raise ValueError("Detection frame numbers must be finite integers")
        detections = detections.copy()
        detections["fn"] = frame_values.astype(int)
        # Unrelated observations outside the requested range are ignored.
        detections = detections[(detections.fn >= 0) & (detections.fn < frames)]

    # GroundByteTracker.update() always discards scores below low_thresh.
    # Drop them *before* creating contact points, homography projections,
    # and Python objects. This does not change any accepted associations.
    detections = detections.loc[detections["score"] >= config.low_thresh].copy()

    # Project the remaining detections once, in a vectorized batch, before
    # running the sequential tracker. In particular this avoids doing a
    # separate Pandas/NumPy conversion for each frame.
    if len(detections):
        xy_image = _contact_points(detections, args)
        # Fit in the same undistorted K-coordinate system as the GUI.
        # This remains vectorized and is performed once before tracking.
        xy_for_homography = undistort_points(xy_image, calibration)
        xy_ground = _project(xy_for_homography, h, scale)
        detections["__ground_x"] = xy_ground[:, 0]
        detections["__ground_y"] = xy_ground[:, 1]

    grouped = {int(fn): group for fn, group in detections.groupby("fn", sort=False)}
    output = []
    for fn in tqdm(range(frames), desc="GroundByteTrack", unit="frame"):
        group = grouped.get(fn)
        frame_dets = []
        if group is not None and len(group):
            boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            scores = group["score"].to_numpy(dtype=float)
            projected = group[["__ground_x", "__ground_y"]].to_numpy(dtype=float)
            classes = (group["class"].to_numpy() if "class" in group
                       else np.full(len(group), -1))
            valid = (np.isfinite(boxes).all(axis=1) &
                     (boxes[:, 2] > boxes[:, 0]) &
                     (boxes[:, 3] > boxes[:, 1]) &
                     np.isfinite(projected).all(axis=1) &
                     np.isfinite(scores))
            for n in np.flatnonzero(valid):
                cls = -1 if pd.isna(classes[n]) else int(classes[n])
                frame_dets.append(GroundDetection(
                    projected[n], boxes[n].copy(), float(scores[n]), cls))
        for t in tracker.update(frame_dets, fn):
            output.append((fn, t.track_id, t.score, *t.box, t.cls))

    result = pd.DataFrame(output, columns=TRACK_COLUMNS)
    df_txt(result, args.TrackingPth)
    print(f"GroundByteTrack: {len(result)} observations, "
          f"{result.id.nunique() if len(result) else 0} tracks; "
          f"scale={scale:.6f} m/pixel, fps={fps:.3f}")


def df(args):
    """Match legacy ByteTrack's tracking DataFrame format, even if empty."""
    path = Path(args.TrackingPth)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=TRACK_COLUMNS).astype({"fn": int, "id": int, "class": int})
    data = pd.read_csv(path, header=None, names=TRACK_COLUMNS)
    for name in ("fn", "id", "class"):
        data[name] = data[name].astype(int)
    return data


def df_txt(frame: pd.DataFrame, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Keep columns and exact order expected by legacy ByteTrack and Track.py.
    with open(out_path, "w", encoding="utf-8") as stream:
        for row in frame[TRACK_COLUMNS].itertuples(index=False, name=None):
            fn, tid, score, x1, y1, x2, y2, cls = row
            stream.write(f"{int(fn)},{int(tid)},{float(score):.6f},"
                         f"{float(x1):.4f},{float(y1):.4f},"
                         f"{float(x2):.4f},{float(y2):.4f},{int(cls)}\n")
