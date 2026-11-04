"""Two-stage, ground-plane distance-based ByteTrack-style multi-object tracker.

This is a new implementation, not an import of the legacy GByteTrack YOLOX fork.
Image boxes are NEVER used for association. A constant-velocity Kalman filter
models each object's contact point in ground-plane metres. Every emitted row
corresponds to an observed detection, rather than an extrapolated image box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class GroundDetection:
    xy: np.ndarray             # ground-plane metres, shape (2,)
    box: np.ndarray            # original image-space xyxy, shape (4,)
    score: float
    cls: int = -1


@dataclass
class GroundTrack:
    track_id: int
    state: np.ndarray          # [x_m, y_m, vx_m/s, vy_m/s]
    covariance: np.ndarray
    cls: int
    hits: int
    last_frame: int
    box: np.ndarray
    score: float
    confirmed: bool
    status: str = "active"


@dataclass
class GroundByteTrackConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_distance_m: float = 3.0
    second_match_distance_m: float = 2.0
    track_buffer: int = 90
    min_hits: int = 1
    measurement_std_m: float = 0.5
    acceleration_std_mps2: float = 8.0
    initial_speed_std_mps: float = 12.0
    class_aware: bool = True

    def __post_init__(self):
        if not 0 <= self.low_thresh < self.high_thresh <= self.new_track_thresh <= 1:
            raise ValueError("Require 0 <= GPLowThresh < GPHighThresh <= GPNewTrackThresh <= 1")
        if min(self.match_distance_m, self.second_match_distance_m,
               self.measurement_std_m, self.acceleration_std_mps2,
               self.initial_speed_std_mps) <= 0:
            raise ValueError("Distance and noise parameters must be positive")
        if self.track_buffer < 0 or self.min_hits < 1:
            raise ValueError("GPTrackBuffer must be nonnegative; GPMinHits >= 1")


class GroundByteTracker:
    """ByteTrack-style: high-score association, then low-score recovery.

    update() must be called once per frame, including frames without detections.
    Detection scores below low_thresh are ignored. Only confirmed active tracks
    matched to a detection *this frame* are returned.
    """

    def __init__(self, config: GroundByteTrackConfig, fps: float = 30.0):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.config = config
        self.dt = 1.0 / fps
        self.tracks: list[GroundTrack] = []
        self._last_frame: Optional[int] = None
        self._next_id = 1
        self._h = np.zeros((2, 4), dtype=float)
        self._h[0, 0] = self._h[1, 1] = 1.0
        self._r = np.eye(2) * config.measurement_std_m ** 2

    def _predict(self, track: GroundTrack, delta_frames: int) -> None:
        dt = self.dt * delta_frames
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        # White-noise acceleration model: Q = (G G^T) sigma_a^2.
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        q = (self.config.acceleration_std_mps2 ** 2) * (g @ g.T)
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + q

    def _correct(self, track: GroundTrack, det: GroundDetection, frame: int) -> None:
        innovation = np.asarray(det.xy, dtype=float) - self._h @ track.state
        s = self._h @ track.covariance @ self._h.T + self._r
        k = np.linalg.solve(s, self._h @ track.covariance).T
        track.state = track.state + k @ innovation
        # Joseph form prevents covariance losing symmetry/PSD numerically.
        ikh = np.eye(4) - k @ self._h
        track.covariance = ikh @ track.covariance @ ikh.T + k @ self._r @ k.T
        track.hits += 1
        track.last_frame = frame
        track.box = np.asarray(det.box, dtype=float).copy()
        track.score = det.score
        if track.cls < 0 and det.cls >= 0:
            track.cls = det.cls
        track.confirmed = track.hits >= self.config.min_hits
        track.status = "active"

    def _new(self, det: GroundDetection, frame: int) -> GroundTrack:
        x, y = map(float, det.xy)
        cfg = self.config
        initial_cov = np.diag([
            cfg.measurement_std_m ** 2, cfg.measurement_std_m ** 2,
            cfg.initial_speed_std_mps ** 2, cfg.initial_speed_std_mps ** 2,
        ])
        track = GroundTrack(
            track_id=self._next_id,
            state=np.array([x, y, 0.0, 0.0]),
            covariance=initial_cov,
            cls=int(det.cls), hits=1, last_frame=frame,
            box=np.asarray(det.box, dtype=float).copy(), score=float(det.score),
            confirmed=cfg.min_hits <= 1,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _associate(self, pool: list[GroundTrack], dets: list[GroundDetection],
                   max_dist: float):
        if not pool or not dets:
            return [], list(range(len(pool))), list(range(len(dets)))
        predicted = np.array([t.state[:2] for t in pool], dtype=float)
        measured = np.array([d.xy for d in dets], dtype=float)
        costs = np.linalg.norm(predicted[:, None, :] - measured[None, :, :], axis=2)
        valid = costs <= max_dist
        if self.config.class_aware:
            for i, track in enumerate(pool):
                if track.cls >= 0:
                    for j, det in enumerate(dets):
                        if det.cls >= 0 and det.cls != track.cls:
                            valid[i, j] = False
        # Assign only feasible distances; never let large invalid edges steal matches.
        matrix = np.where(valid, costs, 1.0e12)
        ri, ci = linear_sum_assignment(matrix)
        matches = [(int(i), int(j)) for i, j in zip(ri, ci) if valid[i, j]]
        matched_tracks = {i for i, _ in matches}
        matched_dets = {j for _, j in matches}
        return (matches, [i for i in range(len(pool)) if i not in matched_tracks],
                [j for j in range(len(dets)) if j not in matched_dets])

    def update(self, detections: list[GroundDetection], frame: int) -> list[GroundTrack]:
        frame = int(frame)
        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError("Frame numbers must strictly increase")
        delta_frames = 1 if self._last_frame is None else frame - self._last_frame
        self._last_frame = frame
        cfg = self.config
        # Expire before matching: otherwise a very old track can be revived.
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= cfg.track_buffer]
        for track in self.tracks:
            self._predict(track, delta_frames)
            if track.status == "active":
                track.status = "lost"

        # Do not let invalid, low-score or malformed detections enter association.
        valid_dets = [d for d in detections if
                      np.isfinite(d.score) and np.isfinite(d.xy).all() and
                      np.isfinite(d.box).all() and d.score >= cfg.low_thresh]
        high = [d for d in valid_dets if d.score >= cfg.high_thresh]
        low = [d for d in valid_dets if d.score < cfg.high_thresh]
        # Unconfirmed tracks can only match high-confidence detections.
        confirmed = [t for t in self.tracks if t.confirmed]
        unconfirmed = [t for t in self.tracks if not t.confirmed]
        matched_output = []

        matches, unmatch_track_ids, unmatch_high = self._associate(
            confirmed, high, cfg.match_distance_m)
        for i, j in matches:
            t = confirmed[i]
            self._correct(t, high[j], frame)
            matched_output.append(t)

        # Stage 2: only previously ACTIVE unmatched tracks, never lost tracks.
        # Before predicting, 'active' status was reset to 'lost'; the last_frame
        # field reveals whether the track was observed the previous frame.
        second_pool = [confirmed[i] for i in unmatch_track_ids
                       if confirmed[i].last_frame == frame - 1]
        second_matches, _, _ = self._associate(
            second_pool, low, cfg.second_match_distance_m)
        for i, j in second_matches:
            t = second_pool[i]
            self._correct(t, low[j], frame)
            matched_output.append(t)

        remaining_high = [high[j] for j in unmatch_high]
        matches, unmatched_pending, unmatched_high = self._associate(
            unconfirmed, remaining_high, cfg.match_distance_m)
        for i, j in matches:
            t = unconfirmed[i]
            self._correct(t, remaining_high[j], frame)
            if t.confirmed:
                matched_output.append(t)
        # ByteTrack drops unmatched unconfirmed tracks immediately.
        pending_to_drop = {unconfirmed[i].track_id for i in unmatched_pending}
        for j in unmatched_high:
            det = remaining_high[j]
            if det.score >= cfg.new_track_thresh:
                t = self._new(det, frame)
                if t.confirmed:
                    matched_output.append(t)

        self.tracks = [t for t in self.tracks
                       if t.track_id not in pending_to_drop and
                       frame - t.last_frame <= cfg.track_buffer]
        return matched_output
