# GroundByteTrack for Transplan

This change adds **`--Tracker GroundByteTrack`** without modifying any existing tracker, including the legacy `GByteTrack`. It is a lightweight, ground-plane, two-stage **ByteTrack-style** implementation with a 2D constant-velocity Kalman filter. It is **not** the upstream YOLOX ByteTrack or BoxMOT implementation; those trackers use image-space bounding-box overlap, which is not appropriate as the only association metric for this variant.

## What changed

- New code: `Trackers/GroundByteTrack/{core.py,track.py,__init__.py}`.
- Updated `Track.py` tracker registry. Existing tracker behavior is unchanged.
- Updated `Utils.py` to populate the homography path for a tracking-only invocation (even if `--Homography` is not requested). It also configures segmentation paths when choosing segmented contact points.
- Updated `main.py` with the `--GP*` arguments listed below.
- New isolated tests: `tests/test_ground_bytetrack.py`.

## Required data

1. Existing `Results/Detection/*.pkl` for the selected detector, with columns `fn`, `x1`, `y1`, `x2`, `y2`, `score`, and preferably `class`. `fn` is **zero-based**.
2. The original video (Transplan locates it through `--Dataset`, as with other trackers).
3. An existing image-to-top-view **homography** in Transplan's usual `Results/Homography/` location, e.g. created for a top-view name `EmeryOrtho`.
4. A **scale in metres per top-view pixel** for orthophoto/custom top views. For example, a 180-metre-wide orthophoto crop rendered at 3000 pixels gives `180 / 3000 = 0.06 m/pixel` (only if the crop really spans 180 metres and pixel dimensions are not rescaled afterward). For `--TopView GoogleMap`, scale is estimated automatically using `metadata.center` and map zoom (default 19); an explicit `--GPMetersPerPixel` overrides it.

The top-view scale must come from the actual orthophoto/map, not the street-view image resolution. Matching thresholds are specified in **metres**. A mistaken scale changes tracking behavior. This first version supports **homographies only**, not DSM reprojection; a DSM metric-coordinate adapter can be added later.

## Run

From the Transplan repo root, with an environment that can run the existing Transplan `main.py`:

```bash
python main.py \
  --Dataset "/path/to/dataset/camera" \
  --Detector YOLOv5 \
  --Tracker GroundByteTrack \
  --Track \
  --TopView EmeryOrtho \
  --BackprojectionMethod Homography \
  --ContactPoint BottomPoint \
  --GPMetersPerPixel 0.06 \
  --GPMatchDistance 3.0 \
  --GPSecondMatchDistance 2.0 \
  --GPTrackBuffer 90 \
  --VisTrack
```

Use your **actual detector name** in place of `YOLOv5`; `--Detector` must match the name used when detection files were generated. You can run tracking from saved detections: `--Detect` and `--Homography` are **not** needed if those assets already exist. For just 300 frames, append `--ForNFrames 300`. The flag now limits GroundByteTrack's tracking range (legacy trackers may treat it differently).

### Options

| Flag | Default | Meaning |
|---|---:|---|
| `--GPMetersPerPixel` | required for non-GoogleMap | Metric scale of top-view pixels |
| `--GPMapZoom` | 19 | Scale estimation for GoogleMap |
| `--GPLowThresh` | 0.1 | Minimum confidence used in second-stage recovery |
| `--GPHighThresh` | 0.5 | Confidence for first-stage association |
| `--GPNewTrackThresh` | 0.6 | Minimum confidence for starting a track |
| `--GPMatchDistance` | 3.0 | First-stage association gate, metres |
| `--GPSecondMatchDistance` | 2.0 | Second-stage association gate, metres |
| `--GPTrackBuffer` | 90 | Keep unmatched tracks for at most this many frames |
| `--GPMinHits` | 1 | Minimum detections to confirm a new track; 2 reduces single-frame false tracks |
| `--GPClassAgnostic` | off | Allow associations across detection classes |

The existing `--ContactPoint` controls which part of the image box gets projected. Supported: `BottomPoint` (default), `Center`, `BottomSeg`, `LineSeg` (the latter two require the normal Transplan segmentation results). Bottom-center is recommended for street-view vehicles; confirm calibration and point suitability before evaluating accuracy.

## Algorithm

For every video frame (including ones without detections):

1. Compute one image-space contact point per detection and project it through the image-to-top-view homography. Convert top-view pixels to metres.
2. Predict track ground coordinates with a constant-velocity Kalman filter; the state is `(x_m, y_m, vx_m_per_s, vy_m_per_s)` using the video's FPS.
3. Match high-confidence detections to confirmed active **or recently lost** tracks by Hungarian assignment of Euclidean distances in metres, with a distance gate and class consistency.
4. Match remaining **recently active** tracks against low-confidence detections (ByteTrack's recovery stage); low-confidence observations never start new tracks.
5. Attempt to confirm pending tracks with remaining high-confidence detections, start tracks for otherwise unmatched detections above `--GPNewTrackThresh`, and expire tracks after the buffer.
6. Save **only observed** image-space bounding boxes, IDs, confidence, and class. No fake image boxes are generated for occluded frames. Reprojected tracks, top-view visualization, counting and evaluation can use the usual downstream Transplan tools.

If a track is occluded, its state is predicted internally so a later high-confidence detection can recover the same ID within the buffer and maximum association distance. **The result is not a faithful copy of the original ByteTrack implementation**: it deliberately replaces box IoU with a ground-plane metric, and its filter is 2D ground-point-based.

## Outputs and compatibility

Tracking file names are unchanged except for the tracker name; for each video, a track record has:

```text
fn,id,score,x1,y1,x2,y2,class
```

This is the 8-column order used by Transplan's existing ByteTrack adapter. `Track.py` writes the normal tracking pickle and backup pickle from this TXT. Original image boxes remain in image coordinates and correspond to actual detections. An empty clip still produces an empty TXT/PKL with the correct schema.

## Tests

Install existing dependencies in your Transplan environment (minimal new-tracker requirements: NumPy, SciPy, pandas, OpenCV). Then:

```bash
python -m pytest -q tests/test_ground_bytetrack.py
```

Tests cover two-stage matching, multi-class association, Kalman prediction across empty frames, track expiration, minimum hits, invalid homography projection, and a small end-to-end synthetic video/pickle run. These tests **do not** replace a real-scene evaluation: compare `ByteTrack` and `GroundByteTrack` on annotated clips using IDF1/HOTA, ID switches, fragmentation, counts, and visual overlays. In particular, fixed Euclidean gates can struggle at crowded intersections or during long occlusions. Tune thresholds on a held-out set.

## Future BoxMOT unification

A good next step is a tracker interface that takes `(detections, frame_index, projection_context)` and returns a standard track table, with a `backend` for standard image-space BoxMOT trackers and a ground-plane association strategy for this tracker. Do **not** attempt to imitate ground-plane matching by directly sending image boxes into BoxMOT's standard ByteTrack. This initial implementation isolates its association/filter logic in `core.py` so the backend boundary can be introduced without changing Transplan output formats.
