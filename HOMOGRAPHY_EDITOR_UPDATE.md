# Transplan homography editor + calibration-aware backprojection (2026-11-03)

## Summary

This incremental patch combines the previous editable correspondence/high-resolution
visualization work with **optional** camera distortion correction. It preserves the
CSV point order and Transplan's `.npy[0]` homography matrix interface.

**By default**, `--HomographyGUI` reads the existing Transplan calibration path
`args.INTRINSICS_PATH` (defined in `Utils.get_intrinsics_path` as
`<Dataset>/<video_stem>.intrinsic_calibration.json`) if that file exists.
`--HomographyIntrinsics` remains an optional override. If the default file is
missing, existing embedded homography calibration is reused when available;
otherwise the legacy uncalibrated behavior is preserved.
Existing calibrated homographies automatically recover their embedded lens
calibration when reopened, visualized, or applied to detection/track points.

## Installation

From an existing Transplan checkout:

```bash
cd /data2/emery_village/src/Transplan
unzip -o /path/to/Transplan_homography_editor_and_undistortion_auto_intrinsics.zip -d .
```

This ZIP includes only added/modified source files, tests, and this README, not `.git`.

## Run: create or edit a distortion-aware homography

The normal command now automatically loads an existing per-video calibration:

```bash
python main.py [your existing --Dataset/--Video/--TopView etc. arguments] \
  --HomographyGUI
```

To override that file:

```bash
python main.py [your existing arguments] --HomographyGUI \
  --HomographyIntrinsics /absolute/path/camera_intrinsics.json
```

If the JSON is keyed by camera (as in Transplan's existing
`{ "sc1": { "intrinsic_matrix": ..., "image_size": ... } }` schema), the
`args.MetaData["camera"]` key is used automatically when present. A single
nested camera is detected even without metadata; a multi-camera file requires
an identifiable camera key.

For an explicit multi-camera calibration JSON, you may still use:

```bash
python main.py [your existing arguments] --HomographyGUI \
  --HomographyIntrinsics /absolute/path/all_cameras.json \
  --HomographyCalibrationCamera cam1
```

Opening the GUI automatically loads *all* existing CSV correspondence rows.
Click to add, drag numbered circles to adjust, use **Actions** to delete a specific
pair, and click **Save and solve homography**. No fixed point-count limit.
The displayed camera image and saved four-column correspondence CSV remain in
**raw street-image pixel coordinates** so existing annotations are reused.
Internally, street points are undistorted with the calibrated camera model,
then `cv2.findHomography` maps *undistorted* pixels to the top-view pixels.

The source images/videos and detection boxes are **not modified**.

**Important:** The selected calibration must describe the exact camera optics and
frame geometry of the original **distorted** street frame. Never enable it for
frames or video that have already been undistorted. Intrinsics for the wrong
camera, lens/focal length, resolution, or crop can worsen the result.
If `image_size` is specified in the JSON, a size mismatch raises an error.
Do not just scale the source frame without updating K and the calibration.

## Calibration JSON

Single-camera **OpenCV pinhole (Brown–Conrady)** example:

```json
{
  "model": "pinhole",
  "camera_matrix": [[1200, 0, 960], [0, 1195, 540], [0, 0, 1]],
  "dist_coeffs": [-0.18, 0.05, 0.001, -0.001, 0.0],
  "image_size": [1920, 1080]
}
```

Use your *measured* calibration values: the above numbers are **illustrative only**.
`camera_matrix` may instead be `K` or `intrinsic_matrix`;
`intrinsic_matrix` may be 3x3 or Transplan's 3x4 `[K | 0]` form;
`dist_coeffs` may instead be `D` or `distortion_coefficients`.
OpenCV pinhole supports 4, 5, 8, 12, or 14 distortion coefficients.

**Do not assume the existing file contains distortion coefficients.** Some
Transplan/DSM calibrations store only `intrinsic_matrix` and `image_size`.
Those files are usable for projections but not for lens undistortion. If the
default calibration file exists without a supported distortion vector, the GUI
fails with a clear error instead of silently inventing zero distortion. Add
measured coefficients to that JSON, or point `--HomographyIntrinsics` at a
complete camera calibration. In particular, do not use synthetic or unrelated
camera parameters.


OpenCV fisheye example:

```json
{
  "model": "fisheye",
  "K": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
  "D": [-0.07, 0.02, 0.0, 0.0],
  "image_size": [1920, 1080]
}
```

This is the **OpenCV fisheye model** (four coefficients), **not** Scaramuzza
`ocam`/polynomial or OpenCV omnidirectional (`omnidir`) models. Such models
are deliberately rejected; they need an appropriate rectifier or conversion.
The `image_size` key, when provided, is `[width, height]`.

## Saving, propagation, and loading

- Existing `.homography.<TopView>.csv` keeps raw camera x/y and top x/y.
- `.homography.<TopView>.npy[0]` remains the 3x3 homography.
- A calibrated `.npy` additionally embeds a calibration snapshot in `[2]`.
- For inspection, `<homography>.npy.calibration.json` contains the same data.
- `--Homography` automatically detects embedded calibration and corrects *raw*
  contact points in detection/track files before applying H.
- `--VisHomographyGUI` and the GUI overlay undistort the source image prior to
  `cv2.warpPerspective`; `--VisTrajectories` uses the same correction.
- Copying a calibrated `.npy` to other video results for the **same camera**
  retains the calibration because it is embedded. If copying to a path with an
  **existing different** `.npy.calibration.json`, remove/copy that mismatched
  sidecar; Transplan fails rather than silently mixing calibrations.
- The original `.npy`, `.txt`, `.csv`, and optional metadata are backed up as
  `.bak` files on the first GUI save when they already exist.
- Loading a *legacy* two-element `.npy` without a sidecar assumes input pixels
  were already undistorted (same as old Transplan). To recalibrate it, reopen
  the **raw-point CSV** in the GUI and solve using the per-video calibration
  or `--HomographyIntrinsics` override.

Do **not** undistort detection points separately before running `--Homography`:
Transplan already handles the correction for calibrated `.npy` files.
The `xcp`/`ycp` columns still store original camera-contact-point coordinates.

## Higher-resolution homography inspection

```bash
python main.py [your existing arguments] --VisHomographyGUI --HomographyVisDPI 300
```

A 3-panel PNG is rendered at adjustable DPI (default 220); the `.allpix.png`
overlay is written at the **original top-view pixel resolution**. Both are
calibration-aware. Increasing output DPI helps inspect the fit; it does not
increase accuracy of the solved homography or source image resolution.

## Tests

```bash
python -m pytest -q tests/test_homography_editor.py tests/test_homography_calibration.py
```

Sixteen noninteractive tests cover automatic per-video calibration loading,
manual override precedence, per-camera selection, missing/malformed calibration,
correspondence reload/edit, native overlay,
legacy matrix compatibility, synthetic pinhole/fisheye undistortion and fit,
metadata propagation, malformed/mismatched metadata, camera selection, and
unsupported model rejection.

Interactive PyQt5 operation was not tested in the patch-building environment;
headless tests and Python compilation passed.
