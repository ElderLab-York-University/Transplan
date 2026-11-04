"""High-resolution, lossless homography diagnostics, independent of the GUI."""
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # These outputs are written to disk, never shown interactively.
import matplotlib.pyplot as plt
import numpy as np

from homography_calibration import check_image_size, undistort_image


def save_homography_visualizations(street_path, top_path, matrix, output_path, dpi=220, calibration=None):
    """Save a higher-DPI 3-panel figure and a native-resolution blended PNG.

    The raw overlay retains exactly the top image's pixel dimensions.
    Upsampling a preview would not introduce additional alignment information.
    """
    if dpi < 50 or dpi > 1200:
        raise ValueError("Homography visualization DPI must be between 50 and 1200")
    street = cv2.imread(street_path, cv2.IMREAD_COLOR)
    top = cv2.imread(top_path, cv2.IMREAD_COLOR)
    if street is None or top is None:
        raise FileNotFoundError("Could not read street/top image for homography visualization")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 homography matrix")

    warped_street = undistort_image(street, calibration)
    height, width = top.shape[:2]
    warped = cv2.warpPerspective(warped_street, matrix, (width, height))
    coverage = cv2.warpPerspective(
        np.ones(warped_street.shape[:2], dtype=np.uint8), matrix,
        (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    blended = cv2.addWeighted(warped, 0.7, top, 0.3, 0)
    overlay = top.copy()
    overlay[coverage] = blended[coverage]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    full_overlay_path = os.path.splitext(output_path)[0] + ".allpix.png"
    if not cv2.imwrite(full_overlay_path, overlay):
        raise OSError("Could not write overlay to {}".format(full_overlay_path))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, image, title in zip(
        axes,
        [warped_street, top, overlay],
        ["Undistorted camera view" if calibration else "Camera view", "Top view", "Camera warped onto top view"],
    ):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path, full_overlay_path
