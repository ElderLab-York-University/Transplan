"""Homography calculation and native-resolution preview compositing."""
import os
import sys

# GUI is launched with cwd=homographygui; make project-root helpers importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from homography_calibration import (check_image_size, load_calibration_json,
    read_metadata, undistort_points, undistort_image, save_homography)


class backend:
    def __init__(self):
        self.fileName1 = ""
        self.fileName2 = ""
        self.img1 = None
        self.img2 = None
        self.txtfile_path = ""
        self.npyfile_path = ""
        self.calibration = None

    def setFileName1(self, filename):
        self.fileName1 = filename
        self.img1 = cv2.imread(filename)
        if self.img1 is None:
            raise ValueError("Cannot read street-view image: {}".format(filename))
        if self.calibration is not None:
            check_image_size(self.calibration, self.img1.shape)

    def setFileName2(self, filename):
        self.fileName2 = filename
        self.img2 = cv2.imread(filename)
        if self.img2 is None:
            raise ValueError("Cannot read top-view image: {}".format(filename))

    def doHomography(self, pts1, pts2):
        if self.img1 is None or self.img2 is None:
            raise ValueError("Open both images before calculating homography")
        if len(pts1) != len(pts2) or len(pts1) < 4:
            raise ValueError("Homography requires at least four complete point pairs")

        src = np.asarray(pts1, dtype=np.float64)
        dst = np.asarray(pts2, dtype=np.float64)
        if not np.isfinite(src).all() or not np.isfinite(dst).all():
            raise ValueError("All point coordinates must be finite")

        if self.calibration is not None:
            check_image_size(self.calibration, self.img1.shape)
        src_for_fit = undistort_points(src, self.calibration)
        matrix, inlier_mask = cv2.findHomography(src_for_fit, dst)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("Could not solve homography: check for duplicate/collinear points")

        for output in (self.npyfile_path, self.txtfile_path):
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        # Retain the original layout expected elsewhere by Transplan:
        # np.load(path, allow_pickle=True)[0] must give the 3x3 matrix.
        save_homography(self.npyfile_path, matrix, inlier_mask, self.calibration)
        np.savetxt(self.txtfile_path, matrix, delimiter=",")

        camera_image = undistort_image(self.img1, self.calibration)
        height, width = self.img2.shape[:2]
        warped = cv2.warpPerspective(camera_image, matrix, (width, height))
        coverage = cv2.warpPerspective(
            np.ones(camera_image.shape[:2], dtype=np.uint8),
            matrix, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
        blended = cv2.addWeighted(warped, 0.7, self.img2, 0.3, 0)
        preview = self.img2.copy()
        preview[coverage] = blended[coverage]
        return preview

    def set_txt_file_path(self, path):
        self.txtfile_path = path

    def set_npy_file_path(self, path):
        self.npyfile_path = path
        # Existing calibrated homographies reopen in the SAME coordinate system.
        self.calibration = read_metadata(path)

    def set_calibration_file(self, path, camera=None):
        self.calibration = load_calibration_json(path, camera=camera)
        if self.img1 is not None:
            check_image_size(self.calibration, self.img1.shape)
