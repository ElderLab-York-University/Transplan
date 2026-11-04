"""Standalone entry point for Transplan's editable homography GUI."""
import argparse
import sys

from PyQt5.QtWidgets import QApplication
from tabbed_ui_func import tabbed_ui_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--StreetView", required=True, help="Original street-view image")
    parser.add_argument("--TopView", required=True, help="Original top-view image")
    parser.add_argument("--Txt", required=True, help="Output homography text path")
    parser.add_argument("--Npy", required=True, help="Output homography NumPy path")
    parser.add_argument("--Csv", required=True, help="Correspondence CSV path (auto-loaded if present)")
    parser.add_argument("--Intrinsics", default=None, help="OpenCV pinhole/fisheye calibration JSON (optional)")
    parser.add_argument("--CalibrationCamera", default=None, help="Camera key for a per-camera calibration JSON")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = tabbed_ui_func(args.StreetView, args.TopView, args.Txt, args.Npy, args.Csv,
                            calibration_path=args.Intrinsics, calibration_camera=args.CalibrationCamera)
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
