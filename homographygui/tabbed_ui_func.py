"""Editable, backward-compatible correspondence GUI for Transplan.

Existing CSVs are loaded as numbered draggable markers. Points are always
stored in source-image pixel coordinates, regardless of GUI zoom and pan.
"""
import os
import shutil

import cv2
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QFont, QImage, QPen
from PyQt5.QtWidgets import (
    QAction, QDialog, QGraphicsEllipseItem, QGraphicsItem, QGraphicsTextItem,
    QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from QtImageViewer import QtImageViewer
from backend import backend
from correspondences import load_correspondences, save_correspondences


class tabbed_ui_func(QMainWindow):
    def __init__(self, image1_path, image2_path, txtfile_path, npyfile_path, csvfile_path,
                 calibration_path=None, calibration_camera=None):
        super(tabbed_ui_func, self).__init__()
        self.setWindowTitle("Transplan — Homography Correspondences")
        self.b = backend()
        self.set_result_paths(txtfile_path, npyfile_path, csvfile_path)
        self.set_image_paths(image1_path, image2_path)
        if calibration_path is not None:
            self.b.set_calibration_file(calibration_path, camera=calibration_camera)

        self.pts1, self.pts2 = [], []
        self.img1DrawnEllipses, self.img1DrawnTexts = [], []
        self.img2DrawnEllipses, self.img2DrawnTexts = [], []
        self._preview_dialog = None

        self._build_ui()
        self._connect_handlers()

        # Show images AND saved correspondences automatically on startup.
        self.callback_OpenImage1()
        self.callback_OpenImage2()
        self._load_existing_correspondences()
        if self.b.calibration is not None:
            self._update_status("Lens correction: {} (raw CSV points retained)".format(self.b.calibration["model"]))

    def _build_ui(self):
        # Some copies of Transplan contain the Designer .ui file; the supplied
        # repository ZIP does not. Both distributions work with this fallback.
        ui_path = os.path.join(os.path.dirname(__file__), "tabbed_ui.ui")
        if os.path.isfile(ui_path):
            uic.loadUi(ui_path, self)
            self.graphicsView_Image1 = self.findChild(QtImageViewer, "graphicsView_Image1")
            self.graphicsView_Image2 = self.findChild(QtImageViewer, "graphicsView_Image2")
            self.pushButton_OpenImage1 = self.findChild(QPushButton, "pushButton_OpenImage1")
            self.pushButton_OpenImage2 = self.findChild(QPushButton, "pushButton_OpenImage2")
            self.pushButton_DoHomography = self.findChild(QPushButton, "pushButton_DoHomography")
            self.pushButton_DeleteLastPoint1 = self.findChild(QPushButton, "pushButton_DeleteLastPoint1")
            self.pushButton_DeleteLastPoint2 = self.findChild(QPushButton, "pushButton_DeleteLastPoint2")
            if all((self.graphicsView_Image1, self.graphicsView_Image2,
                    self.pushButton_OpenImage1, self.pushButton_OpenImage2,
                    self.pushButton_DoHomography, self.pushButton_DeleteLastPoint1,
                    self.pushButton_DeleteLastPoint2)):
                self._create_edit_menu()
                return
            raise ValueError("tabbed_ui.ui does not contain the expected homography controls")

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)

        self.pushButton_OpenImage1 = QPushButton("Reload camera image")
        self.pushButton_OpenImage2 = QPushButton("Reload top image")
        self.pushButton_DeleteLastPoint1 = QPushButton("Undo camera point")
        self.pushButton_DeleteLastPoint2 = QPushButton("Undo top point")
        self.pushButton_DoHomography = QPushButton("Save and solve homography")
        for button in (
            self.pushButton_OpenImage1, self.pushButton_OpenImage2,
            self.pushButton_DeleteLastPoint1, self.pushButton_DeleteLastPoint2,
            self.pushButton_DoHomography,
        ):
            toolbar.addWidget(button)

        help_label = QLabel(
            "Click either image to add points in matching order. Drag any numbered marker "
            "to correct it. Right-drag: zoom area; right-double-click: fit image. "
            "Actions → Remove pair by number deletes an old match."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        panels = QHBoxLayout()
        layout.addLayout(panels, stretch=1)
        for name, title in (("graphicsView_Image1", "Camera view"),
                            ("graphicsView_Image2", "Top view")):
            panel = QVBoxLayout()
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold; font-size: 13px")
            panel.addWidget(label)
            viewer = QtImageViewer(self)
            viewer.setObjectName(name)
            viewer.setMinimumSize(300, 350)
            panel.addWidget(viewer)
            panels.addLayout(panel, stretch=1)
            setattr(self, name, viewer)

        self._create_edit_menu()

    def _create_edit_menu(self):
        menu = self.menuBar().addMenu("Actions")
        remove_action = QAction("Remove pair by number…", self)
        remove_action.setShortcut("Ctrl+Shift+D")
        remove_action.triggered.connect(self.remove_pair_by_number)
        menu.addAction(remove_action)
        remove_last_action = QAction("Remove last complete pair", self)
        remove_last_action.setShortcut("Ctrl+Shift+Z")
        remove_last_action.triggered.connect(self.remove_last_pair)
        menu.addAction(remove_last_action)

    def _connect_handlers(self):
        self.pushButton_DoHomography.clicked.connect(self.callback_DoHomography)
        self.pushButton_OpenImage1.clicked.connect(self.callback_OpenImage1)
        self.pushButton_OpenImage2.clicked.connect(self.callback_OpenImage2)
        self.pushButton_DeleteLastPoint1.clicked.connect(self.callback_DeleteLastPoint1)
        self.pushButton_DeleteLastPoint2.clicked.connect(self.callback_DeleteLastPoint2)
        # Connect once only; reloading an image must not duplicate signal hooks.
        self.graphicsView_Image1.leftMouseButtonPressed.connect(self.img1HandleLeftClick)
        self.graphicsView_Image2.leftMouseButtonPressed.connect(self.img2HandleLeftClick)

    def set_result_paths(self, txtfile_path, npyfile_path, csvfile_path):
        self.txtfile_path = txtfile_path
        self.npyfile_path = npyfile_path
        self.csvfile_path = csvfile_path
        self.b.set_txt_file_path(txtfile_path)
        self.b.set_npy_file_path(npyfile_path)

    def set_image_paths(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def _open_image(self, viewer, path, update_backend):
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError("Image does not exist: {}".format(path))
            update_backend(path)
            viewer.loadImageFromFile(path)
            viewer.show()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open image", str(exc))

    def callback_OpenImage1(self):
        self._open_image(self.graphicsView_Image1, self.image1_path, self.b.setFileName1)

    def callback_OpenImage2(self):
        self._open_image(self.graphicsView_Image2, self.image2_path, self.b.setFileName2)

    def _load_existing_correspondences(self):
        if not os.path.isfile(self.csvfile_path):
            self.statusBar().showMessage("No previous CSV; starting with empty correspondences.")
            return
        try:
            points1, points2 = load_correspondences(self.csvfile_path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot read correspondences", str(exc))
            return
        for point in points1:
            self._append_marker(1, point[0], point[1])
        for point in points2:
            self._append_marker(2, point[0], point[1])
        self._update_status("Loaded {} existing point pairs from CSV".format(len(points1)))

    def _append_marker(self, which, x, y):
        if which == 1:
            points = self.pts1
            markers = self.img1DrawnEllipses
            labels = self.img1DrawnTexts
            scene = self.graphicsView_Image1.scene
            color = QColor(255, 210, 25)
        else:
            points = self.pts2
            markers = self.img2DrawnEllipses
            labels = self.img2DrawnTexts
            scene = self.graphicsView_Image2.scene
            color = QColor(0, 220, 240)

        points.append([float(x), float(y)])
        marker = QGraphicsEllipseItem(-7, -7, 14, 14)
        marker.setPos(float(x), float(y))
        marker.setPen(QPen(QColor(0, 0, 0), 1.5))
        marker.setBrush(QBrush(color))
        marker.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIgnoresTransformations
        )
        marker.setZValue(50)
        label = QGraphicsTextItem(str(len(points)), marker)
        label.setFont(QFont("Arial", 11, QFont.Bold))
        label.setDefaultTextColor(color)
        label.setPos(8, -17)
        marker.setToolTip("Point {} — drag to correct position".format(len(points)))
        scene.addItem(marker)
        markers.append(marker)
        labels.append(label)

    def _update_status(self, prefix=""):
        counts = "Camera: {} points | Top: {} points".format(
            len(self.img1DrawnEllipses), len(self.img2DrawnEllipses))
        self.statusBar().showMessage("{} | {}".format(prefix, counts) if prefix else counts)

    def _delete_last(self, which):
        if which == 1:
            markers, labels, points = self.img1DrawnEllipses, self.img1DrawnTexts, self.pts1
            scene = self.graphicsView_Image1.scene
        else:
            markers, labels, points = self.img2DrawnEllipses, self.img2DrawnTexts, self.pts2
            scene = self.graphicsView_Image2.scene
        if markers:
            scene.removeItem(markers.pop())
            labels.pop()
            points.pop()
        self._update_status()

    def callback_DeleteLastPoint1(self):
        self._delete_last(1)

    def callback_DeleteLastPoint2(self):
        self._delete_last(2)

    def remove_last_pair(self):
        if self.img1DrawnEllipses and self.img2DrawnEllipses:
            self._delete_last(1)
            self._delete_last(2)

    def remove_pair_by_number(self):
        count = min(len(self.img1DrawnEllipses), len(self.img2DrawnEllipses))
        if not count:
            return
        number, accepted = QInputDialog.getInt(
            self, "Remove correspondence", "Correspondence number (1-based):", 1, 1, count
        )
        if not accepted:
            return
        i = number - 1
        for scene, markers, labels, points in (
            (self.graphicsView_Image1.scene, self.img1DrawnEllipses, self.img1DrawnTexts, self.pts1),
            (self.graphicsView_Image2.scene, self.img2DrawnEllipses, self.img2DrawnTexts, self.pts2),
        ):
            scene.removeItem(markers.pop(i))
            labels.pop(i)
            points.pop(i)
            for new_idx, label in enumerate(labels, start=1):
                label.setPlainText(str(new_idx))
                markers[new_idx - 1].setToolTip("Point {} — drag to correct position".format(new_idx))
        self._update_status("Removed correspondence #{}".format(number))

    def img1HandleLeftClick(self, x, y):
        if self.b.img1 is not None and 0 <= x < self.b.img1.shape[1] and 0 <= y < self.b.img1.shape[0]:
            self._append_marker(1, int(x), int(y))
            self._update_status()

    def img2HandleLeftClick(self, x, y):
        if self.b.img2 is not None and 0 <= x < self.b.img2.shape[1] and 0 <= y < self.b.img2.shape[0]:
            self._append_marker(2, int(x), int(y))
            self._update_status()

    def _current_points(self, markers):
        # Read positions from the *movable graphics items*; this captures edits
        # to already-loaded correspondences rather than saving stale lists.
        return [[float(m.scenePos().x()), float(m.scenePos().y())] for m in markers]

    def callback_DoHomography(self):
        points1 = self._current_points(self.img1DrawnEllipses)
        points2 = self._current_points(self.img2DrawnEllipses)
        if len(points1) != len(points2):
            QMessageBox.warning(
                self, "Incomplete pairs", "There are {} camera points and {} top-view points. "
                "Add/delete points until the counts match.".format(len(points1), len(points2))
            )
            return
        if len(points1) < 4:
            QMessageBox.warning(self, "Insufficient points", "At least four pairs are required.")
            return
        try:
            for name, points, image in (
                ("Camera", points1, self.b.img1), ("Top", points2, self.b.img2),
            ):
                if image is None:
                    raise ValueError("{} image is not loaded".format(name))
                height, width = image.shape[:2]
                for i, (x, y) in enumerate(points, 1):
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError("{} point {} is outside the original image bounds".format(name, i))
            # Preserve the original calibration on the first subsequent edit.
            for path in (self.csvfile_path, self.npyfile_path, self.txtfile_path,
                     self.npyfile_path + ".calibration.json"):
                if os.path.isfile(path) and not os.path.exists(path + ".bak"):
                    shutil.copy2(path, path + ".bak")
            preview = self.b.doHomography(points1, points2)
            save_correspondences(self.csvfile_path, points1, points2)
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "Could not save homography", str(exc))
            return

        self.pts1, self.pts2 = points1, points2
        lens = " ({}, undistorted input)".format(self.b.calibration["model"]) if self.b.calibration else " (legacy coordinates)"
        self._update_status("Saved {} pairs, homography .npy/.txt/.csv{}".format(len(points1), lens))
        print("Saved {} correspondences to {}".format(len(points1), self.csvfile_path))
        self._show_preview(preview)

    def _show_preview(self, preview_bgr):
        """Zoomable full-resolution preview instead of a fixed-size cv2.imshow."""
        rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        if self._preview_dialog is not None:
            self._preview_dialog.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("Homography overlay — {}×{} px (right-drag to zoom)".format(width, height))
        dialog.resize(1280, 850)
        layout = QVBoxLayout(dialog)
        viewer = QtImageViewer(dialog)
        viewer.setImage(qimage)
        layout.addWidget(viewer)
        dialog.show()
        self._preview_dialog = dialog
    def closeEvent(self, event):
        if self._preview_dialog is not None:
            self._preview_dialog.close()
            self._preview_dialog = None

        super().closeEvent(event)