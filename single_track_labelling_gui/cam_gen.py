import sys
from PyQt5.QtCore import *  # QDir, Qt, QUrl
from PyQt5.QtMultimedia import *  # QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import *  # QVideoWidget
from PyQt5.QtWidgets import *  # (QApplication, QFileDialog, QHBoxLayout, QLabel,

# QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5 import uic

import OpenGL.GL as gl  # python wrapping of OpenGL
from OpenGL import GLU  # OpenGL Utility Library, extends OpenGL functionality
import argparse
from pathlib import Path
import os


import cam_gen_ui as tui

# get export path usig argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ExportPKL", help="pkl path for exporting common trajectories", type=str
    )
    parser.add_argument(
        "--ExportCSV", help="pkl path for exporting common trajectories", type=str
    )
    parser.add_argument(
        "--CamImage", help="path to topview intersection image", type=str
    )
    parser.add_argument(
        "--ListClassIdsToConsider",
        type=int,
        nargs="+",
        help="A list of class ids to consider in single track labeling gui",
    )
    parser.add_argument(
        "--LoadExistingAnnotations",
        help="flag to load existing annotations for single track labelling gui",
        action="store_true",
    )

    parser.add_argument("--TracksPath", help="path to clustering pkl", type=str)
    args = parser.parse_args()

    print(args)
    export_path_pkl = args.ExportPKL.strip("'")
    export_path_csv = args.ExportCSV.strip("'")
    cam_image = args.CamImage.strip("'")
    tracks_path = args.TracksPath.strip("'")
    list_class_ids_to_consider = args.ListClassIdsToConsider

    app = QApplication(sys.argv)

    myapp = tui.ui_func(
        export_path_pkl=export_path_pkl,
        export_path_csv=export_path_csv,
        cam_image=cam_image,
        tracks_path=tracks_path,
        list_class_ids_to_consider=list_class_ids_to_consider,
        load_existing_annotations=args.LoadExistingAnnotations,
    )
    # myapp.setupUi(self)
    # myapp.showMaximized()
    myapp.show()
    sys.exit(app.exec_())
