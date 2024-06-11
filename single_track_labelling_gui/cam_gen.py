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
        "--Export", help="pkl path for exporting common trajectories", type=str
    )
    parser.add_argument(
        "--CamImage", help="path to topview intersection image", type=str
    )
    parser.add_argument("--TracksPath", help="path to clustering pkl", type=str)
    args = parser.parse_args()

    print(args)
    export_path = args.Export.strip("'")
    cam_image = args.CamImage.strip("'")
    tracks_path = args.TracksPath.strip("'")
    #     export_path = os.path.abspath(args.Export)
    #     topview = os.path.abspath(args.TopView)
    #     clusterspath = os.path.abspath(args.ClustersPath)

    app = QApplication(sys.argv)

    myapp = tui.ui_func(export_path, cam_image, tracks_path)
    # myapp.setupUi(self)
    # myapp.showMaximized()
    myapp.show()
    sys.exit(app.exec_())
