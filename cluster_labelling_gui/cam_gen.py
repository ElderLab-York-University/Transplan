import sys
from PyQt5.QtCore import *#QDir, Qt, QUrl
from PyQt5.QtMultimedia import *#QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import *#QVideoWidget
from PyQt5.QtWidgets import *#(QApplication, QFileDialog, QHBoxLayout, QLabel,
        #QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5 import uic

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
import argparse


import cam_gen_ui as tui

# get export path usig argparse
if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--Export", help="pkl path for exporting common trajectories", type=str)
        parser.add_argument("--TopView", help="path to topview intersection image", type=str)
        parser.add_argument("--ClustersPath", help="path to clustering pkl", type=str)
        args = parser.parse_args()

        app = QApplication(sys.argv)

        myapp = tui.ui_func(args.Export, args.TopView, args.ClustersPath)
        #myapp.setupUi(self)
        #myapp.showMaximized()
        myapp.show()
        sys.exit(app.exec_())
