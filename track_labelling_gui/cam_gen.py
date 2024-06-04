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


import cam_gen_ui as tui

# get export path usig argparse
app = QApplication(sys.argv)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--Export", help="pkl path for exporting common trajectories", type=str
)
args = parser.parse_args()

myapp = tui.ui_func(args.Export)
# myapp.setupUi(self)
# myapp.showMaximized()
myapp.show()
sys.exit(app.exec_())
