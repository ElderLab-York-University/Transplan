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


import tabbed_ui_func as tui

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--StreetView", help="path to streetview intersection image", type=str)
        parser.add_argument("--TopView", help="path to topview intersection image", type=str)
        parser.add_argument("--Txt", help="path to topview intersection image", type=str)
        parser.add_argument("--Npy", help="path to topview intersection image", type=str)
        parser.add_argument("--Csv", help="path to topview intersection image", type=str)
        args = parser.parse_args()

        app = QApplication(sys.argv)

        myapp = tui.tabbed_ui_func(args.StreetView, args.TopView, args.Txt, args.Npy, args.Csv)
        #myapp.setupUi(self)
        myapp.showMaximized()
        sys.exit(app.exec_())
