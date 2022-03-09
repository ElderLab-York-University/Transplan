import sys
from PyQt5.QtCore import *#QDir, Qt, QUrl
from PyQt5.QtMultimedia import *#QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import *#QVideoWidget
from PyQt5.QtWidgets import *#(QApplication, QFileDialog, QHBoxLayout, QLabel,
        #QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5 import uic

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality


import cam_gen_ui as tui
app = QApplication(sys.argv)

myapp = tui.ui_func()
#myapp.setupUi(self)
#myapp.showMaximized()
myapp.show()
sys.exit(app.exec_())
