from PyQt5.QtCore import *#QDir, Qt, QUrl
from PyQt5.QtMultimedia import *#QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import *#QVideoWidget
from PyQt5.QtWidgets import *#(QApplication, QFileDialog, QHBoxLayout, QLabel,
        #QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

from QtImageViewer import QtImageViewer
from QGlWidget import QGlWidget
from backend import backend
import csv
import time


class tabbed_ui_func(QMainWindow):
    def __init__(self, image1_path, image2_path, txtfile_path, npyfile_path, csvfile_path):
        super(tabbed_ui_func, self).__init__()
        uic.loadUi('tabbed_ui.ui',self)

        # Tab dohomography 2 images elements
        self.b = backend()
        self.set_result_paths(txtfile_path, npyfile_path, csvfile_path)
        self.set_image_paths(image1_path, image2_path)
        self.pts1 = []
        self.pts2 = []
        self.img1DrawnEllipses = []
        self.img1DrawnTexts = []
        self.img2DrawnEllipses = []
        self.img2DrawnTexts = []

        # 1 Tab Do Homography 2 images
        self.pushButton_DoHomography = self.findChild(QPushButton,name='pushButton_DoHomography')
        self.pushButton_DoHomography.clicked.connect(self.callback_DoHomography)

        self.pushButton_OpenImage1 = self.findChild(QPushButton,name='pushButton_OpenImage1')
        self.pushButton_OpenImage1.clicked.connect(self.callback_OpenImage1)

        self.pushButton_OpenImage2 = self.findChild(QPushButton,name='pushButton_OpenImage2')
        self.pushButton_OpenImage2.clicked.connect(self.callback_OpenImage2)

        self.pushButton_DeleteLastPoint1 = self.findChild(QPushButton,name='pushButton_DeleteLastPoint1')
        self.pushButton_DeleteLastPoint1.clicked.connect(self.callback_DeleteLastPoint1)

        self.pushButton_DeleteLastPoint2 = self.findChild(QPushButton,name='pushButton_DeleteLastPoint2')
        self.pushButton_DeleteLastPoint2.clicked.connect(self.callback_DeleteLastPoint2)

        self.graphicsView_Image1 = self.findChild(QtImageViewer, name= 'graphicsView_Image1')
        self.graphicsView_Image2 = self.findChild(QtImageViewer, name= 'graphicsView_Image2')

    def set_result_paths(self, txtfile_path, npyfile_path, csvfile_path):
        self.txtfile_path = txtfile_path
        self.npyfile_path = npyfile_path
        self.csvfile_path = csvfile_path
        self.b.set_txt_file_path(self.txtfile_path)
        self.b.set_npy_file_path(self.npyfile_path)

    def set_image_paths(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path
    # complete here

    def callback_OpenImage1(self):
        fName = self.graphicsView_Image1.loadImageFromFile()
        self.b.setFileName1(fName)
        self.graphicsView_Image1.leftMouseButtonPressed.connect(self.img1HandleLeftClick)

        self.graphicsView_Image1.show()
        

    def callback_OpenImage2(self):
        fName = self.graphicsView_Image2.loadImageFromFile()
        self.b.setFileName2(fName)
        self.graphicsView_Image2.leftMouseButtonPressed.connect(self.img2HandleLeftClick)
        self.graphicsView_Image2.show()

    def callback_DeleteLastPoint1(self):
        self.graphicsView_Image1.scene.removeItem(self.img1DrawnEllipses[-1])
        self.graphicsView_Image1.scene.removeItem(self.img1DrawnTexts[-1])
        del self.pts1[-1]
        del self.img1DrawnTexts[-1]
        del self.img1DrawnEllipses[-1]

    def callback_DeleteLastPoint2(self):
        self.graphicsView_Image2.scene.removeItem(self.img2DrawnEllipses[-1])
        self.graphicsView_Image2.scene.removeItem(self.img2DrawnTexts[-1])
        del self.pts2[-1]
        del self.img2DrawnTexts[-1]
        del self.img2DrawnEllipses[-1]

    def callback_DoHomography(self):
        # store points to self.csvfile_path
        all_points = []
        for p1, p2 in zip(self.pts1, self.pts2):
            all_points.append([p1[0], p1[1], p2[0], p2[1]])
        fname=self.csvfile_path
        with open(fname, "w") as f:
            write = csv.writer(f)
            write.writerows(all_points)

        # fname = 'Homography_1_'+'.csv'
        # with open(fname, 'w') as f: 
        #     # using csv.writer method from CSV package 
        #     write = csv.writer(f)
        #     write.writerows(self.pts1)
        # fname = 'Homography_2_'+'.csv'
        # with open(fname, 'w') as f: 
        #     # using csv.writer method from CSV package 
        #     write = csv.writer(f)
        #     write.writerows(self.pts2)


        if self.b.doHomography(self.pts1, self.pts2):
            print('Done!')

    def img1HandleLeftClick(self, x, y):
        y = int(y)
        x = int(x)
        self.pts1.append([x,y])
        currEllipse = self.graphicsView_Image1.scene.addEllipse(x-2.5, y-2.5, 5, 5)
        self.img1DrawnEllipses.append(currEllipse)
        lastPt = self.graphicsView_Image1.scene.addText(str(len(self.pts1)), QtGui.QFont('Arial Black', 15, QtGui.QFont.Light))
        lastPt.setPos(x+3, y)
        self.img1DrawnTexts.append(lastPt)

    def img2HandleLeftClick(self, x, y):
        y = int(y)
        x = int(x)
        self.pts2.append([x,y])
        currEllipse = self.graphicsView_Image2.scene.addEllipse(x-2.5, y-2.5, 5, 5)
        self.img2DrawnEllipses.append(currEllipse)
        lastPt = self.graphicsView_Image2.scene.addText(str(len(self.pts2)), QtGui.QFont('Arial Black', 15, QtGui.QFont.Light))
        lastPt.setPos(x+3, y)
        self.img2DrawnTexts.append(lastPt)
