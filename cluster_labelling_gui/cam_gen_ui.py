from collections import defaultdict

from PyQt5.QtCore import *  # QDir, Qt, QUrl
from PyQt5.QtMultimedia import *  # QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import *  # QVideoWidget
from PyQt5.QtWidgets import *  # (QApplication, QFileDialog, QHBoxLayout, QLabel,

# QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtOpenGL

# from PyQt5 import QtWebEngineWidgets
import matplotlib
from matplotlib import cm

norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
import pandas as pd
import numpy as np
import cv2
import csv
import os
import scipy.io
from pymatreader import read_mat
from tqdm import tqdm
from pathlib import Path


# from counting import Counting

MIN_PTS_IN_TRACK = 80


def df_from_pickle(pickle_path):
    return pd.read_pickle(pickle_path)


# tracks_path = "/home/savoji/Desktop/TransPlan Project/Results/GX010069_tracking_sort_reprojected.txt"
# def group_tracks_by_id(df):
#     # this function was writtern for grouping the tracks with the same id
#     # usinig this one can load the data from a .txt file rather than .mat file
#     all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
#     data = {"id":[], "trajectory":[], "frames":[]}
#     for idd in tqdm(all_ids):
#         frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
#         id = idd
#         trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)

#         data["id"].append(id)
#         data["frames"].append(frames)
#         data["trajectory"].append(trajectory)
#     df2 = pd.DataFrame(data)
#     return df2

cam_mois = [4, 4, 4, 12, 12, 12, 12, 6, 12, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]


class ui_func(QMainWindow):
    def __init__(self, export_path, topview, clusters_path):

        # topview = r"C:\\NonOneDrive\\ElderLab\\data\\train\\Seg01\\Seg01sc1\\seg_01_sc1_b3s_anon_und.homography.top.GoogleMap.png"
        # export_path = r""
        # clusters_path = r""

        super(ui_func, self).__init__()
        # should be a pickle path
        self.export_path = export_path
        self.topview = topview
        self.clusters_path = clusters_path

        uic.loadUi("cam_gen.ui", self)
        # self.Cnt = Counting()

        self.pushButton_openimg = self.findChild(QPushButton, name="pushButton_openimg")
        self.pushButton_openimg.clicked.connect(self.c_pushButton_openimg)

        self.scaleFactor = 1

        self.count_1 = 0
        self.count_2 = 0
        self.count_3 = 0
        self.count_4 = 0
        self.count_5 = 0
        self.count_6 = 0
        self.count_7 = 0
        self.count_8 = 0
        self.count_9 = 0
        self.count_10 = 0
        self.count_11 = 0
        self.count_12 = 0

        self.typical_gt = defaultdict(list)
        self.tracks_gt = {"1": [], "2": [], "3": [], "4": []}
        self.pushButtons = []
        self.countlabels = []
        self.pushButton_1 = self.findChild(QPushButton, name="pushButton_1")
        self.pushButton_1.clicked.connect(self.c_pushButton_1)
        self.pushButtons.append(self.pushButton_1)

        self.pushButton_2 = self.findChild(QPushButton, name="pushButton_2")
        self.pushButton_2.clicked.connect(self.c_pushButton_2)
        self.pushButtons.append(self.pushButton_2)

        self.pushButton_3 = self.findChild(QPushButton, name="pushButton_3")
        self.pushButton_3.clicked.connect(self.c_pushButton_3)
        self.pushButtons.append(self.pushButton_3)

        self.pushButton_4 = self.findChild(QPushButton, name="pushButton_4")
        self.pushButton_4.clicked.connect(self.c_pushButton_4)
        self.pushButtons.append(self.pushButton_4)

        self.pushButton_5 = self.findChild(QPushButton, name="pushButton_5")
        self.pushButton_5.clicked.connect(self.c_pushButton_5)
        self.pushButtons.append(self.pushButton_5)

        self.pushButton_6 = self.findChild(QPushButton, name="pushButton_6")
        self.pushButton_6.clicked.connect(self.c_pushButton_6)
        self.pushButtons.append(self.pushButton_6)

        self.pushButton_7 = self.findChild(QPushButton, name="pushButton_7")
        self.pushButton_7.clicked.connect(self.c_pushButton_7)
        self.pushButtons.append(self.pushButton_7)

        self.pushButton_8 = self.findChild(QPushButton, name="pushButton_8")
        self.pushButton_8.clicked.connect(self.c_pushButton_8)
        self.pushButtons.append(self.pushButton_8)

        self.pushButton_9 = self.findChild(QPushButton, name="pushButton_9")
        self.pushButton_9.clicked.connect(self.c_pushButton_9)
        self.pushButtons.append(self.pushButton_9)

        self.pushButton_10 = self.findChild(QPushButton, name="pushButton_10")
        self.pushButton_10.clicked.connect(self.c_pushButton_10)
        self.pushButtons.append(self.pushButton_10)

        self.pushButton_11 = self.findChild(QPushButton, name="pushButton_11")
        self.pushButton_11.clicked.connect(self.c_pushButton_11)
        self.pushButtons.append(self.pushButton_11)

        self.pushButton_12 = self.findChild(QPushButton, name="pushButton_12")
        self.pushButton_12.clicked.connect(self.c_pushButton_12)
        self.pushButtons.append(self.pushButton_12)

        self.pushButton_opentrk = self.findChild(QPushButton, name="pushButton_opentrk")
        self.pushButton_opentrk.clicked.connect(self.c_pushButton_opentrk)

        self.pushButton_export = self.findChild(QPushButton, name="pushButton_export")
        self.pushButton_export.clicked.connect(self.c_pushButton_export)

        self.pushButton_skip = self.findChild(QPushButton, name="pushButton_skip")
        self.pushButton_skip.clicked.connect(self.c_pushButton_skip)

        self.pushButton_undo = self.findChild(QPushButton, name="pushButton_undo")
        self.pushButton_undo.clicked.connect(self.delete_last_trajectory)
        # self.pushButton_undo.clicked.connect(self.plot_typical)

        self.pushButton_next = self.findChild(QPushButton, name="pushButton_next")
        self.pushButton_next.clicked.connect(self.c_pushButton_next)

        self.all_labels = []
        self.label_1 = self.findChild(QLabel, name="count_1")
        self.all_labels.append(self.label_1)
        self.label_2 = self.findChild(QLabel, name="count_2")
        self.all_labels.append(self.label_2)
        self.label_3 = self.findChild(QLabel, name="count_3")
        self.all_labels.append(self.label_3)
        self.label_4 = self.findChild(QLabel, name="count_4")
        self.all_labels.append(self.label_4)
        self.label_5 = self.findChild(QLabel, name="count_5")
        self.all_labels.append(self.label_5)
        self.label_6 = self.findChild(QLabel, name="count_6")
        self.all_labels.append(self.label_6)
        self.label_7 = self.findChild(QLabel, name="count_7")
        self.all_labels.append(self.label_7)
        self.label_8 = self.findChild(QLabel, name="count_8")
        self.all_labels.append(self.label_8)
        self.label_9 = self.findChild(QLabel, name="count_9")
        self.all_labels.append(self.label_9)
        self.label_10 = self.findChild(QLabel, name="count_10")
        self.all_labels.append(self.label_10)
        self.label_11 = self.findChild(QLabel, name="count_11")
        self.all_labels.append(self.label_11)
        self.label_12 = self.findChild(QLabel, name="count_12")
        self.all_labels.append(self.label_12)

        self.tracks = {
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
            "11": [],
            "12": [],
        }
        self.tracks_df = []
        for i in range(12):
            self.pushButtons[i].setEnabled(False)
            self.pushButtons[i].setHidden(True)
            self.all_labels[i].setHidden(True)
        # self.plot_next_track()

    def calculate_distance(self, current_trac):
        dis = self.Cnt.calculate_two_trajactory(current_trac, self.typical_gt)
        return dis

    # def select_next(self):
    #     self.current_id = self.tids[self.id_index]
    #     self.current_track = self.df[self.df['id'] == self.current_id]
    #     if self.id_index < len(self.tids) - 1:

    #         while len(list(self.current_track['trajectory'])[0]) < 100:
    #             self.id_index = self.id_index + 1
    #             self.current_id = self.tids[self.id_index]
    #             self.current_track = self.df[self.df['id'] == self.current_id]

    #         current_trac = self.current_track['trajectory'].values.tolist()[0]
    #         # dis = self.calculate_distance(current_trac)
    #         # print(dis)
    #         #
    #         # while min(dis) < 50:
    #         #     print("skipped")
    #         #     self.id_index = self.id_index + 1
    #         #     self.current_id = self.tids[self.id_index]
    #         #     self.current_track = self.df[self.df['id'] == self.current_id]
    #         #     while len(list(self.current_track['trajectory'])[0]) < 80:
    #         #         self.id_index = self.id_index + 1
    #         #         self.current_id = self.tids[self.id_index]
    #         #         self.current_track = self.df[self.df['id'] == self.current_id]
    #         #     current_trac = self.current_track['trajectory'].values.tolist()[0]
    #         #     dis = self.calculate_distance(current_trac)
    #         #     print(dis)

    #         if len(list(self.current_track['trajectory'])[0]) > 100:
    #             print(self.current_id)
    #             self.temp_img = self.img.copy()
    #             colormap = cm.get_cmap('rainbow', len(self.current_track))
    #             newcolors = colormap(np.linspace(0, 1, len(list(self.current_track['trajectory'])[0])))
    #             index = 0
    #             lastpt = []
    #             firstpt = []
    #             prevpt = []

    #             # print(list(self.current_track['trajectory'])[0])
    #             for x, y in list(self.current_track['trajectory'])[0]:
    #                 pt1 = int(x)
    #                 pt2 = int(y)
    #                 pt = [pt1, pt2]
    #                 lastpt = pt
    #                 rgba_color = newcolors[index]
    #                 cv2.circle(self.temp_img, (int(pt[0]), int(pt[1])), 3,
    #                            (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)), -1)
    #                 if index == 0:
    #                     firstpt = pt
    #                 if index > 0:
    #                     cv2.line(self.temp_img, (int(pt[0]), int(pt[1])), (int(prevpt[0]), int(prevpt[1])), 5)
    #                 index = index + 1
    #                 prevpt = pt
    #             cv2.circle(self.temp_img, (int(firstpt[0]), int(firstpt[1])), 5, (0, 255, 0), -1)
    #             cv2.circle(self.temp_img, (int(lastpt[0]), int(lastpt[1])), 5, (0, 0, 255), -1)
    #             bytesPerLine = 3 * self.temp_img.shape[1]
    #             self.image = QtGui.QImage(self.temp_img.data, self.temp_img.shape[1], self.temp_img.shape[0],
    #                                       bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
    #             self.image_frame = self.findChild(QLabel, name='label_img')
    #             pix = QtGui.QPixmap.fromImage(self.image)
    #             self.image_frame.setPixmap(
    #                 pix.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    #         self.id_index = self.id_index + 1
    #         self.current_id = self.tids[self.id_index]
    #         self.current_track = self.df[self.df['id'] == self.current_id]

    def c_pushButton_openimg(self):
        for i in range(12):
            self.pushButtons[i].setEnabled(False)
            self.pushButtons[i].setHidden(True)
            self.all_labels[i].setHidden(True)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = self.topview
        # fileName = "C:\\NonOneDrive\\ElderLab\\data\\train\\Seg01\\Seg01sc1\\seg_01_sc1_b3s_anon_und.homography.top.GoogleMap.png"
        # print(os.path.isfile("C:\\NonOneDrive\\ElderLab\\data\\train\\Seg01\\Seg01sc1\\seg_01_sc1_b3s_anon_und.homography.top.GoogleMap.png"))
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)

        self.nummoi = 12

        for i in range(12):
            self.pushButtons[i].setEnabled(True)
            self.pushButtons[i].setHidden(False)
            self.all_labels[i].setHidden(False)

        self.img = cv2.imread(fileName)

        [h, w, c] = self.img.shape
        if w > 1500:  # 1280
            self.scaleFactor = 2
        else:
            self.scaleFactor = 1
        height, width, byteValue = self.img.shape
        byteValue = byteValue / self.scaleFactor * width

        self.img = cv2.resize(
            self.img,
            (
                int(self.img.shape[1] / self.scaleFactor),
                int(self.img.shape[0] / self.scaleFactor),
            ),
        )
        bytesPerLine = 3 * self.img.shape[1]
        self.image = QtGui.QImage(
            self.img.data,
            self.img.shape[1],
            self.img.shape[0],
            bytesPerLine,
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()
        # self.image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame = self.findChild(QLabel, name="label_img")
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

        # self.temp_img = self.img.copy()
        # self.img = cv2.resize(self.img,(int(self.img.shape[1]/2),int(self.img.shape[0]/2)))

    def c_pushButton_opentrk(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        fileName = self.clusters_path
        self.tracks_file = fileName  #'D:/projects/aicitychallenge/data/AIC21_Track1_Vehicle_Counting/AIC21_Track1_Vehicle_Counting/screen_shot_with_roi_and_movement/cam_5_3000.txt'
        # data = read_mat(self.tracks_file)
        # # self.df = pd.DataFrame(data['recorded_tracks'])
        # self.df1 = pd.DataFrame(data['recorded_tracks'])
        # self.df = self.df1[['id', 'trajectory']]
        # self.df.columns = ['id', 'trajectory']
        # # self.df = pd.read_csv(self.tracks_file,names=['f','id','x','y','w','h','e1','e2','e3'])
        self.df = df_from_pickle(self.tracks_file)
        self.df.rename(columns={"class": "cid"}, inplace=True)

        print(self.df)
        # self.tids = np.unique(self.df['id'].tolist())
        self.tids = None
        self.cids = np.unique(self.df["cid"].tolist())

        self.id_index = 0
        self.cid_index = 0
        self.cluster_mode = True

        self.current_track = None  # use this for geting a specific track
        self.df_c = None  # user this df for geting a specific cluster
        self.plot_next_cluster()

        # self.plot_next_track()

    def plot_next_cluster(self):
        if not self.cluster_mode:
            raise "something went wrong! plot_next_cluster() should be only called with self.cluster_mode=True"
        if self.cid_index >= len(self.cids):
            # prompt user that al clusters were shown to them
            QMessageBox.information(
                self, "No more data", "you have already labelled all the clusters!"
            )
        self.id_index = 0
        self.df_c = self.df[self.df["cid"] == self.cids[self.cid_index]]
        self.tids = np.unique(self.df_c["id"].to_list())

        temp_image = self.img.copy()
        for temp_tid in self.tids:
            current_track = self.df_c[self.df_c["id"] == temp_tid]
            temp_image = self.draw_track_on_image(temp_image, current_track)
        # for i, current_track in self.df_c.iterrows():
        #     print(type(current_track))
        #     print(current_track)

        self.show_image_on_GUI(temp_image)
        self.cid_index += 1

    def draw_track_on_image(self, image, current_track):
        # gets and image and draws

        temp_img = image.copy()
        colormap = cm.get_cmap("rainbow", len(current_track))
        newcolors = colormap(
            np.linspace(0, 1, len(list(current_track["trajectory"])[0]))
        )
        index = 0
        lastpt = []
        firstpt = []
        prevpt = []
        # if len(self.current_track['trajectory'])>15:
        # print(list(current_track['trajectory'])[0])
        for x, y in list(current_track["trajectory"])[0]:
            # pt1 = int(row['x']+row['w']/2)
            # pt2 = int(row['y']+row['h']/2)
            # pt = [int(pt1/self.scaleFactor), int(pt2/self.scaleFactor)]
            pt1 = int(x)
            pt2 = int(y)
            pt = [pt1, pt2]
            lastpt = pt
            rgba_color = newcolors[index]  # cm.rainbow(norm(i),bytes=True)[0:3]
            cv2.circle(
                temp_img,
                (int(pt[0]), int(pt[1])),
                3,
                (
                    int(rgba_color[0] * 255),
                    int(rgba_color[1] * 255),
                    int(rgba_color[2] * 255),
                ),
                -1,
            )
            if index == 0:
                firstpt = pt
            if index > 0:
                cv2.line(
                    temp_img,
                    (int(pt[0]), int(pt[1])),
                    (int(prevpt[0]), int(prevpt[1])),
                    5,
                )
            index = index + 1
            prevpt = pt
        cv2.circle(temp_img, (int(firstpt[0]), int(firstpt[1])), 5, (0, 255, 0), -1)
        cv2.circle(temp_img, (int(lastpt[0]), int(lastpt[1])), 5, (0, 0, 255), -1)
        return temp_img

    def show_image_on_GUI(self, temp_img):
        # actually plots the image into the GUI
        bytesPerLine = 3 * temp_img.shape[1]

        self.image = QtGui.QImage(
            temp_img.data,
            temp_img.shape[1],
            temp_img.shape[0],
            bytesPerLine,
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()
        self.image_frame = self.findChild(QLabel, name="label_img")
        pix = QtGui.QPixmap.fromImage(self.image)
        self.image_frame.setPixmap(
            pix.scaled(
                self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def plot_next_track(self):
        if self.cluster_mode:
            raise "Something is wrong. plot_next_track should only be called when cluster_mode is False"
        if self.id_index >= len(self.tids):
            print(
                "already ploted all the trajectories in the current cluster! moved to the next cluster"
            )
            self.cluster_mode = True
            self.plot_next_cluster()
            return

        self.current_id = self.tids[self.id_index]
        self.current_track = self.df_c[self.df_c["id"] == self.current_id]
        # print(self.current_track['trajectory'])

        while (
            self.id_index < len(self.tids) - 1
            and len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK
        ):
            # print(self.current_id)
            # print(len(list(self.current_track['trajectory'])[0]))
            self.id_index = self.id_index + 1
            self.current_id = self.tids[self.id_index]
            self.current_track = self.df_c[self.df_c["id"] == self.current_id]
        # print(self.current_id)
        # print(len(list(self.current_track['trajectory'])[0]))

        if self.id_index < len(self.tids):
            # print(self.current_id)
            self.temp_img = self.draw_track_on_image(self.img, self.current_track)

            ######################################
            # self.temp_img = self.img.copy()
            # colormap = cm.get_cmap('rainbow',len(self.current_track))
            # newcolors = colormap(np.linspace(0, 1, len(list(self.current_track['trajectory'])[0])))
            # index = 0
            # lastpt = []
            # firstpt = []
            # prevpt = []
            # # if len(self.current_track['trajectory'])>15:
            # print(list(self.current_track['trajectory'])[0])
            # for x, y in list(self.current_track['trajectory'])[0]:
            #     # pt1 = int(row['x']+row['w']/2)
            #     # pt2 = int(row['y']+row['h']/2)
            #     # pt = [int(pt1/self.scaleFactor), int(pt2/self.scaleFactor)]
            #     pt1 = int(x)
            #     pt2 = int(y)
            #     pt = [pt1, pt2]
            #     lastpt = pt
            #     rgba_color = newcolors[index]#cm.rainbow(norm(i),bytes=True)[0:3]
            #     cv2.circle(self.temp_img, (int(pt[0]), int(pt[1])), 3, (int(rgba_color[0]*255),int(rgba_color[1]*255),int(rgba_color[2]*255)), -1)
            #     if index == 0:
            #         firstpt = pt
            #     if index > 0:
            #         cv2.line(self.temp_img, (int(pt[0]), int(pt[1])), (int(prevpt[0]), int(prevpt[1])), 5)
            #     index = index + 1
            #     prevpt = pt
            # cv2.circle(self.temp_img, (int(firstpt[0]), int(firstpt[1])), 5, (0, 255, 0), -1)
            # cv2.circle(self.temp_img, (int(lastpt[0]), int(lastpt[1])), 5, (0, 0, 255), -1)
            ################################

            self.show_image_on_GUI(self.temp_img)
            #############################
            # bytesPerLine = 3 * self.temp_img.shape[1]
            # self.image = QtGui.QImage(self.temp_img.data, self.temp_img.shape[1], self.temp_img.shape[0], bytesPerLine,QtGui.QImage.Format_RGB888).rgbSwapped()
            # self.image_frame = self.findChild(QLabel,name='label_img')
            # pix = QtGui.QPixmap.fromImage(self.image)
            # self.image_frame.setPixmap(pix.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            #############################

            self.id_index = self.id_index + 1
        # else:
        #     fname = self.tracks_file.replace(".txt", ".csv")
        #     writedf = pd.concat(self.tracks_df)
        #     writedf.to_csv(fname, sep=',')
        #     QMessageBox.information(self, 'Last trajectory saved', 'Exported to ' + fname)

    # def plot_typical(self):

    #     self.current_id = self.tids[self.id_index]
    #     self.current_track = self.df[self.df['id'] == self.current_id]
    #     lst = [47, 132, 240, 405, 474, 542, 547, 639, 1445, 1818, 1936, 2247, 2506, 3094]
    #     while self.id_index < len(self.tids) - 1:
    #         if len(list(self.current_track['trajectory'])[0]) > 80 and self.current_id < 3100 and self.current_id in lst:
    #             print(self.current_id)
    #             # self.temp_img = self.img.copy()
    #             colormap = cm.get_cmap('rainbow', len(self.current_track))
    #             newcolors = colormap(np.linspace(0, 1, len(list(self.current_track['trajectory'])[0])))
    #             index = 0
    #             lastpt = []
    #             firstpt = []
    #             prevpt = []

    #             for x, y in list(self.current_track['trajectory'])[0]:
    #                 pt1 = int(x)
    #                 pt2 = int(y)
    #                 pt = [pt1, pt2]
    #                 lastpt = pt
    #                 rgba_color = newcolors[index]
    #                 cv2.circle(self.temp_img, (int(pt[0]), int(pt[1])), 3,
    #                            (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)), -1)
    #                 if index == 0:
    #                     firstpt = pt
    #                 if index > 0:
    #                     cv2.line(self.temp_img, (int(pt[0]), int(pt[1])), (int(prevpt[0]), int(prevpt[1])), 5)
    #                 index = index + 1
    #                 prevpt = pt
    #             cv2.circle(self.temp_img, (int(firstpt[0]), int(firstpt[1])), 5, (0, 255, 0), -1)
    #             cv2.circle(self.temp_img, (int(lastpt[0]), int(lastpt[1])), 5, (0, 0, 255), -1)
    #             bytesPerLine = 3 * self.temp_img.shape[1]
    #             self.image = QtGui.QImage(self.temp_img.data, self.temp_img.shape[1], self.temp_img.shape[0],
    #                                       bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
    #             self.image_frame = self.findChild(QLabel, name='label_img')
    #             pix = QtGui.QPixmap.fromImage(self.image)
    #             self.image_frame.setPixmap(
    #                 pix.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    #         self.id_index = self.id_index + 1
    #         self.current_id = self.tids[self.id_index]
    #         self.current_track = self.df[self.df['id'] == self.current_id]

    def track_resample(self, track, threshold=5):
        """
        :param track: input track numpy array (M, 2)
        :param threshold: default 20 pixel interval for neighbouring points
        :return:
        """
        assert track.shape[1] == 2

        accum_dist = 0
        index_keep = [0]
        for i in range(1, track.shape[0]):
            dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
            # dist pixel == 1
            if dist_ >= 1:
                accum_dist += dist_
                if accum_dist >= threshold:
                    index_keep.append(i)
                    accum_dist = 0
        # print(track[index_keep, :].shape[0])
        if track[index_keep, :].shape[0] < 400:
            return track[index_keep, :]
        else:
            threshold += 3
            return self.track_resample(track, threshold)

    def c_pushButton_1(self):
        if not self.cluster_mode:
            self.count_1 = self.count_1 + 1
            self.current_track["moi"] = 1
            self.tracks_df.append(self.current_track)
            self.label_1.setText(str(self.count_1))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_1 = self.count_1 + 1
                self.current_track["moi"] = 1
                self.tracks_df.append(self.current_track)
            self.label_1.setText(str(self.count_1))
            self.plot_next_cluster()
        # self.count_1 = self.count_1 + 1
        # self.label_1.setText(str(self.count_1))
        # # self.current_track.loc[0, ['moi']] = 1
        # self.current_track['moi'] = 1
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['1'].append(self.current_track)
        # self.typical_gt['1'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_2(self):
        if not self.cluster_mode:
            self.count_2 = self.count_2 + 1
            self.current_track["moi"] = 2
            self.tracks_df.append(self.current_track)
            self.label_2.setText(str(self.count_2))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_2 = self.count_2 + 1
                self.current_track["moi"] = 2
                self.tracks_df.append(self.current_track)
            self.label_2.setText(str(self.count_2))
            self.plot_next_cluster()
        # self.count_2 = self.count_2 + 1
        # self.label_2.setText(str(self.count_2))
        # self.current_track['moi'] = 2
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['2'].append(self.current_track)
        # self.typical_gt['2'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_3(self):
        if not self.cluster_mode:
            self.count_3 = self.count_3 + 1
            self.current_track["moi"] = 3
            self.tracks_df.append(self.current_track)
            self.label_3.setText(str(self.count_3))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_3 = self.count_3 + 1
                self.current_track["moi"] = 3
                self.tracks_df.append(self.current_track)
            self.label_3.setText(str(self.count_3))
            self.plot_next_cluster()
        # self.count_3 = self.count_3 + 1
        # self.label_3.setText(str(self.count_3))
        # self.current_track['moi'] = 3
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['3'].append(self.current_track)
        # self.typical_gt['3'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_4(self):
        if not self.cluster_mode:
            self.count_4 = self.count_4 + 1
            self.current_track["moi"] = 4
            self.tracks_df.append(self.current_track)
            self.label_4.setText(str(self.count_4))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_4 = self.count_4 + 1
                self.current_track["moi"] = 4
                self.tracks_df.append(self.current_track)
            self.label_4.setText(str(self.count_4))
            self.plot_next_cluster()
        # self.count_4 = self.count_4 + 1
        # self.label_4.setText(str(self.count_4))
        # self.current_track['moi'] = 4
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['4'].append(self.current_track)
        # self.typical_gt['4'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_5(self):
        if not self.cluster_mode:
            self.count_5 = self.count_5 + 1
            self.current_track["moi"] = 5
            self.tracks_df.append(self.current_track)
            self.label_5.setText(str(self.count_5))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_5 = self.count_5 + 1
                self.current_track["moi"] = 5
                self.tracks_df.append(self.current_track)
            self.label_5.setText(str(self.count_5))
            self.plot_next_cluster()
        # self.count_5 = self.count_5 + 1
        # self.label_5.setText(str(self.count_5))
        # self.current_track['moi'] = 5
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['5'].append(self.current_track)
        # self.typical_gt['5'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_6(self):
        if not self.cluster_mode:
            self.count_6 = self.count_6 + 1
            self.current_track["moi"] = 6
            self.tracks_df.append(self.current_track)
            self.label_6.setText(str(self.count_6))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_6 = self.count_6 + 1
                self.current_track["moi"] = 6
                self.tracks_df.append(self.current_track)
            self.label_6.setText(str(self.count_6))
            self.plot_next_cluster()
        # self.count_6 = self.count_6 + 1
        # self.label_6.setText(str(self.count_6))
        # self.current_track['moi'] = 6
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['6'].append(self.current_track)
        # self.typical_gt['6'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_7(self):
        if not self.cluster_mode:
            self.count_7 = self.count_7 + 1
            self.current_track["moi"] = 7
            self.tracks_df.append(self.current_track)
            self.label_7.setText(str(self.count_7))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_7 = self.count_7 + 1
                self.current_track["moi"] = 7
                self.tracks_df.append(self.current_track)
            self.label_7.setText(str(self.count_7))
            self.plot_next_cluster()
        # self.count_7 = self.count_7 + 1
        # self.label_7.setText(str(self.count_7))
        # self.current_track['moi'] = 7
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['7'].append(self.current_track)
        # self.typical_gt['7'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_8(self):
        if not self.cluster_mode:
            self.count_8 = self.count_8 + 1
            self.current_track["moi"] = 8
            self.tracks_df.append(self.current_track)
            self.label_8.setText(str(self.count_8))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_8 = self.count_8 + 1
                self.current_track["moi"] = 8
                self.tracks_df.append(self.current_track)
            self.label_8.setText(str(self.count_8))
            self.plot_next_cluster()
        # self.count_8 = self.count_8 + 1
        # self.label_8.setText(str(self.count_8))
        # self.current_track['moi'] = 8
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['8'].append(self.current_track)
        # self.typical_gt['8'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_9(self):
        if not self.cluster_mode:
            self.count_9 = self.count_9 + 1
            self.current_track["moi"] = 9
            self.tracks_df.append(self.current_track)
            self.label_9.setText(str(self.count_9))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_9 = self.count_9 + 1
                self.current_track["moi"] = 9
                self.tracks_df.append(self.current_track)
            self.label_9.setText(str(self.count_9))
            self.plot_next_cluster()
        # self.count_9 = self.count_9 + 1
        # self.label_9.setText(str(self.count_9))
        # self.current_track['moi'] = 9
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['9'].append(self.current_track)
        # self.typical_gt['9'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_10(self):
        if not self.cluster_mode:
            self.count_10 = self.count_10 + 1
            self.current_track["moi"] = 10
            self.tracks_df.append(self.current_track)
            self.label_10.setText(str(self.count_10))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_10 = self.count_10 + 1
                self.current_track["moi"] = 10
                self.tracks_df.append(self.current_track)
            self.label_10.setText(str(self.count_10))
            self.plot_next_cluster()
        # self.count_10 = self.count_10 + 1
        # self.label_10.setText(str(self.count_10))
        # self.current_track['moi'] = 10
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['10'].append(self.current_track)
        # self.typical_gt['10'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_11(self):
        if not self.cluster_mode:
            self.count_11 = self.count_11 + 1
            self.current_track["moi"] = 11
            self.tracks_df.append(self.current_track)
            self.label_11.setText(str(self.count_11))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_11 = self.count_11 + 1
                self.current_track["moi"] = 11
                self.tracks_df.append(self.current_track)
            self.label_11.setText(str(self.count_11))
            self.plot_next_cluster()
        # self.count_11 = self.count_11 + 1
        # self.label_11.setText(str(self.count_11))
        # self.current_track['moi'] = 11
        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # self.tracks_df.append(self.current_track)
        # self.tracks['11'].append(self.current_track)
        # self.typical_gt['11'].append(list(self.current_track['trajectory'])[0])
        # self.plot_next_track()

    def c_pushButton_12(self):
        if not self.cluster_mode:
            self.count_12 = self.count_12 + 1
            self.current_track["moi"] = 12
            self.tracks_df.append(self.current_track)
            self.label_12.setText(str(self.count_12))
            self.plot_next_track()

        else:
            for temp_id in self.tids:
                self.current_track = self.df_c[self.df_c["id"] == temp_id]
                if len(list(self.current_track["trajectory"])[0]) < MIN_PTS_IN_TRACK:
                    continue
                self.count_12 = self.count_12 + 1
                self.current_track["moi"] = 12
                self.tracks_df.append(self.current_track)
            self.label_12.setText(str(self.count_12))
            self.plot_next_cluster()

        # a = self.current_track['trajectory'].values
        # a = self.track_resample(a[0])
        # self.current_track['trajectory'] = [a]
        # append cur_row to dataframe

        # self.tracks['12'].append(self.current_track)
        # self.typical_gt['12'].append(list(self.current_track['trajectory'])[0])

    def delete_last_trajectory(self):
        while len(self.tracks_df) > 1:
            # self.tracks_df.drop(self.tracks_df.tail(1).index, inplace=True)
            self.tracks_df = self.tracks_df[:-1]
        self.id_index = max(0, self.id_index - 1)
        self.plot_next_track()

    def c_pushButton_export(self):
        # fname = self.tracks_file.replace(".txt",".csv")
        fname = self.export_path
        print(self.tracks_df)
        if len(self.tracks_df) > 0:
            writedf = pd.concat(self.tracks_df)
            writedf.to_pickle(fname)
            # with open(fname, 'w') as f:
            #     # using csv.writer method from CSV package
            #     write = csv.writer(f)
            #     write.writerows(self.tracks)
            QMessageBox.information(self, "File saved", "Exported to " + fname)

    def c_pushButton_skip(self):
        self.cluster_mode = True
        self.plot_next_cluster()

    def c_pushButton_next(self):
        self.cluster_mode = False
        self.plot_next_track()
