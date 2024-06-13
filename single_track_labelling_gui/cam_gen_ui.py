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


cam_mois = [4, 4, 4, 12, 12, 12, 12, 6, 12, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]


class ui_func(QMainWindow):
    def __init__(self, export_path_pkl, export_path_csv, cam_image, tracks_path):
        super(ui_func, self).__init__()
        # should be a pickle path
        self.export_path_pkl = export_path_pkl
        self.export_path_csv = export_path_csv
        self.cam_image = cam_image
        self.tracks_path = tracks_path

        uic.loadUi("cam_gen.ui", self)
        # self.Cnt = Counting()

        self.pushButton_openimg = self.findChild(QPushButton, name="pushButton_openimg")
        self.pushButton_openimg.clicked.connect(self.c_pushButton_openimg)

        self.scaleFactor = 1

        self.counts = [0] * 13

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

        self.pushButton_skip = self.findChild(QPushButton, name="pushButton_skip")
        self.pushButton_skip.clicked.connect(self.c_pushButton_skip)
        self.pushButtons.append(self.pushButton_skip)

        self.pushButton_opentrk = self.findChild(QPushButton, name="pushButton_opentrk")
        self.pushButton_opentrk.clicked.connect(self.c_pushButton_opentrk)

        self.pushButton_export = self.findChild(QPushButton, name="pushButton_export")
        self.pushButton_export.clicked.connect(self.c_pushButton_export)

        # self.pushButton_skip = self.findChild(QPushButton, name="pushButton_skip")
        # self.pushButton_skip.clicked.connect(self.c_pushButton_skip)

        # self.pushButton_undo = self.findChild(QPushButton, name="pushButton_undo")
        # self.pushButton_undo.clicked.connect(self.delete_last_trajectory)
        self.pushButton_prev = self.findChild(QPushButton, name="pushButton_prev")
        self.pushButton_prev.clicked.connect(self.c_pushButton_prev)
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
        self.label_skip = self.findChild(QLabel, name="count_skip")
        self.all_labels.append(self.label_skip)

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
        self.tracks_df = pd.DataFrame(
            columns=["fn", "id", "x1", "y1", "x2", "y2", "class", "moi"]
        )
        for i in range(13):
            self.pushButtons[i].setEnabled(False)
            self.pushButtons[i].setHidden(True)
            self.all_labels[i].setHidden(True)
        # self.plot_next_track()

    def calculate_distance(self, current_trac):
        dis = self.Cnt.calculate_two_trajactory(current_trac, self.typical_gt)
        return dis

    def c_pushButton_openimg(self):
        for i in range(13):
            self.pushButtons[i].setEnabled(False)
            self.pushButtons[i].setHidden(True)
            self.all_labels[i].setHidden(True)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName = self.cam_image

        self.nummoi = 12

        for i in range(13):
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

        fileName = self.tracks_path
        self.tracks_file = fileName
        self.df = df_from_pickle(self.tracks_file)

        self.tids = np.unique(self.df["id"].tolist())

        self.id_index = 0

        self.current_track = None

        self.plot_next_track()

    def update_moi_button(self, current_id):
        for i in range(13):
            self.pushButtons[i].setStyleSheet("color: black;")

        if (self.tracks_df["id"] == current_id).any():
            current_moi = self.tracks_df.loc[
                self.tracks_df["id"] == current_id, "moi"
            ].unique()[0]

            if current_moi == -1:
                self.pushButtons[-1].setStyleSheet("color: green;")
            else:
                self.pushButtons[current_moi - 1].setStyleSheet("color: green;")

    def draw_track_on_image(self, image, current_track):
        temp_img = image.copy()
        colormap = cm.get_cmap("rainbow", len(current_track))
        newcolors = colormap(np.linspace(0, 1, current_track.shape[0]))
        index = 0
        lastpt = []
        firstpt = []
        prevpt = []

        for _, track in current_track.iterrows():
            x_track = int((track["x1"] + track["x2"]) / 2)
            y_track = int(track["y2"])
            pt = [x_track, y_track]
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
        # if self.track_mode:
        #     raise "Something is wrong. plot_next_track should only be called when track_mode is False"
        if self.id_index >= len(self.tids):
            print("already ploted all the trajectories")
            # self.track_mode = True
            # self.plot_next_cluster()
            return

        self.current_id = self.tids[self.id_index]
        self.current_track = self.df[self.df["id"] == self.current_id]
        # print(self.current_track['trajectory'])

        while (
            self.id_index < len(self.tids) - 1
            and self.current_track.shape[0] < MIN_PTS_IN_TRACK
        ):
            # print(self.current_id)
            # print(len(list(self.current_track['trajectory'])[0]))
            self.id_index = self.id_index + 1
            self.current_id = self.tids[self.id_index]
            self.current_track = self.df[self.df["id"] == self.current_id]
        # print(self.current_id)
        # print(len(list(self.current_track['trajectory'])[0]))

        if self.id_index < len(self.tids):
            self.temp_img = self.draw_track_on_image(self.img, self.current_track)
            self.update_moi_button(self.current_id)

            self.show_image_on_GUI(self.temp_img)

            self.id_index = self.id_index + 1

    def plot_previous_track(self):
        if self.id_index == 1:
            print("There is no previous track")
            return

        self.id_index -= 1

        self.current_id = self.tids[self.id_index - 1]
        self.current_track = self.df[self.df["id"] == self.current_id]

        while self.id_index > 0 and self.current_track.shape[0] < MIN_PTS_IN_TRACK:
            # print(self.current_id)
            # print(len(list(self.current_track['trajectory'])[0]))
            self.id_index -= 1
            self.current_id = self.tids[self.id_index]
            self.current_track = self.df[self.df["id"] == self.current_id]

        if self.id_index < len(self.tids) and self.id_index > 0:
            # print(self.current_id)
            self.temp_img = self.draw_track_on_image(self.img, self.current_track)
            self.update_moi_button(self.current_id)
            self.show_image_on_GUI(self.temp_img)

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

    def update_moi(self, moi):
        if moi == -1:
            self.counts[-1] += 1
        else:
            self.counts[moi - 1] += 1

        if (self.tracks_df["id"] == self.current_id).any():
            prev_moi = self.tracks_df.loc[
                self.tracks_df["id"] == self.current_id, "moi"
            ].unique()[0]

            if prev_moi == -1:
                self.counts[-1] = max(0, self.counts[-1] - 1)
            else:
                self.counts[prev_moi - 1] = max(0, self.counts[prev_moi - 1] - 1)

            self.tracks_df.loc[self.tracks_df["id"] == self.current_id, "moi"] = moi
        else:
            self.current_track["moi"] = moi
            self.tracks_df = pd.concat(
                [self.tracks_df, self.current_track], ignore_index=True
            )

    def update_moi_labels(self):
        for i in range(13):
            self.all_labels[i].setText(str(self.counts[i]))

    def c_pushButton_1(self):
        self.update_moi(1)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_2(self):
        self.update_moi(2)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_3(self):
        self.update_moi(3)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_4(self):
        self.update_moi(4)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_5(self):
        self.update_moi(5)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_6(self):
        self.update_moi(6)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_7(self):
        self.update_moi(7)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_8(self):
        self.update_moi(8)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_9(self):
        self.update_moi(9)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_10(self):
        self.update_moi(10)
        self.update_moi_labels()
        # self.tracks['10'].append(self.current_track)
        self.plot_next_track()

    def c_pushButton_11(self):
        self.update_moi(11)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_12(self):
        self.update_moi(12)
        self.update_moi_labels()
        self.plot_next_track()

    def c_pushButton_skip(self):
        self.update_moi(-1)
        self.update_moi_labels()
        self.plot_next_track()

    def delete_last_trajectory(self):
        while len(self.tracks_df) > 1:
            # self.tracks_df.drop(self.tracks_df.tail(1).index, inplace=True)
            self.tracks_df = self.tracks_df[:-1]
        self.id_index = max(0, self.id_index - 1)
        self.plot_next_track()

    def c_pushButton_export(self):
        print(self.tracks_df)
        if self.tracks_df.shape[0] > 0:
            self.tracks_df.to_pickle(self.export_path_pkl)
            self.tracks_df[["id", "moi"]].drop_duplicates(subset="id").to_csv(
                self.export_path_csv, index=False
            )
            QMessageBox.information(
                self,
                "Results saved",
                f"Exported to {self.export_path_pkl} and {self.export_path_csv}",
            )

    def c_pushButton_next(self):
        self.track_mode = False
        self.plot_next_track()

    def c_pushButton_prev(self):
        self.track_mode = False
        self.plot_previous_track()
