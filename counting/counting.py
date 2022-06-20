"""
# ################### Counting ##################################
counting trajectories by the movement categories
"""
import glob
import re
import time
import json
import ctypes
from collections import defaultdict
import cv2
import matplotlib
import numpy as np
from pymatreader import read_mat
import pandas as pd
from matplotlib import cm
from .resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm
import cv2 as cv

from Utils import *

#  Hyperparameters
MIN_TRAJ_POINTS = 10
MIN_TRAJ_Length = 50
MAX_MATCHED_Distance = 90


# def group_tracks_by_id(tracks_path):
#     # this function was writtern for grouping the tracks with the same id
#     # usinig this one can load the data from a .txt file rather than .mat file
#     tracks = np.loadtxt(tracks_path, delimiter=",")
#     all_ids = np.unique(tracks[:, 1])
#     data = {"id":[], "trajectory":[], "frames":[]}
#     for idd in tqdm(all_ids):
#         mask = tracks[:, 1]==idd
#         selected_tracks = tracks[mask]
#         frames = [selected_tracks[: ,0]]
#         id = selected_tracks[0][1]
#         trajectory = selected_tracks[:, 2:4]
#         data["id"].append(id)
#         data["frames"].append(frames)
#         data["trajectory"].append(trajectory)
#     df = pd.DataFrame(data)
#     return df

def group_tracks_by_id(df):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids):
        frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
        id = idd
        trajectory = df[df['id']==idd][["x", "y"]].to_numpy(np.float32)
        
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df2 = pd.DataFrame(data)
    return df2

class Counting:
    def __init__(self, args):
        #ground truth labelled trajactories
        # validated tracks with moi labels
        # args.ReprojectedPklMeter
        # args.TrackLabellingExportPthMeter
        self.args = args
        validated_trakcs_path = self.args.TrackLabellingExportPthMeter

        df = pd.read_pickle(validated_trakcs_path)
        self.typical_mois = defaultdict(list)
        for index, row in df.iterrows():
            self.typical_mois[row['moi']].append(row["trajectory"])

        ################ CMM LIB init ################################
        # 0. find the shared library, you need to do first: "python setup.py build", and you will get it from the build folder
        libfile = "./counting/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so"
        # libfile = 'York/Elderlab/yorku_pipeline_deepsort_features/yorku_pipeline_deepsort_features/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so'
        self.cmmlib = ctypes.CDLL(libfile)

        # 2. tell Python the argument and result types of function cmm
        self.cmmlib.cmm_truncate_sides.restype = ctypes.c_double
        self.cmmlib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                                   np.ctypeslib.ndpointer(dtype=np.float64),
                                                   np.ctypeslib.ndpointer(dtype=np.float64),
                                                   np.ctypeslib.ndpointer(dtype=np.float64),
                                                   ctypes.c_int, ctypes.c_int]
        self.counter = defaultdict(int)
        self.traject_couter = 0
        self.tem = []


    def counting(self, current_trajectory, cmmlib):
        counting_start_time = time.time()
        resampled_trajectory = track_resample(np.array(current_trajectory, dtype=np.float64))

        # distance = pow(pow(resampled_trajectory[0][0] - resampled_trajectory[-1][0], 2) + pow(resampled_trajectory[0][1] - resampled_trajectory[-1][1], 2), 0.5)
        # distance = abs(resampled_trajectory[0][0] - resampled_trajectory[-1][0]) + abs(resampled_trajectory[0][1] - resampled_trajectory[-1][1])
        # print(distance)

        # if 65< distance < 100:
        #     self.tem.append(current_trajectory)

        # if len(current_trajectory) > MIN_TRAJ_POINTS and distance > MIN_TRAJ_Length:
        if True:
            min_c = float('inf')
            matched_id = -1
            tem = []
            key = []
            for keys, values in self.typical_mois.items():
                for gt_trajectory in values:
                    traj_a, traj_b = gt_trajectory, resampled_trajectory
                    if traj_a.shape[0] >= traj_b.shape[0]:
                        c = cmmlib.cmm_truncate_sides(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0],
                                                      traj_b.shape[0])
                    else:
                        c = cmmlib.cmm_truncate_sides(traj_b[:, 0], traj_b[:, 1], traj_a[:, 0], traj_a[:, 1], traj_b.shape[0],
                                                      traj_a.shape[0])

                    c = np.abs(c)
                    tem.append(c)
                    key.append(keys)
                    if c < min_c:
                        min_c = c
                        matched_id = keys

            for t , k in zip(tem, key):
                print(f"tem = {t}, key = {k}")
            # self.viz_CMM(current_trajectory)

            # having this threshold allows classification rejection
            # if min_c < MAX_MATCHED_Distance:
            if self.counter[matched_id] == 0:
                self.tem.append(current_trajectory)
            self.counter[matched_id] += 1
            self.traject_couter += 1

    def viz_CMM(self, current_track):
        image_path = "./../../Dataset/DundasStAtNinthLine.jpg"
        img = cv.imread(image_path)
        rows, cols, dim = img.shape
        for keys, values in self.typical_mois.items():
            for gt_trajectory in values:
                for p in gt_trajectory:
                    x, y = int(p[0]), int(p[1])
                    img = cv.circle(img, (x,y), radius=2, color=(int(keys/12*255), 70, int(255 - keys/12*255)), thickness=2)
        for p in current_track:
            x, y = int(p[0]), int(p[1])
            img = cv.circle(img, (x,y), radius=2, color=(70, 255, 70), thickness=2)
        cv.imshow('image',img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # def draw_trajectory(self):
    #     img_path = "./../../Dataset/DundasStAtNinthLine.jpg"
    #     # img_path = 'York/Elderlab/tracks_labelling_gui-master/Dundas_Street_at_Ninth_Line/Dundas Street at Ninth Line.jpg'
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
    #     frame = cv2.imread(img_path)
    #     for track_num, track in enumerate(self.tem):
    #         trajectory = track
    #         color = (
    #         np.random.randint(low=0, high=128), np.random.randint(low=0, high=128), np.random.randint(low=0, high=128))
    #         index = 0
    #         for i, pt in enumerate(trajectory):
    #             rgba_color = cm.rainbow(norm(index), bytes=True)[0:3]
    #             if pt[0] < 0 or pt[1] < 0 or pt[0] >= frame.shape[1] or pt[1] >= frame.shape[0]:
    #                 continue
    #             index += 1
    #             cv2.circle(frame, (int(pt[0]), int(pt[1])), 3,
    #                        (int(rgba_color[0]), int(rgba_color[1]), int(rgba_color[2])), -1)
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def main(self):
        # file_path to all trajectories .txt file(at the moment
        # ** do not confuse it with selected trajectories
        file_name = self.args.ReprojectedPklMeter
        result_paht = self.args.CountingResPth

        df_temp = pd.read_pickle(file_name)
        df = group_tracks_by_id(df_temp)
        tids = np.unique(df['id'].tolist())
        for idx in tids:
            current_track = df[df['id'] == idx]
            a = current_track['trajectory'].values.tolist()
            if a[0].shape[0] > 50:
                self.counting(a[0], self.cmmlib)
                # print(f"couning a[0] with shape {a[0].shape}")

        for i in range(12):
            print(f"{i+1}: {self.counter[i+1]}")
        # print(self.counter)
        print(self.traject_couter)
        with open(result_paht, "w") as f:
            json.dump(self.counter, f, indent=2)

    def arc_length(self, track):
        """
        :param track: input track numpy array (M, 2)
        :return: the estimated arc length of the track
        """
        assert track.shape[1] == 2
        accum_dist = 0
        for i in range(1, track.shape[0]):
            dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
            accum_dist += dist_
        return accum_dist

def main(args):
    # some relative path form the args
        # args.ReprojectedPklMeter
        # args.TrackLabellingExportPthMeter
    counter = Counting(args)
    # input Trajectory_path
    counter.main()
    return SucLog("counting part executed successfully")