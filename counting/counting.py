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


from resample_gt_MOI.resample_typical_tracks import track_resample

class Counting:
    def __init__(self):
        #ground truth labelled trajactories
        jsonFile = 'tracks_labelling_gui-master/Dundas_Street_at_Ninth_Line/NMSvalidated_trajectories.csv'

        DF = pd.read_csv(jsonFile, sep=',', usecols= ['id', 'trajectory', 'moi'])
        self.typical_mois = defaultdict(list)
        for index, row in DF.iterrows():
            a = self.decode_sring(row['trajectory'])
            self.typical_mois[row['moi']].append(a)

        ################ CMM LIB init ################################
        # 0. find the shared library, you need to do first: "python setup.py build", and you will get it from the build folder
        libfile = 'York/Elderlab/yorku_pipeline_deepsort_features/yorku_pipeline_deepsort_features/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so'
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

    def decode_sring(self, aa):
        res = []
        my_list = aa.split("\n")[1: -1]
        for tem in my_list:
            # tem = tem.replace("' [", "'[")
            # tem = tem.replace("[ ", "[")
            tem = tem[2: -2]
            tem = re.sub("\s+", ",", tem)
            tem = tem.split(",")
            # print(tem)
            cur_tem = []
            for item in tem:
                if item != '':
                    cur_tem.append(float(item))
            # print(cur_tem)
            if cur_tem:
                res.append(cur_tem)
        arr_res = np.array(res)
        return arr_res



    def counting(self, current_trajectory, cmmlib):
        counting_start_time = time.time()
        resampled_trajectory = track_resample(np.array(current_trajectory, dtype=np.float64))

        # distance = pow(pow(resampled_trajectory[0][0] - resampled_trajectory[-1][0], 2) + pow(resampled_trajectory[0][1] - resampled_trajectory[-1][1], 2), 0.5)
        distance = abs(resampled_trajectory[0][0] - resampled_trajectory[-1][0]) + abs(resampled_trajectory[0][1] - resampled_trajectory[-1][1])
        print(distance)

        # if 65< distance < 100:
        #     self.tem.append(current_trajectory)

        if len(current_trajectory) > 35 and distance > 65:
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
                    tem.append(c)
                    key.append(keys)
                    if c < min_c:
                        min_c = c
                        matched_id = keys
            print(matched_id)
            if min_c < 99:
                if self.counter[matched_id] == 0:
                    self.tem.append(current_trajectory)
                self.counter[matched_id] += 1
            self.traject_couter += 1

    def draw_trajectory(self):
        img_path = 'York/Elderlab/tracks_labelling_gui-master/Dundas_Street_at_Ninth_Line/Dundas Street at Ninth Line.jpg'
        norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
        frame = cv2.imread(img_path)
        for track_num, track in enumerate(self.tem):
            trajectory = track
            color = (
            np.random.randint(low=0, high=128), np.random.randint(low=0, high=128), np.random.randint(low=0, high=128))
            index = 0
            for i, pt in enumerate(trajectory):
                rgba_color = cm.rainbow(norm(index), bytes=True)[0:3]
                if pt[0] < 0 or pt[1] < 0 or pt[0] >= frame.shape[1] or pt[1] >= frame.shape[0]:
                    continue
                index += 1
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3,
                           (int(rgba_color[0]), int(rgba_color[1]), int(rgba_color[2])), -1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def main(self, filename):
        data = read_mat(filename)
        df1 = pd.DataFrame(data['recorded_tracks'])
        df = df1[['id', 'trajectory']]
        df.columns = ['id', 'trajectory']
        tids = np.unique(df['id'].tolist())
        for idx in tids:
            current_track = df[df['id'] == idx]
            a = current_track['trajectory'].values.tolist()
            if a[0].shape[0] > 30:
                self.counting(a[0], self.cmmlib)
        print(self.counter)
        print(self.traject_couter)

if __name__ == "__main__":

    ob1 = Counting()
    # input Trajectory_path
    print(ob1.main(Trajectory_path))




