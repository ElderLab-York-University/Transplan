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
# from pymatreader import read_mat
import pandas as pd
from matplotlib import cm
from .resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm
import cv2 as cv

from Utils import *
from Maps import *
from Libs import *

#  Hyperparameters
MIN_TRAJ_POINTS = 10
MIN_TRAJ_Length = 50
MAX_MATCHED_Distance = 90

color_dict = moi_color_dict

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
        
        self.metric = Metric_Dict[args.CountMetric]
        self.args = args
        if self.args.LabelledTrajectories is None:
            print("loaded track labelling from previous path")
            validated_trakcs_path = self.args.TrackLabellingExportPthMeter
        else: validated_trakcs_path = self.args.LabelledTrajectories; print("loaded track labelling from external source")

        df = pd.read_pickle(validated_trakcs_path)
        # print(len(df))
        df['trajectory'] = df['trajectory'].apply(lambda x: track_resample(x))

        self.typical_mois = defaultdict(list)
        for index, row in df.iterrows():
            self.typical_mois[row['moi']].append(row["trajectory"])

        self.counter = defaultdict(int)
        self.traject_couter = 0
        self.tem = []       

    def counting(self, current_trajectory):
        # counting_start_time = time.time()
        resampled_trajectory = track_resample(np.array(current_trajectory, dtype=np.float64))

        if True:
            min_c = float('inf')
            matched_id = -1
            tem = []
            key = []
            for keys, values in self.typical_mois.items():
                for gt_trajectory in values:
                    traj_a, traj_b = gt_trajectory, resampled_trajectory
                    c = self.metric(traj_a, traj_b)
                    tem.append(c)
                    key.append(keys)

            tem = np.array(tem)
            key = np.array(key)
            idxs = np.argpartition(tem, 1)
            votes = key[idxs[:1]]
            matched_id = int(np.argmax(np.bincount(votes)))

            if self.args.CountVisPrompt:
                for t , k in zip(tem, key):
                    print(f"tem = {t}, key = {k}")
                print(f"matched id:{matched_id}")
                self.viz_CMM(resampled_trajectory, matched_id=matched_id)
                
            self.counter[matched_id] += 1
            self.traject_couter += 1
            return matched_id

    def viz_CMM(self, current_track, alpha=0.3, matched_id=0):
        r = meter_per_pixel(self.args.MetaData['center'])
        image_path = self.args.HomographyTopView
        img = cv.imread(image_path)
        back_ground = cv.imread(image_path)
        rows, cols, dim = img.shape
        for keys, values in self.typical_mois.items():
            for gt_trajectory in values:
                if not keys == matched_id:
                    for i in range(1, len(gt_trajectory)):
                        p1 = gt_trajectory[i-1]
                        p2 = gt_trajectory[i]
                        x1, y1 = int(p1[0]/r), int(p1[1]/r)
                        x2, y2 = int(p2[0]/r), int(p2[1]/r)
                        c = color_dict[keys]
                        img = cv2.line(img, (x1, y1), (x2, y2), c, thickness=1) 
                    for p in gt_trajectory:
                        x, y = int(p[0]/r), int(p[1]/r)
                        c = color_dict[keys]
                        img = cv.circle(img, (x,y), radius=1, color=c, thickness=1)
                else: 
                    for i in range(1, len(gt_trajectory)):
                        p1 = gt_trajectory[i-1]
                        p2 = gt_trajectory[i]
                        x1, y1 = int(p1[0]/r), int(p1[1]/r)
                        x2, y2 = int(p2[0]/r), int(p2[1]/r)
                        c = color_dict[keys]
                        back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), c, thickness=2) 

                    for p in gt_trajectory:
                        x, y = int(p[0]/r), int(p[1]/r)
                        c = color_dict[keys]
                        back_ground = cv.circle(back_ground, (x,y), radius=2, color=c, thickness=2)

        for i in range(1, len(current_track)):
            p1 = current_track[i-1]
            p2 = current_track[i]
            x1, y1 = int(p1[0]/r), int(p1[1]/r)
            x2, y2 = int(p2[0]/r), int(p2[1]/r)
            back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), (240, 50, 230), thickness=2) 

        for p in current_track:
            x, y = int(p[0]/r), int(p[1]/r)
            back_ground = cv.circle(back_ground, (x,y), radius=2, color=(240, 50, 230), thickness=2)

        p = current_track[0]
        x, y = int(p[0]/r), int(p[1]/r)
        back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 255, 0), thickness=2)

        p = current_track[-1]
        x, y = int(p[0]/r), int(p[1]/r)
        back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 0, 255), thickness=2)

        img_new = cv2.addWeighted(img, alpha, back_ground, 1 - alpha, 0)
        img_new = cv.cvtColor(img_new, cv.COLOR_BGR2RGB)
        plt.imshow(img_new)
        plt.show()

    def main(self):
        # file_path to all trajectories .txt file(at the moment
        # ** do not confuse it with selected trajectories
        file_name = self.args.ReprojectedPklMeter
        result_paht = self.args.CountingResPth

        data = {}
        data['id'], data['moi'] = [], []

        df_temp = pd.read_pickle(file_name)
        df = group_tracks_by_id(df_temp)
        tids = np.unique(df['id'].tolist())
        for idx in tqdm(tids):
            current_track = df[df['id'] == idx]
            a = current_track['trajectory'].values.tolist()
            matched_moi = self.counting(a[0])
            data['id'].append(idx)
            data['moi'].append(matched_moi)
                # print(f"couning a[0] with shape {a[0].shape}")

        for i in range(12):
            print(f"{i+1}: {self.counter[i+1]}")
        # print(self.counter)
        print(self.traject_couter)
        with open(result_paht, "w") as f:
            json.dump(self.counter, f, indent=2)

        df = pd.DataFrame.from_dict(data)
        df.to_csv(self.args.CountingIdMatchPth, index=False)

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

def eval_count(args):
    # args.CountingResPth a json file
    # args.CountingStatPth a csv file
    # args.MetaData.gt has the gt numbers
    estimated = None
    with open(args.CountingResPth) as f:
        estimated = json.load(f)
    data = {}
    data["moi"] = [i for i in args.MetaData["gt"].keys()]
    data["gt"] = [args.MetaData["gt"][i] for i in args.MetaData["gt"].keys()]
    data["estimated"] = [estimated[i] for i in args.MetaData["gt"].keys()]
    df = pd.DataFrame.from_dict(data)
    df["diff"] = (df["gt"] - df["estimated"]).abs()
    df["err"] = df["diff"]/df["gt"]
    
    data2 = {}
    data2["moi"] = ["all"]
    data2["gt"] = [df["gt"].sum()]
    data2["estimated"] = [df["estimated"].sum()]
    data2["diff"] = [df["diff"].sum()]
    data2["err"] = data2["diff"][0]/data2["gt"][0]
    df2 = pd.DataFrame.from_dict(data2)
    df = df.append(df2, ignore_index=True)
    df.to_csv(args.CountingStatPth, index=False)

def main(args):
    # some relative path form the args
        # args.ReprojectedPklMeter
        # args.TrackLabellingExportPthMeter
    counter = Counting(args)
    counter.main()

    if args.EvalCount:
        eval_count(args)
        return SucLog("counting part executed successfully with stats saved in counting/")
    return SucLog("counting part executed successfully")