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
from TrackLabeling import *
from hmmlearn import hmm 
import copy

#  Hyperparameters
# MIN_TRAJ_POINTS = 10
# MIN_TRAJ_Length = 50
# MAX_MATCHED_Distance = 90

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

    def counting(self, current_trajectory, track_id=-1):
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
                self.viz_CMM(resampled_trajectory, matched_id=matched_id, track_id = track_id)
                
            self.counter[matched_id] += 1
            self.traject_couter += 1
            return matched_id

    def viz_CMM(self, current_track, alpha=0.3, matched_id=0, track_id=-1):
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
        plt.title(f"id;{track_id}")
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
            matched_moi = self.counting(a[0], track_id=idx)
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

class IKDE():
    def __init__(self, kernel="gaussian", bandwidth=3.2, os_ratio = 2):
        self.kernel = kernel
        self.bw = bandwidth
        self.osr = os_ratio

    def fit(self, tracks):
        self.mois = np.unique(tracks["moi"])
        self.kdes = {}
        for moi in self.mois:
            self.kdes[moi] = sklearn.neighbors.KernelDensity(kernel=self.kernel, bandwidth=self.bw)
        for moi in self.mois:
            kde_data = []
            sequence_data = []
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                traj = row["trajectory"]
                for i in range(1, len(traj)):
                    p1, p2 = traj[i-1], traj[i]
                    for r in np.linspace(0, 1, num=self.osr):
                        p_temp = (1-r)*p1 + r*p2
                        sequence_data.append(p_temp)
                for i , p in enumerate(sequence_data):
                    x, y = p
                    kde_data.append([x, y])
            kde_data = np.array(kde_data)
            self.kdes[moi].fit(kde_data)

    def get_traj_score(self, traj, moi):
        traj_data = []
        for i in range(len(traj)):
            x, y = traj[i]
            traj_data.append([x, y])
        traj_data = np.array(traj_data)
        return np.sum(self.kdes[moi].score_samples(traj_data))
    
    def predict_traj(self, traj):
        moi_scores = []
        for moi in self.mois:
            moi_scores.append(self.get_traj_score(traj, moi))
        max_moi = self.mois[np.argmax(moi_scores)]
        return max_moi

    def predict_tracks(self, tracks):
        max_mois = []
        for i, row in tracks.iterrows():
            traj = row["trajectory"]
            max_mois.append(self.predict_traj(traj))
        return max_mois

class LOSIKDE(IKDE):
    def __init__(self, kernel="gaussian", bandwidth=3.2, os_ratio = 2):
        super().__init__(kernel, bandwidth, os_ratio)

    def fit(self, tracks):
        self.mois = np.unique(tracks["moi"])
        self.kdes = {}
        for moi in self.mois:
            self.kdes[moi] = sklearn.neighbors.KernelDensity(kernel=self.kernel, bandwidth=self.bw)
        for moi in self.mois:
            kde_data = []
            sequence_data = []
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                traj = row["trajectory"]
                for i in range(1, len(traj)):
                    p1, p2 = traj[i-1], traj[i]
                    for r in np.linspace(0, 1, num=self.osr):
                        p_temp = (1-r)*p1 + r*p2
                        sequence_data.append(p_temp)
                for i , p in enumerate(sequence_data):
                    x, y = p
                    kde_data.append([x, y, i/len(sequence_data)])
            kde_data = np.array(kde_data)
            self.kdes[moi].fit(kde_data)

    def get_traj_score(self, traj, moi):
        traj_data = []
        for i in range(len(traj)):
            x, y = traj[i]
            traj_data.append([x, y, i/len(traj)])
        traj_data = np.array(traj_data)
        return np.sum(self.kdes[moi].score_samples(traj_data))

class HMMG():
    def __init__(self, n_components=5):
        self.nc = n_components

    def fit(self, tracks):
        self.mois = np.unique(tracks["moi"])
        self.hmms = {}
        for moi in self.mois:
            self.hmms[moi] = hmm.GaussianHMM(n_components=self.nc)
        for moi in self.mois:
            hmm_data = []
            hmm_length = []
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                hmm_data.append(row["trajectory"])
                hmm_length.append(len(row["trajectory"]))
            hmm_data = np.concatenate(hmm_data)
            self.hmms[moi].fit(hmm_data, hmm_length)

    def get_traj_score(self, traj, moi):
        return self.hmms[moi].score(traj)

    def predict_traj(self, traj):
        moi_scores = []
        for moi in self.mois:
            moi_scores.append(self.get_traj_score(traj, moi))
        max_moi = self.mois[np.argmax(moi_scores)]
        return max_moi

    def predict_tracks(self, tracks):
        max_mois = []
        for i, row in tracks.iterrows():
            traj = row["trajectory"]
            max_mois.append(self.predict_traj(traj))
        return max_mois

class MyPoly():
    def __init__(self, roi):
        self.poly = sympy.Polygon(*roi)
        self.lines = []
        for i in range(len(roi)-1):
            self.lines.append(sympy.Line(sympy.Point(roi[i]), sympy.Point(roi[i+1])))
        self.lines.append(sympy.Line(sympy.Point(*roi[-1]), sympy.Point(*roi[0])))
        
    def distance(self, point):
        p = sympy.Point(*point)
        distances = []
        for line in self.lines:
            distances.append(float(line.distance(p)))
        distances = np.array(distances)
        min_pos = np.argmin(distances)
        return distances[min_pos], int(min_pos)

    def distance_angle_filt(self, p_main, p_second):
        imaginary_line = sympy.Line(sympy.Point(p_main), sympy.Point(p_second))
        min_angle = float('inf')
        min_distance = float('inf')
        min_distance_indx = None
        for i, line in enumerate(self.lines):
            d_main = float(line.distance(p_main))
            d_secn = float(line.distance(p_second))
            if d_secn <  d_main : continue
            else:
                if np.abs(float(imaginary_line.angle_between(line)) - np.pi/2 ) < min_angle:
                    min_distance = float(line.distance(p_main))
                    min_distance_indx  = i
        return min_distance, min_distance_indx

    @property
    def area(self):
        return float(self.poly.area)

    
class KDECounting(Counting):
    def __init__(self, args):
        # load tracks
        # isolate complete tracks
        # train kdes and store them
        # classify based on kdes
        self.args = args
        tracks_path = args.ReprojectedPkl
        tracks_meter_path = args.ReprojectedPklMeter
        top_image = args.HomographyTopView
        meta_data = args.MetaData # dict is already loaded
        HomographyNPY = args.HomographyNPY

        # load data
        M = np.load(HomographyNPY, allow_pickle=True)[0]
        tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
        tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path))
        tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        img = plt.imread(top_image)
        img1 = cv.imread(top_image)
        H, W, C = img1.shape
        X_cord, Y_cord = np.meshgrid(range(int(W)), range(int(H)))
        # X_cord, Y_cord = np.meshgrid(range(200, 600), range(200, 600))
        Data_cord = np.stack([X_cord, Y_cord]).reshape(2, -1).T

        # create roi polygon
        roi_rep = []
        for p in args.MetaData["roi"]:
            point = np.array([p[0], p[1], 1])
            new_point = M.dot(point)
            new_point /= new_point[2]
            roi_rep.append([new_point[0], new_point[1]])

        pg = MyPoly(roi_rep)
        th = args.MetaData["roi_percent"] * np.sqrt(pg.area)

        counter = 0
        mask = []
        i_strs = []
        i_ends = []
        moi = []

        # find proper tracks(complete and monotonic)
        for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
            traj = row["trajectory"]
            d_str, i_str = pg.distance(traj[0])
            d_end, i_end = pg.distance(traj[-1])
            i_strs.append(i_str)
            i_ends.append(i_end)
            moi.append(str_end_to_moi(i_str, i_end))

            if (d_str < th) and (d_end < th) and (not i_str == i_end) and is_monotonic(traj[row["index_mask"]]):
                mask.append(True)
                c=0 
                for x, y in traj:
                    x, y = int(x), int(y)
                    img1 = cv.circle(img1, (x,y), radius=1, color=(int(c/len(traj)*255), 70, int(255 - c/len(traj)*255)), thickness=1)
                    c+=1
            else:
                mask.append(False)

        plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        # plt.show()


        # temporarily add info to track dataframe
        tracks['i_str'] = i_strs
        tracks['i_end'] = i_ends
        tracks['moi'] = moi
        tracks_meter['i_str'] = i_strs
        tracks_meter['i_end'] = i_ends 
        tracks_meter['moi'] = moi
        plt.hist(tracks[mask]["moi"])
        # plt.show()
        # resample on the ground plane but not in meter
        tracks["trajectory"] = tracks.apply(lambda x: x['trajectory'][x["index_mask"]], axis=1)
      
        if self.args.CountMetric == "kde":
            self.ikde = IKDE()
        elif self.args.CountMetric == "loskde":
            self.ikde = LOSIKDE()
        elif self.args.CountMetric == "hmmg":
            self.ikde = HMMG()
        else: raise "it should not happen"
        self.ikde.fit(tracks[mask])

    def main(self, args=None):
        # where the counting happens
        if args is not None:
            self.args = args
        args = self.args
        tracks_path = args.ReprojectedPkl
        tracks_meter_path = args.ReprojectedPklMeter
        top_image = args.HomographyTopView
        meta_data = args.MetaData # dict is already loaded
        HomographyNPY = args.HomographyNPY
        result_paht = args.CountingResPth
        # load data
        M = np.load(HomographyNPY, allow_pickle=True)[0]
        tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
        tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path))
        tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True))
        img = plt.imread(top_image)
        img1 = cv.imread(top_image)
        # resample gp tracks
        tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        tracks["moi"] = self.ikde.predict_tracks(tracks)

        counted_tracks  = tracks[["id", "moi"]]
        counted_tracks.to_csv(self.args.CountingIdMatchPth, index=False)

        counter = {}
        for moi in range(1, 13):
            counter[moi]  = 0
        for i, row in counted_tracks.iterrows():
                counter[int(row["moi"])] += 1
        print(counter)
        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)

        print("right before count vis prompt")

        # get roi on groundpalane
        roi_rep = []
        for p in args.MetaData["roi"]:
            point = np.array([p[0], p[1], 1])
            new_point = M.dot(point)
            new_point /= new_point[2]
            roi_rep.append([new_point[0], new_point[1]])
        pg = MyPoly(roi_rep)
        th = args.MetaData["roi_percent"] * np.sqrt(pg.area)

        if self.args.CountVisPrompt:
            print("will initiate count vis prompt")
            for i, row in tracks.iterrows():
                self.plot_track_on_gp(row["trajectory"], matched_id=row["moi"], track_id=row["id"], roi=roi_rep, roi_th=th, pg=pg)

    def add_roi_line_to_img(self,img, roi):
        for i in range(1, len(roi)):
            x1, y1 = roi[i-1]
            x2, y2 = roi[i]
            img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
        return img
    
    def add_roi_gap_to_img(self,img, pg, roi_th):
        alpha = 0.2
        bg_img = copy.deepcopy(img)
        for y in range(bg_image.shape[0]):
            for x in range(bg_image.shape[1]):
                d, i = pg.distance([x, y])
                if d > roi_th:
                    bg_img[y, x] = 0
        return cv2.addWeighted(img, alpha, bg_img, 1 - alpha, 0)


    def plot_track_on_gp(self, current_track, matched_id=0, alpha=0.4, track_id=None, roi=None, roi_th=None, pg=None):
        c = color_dict[int(matched_id)]
        image_path = self.args.HomographyTopView
        img = cv.imread(image_path)
        back_ground = cv.imread(image_path)

        for i in range(1, len(current_track)):
            p1 = current_track[i-1]
            p2 = current_track[i]
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), c, thickness=2) 

        for p in current_track:
            x, y = int(p[0]), int(p[1])
            back_ground = cv.circle(back_ground, (x,y), radius=2, color=c, thickness=2)

        p = current_track[0]
        x, y = int(p[0]), int(p[1])
        back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 255, 0), thickness=2)

        p = current_track[-1]
        x, y = int(p[0]), int(p[1])
        back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 0, 255), thickness=2)

        if roi is not None:
            back_ground = self.add_roi_line_to_img(back_ground, roi)
            if (roi_th is not None) and (pg is not None):
                back_ground = self.add_roi_gap_to_img(back_ground, pg, roi_th)

        img_new = cv2.addWeighted(img, alpha, back_ground, 1 - alpha, 0)
        img_new = cv.cvtColor(img_new, cv.COLOR_BGR2RGB)
        plt.imshow(img_new)
        if track_id is not None:
            plt.title(f"matched id:{matched_id} track id:{track_id}")
        else:
            plt.title(f"matched id:{matched_id}")
        plt.show()
        print(len(current_track))
        print(current_track)

class ROICounting(KDECounting):
    def __init__(self, args):
        self.args = args
        # load ROI 
        meta_data = args.MetaData # dict is already loaded
        HomographyNPY = args.HomographyNPY
        self.M = np.load(HomographyNPY, allow_pickle=True)[0]
        roi_rep = []

        for p in args.MetaData["roi"]:
            point = np.array([p[0], p[1], 1])
            new_point = self.M.dot(point)
            new_point /= new_point[2]
            roi_rep.append([new_point[0], new_point[1]])

        self.pg = MyPoly(roi_rep)

    def main(self):
        args = self.args
        pg = self.pg
        tracks_path = args.ReprojectedPkl
        result_paht = args.CountingResPth

        tracks = group_tracks_by_id(pd.read_pickle(tracks_path))
        i_strs = []
        i_ends = []
        moi = []
        # perform the actual counting paradigm
        for i, row in tqdm(tracks.iterrows(), total=len(tracks)):
            traj = row["trajectory"]
            d_str, i_str = pg.distance(traj[0])
            d_end, i_end = pg.distance(traj[-1])
            i_strs.append(i_str)
            i_ends.append(i_end)
            moi.append(str_end_to_moi(i_str, i_end))

        tracks['i_str'] = i_strs
        tracks['i_end'] = i_ends
        tracks['moi'] = moi
        counted_tracks  = tracks[["id", "moi"]][tracks["moi"]!=-1]
        counted_tracks.to_csv(self.args.CountingIdMatchPth, index=False)


        counter = {}
        for moi in range(1, 13):
            counter[moi]  = 0
        for i, row in counted_tracks.iterrows():
                counter[int(row["moi"])] += 1
        print(counter)
        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)

        if self.args.CountVisPrompt:
            for i, row in tracks.iterrows():
                self.plot_track_on_gp(row["trajectory"], matched_id=row["moi"])
                
def str_end_to_moi(str, end):
    str_end_moi = {}
    str_end_moi[(3, 0)] = '1'
    str_end_moi[(3, 1)] = '2'
    str_end_moi[(3, 2)] = '3'
    str_end_moi[(2, 3)] = '4'
    str_end_moi[(2, 0)] = '5'
    str_end_moi[(2, 1)] = '6'
    str_end_moi[(1, 2)] = '7'
    str_end_moi[(1, 3)] = '8'
    str_end_moi[(1, 0)] = '9'
    str_end_moi[(0, 1)] = '10'
    str_end_moi[(0, 2)] = '11'
    str_end_moi[(0, 3)] = '12'
    if (str ,end) in str_end_moi:
        return str_end_moi[(str, end)]
    return -1
        

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

    # check if use cached counter
    if args.UseCachedCounter:
        with open(args.CachedCounterPth, "rb") as f:
            counter = pkl.load(f)
    else:
        if args.CountMetric in ["kde", "loskde", "hmmg"] :
            counter = KDECounting(args)
        elif args.CountMetric == "roi":
            counter = ROICounting(args)
        else:
            counter = Counting(args)

    # perfom counting here
    counter.main(args)

    # save counter object for later use
    if args.CacheCounter:
        with open(args.CachedCounterPth, "wb") as f:
            print(f"counter being saved to {args.CachedCounterPth}")
            pkl.dump(counter, f)

    if args.EvalCount:
        eval_count(args)
        return SucLog("counting part executed successfully with stats saved in counting/")
    return SucLog("counting part executed successfully")