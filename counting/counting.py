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
import Track
from hmmlearn import hmm 
import copy
import os

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve2d
from sklearn.neighbors import KNeighborsClassifier

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

def over_sample(traj, osr):
    # oversample a downsampled trajectory to fill in gaps
    osr_traj = []
    tot_ps_to_add = len(traj) * osr
    # compute arch length of traj
    t1 = np.array(traj[1:])
    t2 = np.array(traj[:-1])
    distances = np.linalg.norm(t1 - t2, axis=1)
    arc = np.sum(distances)
    num_points = distances / arc * tot_ps_to_add

    for i in range(1, len(traj)):
        p1, p2 = traj[i-1], traj[i]
        for r in np.linspace(0, 1, num=int(num_points[i-1]+1)):
            p_temp = (1-r)*p1 + r*p2
            x, y = p_temp
            osr_traj.append([x, y])
    osr_traj.append(traj[-1])
    return np.array(osr_traj)

def my_viz_CMM(current_track, img, alpha=0.3, matched_id=0, track_id=-1):

    back_ground = copy.deepcopy(img)
    rows, cols, dim = img.shape

    for i in range(1, len(current_track)):
        p1 = current_track[i-1]
        p2 = current_track[i]
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), (240, 50, 230), thickness=2) 

    for p in current_track:
        x, y = int(p[0]), int(p[1])
        back_ground = cv.circle(back_ground, (x,y), radius=2, color=(240, 50, 230), thickness=2)

    p = current_track[0]
    x, y = int(p[0]), int(p[1])
    back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 255, 0), thickness=2)

    p = current_track[-1]
    x, y = int(p[0]), int(p[1])
    back_ground = cv.circle(back_ground, (x,y), radius=3, color=(0, 0, 255), thickness=2)

    img_new = cv2.addWeighted(img, alpha, back_ground, 1 - alpha, 0)
    img_new = cv.cvtColor(img_new, cv.COLOR_BGR2RGB)
    plt.imshow(img_new)
    plt.title(f"id;{track_id}")
    plt.savefig(f"sample_track_moi{matched_id}_id{track_id}.png")


def group_tracks_by_id(df, gp, verbose=True):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    # if gp choose backprojected points[x, y]
    # if not gp choose contact point [xcp, ycp]
    if gp:
        traj_fields = ["x", "y"]
    else:
        traj_fields = ["xcp", "ycp"]

    all_ids = np.unique(df['id'].to_numpy(dtype=np.int64))
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids, disable=not verbose):
        frames = df[df['id']==idd]["fn"].to_numpy(np.float32)
        id = idd
        trajectory = df[df['id']==idd][traj_fields].to_numpy(np.float32)
        
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df2 = pd.DataFrame(data)
    return df2

def interpolate_traj(traj, frames):
    int_traj, int_frames = [traj[0]], [frames[0]]
    assert len(traj)==len(frames)
    for i in range(1, len(frames)):
        fm = int(frames[i-1])
        fn = int(frames[i])
        assert fn > fm
        if not(fm+1 == fn): # interpolate between two tracks
            for fi in range(fm+1, fn+1):
                r = (fi-fm)/(fn-fm)
                int_frames.append(fi)
                int_traj.append(r*traj[i] + (1-r)*traj[i-1])
        else:
            int_frames.append(fn)
            int_traj.append(traj[i])

    return int_traj, int_frames

def find_opt_bw_of_surf(args, gp):
    osr = args.OSR
    tracks_path = args.ReprojectedPkl
    tracks_meter_path = args.ReprojectedPklMeter
    if gp:
        top_image = args.HomographyTopView
    else:
        top_image = args.HomographyStreetView
    meta_data = args.MetaData # dict is already loaded
    # HomographyNPY = args.HomographyNPY @Del

    # load data
    tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=gp)
    tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=gp)
    # resample gp tracks
    tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
    tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
    img = plt.imread(top_image)
    # img1 = cv.imread(top_image) @Del

    # make data ready for kde training
    kde_data = []
    sequence_data = []
    for i, row in tracks.iterrows():
        traj = row["trajectory"]
        # total points to add
        tot_ps_to_add = len(traj) * osr
        # compute arch length of traj
        t1 = np.array(traj[1:])
        t2 = np.array(traj[:-1])
        distances = np.linalg.norm(t1 - t2, axis=1)
        arc = np.sum(distances)
        num_points = distances / arc * tot_ps_to_add

        for i in range(1, len(traj)):
            p1, p2 = traj[i-1], traj[i]
            for r in np.linspace(0, 1, num=int(num_points[i-1]+0.5)):
                p_temp = (1-r)*p1 + r*p2
                x, y = p_temp
                kde_data.append([x, y])

    kde_data = np.array(kde_data)

    # our data is repatative in time, temporal split
    split_idx = int(len(kde_data) * 0.5)
    
    # 50 percent train and 50 percent test
    kde_data_train = kde_data[:split_idx]
    kde_data_test  = kde_data[split_idx:]
    
    # use optimized 2D KDE
    max_d = np.max(img.shape)
    h_range = np.linspace(1, int(max_d * 0.01), 100)
    scores = []
    for h in tqdm(h_range):
        kde = KDE2D()
        kde.fit(kde_data_train, img, h)
        scores.append(np.sum(kde.score_samples(kde_data_test)))

    scores = np.array(scores)

    plt.plot(h_range, scores, color="black")
    plt.xlabel('Bandwidth [pixels]')
    plt.ylabel('log likelihood')
    h_opt = h_range[np.argmax(scores)]
    plt.plot(h_opt,np.max(scores),marker='*', color="red")

    plt.plot([h_opt, h_opt], [np.min(scores), np.max(scores)], color="red", linestyle="dashed")
    plt.plot([np.min(h_range), h_opt], [np.max(scores), np.max(scores)], color="red", linestyle="dashed")
    if gp:
        plt.savefig(args.OptBWGround)
    else:
        plt.savefig(args.OptBWImage)

    plt.close("all")

    return h_opt

def find_opt_bw(args):
    h_opt_ground = find_opt_bw_of_surf(args, True)
    h_opt_image  = find_opt_bw_of_surf(args, False)
    return h_opt_ground, h_opt_image

class KDE2D(object):
    def __init__(self):
        self.pmap = None
        self.im = None
        self.eps = 1e-20

    def fit(self, data, im, h):
        # Fit KDE model to 2D data points and plots results, overlaid on image im.
        # Input:
        # data: Nx2 data matrix
        # im: Image from which data point were derived
        # h: kernel bw
        self.im = im
        imsize = im.shape[0:2]
        n = data.shape[0]

        datamap = np.zeros(imsize) + self.eps
        # note reversal of x and y
        #
        # datamap[data[:,1].astype(int), data[:,0].astype(int)] += 1 ?
        missed_counts = 0
        for point in data.astype(int):
            try:
                datamap[point[1],point[0]] += 1
            except: #TODO some points are on the edge of image with incorrect coordinates
                missed_counts += 1
                pass
                # print(f"datamap shape:{datamap.shape} y:{point[1]} x:{point[0]}")


        self.pmap = self.KDE2D(datamap, h, n - missed_counts)

    def score_samples(self, test_data):
        scores = []
        for test_point in test_data.astype(int):
            try:
                scores.append(np.log(self.pmap[test_point[1], test_point[0]]))
            except: #TODO some points are on the edge of image with incorrect coordinates
                pass
                # print(f"test_point coordinates: y:{test_point[1]} x:{test_point[0]}")   
        return np.array(scores)
    
    def find_opt_h(self, data, im, h_range = range(5, 55, 5), M=10):
        imsize = im.shape[0:2]
        hopt = self.KDE2Dopt(data, h_range, imsize, M)
        return hopt

    def KDE2D(self, datamap=None, h=None, n=None):
        # Compute KDE map over 2D data, assuming points are at integer locations
        # Input:
        # data: Nx2 data matrix
        # h: bandwidth
        # n: number of points
        # Output:
        # pmap: probability density at each pixel of image

        # Truncate Gaussian kernel at +/-20h
        xlim = math.ceil(20 * h)
        x = range(-xlim, xlim + 1)
        kern = np.reshape(norm.pdf(x, 0, h), (-1, 1))
        pmap = convolve2d(convolve2d(datamap, kern, 'same'), kern.T, 'same') / n
        return pmap


    def KDE2DXval(self, data=None, h=None, M=None, imsize=None):
        # Compute M-fold cross-validated pseudo log likelihood
        # Input:
        # data: Nx2 data matrix
        # h: bandwidth
        # M: number of folds
        # imsize: size of image
        # Output:
        # pll: pseudo log likelihood of model
        N = data.shape[0]
        pllM = np.zeros(M)
        W = math.ceil(N / M)  # Fold size

        for i in range(M):
            tstart = W * i
            tend = min(W * (i + 1), N)  # Last fold may be slightly smaller
            trainidx = list(range(0, tstart)) + list(range(tend, N))
            datamap = np.zeros(imsize) + self.eps
            # note reversal of x and y and conversion
            # datamap[data[trainidx, 1].astype(int), data[trainidx, 0].astype(int)] += 1 ?
            for point in data.astype(int)[trainidx]:
                try:
                    datamap[point[1],point[0]] += 1
                except:
                    print(f"datamap shape:{datamap.shape} y:{point[1]} x:{point[0]}")

            pmap = self.KDE2D(datamap, h, len(trainidx))
            testdata = data[tstart:tend]
            # note reversal of x and y
            pllM[i] = np.sum(np.log(pmap[testdata[:, 1].astype(int), testdata[:, 0].astype(int)]))

        pll = np.mean(pllM)
        return pll


    def KDE2Dopt(self, data=None, hs=None, imsize=None, M=10):
        # Optimize h by cross-validation
        # Input:
        # data: Nx2 data matrix
        # hs: bandwidths
        # imsize: size of image
        # Output:
        # hopt: pseudo log likelihood of model

        N = data.shape[0]
        data = np.random.permutation(data) #randomly permute data to shuffle any correlations
        pll = []
        for h in tqdm(hs, desc="KDE BWs:"):
            pll.append(self.KDE2DXval(data, h, M, imsize))
        idx=np.argmax(pll)
        hopt = hs[idx]

        # Plot pseudo log likelihood as a function of bandwidth h
        plt.figure()
        plt.plot(hs, pll)
        plt.xlabel('Bandwidth h (pixels)')
        plt.ylabel('Pseudo log likelihood')
        plt.savefig("find_opt_h.png")

        return hopt

class Counting:
    def __init__(self, args, gp):
        # ground truth labelled trajactories
        # validated tracks with moi labels
        # args.ReprojectedPklMeter
        # args.TrackLabellingExportPthMeter
        self.gp = gp
        if gp:
            self.init_gp(args)
        else:
            self.init_image(args)

    def init_gp(self, args):
        self.CountMetric = args.CountMetric
        self.metric = Metric_Dict[args.CountMetric]
        self.args = args
        if self.args.LabelledTrajectories is None:
            print("loaded track labelling from previous path")
            validated_trakcs_path = self.args.TrackLabellingExportPth
            validated_tracks_path_meter = self.args.TrackLabellingExportPthMeter
        else: validated_trakcs_path = self.args.LabelledTrajectories; print("loaded track labelling from external source")

        df = pd.read_pickle(validated_trakcs_path)
        df_meter = pd.read_pickle(validated_tracks_path_meter)
        df['index_mask'] = df_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        df["trajectory"] = df.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        df["trajectory"] = df.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)

        # create the intersection ROI and filter prototypes and tracks
        # to be within intersection
        # create roi polygon
        method = args.BackprojectionMethod
        M = None
        GroundRaster = None
        orthophoto_win_tif_obj = None

        if method == "Homography":
            homography_path = args.HomographyNPY
            M = np.load(homography_path, allow_pickle=True)[0]

        elif method == "DSM":
            # load OrthophotoTiffile
            orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
            # creat/load raster
            if not os.path.exists(args.ToGroundRaster):
                coords = DSM.load_dsm_points(args)
                intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
                projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
                DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
            
            with open(args.ToGroundRaster, 'rb') as f:
                GroundRaster = DSM.pickle.load(f)

        roi_rep = []
        for p in args.MetaData["roi"]:
            x , y = p[0], p[1]
            new_point = Homography.reproj_point(args, x, y, method,
                        M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
            roi_rep.append([new_point[0], new_point[1]])

        self.pg = MyPoly(roi_rep, args.MetaData["roi_group"])
        self.poly_path = mplPath.Path(np.array(roi_rep))

        # df['trajectory'] = df['trajectory'].apply(lambda x: get_in_roi_points(x, self.poly_path, return_mask=False))

        self.typical_mois = defaultdict(list)
        for index, row in df.iterrows():
            self.typical_mois[row['moi']].append(get_in_roi_points(row["trajectory"], self.poly_path, return_mask=False))

    def init_image(self, args):
        self.CountMetric = args.CountMetric
        self.metric = Metric_Dict[args.CountMetric]
        self.args = args
        if self.args.LabelledTrajectories is None:
            print("loaded track labelling from previous path")
            # load typical tracks from image
            validated_trakcs_path = self.args.TrackLabellingExportPthImage
        else: validated_trakcs_path = self.args.LabelledTrajectories; print("loaded track labelling from external source")

        df = pd.read_pickle(validated_trakcs_path)
        df['index_mask'] = df['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        df["trajectory"] = df.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        df["trajectory"] = df.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)

        roi_rep = [p for p in args.MetaData["roi"]]
        self.pg = MyPoly(roi_rep, args.MetaData["roi_group"])
        self.poly_path = mplPath.Path(np.array(roi_rep))
        # df['trajectory'] = df['trajectory'].apply(lambda x: get_in_roi_points(x, self.poly_path, return_mask=False))

        self.typical_mois = defaultdict(list)
        for index, row in df.iterrows():
            self.typical_mois[row['moi']].append(get_in_roi_points(row["trajectory"], self.poly_path, return_mask=False))
    
    def counting(self, current_trajectory, track_id=-1):
        # only get the points in intersectin ROI
        resampled_trajectory = get_in_roi_points(current_trajectory, self.poly_path, return_mask=False)
        # this method of counting should only work for tracks that move in roi
        if len(resampled_trajectory) <= 1:
            return -1
        
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

        tem = np.array(tem).reshape(-1,)
        key = np.array(key).reshape(-1,)
        if len(tem) > 1:
            idxs = np.argpartition(tem, 1)
            votes = key[idxs[:1]]
        else:
            votes = key
        matched_id = int(np.argmax(np.bincount(votes)))

        if self.args.CountVisPrompt:
            for t , k in zip(tem, key):
                print(f"tem = {t}, key = {k}")
            print(f"matched id:{matched_id}")
            self.viz_CMM(resampled_trajectory, matched_id=matched_id, track_id = track_id)
            
        return matched_id

    def viz_CMM(self, current_track, alpha=0.3, matched_id=0, track_id=-1):
        input("enter smth to plot")
        if self.gp:
            image_path = self.args.HomographyTopView
            thickness = 1
            radius = 2
            r = meter_per_pixel(self.args.MetaData['center'])
        else: 
            image_path = self.args.HomographyStreetView
            thickness = 5
            radius = 10
            r = 1

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
                        img = cv2.line(img, (x1, y1), (x2, y2), c, thickness=thickness) 
                    for p in gt_trajectory:
                        x, y = int(p[0]/r), int(p[1]/r)
                        c = color_dict[keys]
                        img = cv.circle(img, (x,y), radius=radius, color=c, thickness=thickness)
                else: 
                    for i in range(1, len(gt_trajectory)):
                        p1 = gt_trajectory[i-1]
                        p2 = gt_trajectory[i]
                        x1, y1 = int(p1[0]/r), int(p1[1]/r)
                        x2, y2 = int(p2[0]/r), int(p2[1]/r)
                        c = color_dict[keys]
                        back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), c, thickness=thickness) 

                    for p in gt_trajectory:
                        x, y = int(p[0]/r), int(p[1]/r)
                        c = color_dict[keys]
                        back_ground = cv.circle(back_ground, (x,y), radius=radius, color=c, thickness=thickness)

        for i in range(1, len(current_track)):
            p1 = current_track[i-1]
            p2 = current_track[i]
            x1, y1 = int(p1[0]/r), int(p1[1]/r)
            x2, y2 = int(p2[0]/r), int(p2[1]/r)
            back_ground = cv2.line(back_ground, (x1, y1), (x2, y2), (240, 50, 230), thickness=thickness) 

        for p in current_track:
            x, y = int(p[0]/r), int(p[1]/r)
            back_ground = cv.circle(back_ground, (x,y), radius=radius, color=(240, 50, 230), thickness=thickness)

        p = current_track[0]
        x, y = int(p[0]/r), int(p[1]/r)
        back_ground = cv.circle(back_ground, (x,y), radius=radius, color=(0, 255, 0), thickness=thickness)

        p = current_track[-1]
        x, y = int(p[0]/r), int(p[1]/r)
        back_ground = cv.circle(back_ground, (x,y), radius=radius, color=(0, 0, 255), thickness=thickness)

        img_new = cv2.addWeighted(img, alpha, back_ground, 1 - alpha, 0)
        img_new = cv.cvtColor(img_new, cv.COLOR_BGR2RGB)
        plt.imshow(img_new)
        plt.title(f"id;{track_id}")
        plt.savefig(f"couting_with_{self.CountMetric}_id_{track_id}_matched_{matched_id}.png")
        plt.show()


    def main(self, args=None, verbose=True):
        if args is None:
            args = self.args

        counter = defaultdict(int)
        traject_couter = 0

        # file_path to all trajectories .txt file(at the moment
        # ** do not confuse it with selected trajectories
        df_meter = pd.read_pickle(args.ReprojectedPklMeter)
        df       = pd.read_pickle(args.ReprojectedPkl)

        result_paht = args.CountingResPth

        # get deffrent trajectory reps depending on if gp or image reasoning
        if self.gp:
            df_meter = group_tracks_by_id(df_meter, gp=True, verbose=verbose)
            df       = group_tracks_by_id(df, gp=True, verbose=verbose)
        else:
            df_meter = group_tracks_by_id(df_meter, gp=False, verbose=verbose)
            df       = group_tracks_by_id(df, gp=False, verbose=verbose)
            
        # resample tracks
        df['index_mask'] = df_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        df["trajectory"] = df.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        df["trajectory"] = df.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)

        data = {}
        data['id'], data['moi'] = [], []
        tids = np.unique(df['id'].tolist())

        for idx in tqdm(tids, disable= not verbose):
            current_track = df[df['id'] == idx]
            a = current_track['trajectory'].values.tolist()
            matched_moi = self.counting(a[0], track_id=idx)
            data['id'].append(idx)
            data['moi'].append(matched_moi)

            counter[matched_moi] += 1
            traject_couter += 1
        if verbose:
            for i in range(12):
                print(f"{i+1}: {counter[i+1]}")
            print(traject_couter)

        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)

        df = pd.DataFrame.from_dict(data)
        df.to_csv(args.CountingIdMatchPth, index=False)

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

class KNNCounting(Counting):
    def __init__(self, args, gp):
        self.gp = gp
        if gp:
            self.init_gp(args)
        else:
            self.init_image(args)

    def init_gp(self, args):
        self.args = args
        self.K = args.K # K in KNN

        if self.args.LabelledTrajectories is None:
            print("loaded track labelling from previous path")
            validated_trakcs_path = self.args.TrackLabellingExportPth
            validated_tracks_path_meter = self.args.TrackLabellingExportPthMeter
        else: validated_trakcs_path = self.args.LabelledTrajectories; print("loaded track labelling from external source")

        df = pd.read_pickle(validated_trakcs_path)
        df_meter = pd.read_pickle(validated_tracks_path_meter)
        # resample tracks on gp
        df['index_mask'] = df_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        df["trajectory"] = df.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        df["trajectory"] = df.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)


        # add prototype tracks as KNN data points
        x = []
        y = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="building KNN data"):
            moi = row.moi
            for p in row.trajectory:
                x.append(p)
                y.append(moi)
        x = np.array(x)
        y = np.array(y)
        self.knn = KNeighborsClassifier(n_neighbors=self.K)
        self.knn.fit(x, y)

        img = args.HomographyTopView
        img = cv.imread(args.HomographyTopView)
        if args.CountVisDensity:
            # self.plot_train_samples(img, x, y, args.DensityVisPath)
            self.plot_boundries(img, args.DensityVisPath)

    def init_image(self, args):
        self.args = args
        self.K = args.K # K in KNN

        if self.args.LabelledTrajectories is None:
            print("loaded track labelling from previous path")
            validated_trakcs_path = self.args.TrackLabellingExportPthImage
        else: validated_trakcs_path = self.args.LabelledTrajectories; print("loaded track labelling from external source")

        df = pd.read_pickle(validated_trakcs_path)
        # resample tracks on gp
        df['index_mask'] = df['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
        df["trajectory"] = df.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        df["trajectory"] = df.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)

        # add prototype tracks as KNN data points
        x = []
        y = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="building KNN data"):
            moi = row.moi
            for p in row.trajectory:
                x.append(p)
                y.append(moi)
        x = np.array(x)
        y = np.array(y)
        self.knn = KNeighborsClassifier(n_neighbors=self.K)
        self.knn.fit(x, y)

        img = cv.imread(args.HomographyStreetView)
        if args.CountVisDensity:
            # self.plot_train_samples(img, x, y, args.DensityVisPath)
            self.plot_boundries(img, args.DensityVisPath)


    def plot_boundries(self, img, vis_path):
        from matplotlib import ticker
        h, w, c = img.shape
        ys, xs  = np.linspace(0, h, num=int(h), endpoint=False), np.linspace(0, w, num=int(w), endpoint=False)
        xx, yy = np.meshgrid(xs, ys)
        xf , yf = xx.flatten(), yy.flatten()
        X = np.vstack([xf, yf]).T
        # scores = np.zeros(shape = (len(ys), len(xs)))

        scores = self.knn.predict(X)
        scores = scores.reshape(xx.shape)

        plt.imshow(img)
        plt.contourf(xs, ys, scores, alpha=0.5, cmap="gist_ncar", levels=24)
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=12)
        cb.locator = tick_locator
        cb.update_ticks()
        plt.title(f"K: {self.K}")
        plt.savefig(vis_path+f"Boundries.png")
        plt.close("all")

    def plot_train_samples(self, img, x, y, vis_path):
        plt.imshow(img)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.title(f"K = {self.K}")
        plt.savefig(vis_path+f"{self.K}NN_TrainSamples.png")
        plt.close("all")

    def main(self, args=None, verbose=True):
        if args is None:
            args = self.args

        tracks_path = args.ReprojectedPkl
        tracks_meter_path = args.ReprojectedPklMeter
        result_paht = args.CountingResPth

        if self.gp:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True, verbose=verbose)
            tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True, verbose=verbose)
            tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        else:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=False, verbose=verbose)
            tracks['index_mask'] = tracks['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)

        tracks["trajectory"] = tracks.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)
        
        moi = []
        for i, row in tqdm(tracks.iterrows(), total=len(tracks), desc="knn classifier infer", disable=not verbose):
            x = [p for p in row.trajectory]
            y = self.knn.predict(x)
            y_star = np.bincount(y).argmax()
            moi.append(y_star)
        tracks["moi"] = moi

        # save id matcing results
        counted_tracks  = tracks[["id", "moi"]]
        counted_tracks.to_csv(args.CountingIdMatchPth, index=False)

        # save coutning results
        counter = defaultdict(int)
        for moi in range(1, 13):
            counter[moi]  = 0
        for i, row in counted_tracks.iterrows():
                counter[int(row["moi"])] += 1
        if verbose:        
            print(counter)
        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)
        
# change IKDE code to propagate bw to 2D KDE instead of hardcoding. this is important for Image Counters @TODO
class IKDE():
    def __init__(self, bandwidth, os_ratio, kernel="gaussian"):
        self.kernel = kernel
        self.bw = bandwidth
        self.osr = os_ratio

    def del_kdes(self):
        del self.kdes

    def fit(self, tracks, img):
        self.mois = np.unique(tracks["moi"])
        self.kdes = {}
        self.priors = {}
        self.n_prototype = {}
        self.infer_maps = {}
        for moi in self.mois:
            # self.kdes[moi] = sklearn.neighbors.KernelDensity(kernel=self.kernel, bandwidth=self.bw)
            self.kdes[moi] = KDE2D()
            self.priors[moi] = 0
            self.infer_maps[moi] = None
            self.n_prototype[moi] = 0

        sum_temp = 0
        for moi in tqdm(self.mois, desc="computing priors"):
            num_traj_per_moi = len(tracks[tracks["moi"] == moi])
            self.n_prototype[moi] = num_traj_per_moi
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                traj = row["trajectory"]
                self.priors[moi] += len(traj)/num_traj_per_moi 
                sum_temp += len(traj)/num_traj_per_moi 
        for moi in self.mois:
            self.priors[moi] /= sum_temp

        for moi in tqdm(self.mois, desc="fit KDEs"):
            kde_data = []
            sequence_data = []
            for i, row in tracks[tracks["moi"] == moi].iterrows():
                traj = row["trajectory"]
                # if row["moi"] == 6:
                #     my_viz_CMM(traj, img, matched_id = row["moi"], track_id = row["id"])
                # total points to add
                tot_ps_to_add = len(traj) * self.osr
                # compute arch length of traj
                t1 = np.array(traj[1:])
                t2 = np.array(traj[:-1])
                distances = np.linalg.norm(t1 - t2, axis=1)
                arc = np.sum(distances)
                num_points = distances / arc * tot_ps_to_add

                for i in range(1, len(traj)):
                    p1, p2 = traj[i-1], traj[i]
                    for r in np.linspace(0, 1, num=int(num_points[i-1]+0.5)):
                        p_temp = (1-r)*p1 + r*p2
                        x, y = p_temp
                        kde_data.append([x, y])

            kde_data = np.array(kde_data)
            self.kdes[moi].fit(kde_data, img, self.bw)

    def make_inference_maps(self, img, vis_path):
        for moi in tqdm(self.kdes.keys(), desc="estimating inference maps"):
            self.make_infer_map(moi, img, vis_path)

    def make_infer_map(self, moi, img, vis_path):
        kde = self.kdes[moi]
        h, w, c = img.shape
        # compute infer map per pixel on image
        y, x  = np.linspace(0, h, num=int(h), endpoint=False), np.linspace(0, w, num=int(w), endpoint=False)
        xx, yy = np.meshgrid(x, y)
        xf , yf = xx.flatten(), yy.flatten()
        X = np.vstack([xf, yf]).T

        scores = kde.score_samples(X)
        scores = scores.reshape(xx.shape)

        self.infer_maps[moi] = self.get_infer_map_dict(xx, yy, scores)

    def get_infer_map_dict(self, xx, yy, ss):
        dict_map = {}
        for x, y, s in zip(xx.flatten(), yy.flatten(), ss.flatten()):
            dict_map[(x, y)] = s

        return dict_map

    def plot_densities(self, img, vis_path):
        for moi in tqdm(self.kdes.keys(), desc="vis KDEs"):
            self.plot_density(moi, img, vis_path)

    def plot_density(self, moi, img, vis_path):
        infer_map = self.infer_maps[moi]
        h, w, c = img.shape
        ys, xs  = np.linspace(0, h, num=int(h), endpoint=False), np.linspace(0, w, num=int(w), endpoint=False)
        scores = np.zeros(shape = (len(ys), len(xs)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                scores[j, i] = infer_map[(x, y)]

        plt.imshow(img)
        plt.axis("off")
        plt.contourf(xs, ys, scores, alpha=0.5, levels=100, cmap='plasma')
        # plt.colorbar()
        plt.title(f"prototpyes used: {self.n_prototype[moi]}")
        plt.savefig(vis_path+f"logdensity_{moi}.png", bbox_inches='tight')
        plt.close("all")

        scores = np.exp(scores)
        plt.imshow(img)
        plt.axis("off")
        plt.contourf(xs, ys, scores, alpha=0.5, levels=100, cmap="plasma")
        # plt.colorbar()
        plt.title(f"prototpyes used: {self.n_prototype[moi]}")
        plt.savefig(vis_path+ f"density_{moi}.png", bbox_inches='tight')
        plt.close("all")

    def get_traj_score(self, traj, moi):
        ## use infer maps to get estimates
        infer_map = self.infer_maps[moi]
        log_score = 0
        for i in range(len(traj)):
            x, y = traj[i]
            x , y = int(x), int(y) # to have map indexes
            try:
                log_score += infer_map[(x, y)]
            except:
                pass

        return log_score
    
    def predict_traj(self, traj):
        moi_scores = []
        for moi in self.mois:
            moi_scores.append(self.get_traj_score(traj, moi))
        max_moi = self.mois[np.argmax(moi_scores)]
        return max_moi

    def predict_tracks(self, tracks, verbose=True):
        max_mois = []
        for i, row in tqdm(tracks.iterrows(), desc="pred moi", total=len(tracks), disable=not verbose):
            traj = row["trajectory"]
            max_mois.append(self.predict_traj(traj))
        return max_mois

# class LOSIKDE(IKDE):
#     def __init__(self, kernel="gaussian", bandwidth=3.2, os_ratio = 2):
#         super().__init__(kernel, bandwidth, os_ratio)

#     def fit(self, tracks):
#         self.mois = np.unique(tracks["moi"])
#         self.kdes = {}
#         for moi in self.mois:
#             self.kdes[moi] = sklearn.neighbors.KernelDensity(kernel=self.kernel, bandwidth=self.bw)
#         for moi in self.mois:
#             kde_data = []
#             sequence_data = []
#             for i, row in tracks[tracks["moi"] == moi].iterrows():
#                 traj = row["trajectory"]
#                 for i in range(1, len(traj)):
#                     p1, p2 = traj[i-1], traj[i]
#                     for r in np.linspace(0, 1, num=self.osr):
#                         p_temp = (1-r)*p1 + r*p2
#                         sequence_data.append(p_temp)
#                 for i , p in enumerate(sequence_data):
#                     x, y = p
#                     kde_data.append([x, y, i/len(sequence_data)])
#             kde_data = np.array(kde_data)
#             self.kdes[moi].fit(kde_data)

#     def get_traj_score(self, traj, moi):
#         traj_data = []
#         for i in range(len(traj)):
#             x, y = traj[i]
#             traj_data.append([x, y, i/len(traj)])
#         traj_data = np.array(traj_data)
#         return np.sum(self.kdes[moi].score_samples(traj_data))

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


class KDECounting(Counting):
    def __init__(self, args, gp):

        self.gp = gp
        if gp:
            self.img_path = args.HomographyTopView
        else:
            self.img_path = args.HomographyStreetView
        self.init(args)

    def init(self, args):
        self.args = args
        tracks = get_proper_tracks(self.args, gp = self.gp)
        # cluster prop_tracks based on their starting point
        tracks = cluster_prop_tracks_based_on_str(tracks, self.args)
        # get same number of longest tracks from each cluster per moi
        # uncomment this line if you want to use the same anumber of prototypes per cluster
        # tracks = make_uniform_clt_per_moi(tracks, self.args)


        # resample on the ground plane but not in meter
        tracks["trajectory"] = tracks.apply(lambda x: x['trajectory'][x["index_mask"]], axis=1)
        tracks["frames"] = tracks.apply(lambda x: x['frames'][x["index_mask"]], axis=1)
        # interpolate tracks to fill in gaps
        # os_trajectories  = []
        # os_frames        = []
        # for i, row in tracks.iterrows():
        #     traj, frames = interpolate_traj(row["trajectory"], row["frames"])
        #     os_trajectories.append(traj)
        #     os_frames.append(frames)
        # tracks["trajectory"] = os_trajectories
        # tracks["frames"] = os_frames
            
        if self.args.CountMetric == "kde":
            self.ikde = IKDE(self.args.KDEBW, self.args.OSR)
        elif self.args.CountMetric == "gkde":
            self.ikde = IKDE(self.args.KDEBW, self.args.OSR)
        elif self.args.CountMetric == "hmmg":
            self.ikde = HMMG()
        else: raise "it should not happen"

        img = cv.imread(self.img_path)
        self.ikde.fit(tracks, img)
        self.ikde.make_inference_maps(img, args.DensityVisPath)
        if args.CountVisDensity:
            self.ikde.plot_densities(img, args.DensityVisPath)
        # delete original kde for each moi to free up space when you are caching object
        self.ikde.del_kdes()

    def main(self, args=None, verbose=True):
        # where the counting happens
        if args is not None:
            self.args = args
        args = self.args
        tracks_path = args.ReprojectedPkl
        tracks_meter_path = args.ReprojectedPklMeter
        top_image = args.HomographyTopView
        street_image = args.HomographyStreetView
        meta_data = args.MetaData # dict is already loaded
        HomographyNPY = args.HomographyNPY
        result_paht = args.CountingResPth
        # load data
        M = np.load(HomographyNPY, allow_pickle=True)[0]
        if self.gp:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True,  verbose=verbose)
            tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True, verbose=verbose)
            tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            img = plt.imread(top_image)
            img1 = cv.imread(top_image)
        else:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=False)
            tracks['index_mask'] = tracks['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            img = plt.imread(street_image)
            img1 = cv.imread(street_image)

        # resample gp tracks
        tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        tracks["trajectory"] = tracks.apply(lambda x: over_sample(x.trajectory, self.args.OSR), axis=1)
        tracks["moi"] = self.ikde.predict_tracks(tracks, verbose=verbose)

        counted_tracks  = tracks[["id", "moi"]]
        counted_tracks.to_csv(args.CountingIdMatchPth, index=False)

        counter = {}
        for moi in range(1, 13):
            counter[moi]  = 0
        for i, row in counted_tracks.iterrows():
                counter[int(row["moi"])] += 1
        if verbose:
            print(counter)
        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)

        # get roi on groundpalane
        if self.gp:
            roi_rep = []
            for p in args.MetaData["roi"]:
                point = np.array([p[0], p[1], 1])
                new_point = M.dot(point)
                new_point /= new_point[2]
                roi_rep.append([new_point[0], new_point[1]])
        else:
            roi_rep = [p for p in args.MetaData["roi"]]
            
        pg = MyPoly(roi_rep, args.MetaData["roi_group"])
        th = args.MetaData["roi_percent"] * np.sqrt(pg.area)

        if self.args.CountVisPrompt:
            print("will initiate count vis prompt")
            for i, row in tracks.iterrows():
                    self.plot_track_on_gp(row["trajectory"], matched_id=row["moi"], track_id=row["id"], roi=roi_rep, roi_th=th, pg=pg)

    def add_roi_line_to_img(self,img, roi):
        for i in range(1, len(roi)):
            x1, y1 = roi[i-1]
            x2, y2 = roi[i]
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=2)
        return img
    
    def add_roi_gap_to_img(self,img, pg, roi_th):
        alpha = 0.2
        bg_img = copy.deepcopy(img)
        for y in range(bg_img.shape[0]):
            for x in range(bg_img.shape[1]):
                d, i = pg.distance([x, y])
                if d > roi_th:
                    bg_img[y, x] = 0
        return cv2.addWeighted(img, alpha, bg_img, 1 - alpha, 0)


    def plot_track_on_gp(self, current_track, matched_id=0, alpha=0.4, track_id=None, roi=None, roi_th=None, pg=None):
        c = color_dict[int(matched_id)]
        image_path = self.image_path
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
            # if (roi_th is not None) and (pg is not None):
                # back_ground = self.add_roi_gap_to_img(back_ground, pg, roi_th)

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

class KDECountingMulti(KDECounting):
    def __init__(self, args, args_mcs, gp):
        assert gp == True
        self.gp = gp
        if gp:
            self.img_path = args.HomographyTopView
        else:
            raise NotImplementedError
        self.init(args, args_mcs, verbose=False)

    def init(self, args, args_mcs, verbose=True):
        self.args = args
        tracks = get_proper_tracks_multi(args_mcs, gp = self.gp, verbose=verbose)
        # cluster prop_tracks based on their starting point
        tracks = cluster_prop_tracks_based_on_str(tracks, self.args)

        # resample on the ground plane but not in meter
        tracks["trajectory"] = tracks.apply(lambda x: x['trajectory'][x["index_mask"]], axis=1)
        tracks["frames"] = tracks.apply(lambda x: x['frames'][x["index_mask"]], axis=1)

        if self.args.CountMetric == "kde":
            raise NotImplementedError
            # it only supports ground plane counting
        elif self.args.CountMetric == "gkde":
            self.ikde = IKDE(self.args.KDEBW, self.args.OSR)

        else: raise "it should not happen. undefined counter metric"

        img = cv.imread(self.img_path)
        self.ikde.fit(tracks, img)
        self.ikde.make_inference_maps(img, args.DensityVisPath)
        if args.CountVisDensity:
            self.ikde.plot_densities(img, args.DensityVisPath)
        # delete original kde for each moi to free up space when you are caching object
        self.ikde.del_kdes()

class ROICounting(KDECounting):
    def __init__(self, args, gp):
        self.gp = gp
        if gp:
            self.init_gp(args)
        else:
            self.init_image(args)


    def init_gp(self, args):
        self.args = args
        # load ROI 
        meta_data = args.MetaData # dict is already loaded
        # HomographyNPY = args.HomographyNPY
        # self.M = np.load(HomographyNPY, allow_pickle=True)[0]
        roi_rep = []
        # create roi polygon according to backproj method
        method = args.BackprojectionMethod
        M = None
        GroundRaster = None
        orthophoto_win_tif_obj = None

        if method == "Homography":
            homography_path = args.HomographyNPY
            M = np.load(homography_path, allow_pickle=True)[0]

        elif method == "DSM":
            # load OrthophotoTiffile
            orthophoto_win_tif_obj, __ = DSM.load_orthophoto(args.OrthoPhotoTif)
            # creat/load raster
            if not os.path.exists(args.ToGroundRaster):
                coords = DSM.load_dsm_points(args)
                intrinsic_dict = DSM.load_json_file(args.INTRINSICS_PATH)
                projection_dict = DSM.load_json_file(args.EXTRINSICS_PATH)
                DSM.create_raster(args, args.MetaData["camera"], coords, projection_dict, intrinsic_dict)
            
            with open(args.ToGroundRaster, 'rb') as f:
                GroundRaster = DSM.pickle.load(f)

        for p in args.MetaData["roi"]:
            x , y = p[0], p[1]
            new_point = Homography.reproj_point(args, x, y, method,
                        M=M, GroundRaster=GroundRaster, TifObj = orthophoto_win_tif_obj)
            roi_rep.append([new_point[0], new_point[1]])

        self.pg = MyPoly(roi_rep, args.MetaData["roi_group"])
        self.poly_path = mplPath.Path(np.array(roi_rep))
        self.th = self.args.MetaData["roi_percent"] * np.sqrt(self.pg.area)
        self.M = M
        self.GroundRaster = GroundRaster
        self.orthophoto_win_tif_obj = orthophoto_win_tif_obj

    def init_image(self, args):
        self.args = args
        # load ROI 
        meta_data = args.MetaData # dict is already loaded
        HomographyNPY = args.HomographyNPY
        self.M = np.load(HomographyNPY, allow_pickle=True)[0]

        # roi rep on image domain
        roi_rep = [p for p in args.MetaData["roi"] ]

        self.pg = MyPoly(roi_rep, args.MetaData["roi_group"])
        self.poly_path = mplPath.Path(np.array(roi_rep))
        self.th = self.args.MetaData["roi_percent"] * np.sqrt(self.pg.area)

    def main(self, args=None, verbose=True):
        if args is None:
            args = self.args
        pg = self.pg
        tracks_path = args.ReprojectedPkl
        tracks_meter_path = args.ReprojectedPklMeter
        result_paht = args.CountingResPth

        if self.gp:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=True, verbose=verbose)
            tracks_meter = group_tracks_by_id(pd.read_pickle(tracks_meter_path), gp=True, verbose=verbose)
            tracks['index_mask'] = tracks_meter['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)
        else:
            tracks = group_tracks_by_id(pd.read_pickle(tracks_path), gp=False)
            tracks['index_mask'] = tracks['trajectory'].apply(lambda x: track_resample(x, return_mask=True, threshold=args.ResampleTH))
            tracks["trajectory"] = tracks.apply(lambda x: x["trajectory"][x["index_mask"]], axis=1)

        # depending on tracks select str and end roi edge
        i_strs = []
        i_ends = []
        moi = []
        for i, row in tqdm(tracks.iterrows(), total=len(tracks), disable=not verbose):
            traj = row["trajectory"]

            if Track.within_roi(self.pg, traj, self.th, self.poly_path):
                d_end, i_end = pg.distance(traj[-1])
                # need to find the closest edge to track start
                # the edge should be different from ending edge
                ds_str, is_str = pg.all_distances(traj[0])
                paired_sorted = sorted(zip(ds_str, is_str))
                ds_sorted, is_reorganized = zip(*paired_sorted)
                ds_sorted = list(ds_sorted)
                is_reorganized = list(is_reorganized)
                for d_s, i_s in zip(ds_sorted, is_reorganized):
                    if not i_s == i_end:
                        d_str, i_str = d_s, i_s

            elif Track.cross_roi_multiple(self.pg, traj, self.th, self.poly_path):
                int_indxes = pg.doIntersect(traj, ret_points=False)
                i_str , i_end = int_indxes[0], int_indxes[-1]

            elif Track.just_enter_roi(self.pg, traj, self.th, self.poly_path):
                int_indxes = pg.doIntersect(traj, ret_points=False)
                i_str = int_indxes[0]

                # find the edge closest to end of track
                # it can not be the edge we had for track beginning
                # Pair the elements of a and b, and sort the pairs based on elements of a
                # Unzip the pairs back into two lists
                # Convert the tuples back to lists if needed
                ds_end , is_end = pg.all_distances(traj[-1])
                paired_sorted = sorted(zip(ds_end, is_end))
                ds_sorted, is_reorganized = zip(*paired_sorted)
                ds_sorted = list(ds_sorted)
                is_reorganized = list(is_reorganized)
                for d_e, i_e in zip(ds_sorted, is_reorganized):
                    if not i_e == i_str:
                        d_end, i_end = d_e, i_e

            elif Track.just_exit_roi(self.pg, traj, self.th, self.poly_path):
                int_indxes = pg.doIntersect(traj, ret_points=False)
                i_end = int_indxes[-1]

                # need to find the closest edge to track start
                # the edge should be different from ending edge
                ds_str, is_str = pg.all_distances(traj[0])
                paired_sorted = sorted(zip(ds_str, is_str))
                ds_sorted, is_reorganized = zip(*paired_sorted)
                ds_sorted = list(ds_sorted)
                is_reorganized = list(is_reorganized)
                for d_s, i_s in zip(ds_sorted, is_reorganized):
                    if not i_s == i_end:
                        d_str, i_str = d_s, i_s

            else:
                d_end, i_end = pg.distance(traj[-1])
                # need to find the closest edge to track start
                # the edge should be different from ending edge
                ds_str, is_str = pg.all_distances(traj[0])
                paired_sorted = sorted(zip(ds_str, is_str))
                ds_sorted, is_reorganized = zip(*paired_sorted)
                ds_sorted = list(ds_sorted)
                is_reorganized = list(is_reorganized)
                for d_s, i_s in zip(ds_sorted, is_reorganized):
                    if not i_s == i_end:
                        d_str, i_str = d_s, i_s

            i_strs.append(i_str)
            i_ends.append(i_end)
            moi.append(str_end_to_moi(i_str, i_end))


        tracks['i_str'] = i_strs
        tracks['i_end'] = i_ends
        tracks['moi'] = moi
        counted_tracks  = tracks[["id", "moi"]][tracks["moi"]!=-1]
        counted_tracks.to_csv(args.CountingIdMatchPth, index=False)


        counter = {}
        for moi in range(1, 13):
            counter[moi]  = 0
        for i, row in counted_tracks.iterrows():
                counter[int(row["moi"])] += 1
        if verbose:
            print(counter)
        with open(result_paht, "w") as f:
            json.dump(counter, f, indent=2)

        if self.args.CountVisPrompt:
            for i, row in tracks.iterrows():
                self.plot_track_on_gp(row["trajectory"], matched_id=row["moi"])      

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
    data["estimated"] = [estimated[i] if i in estimated else 0 for i in args.MetaData["gt"].keys()]
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
    df = pd.concat([df,df2], ignore_index=True)
    df.to_csv(args.CountingStatPth, index=False)
    return SucLog("counts evaluated")

def eval_count_multi(args, args_mcs):
    # evaluate each video first
    # to get metrics
    # then call another function for evaluate on multi args level
    args_flat = flatten_args(args_mcs)
    stats = []
    if args.EvalCount or args.EvalCountMSfromMC:
        for arg_i in tqdm(args_flat):
            eval_count(arg_i)
            stats.append(pd.read_csv(arg_i.CountingStatPth))
    
    df_multi = copy.deepcopy(stats[0])
    cols_to_sum = ["gt", "estimated", "diff"]
    for stat in stats[1:]:
        for col in cols_to_sum:
            df_multi[col] = df_multi[col] + stat[col]
    df_multi["err"]  = df_multi["diff"]/df_multi["gt"]

    df_multi.to_csv(args.CountingStatPth, index=False)
    return SucLog("counts Multi evaluated")

def eval_count_multi_per_camera(args, args_mcs):
    # evaluate each video first
    # to get metrics
    # then call another function for evaluate on multi args level
    def accumulate(stats):
        df_multi = copy.deepcopy(stats[0])
        cols_to_sum = ["gt", "estimated", "diff"]
        for stat in stats[1:]:
            for col in cols_to_sum:
                df_multi[col] = df_multi[col] + stat[col]
        df_multi["err"]  = df_multi["diff"]/df_multi["gt"]
        return df_multi

    args_flat = flatten_args(args_mcs)
    stats = []
    cams = []

    for arg_i in tqdm(args_flat):
        stats.append(pd.read_csv(arg_i.CountingStatPth))

    for arg_i in tqdm(args_flat):
        cams.append(arg_i.SubID[-3:])

    cam_based_df_data = {}
    unique_cams = np.unique(cams)
    for cam in unique_cams:
        stats_cam = []
        for cam_i, stat_i in zip(cams, stats):
            if cam_i == cam:
                stats_cam.append(stat_i)
        df_multi_cam = accumulate(stats_cam)
        if not "moi" in cam_based_df_data: cam_based_df_data["moi"] = df_multi_cam["moi"].tolist()
        cam_based_df_data[cam] = df_multi_cam["err"].tolist()

    df_final = pd.DataFrame.from_dict(cam_based_df_data)
    df_final.to_csv(args.CountingStatCameraBasedPth, index=False)

    # add another csvfile for total tracks per camera
    args_flat = flatten_args(args_mcs)
    stats = []
    cams = []

    for arg_i in tqdm(args_flat):
        stats.append(pd.read_csv(arg_i.CountingStatPth))

    for arg_i in tqdm(args_flat):
        cams.append(arg_i.SubID[-3:])

    cam_based_df_data = {}
    unique_cams = np.unique(cams)
    for cam in unique_cams:
        stats_cam = []
        for cam_i, stat_i in zip(cams, stats):
            if cam_i == cam:
                stats_cam.append(stat_i)
        df_multi_cam = accumulate(stats_cam)
        if not "moi" in cam_based_df_data: cam_based_df_data["moi"] = df_multi_cam["moi"].tolist()
        cam_based_df_data[cam] = df_multi_cam["estimated"].tolist()

    df_final = pd.DataFrame.from_dict(cam_based_df_data)
    df_final.to_csv(args.CountingStatCameraBasedPth+".TotalCounts.csv", index=False)

    return SucLog("counts Cameara Multi evaluated")

def main(args):
    # some relative path form the args
    # args.ReprojectedPklMeter
    # args.TrackLabellingExportPthMeter

    # check if use cached counter
    if args.UseCachedCounter:
        with open(args.CachedCounterPth, "rb") as f:
            counter = pkl.load(f)
    else:
        if   args.CountMetric == "kde" :
            counter = KDECounting(args, gp=False)
        elif args.CountMetric == "gkde":
            counter = KDECounting(args, gp=True)
        elif args.CountMetric == "groi":
            counter = ROICounting(args, gp=True)
        elif args.CountMetric == "roi":
            counter = ROICounting(args, gp=False)
        elif args.CountMetric == "gknn":
            counter = KNNCounting(args, gp=True)
        elif args.CountMetric == "knn":
            counter = KNNCounting(args, gp=False)
        else:
            if args.CountMetric[0] == "g":
                counter = Counting(args, gp=True)
            else:
                counter = Counting(args, gp=False)

    # perfom counting here
    counter.main(args)

    # save counter object for later use
    if args.CacheCounter:
        with open(args.CachedCounterPth, "wb") as f:
            print(f"counter being saved to {args.CachedCounterPth}")
            pkl.dump(counter, f)

    return SucLog("counting part executed successfully")

def mainMulti(args, args_mcs):

    # will save counters both on args level(eg MS)
    # and for each subarg(eg videos)
    # only applicable for ground based counting for now
    args_flat = flatten_args(args_mcs)

    # check if use cached counter
    if args.UseCachedCounter:
        with open(args.CachedCounterPth, "rb") as f:
            counter = pkl.load(f)
    else:
        if args.CountMetric == "gkde":
            counter = KDECountingMulti(args, args_mcs, gp=True)
        elif args.CountMetric == "groi":
            counter = ROICounting(args, gp=True)
        elif args.CountMetric == "gknn":
            counter = KNNCounting(args, gp=True)
        else:
            if args.CountMetric[0] == "g":
                counter = Counting(args, gp=True)
            else:
                # only supports gournd based counting
                raise NotImplementedError

    # perfom counting here
    for arg_i in tqdm(args_flat):
        counter.main(arg_i, verbose=False)

    # save counter object for later use
    # save counters both on args level
    # and on Multi level for later use
    if args.CacheCounter:
        with open(args.CachedCounterPth, "wb") as f:
            print(f"counter being saved to {args.CachedCounterPth}")
            pkl.dump(counter, f)
        for arg_i in args_flat:
            with open(arg_i.CachedCounterPth, "wb") as f:
                print(f"counter being saved to {arg_i.CachedCounterPth}")
                pkl.dump(counter, f)

    return SucLog("counting Multi executed successfully")