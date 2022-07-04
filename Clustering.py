from Libs import *
from Utils import *


import glob
from operator import index
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
from counting.resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import SpectralClustering
import pickle as pkl
import copy

# add your clustering algorithms here
# follow a sklearn-like API for consistency
# this means that your clustering object should have the fit_predict method

clusterers = {}
clusterers["DBSCAN"] = DBSCAN(eps = 0.8, min_samples=2, metric="precomputed")
clusterers["Spectral"] = SpectralClustering(affinity="precomputed_nearest_neighbors", n_clusters=36)

def viz_CMM(current_track):
    image_path = "./../../Dataset/DundasStAtNinthLine.jpg"
    img = cv.imread(image_path)
    rows, cols, dim = img.shape
    for p in current_track:
        x, y = int(p[0]), int(p[1])
        img = cv.circle(img, (x,y), radius=2, color=(70, 255, 70), thickness=2)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def viz_all_tracks(all_tracks, labels, save_path, image_path):
    img = cv.imread(image_path)
    rows, cols, dim = img.shape

    for current_track, current_label in zip(all_tracks, labels):
        if current_label<0: continue
        for p in current_track:
            x, y = int(p[0]), int(p[1])
            np.random.seed(int(current_label))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            img = cv.circle(img, (x,y), radius=2, color=color, thickness=2)

    total_clusters = int(len(np.unique(labels)) - 1)
    # add description for the visualization on the top part of image
    text = f"{total_clusters} clusters annotated with different colors"
    coordinates = (15,15)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0,0,0)
    thickness = 1
    img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
    
    cv.destroyAllWindows()
    if os.path.exists(save_path):
        os.remove(save_path)
    cv2.imwrite(save_path, img)


def cmm_ref_distance(index_1, index_2, df, cmmlib):
    index_1, index_2 = int(index_1), int(index_2)
    traj_a, traj_b = df['trajectory'].iloc[index_1], df['trajectory'].iloc[index_2]
    return cmm_distance(traj_a, traj_b, cmmlib)


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

def cmm_distance(traj_a, traj_b, cmmlib):
    traj_a = traj_a.astype(np.float64)
    traj_b = traj_b.astype(np.float64)
    if traj_a.shape[0] >= traj_b.shape[0]:
        c = cmmlib.cmm_truncate_sides(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0],
                                        traj_b.shape[0])
    else:
        c = cmmlib.cmm_truncate_sides(traj_b[:, 0], traj_b[:, 1], traj_a[:, 0], traj_a[:, 1], traj_b.shape[0],
                                        traj_a.shape[0])
    return c


def find_centers_MNAVG(df, labels, M):
    unique_labels = np.unique(labels)
    center_indexes = []
    for ul in unique_labels:
        if ul < 0 : continue
        M_ul = M[labels==ul, :][:, labels==ul]
        indexes_ul = indexes[labels==ul].reshape(-1,)
        M_ul_avg = np.mean(M_ul, axis=0)
        i = np.argmin(M_ul_avg)
        center_indexes.append(int(indexes_ul[i]))
    return center_indexes

def find_centers_MXLEN(df, labels, M):
    unique_labels = np.unique(labels)
    center_indexes = []
    for ul in unique_labels:
        if ul < 0 : continue
        indexes_ul = indexes[labels==ul].reshape(-1,)
        traj_ul_len = [len(traj) for traj in df["trajectory"].iloc[indexes_ul]]
        i = np.argmax(traj_ul_len)
        center_indexes.append(int(indexes_ul[i]))
    return center_indexes

def cluster(args):
    # variables to be used from the args
        # args.ReprojectedPklMeter
        # args.ReprojectedPkl

    df_meter_ungrouped = pd.read_pickle(args.ReprojectedPklMeter)
    df_reg_ungrouped   = pd.read_pickle(args.ReprojectedPkl)

    # save clustering result to the following args variables 
        # args.ReprojectedPklMeterCluster
        # args.ReprojectedPklCluster
        # args.ClusteringDistanceMatrix
        # args.ClusteringVis

    libfile = "./counting/cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so"
    cmmlib = ctypes.CDLL(libfile)
    # 2. tell Python the argument and result types of function cmm
    cmmlib.cmm_truncate_sides.restype = ctypes.c_double
    cmmlib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                np.ctypeslib.ndpointer(dtype=np.float64),
                                                ctypes.c_int, ctypes.c_int]

    df_meter = group_tracks_by_id(df_meter_ungrouped)
    df_reg   = group_tracks_by_id(df_reg_ungrouped)
    
    df_meter_resampled = copy.deepcopy(df_meter)
    df_reg_resampled   = copy.deepcopy(df_reg)

    df_meter_resampled['trajectory'] = df_meter_resampled['trajectory'].apply(lambda x: track_resample(x))
    df_reg_resampled['trajectory'] = df_reg_resampled['trajectory'].apply(lambda x: track_resample(x))

    indexes = np.array([i for i in range(len(df_meter))]).reshape(-1, 1)

    M = None
    # check if the distance matrix is available first
    if os.path.exists(args.ClusteringDistanceMatrix):
        with open(args.ClusteringDistanceMatrix, "rb") as f:
            M = np.load(f)
    else:
        # compute distance matrix
        M = np.zeros(shape=(len(indexes), len(indexes)))
        for i in tqdm(indexes):
            for j in range(int(i)+1, len(indexes)):
                c =  np.abs(cmm_ref_distance(i, j, df_meter_resampled, cmmlib))
                M[int(i), int(j)] = c
                M[int(j), int(i)] = c
        with open(args.ClusteringDistanceMatrix, "wb") as f:
            np.save(f, M)

    clt = clusterers[args.ClusteringAlgo]
    labels = clt.fit_predict(M)
    sns.histplot(labels)
    print(np.unique(labels))


    # save clustered trackes with labels for both meter and regular
    df_meter['cid'] = labels
    df_meter.to_pickle(args.ReprojectedPklMeterCluster)
    df_reg['cid'] = labels
    df_reg.to_pickle(args.ReprojectedPklCluster)

    viz_all_tracks(df_reg["trajectory"], labels, save_path = args.ClusteringVis, image_path=args.HomographyTopView)
    return SucLog("clustering executed successfully")

# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/DBSCAN_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/DBSCAN_centers_MXLN.png")

##### Agglomorative Clustering
# clt = AgglomerativeClustering(n_clusters = 12, affinity="precomputed", linkage='average')
# labels = clt.fit_predict(M)
# sns.histplot(labels)
# print(np.unique(labels))
# plt.show()
# print(labels)
# viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/Agglomorattive.png")

# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MXLN.png")

#### Affinity Propagation 
# clt = AffinityPropagation(damping =0.5 , affinity="precomputed", max_iter=2000)
# labels = clt.fit_predict(M)
# sns.histplot(labels)
# print(np.unique(labels))
# plt.show()
# print(labels)
# viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/AffinityPropagation.png")
# center_indexes = find_centers_MNAVG(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MAVD.png")
# center_indexes = find_centers_MXLEN(df, labels, M)
# viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MXLN.png")


# how to find cluster centers ??
# idea 1: for each cluster find the trajectory that has the least average distance from all the other clusters






