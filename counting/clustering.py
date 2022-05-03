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
from resample_gt_MOI.resample_typical_tracks import track_resample
from tqdm import tqdm
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation
import pickle as pkl

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

def viz_all_tracks(all_tracks, labels, save_path):
    image_path = "./../../Dataset/DundasStAtNinthLine.jpg"
    img = cv.imread(image_path)
    rows, cols, dim = img.shape
    for current_track, current_label in zip(all_tracks, labels):
        if current_label<0: continue
        for p in current_track:
            x, y = int(p[0]), int(p[1])
            np.random.seed(int(current_label))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            img = cv.circle(img, (x,y), radius=2, color=color, thickness=2)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv2.imwrite(save_path, img)


def cmm_ref_distance(index_1, index_2):
    index_1, index_2 = int(index_1), int(index_2)
    global df
    traj_a, traj_b = df['trajectory'].iloc[index_1], df['trajectory'].iloc[index_2]
    return cmm_distance(traj_a, traj_b)


def group_tracks_by_id(tracks_path):
    # this function was writtern for grouping the tracks with the same id
    # usinig this one can load the data from a .txt file rather than .mat file
    tracks = np.loadtxt(tracks_path, delimiter=",")
    all_ids = np.unique(tracks[:, 1])
    data = {"id":[], "trajectory":[], "frames":[]}
    for idd in tqdm(all_ids):
        mask = tracks[:, 1]==idd
        selected_tracks = tracks[mask]
        frames = [selected_tracks[: ,0]]
        id = selected_tracks[0][1]
        trajectory = selected_tracks[:, 2:4]
        data["id"].append(id)
        data["frames"].append(frames)
        data["trajectory"].append(trajectory)
    df = pd.DataFrame(data)
    return df

def cmm_distance(traj_a, traj_b):

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



libfile = "./cmm_truncate_linux/build/lib.linux-x86_64-3.7/cmm.cpython-37m-x86_64-linux-gnu.so"
cmmlib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function cmm
cmmlib.cmm_truncate_sides.restype = ctypes.c_double
cmmlib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            ctypes.c_int, ctypes.c_int]


tracks_path = "./../../Results/GX010069_tracking_sort_reprojected.txt"
df = group_tracks_by_id(tracks_path)
df['trajectory'] = df['trajectory'].apply(lambda x: track_resample(x))
mask = [len(t)>5 for t in df['trajectory']]
df = df[mask]
indexes = np.array([i for i in range(len(df))]).reshape(-1, 1)

# M = np.zeros(shape=(len(indexes), len(indexes)))
# for i in tqdm(indexes):
#     for j in range(int(i)+1, len(indexes)):
#         c =  np.abs(cmm_ref_distance(i, j))
#         M[int(i), int(j)] = c
#         M[int(j), int(i)] = c
# with open("pairdistances.npy", "wb") as f:
#     np.save(f, M)

## Clustering Part
M = None
with open("pairdistances.npy", "rb") as f:
    M = np.load(f)

# clt = DBSCAN(eps = 10, min_samples=5, metric="precomputed")
# labels = clt.fit_predict(M)
# sns.histplot(labels)
# print(np.unique(labels))
# plt.show()
# print(labels)
# viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/DBSCAN.png")

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
clt = AffinityPropagation(damping =0.5 , affinity="precomputed", max_iter=2000)
labels = clt.fit_predict(M)
sns.histplot(labels)
print(np.unique(labels))
plt.show()
print(labels)
viz_all_tracks(df["trajectory"], labels, save_path = "./../../Results/AffinityPropagation.png")
center_indexes = find_centers_MNAVG(df, labels, M)
viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MAVD.png")
center_indexes = find_centers_MXLEN(df, labels, M)
viz_all_tracks(df["trajectory"].iloc[center_indexes], labels[center_indexes], save_path = "./../../Results/Agglomorative_centers_MXLN.png")


# how to find cluster centers ??
# idea 1: for each cluster find the trajectory that has the least average distance from all the other clusters






