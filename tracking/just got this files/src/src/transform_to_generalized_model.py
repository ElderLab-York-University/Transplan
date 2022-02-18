import numpy as np
import cv2
import os
import scipy.io
import csv
import numpy as np
import glob
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib import cm
import matplotlib
norm = matplotlib.colors.Normalize(vmin=0, vmax=50)

base_folder = '/media/poorna/D_drive/projects/trans-plan/05March2021/'
intersection_folder = 'Derry Road W at RR25(Bronte Road) (Milton)/'

gen_model = np.loadtxt(base_folder + 'generalized_model.txt')
intersection_9points = np.loadtxt(base_folder + intersection_folder +'/Homography/Homography_drawnworld_9points.txt')
cv2hom = cv2.findHomography(np.array(intersection_9points), np.array(gen_model))

new_pts = cv2.perspectiveTransform(np.array([intersection_9points], dtype='float32'), cv2hom[0].transpose())
tracks_folder = base_folder + intersection_folder +'/Tracking/'
tracks_files = glob.glob(tracks_folder + '/*validated_trajectories.mat')
label_files = glob.glob(base_folder + intersection_folder +'Labeling/*validated_trajectories.xlsx')

labelled_trajectories_df = pd.DataFrame(columns=['labelid','trackid','transformed_trjectory','label'])
index = 0
new_trajectories = []
old_trajectories = []
labels = []
for track_file, label_file in zip(tracks_files, label_files):
    tracks_mat = scipy.io.loadmat(track_file)
    tracks = tracks_mat['recorded_tracks']
    labels_df = pd.read_excel(label_file,engine='openpyxl')
    #labels_df.columns=['trackid','label']
    columns = list(labels_df) 
    

    for track_num,track in enumerate(tracks):
        id_label = labels_df['trackid'][track_num]
        id_track = track[0][0][0][0]
        if id_label != id_track:
            print("ID MISMATCH!")
            break
        trajectory = track[0][-1]
        old_trajectories.append(trajectory)
        new_trajectory = cv2.perspectiveTransform(np.array([trajectory], dtype='float32'), cv2hom[0])
        new_trajectories.append(new_trajectory[0])
        labels.append(labels_df['label'][track_num])
        #labelled_trajectories_df.loc[index] = [id_label, id_track, new_trajectory, labels_df['label'][track_num]]
        index = index+1
#print(labelled_trajectories_df.head())
#labelled_trajectories_df.to_excel(base_folder + intersection_folder +'labelled_trajectories.xlsx')
obj_arr = np.zeros((3,), dtype=np.object)
obj_arr[0] = [old_trajectories]
obj_arr[1] = [new_trajectories]
obj_arr[2] = [labels]
scipy.io.savemat(base_folder + intersection_folder +'labelled_trajectories.mat', mdict={'labelled_trajectories': obj_arr})
temp = 0




