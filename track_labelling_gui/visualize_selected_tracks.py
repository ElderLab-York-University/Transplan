from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import re
from ast import literal_eval
from enum import unique
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

labeled_trakcs_path = "./../../Results/GX010069_tracking_sort_reprojected.pkl"
tracks = pd.read_pickle(labeled_trakcs_path)
tracks = tracks.sort_values("moi")
second_image_path = "./../../Dataset/DundasStAtNinthLine.jpg"

img2 = cv.imread(second_image_path)
rows2, cols2, dim2 = img2.shape
for i in range(len(tracks)):
    track = tracks.iloc[i]
    traj = track['trajectory']
    moi = track["moi"]
    for j , p in enumerate(traj):
        x , y = int(p[0]), int(p[1])
        img2 = cv.circle(img2, (x,y), radius=2, color=(int(i/len(tracks)*255), int(moi/12*255), int(255 - j/len(traj)*255)), thickness=2)

plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
save_path = f"./../../Results/GX010069_selected_tracks.png"
plt.savefig(save_path)
