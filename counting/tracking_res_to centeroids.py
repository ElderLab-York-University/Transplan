# create the VideoCapture Object
# Some basic setup:
import os
from tqdm import tqdm

import numpy as np
import os, json, cv2, random


output_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/trackinig_result_centeroids.txt"
tracks_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/trackinig_result.txt"


tracks = np.loadtxt(tracks_path, delimiter=',')
with open(output_path,'w') as out_file:
    for track in tqdm(tracks):
        fnum, bbid, x1 , y1, x2, y2 =int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4]), int(track[5])
        print(f'{fnum},{bbid},{(x1 + x2)/2},{(y1+y2)/2}', file=out_file)