import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

first_image_path = "./../Dataset/GX010069_frame_4400.png"
second_image_path = "./../Dataset/DundasStAtNinthLine.jpg"
tracks_path = "./../Results/GX010069_tracking_sort.txt"
transformed_tracks_path = "./../Results/GX010069_tracking_sort_reprojected.txt"


tracks = np.loadtxt(tracks_path, delimiter=",")
transformed_tracks = np.loadtxt(transformed_tracks_path, delimiter=",")
img1 = cv.imread(first_image_path)
img2 = cv.imread(second_image_path)
rows1, cols1, dim1 = img1.shape
rows2, cols2, dim2 = img2.shape
track_id = 548
# M = np.load(homography_path, allow_pickle=True)[0]
# img12 = cv.warpPerspective(img1, M, (cols2, rows2))

mask = tracks[:, 1]==track_id
tracks_id = tracks[mask]
transformed_tracks_id = transformed_tracks[mask]

for i, track in enumerate(tracks_id):
    x, y = int((track[2] + track[4])/2), int((track[3]+track[5])/2)
    img1 = cv.circle(img1, (x,y), radius=5, color=(int(i/len(tracks_id)*255), 70, int(255 - i/len(tracks_id)*255)), thickness=3)



for i, track in enumerate(transformed_tracks_id):
    x, y = int(track[2]), int(track[3])
    img2 = cv.circle(img2, (x,y), radius=2, color=(int(i/len(transformed_tracks_id)*255), 70, int(255 - i/len(transformed_tracks_id)*255)), thickness=2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
save_path = f"./../Results/GX010069_trajectory_id={track_id}.png"
plt.savefig(save_path)