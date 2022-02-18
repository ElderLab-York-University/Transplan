import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

first_image_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Dataset/GX010069_frame_4400.png"
second_image_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Dataset/DundasStAtNinthLine.jpg"
# homography_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/TransPlan Local/homography-gui/homography_125.npy"
tracks_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/trackinig_result_centeroids.txt"
transformed_tracks_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/trackinig_result_centeroids_transformed.txt"

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

for track in tracks_id:
    x, y = int(track[2]), int(track[3])
    img1 = cv.circle(img1, (x,y), radius=2, color=(0, 0, 255), thickness=2)

plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.show()


for track in transformed_tracks_id:
    x, y = int(track[2]), int(track[3])
    img2 = cv.circle(img2, (x,y), radius=2, color=(0, 0, 255), thickness=2)

plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.show()

# print(M.shape)
# print(M)
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
# plt.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB), alpha=0.4)
# plt.show()


