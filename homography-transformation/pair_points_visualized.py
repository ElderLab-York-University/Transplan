import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

first_image_path = "./../Dataset/GX010069_frame_4400.png"
second_image_path = "./../Dataset/DundasStAtNinthLine.jpg"
first_set_of_points_path = "./homography-gui/Homography_1_.csv"
second_set_of_points_path = "./homography-gui/Homography_2_.csv"
homography_path = "./homography-gui/homography.npy"

first_points = pd.read_csv(first_set_of_points_path)
second_points = pd.read_csv(second_set_of_points_path)

img1 = cv.imread(first_image_path)
img2 = cv.imread(second_image_path)

img1p = cv.imread(first_image_path)
img2p = cv.imread(second_image_path)

rows1, cols1, dim1 = img1.shape
rows2, cols2, dim2 = img2.shape
M = np.load(homography_path, allow_pickle=True)[0]
Mp = np.linalg.inv(M)

for i , row in tqdm(second_points.iterrows()):
    x, y = row[0], row[1]
    img2 = cv.circle(img2, (x,y), radius=4, color=(int(i/len(second_points)*255), 70, int(255 - i/len(second_points)*255)), thickness=3)
for i, row in first_points.iterrows():
    x, y = row[0], row[1]
    point = np.array([x, y, 1])
    new_point = M.dot(point)
    new_point /= new_point[2]
    xp, yp = int(new_point[0]), int(new_point[1]) 
    img2p = cv.circle(img2p, (xp,yp), radius=4, color=(int(i/len(first_points)*255), 70, int(255 - i/len(first_points)*255)), thickness=3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
ax1.set_title("original")
ax2.imshow(cv.cvtColor(img2p, cv.COLOR_BGR2RGB))
ax2.set_title("reprojected")
save_path = f"./../Results/GX010069_paired_points_topview.png"
plt.savefig(save_path)


for i , row in tqdm(first_points.iterrows()):
    x, y = row[0], row[1]
    img1 = cv.circle(img1, (x,y), radius=20, color=(int(i/len(first_points)*255), 70, int(255 - i/len(first_points)*255)), thickness=10)
for i, row in second_points.iterrows():
    x, y = row[0], row[1]
    point = np.array([x, y, 1])
    new_point = Mp.dot(point)
    new_point /= new_point[2]
    xp, yp = int(new_point[0]), int(new_point[1]) 
    img1p = cv.circle(img1p, (xp,yp), radius=20, color=(int(i/len(second_points)*255), 70, int(255 - i/len(second_points)*255)), thickness=10)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax1.set_title("original")
ax2.imshow(cv.cvtColor(img1p, cv.COLOR_BGR2RGB))
ax2.set_title("reprojected")
save_path = f"./../Results/GX010069_paired_points_steetview.png"
plt.savefig(save_path)