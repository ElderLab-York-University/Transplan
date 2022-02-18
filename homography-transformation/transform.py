import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

first_image_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Dataset/GX010069_frame_4400.png"
second_image_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Dataset/DundasStAtNinthLine.jpg"
homography_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/TransPlan Local/homography-gui/homography.npy"


img1 = cv.imread(first_image_path)
img2 = cv.imread(second_image_path)
rows1, cols1, dim1 = img1.shape
rows2, cols2, dim2 = img2.shape
M = np.load(homography_path, allow_pickle=True)[0]

img12 = cv.warpPerspective(img1, M, (cols2, rows2))

print(M.shape)
print(M)
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB), alpha=0.4)
plt.show()


