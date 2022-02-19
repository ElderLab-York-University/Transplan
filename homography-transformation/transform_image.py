import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

first_image_path = "./../Dataset/GX010069_frame_4400.png"
second_image_path = "./../Dataset/DundasStAtNinthLine.jpg"
homography_path = "./homography-gui/homography.npy"
save_path = "./../Results/GX010069_homography.jpg"

img1 = cv.imread(first_image_path)
img2 = cv.imread(second_image_path)
rows1, cols1, dim1 = img1.shape
rows2, cols2, dim2 = img2.shape
M = np.load(homography_path, allow_pickle=True)[0]

img12 = cv.warpPerspective(img1, M, (cols2, rows2))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
ax1.set_title("camera view")
ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
ax2.set_title("top view")

ax3.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB))
ax3.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB), alpha=0.3)
ax3.set_title("camera view reprojected on top view")


plt.savefig(save_path)


