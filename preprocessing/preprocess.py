import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./../Dataset/GX010069_frame_4400.png")
K = np.load("./../Dataset/intrinsics.npy")
D = np.array([1.198095304735928e+03, -2.374239692518112e-04, -1.666034659090048e-09, 7.431175936706124e-13])
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

undistorted_image = cv2.fisheye.undistortImage(image, K, D)
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.show()
print(D)
print(K)