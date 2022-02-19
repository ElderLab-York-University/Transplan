# uses the homograpy matrix to reproject the tracking points
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

first_image_path = "./../Dataset/GX010069_frame_4400.png"
second_image_path = "./../Dataset/DundasStAtNinthLine.jpg"
homography_path = "./homography-gui/homography.npy"
tracks_path = "./../Results/GX010069_tracking_sort.txt"
out_path = "./../Results/GX010069_tracking_sort_reprojected.txt"

tracks = np.loadtxt(tracks_path, delimiter=",")
transformed_tracks = np.zeros_like(tracks)
img1 = cv.imread(first_image_path)
img2 = cv.imread(second_image_path)
rows1, cols1, dim1 = img1.shape
rows2, cols2, dim2 = img2.shape
M = np.load(homography_path, allow_pickle=True)[0]
img12 = cv.warpPerspective(img1, M, (cols2, rows2))

for index, track in tqdm(enumerate(tracks)):
    fn, idd, x, y = track[0], track[1], (track[2] + track[4]/2), (track[3] + track[5])/2
    transformed_tracks[index, 0] = fn
    transformed_tracks[index, 1] = idd 
    point = np.array([x, y, 1])
    new_point = M.dot(point)
    new_point /= new_point[2]
    transformed_tracks[index, 2] = new_point[0]
    transformed_tracks[index, 3] = new_point[1]  

with open(out_path, 'w') as out_file:
    for track in transformed_tracks:
        fn, idd, x, y = int(track[0]), int(track[1]), track[2], track[3]
        print(f'{fn},{idd},{x},{y}', file=out_file)  



# print(M.shape)
# print(M)
# plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
# plt.imshow(cv.cvtColor(img12, cv.COLOR_BGR2RGB), alpha=0.4)
# plt.show()


