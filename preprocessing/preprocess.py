import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


video_path = "/mnt/dataB/CityFlowV2Local/train/S06C044/c044/vdo.avi"
out_video_path = "/mnt/dataB/CityFlowV2Local/train/S06C044/c044/vdo.mp4"
mtx = np.array([[1280.000000000000000, 0.000000000000000, 640.000000000000000],[0.000000000000000, 1280.000000000000000, 480.000000000000000],[0.000000000000000,0.000000000000000,1.000000000000000]])
dist = np.array([-0.400000005960464, 0.000000000000000, 0.000000000000000, 0.000000000000000])

cap = cv2.VideoCapture(video_path)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
out_cap = cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0.5, (w,h))

for frame_num in tqdm(range(frames)):
    if (not cap.isOpened()):
        print("error reading the video")
    ret, img = cap.read()

    if not ret:
        print("cap did not return a frame")

    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    out_cap.write(undistorted_img)

cap.release()
out_cap.release()