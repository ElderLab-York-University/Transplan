# create the VideoCapture Object
# captures a specific frame on a videl file and stores it in the determined destination
# Some basic setup:
import os
from tqdm import tqdm

import numpy as np
import os, json, cv2, random

chosen_frame = 4400
video_path = "./../Dataset/GX010069.MP4"
image_out_path = f"./../Dataset/GX010069_frame_{chosen_frame}.png"
# Opens the Video file
cap = cv2.VideoCapture(video_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


for frame_num in tqdm(range(frames)):
    if (not cap.isOpened()):
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret and (frame_num == chosen_frame):
        cv2.imwrite(image_out_path,frame)
        break
print("done")
cap.release()
cv2.destroyAllWindows()