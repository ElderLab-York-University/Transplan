# create the VideoCapture Object
# Some basic setup:
import os
from tqdm import tqdm

import numpy as np
import os, json, cv2, random


video_path = "./../Dataset/GX010069.avi"
annotated_video_path = "./../Results/GX010069_tracking_sort.MP4"
tracks_path = "./../Results/GX010069_tracking_sort.txt"

color = (0, 0, 102)

# load tracks
tracks = np.loadtxt(tracks_path, delimiter=',')


cap = cv2.VideoCapture(video_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc('X','2','6','4'), fps, (frame_width,frame_height))

# Read until video is completed
for frame_num in tqdm(range(frames)):
    if (not cap.isOpened()):
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        this_frames_tracks = tracks[tracks[:, 0]==(frame_num+1)]
        for track in this_frames_tracks:
            # plot the bbox + id with colors
            bbid, x1 , y1, x2, y2 = track[1], int(track[2]), int(track[3]), int(track[4]), int(track[5])
            # print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'id:{bbid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        out_cap.write(frame)


# When everything done, release the video capture object
cap.release()
out_cap.release()

# Closes all the frames
cv2.destroyAllWindows()