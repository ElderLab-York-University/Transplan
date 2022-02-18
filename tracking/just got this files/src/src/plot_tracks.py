import imageio
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormap
from matplotlib import cm
norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
#norm = norm[0:20]
#tracks = scipy.io.loadmat('D:/projects/trans-plan/yongming-results/trans-plan/trans-plan/training_data/barrie107_validated_trajectories.mat')['recorded_tracks']

tracks = scipy.io.loadmat('/home/yufei/Documents/transplan-master (1)/res/NMSvalidated_trajectories.mat')['recorded_tracks']
frame = cv2.imread('/home/yufei/Pictures/Dundas Street at Ninth Line.jpg')
color = (100,100,100)
cv2.namedWindow('tracks',cv2.WINDOW_NORMAL)
cv2.resizeWindow('tracks', 640,640)
index = 0
frames = []
for track_num,track in enumerate(tracks):
    trajectory = track[0][-1]
    color = (np.random.randint(low=0,high=128),np.random.randint(low=0,high=128),np.random.randint(low=0,high=128))
    
    index = 0
    #frame = cv2.imread('D:/projects/trans-plan/updated_algo/Refactored/EssaRoad_BurtonAvenue.png')
    print(len(trajectory))
    if len(trajectory) < 50:
        continue
    for i,pt in enumerate(trajectory):

        rgba_color = cm.rainbow(norm(index),bytes=True)[0:3]
        if pt[0] < 0 or pt[1] < 0 or pt[0] >= frame.shape[0] or pt[1] >= frame.shape[1]:
            continue
        # if i == 0:
        #     rgba_color = (0,255,0)
        # elif i == len(trajectory)-1:
        #     rgba_color = (255,0,0)
        # else:
        #     rgba_color = (0,255,0)
        index+=1
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (int(rgba_color[0]),int(rgba_color[1]),int(rgba_color[2])), -1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)
    if index > 2000:
        break
    cv2.waitKey(1)
    cv2.imshow('tracks',frame)
    #cv2.waitKey(10)
cv2.waitKey(0)
cv2.imwrite('/home/yufei/Pictures/tracks.png', frame)
imageio.mimsave('/home/yufei/Pictures/movie.gif', frames)

# print("Saving GIF file")
# with imageio.get_writer("/home/yufei/Pictures/tracking.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         print("Adding frame to GIF file: ", idx + 1)
#         # writer.append_data(rgb_frame)
