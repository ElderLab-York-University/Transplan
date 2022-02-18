import cv2
import numpy as np
 
# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('/home/sajjad/Desktop/TransPlan Project/transplan dataset/GX010069.MP4')
 
 
# Loop until the end of the video
while (cap.isOpened()):
 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue
    # frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
    #                      interpolation = cv2.INTER_CUBIC)
 
    # Display the resulting frame
    # cv2.imshow('Frame', frame)
 
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
 
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()
print('exited normally')