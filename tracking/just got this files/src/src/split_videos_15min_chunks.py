import cv2
import glob
folderPath = 'D:/projects/trans-plan/data/2020/Dec/DorvalAtNorthServiceRoad/'
vidfiles = glob.glob(folderPath+'*.mp4')
chunkCount = 1
currframecount = 0
vid_cap = cv2.VideoCapture(vidfiles[0])
fps = vid_cap.get(cv2.CAP_PROP_FPS)
w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = 'mp4v'  # output video codec
# fps = 60
# w = 2704
# h = 2032
vid_writer = cv2.VideoWriter(folderPath+'15min_'+str(chunkCount)+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for vidfile in vidfiles:
    cap = cv2.VideoCapture(vidfile)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        currframecount = currframecount+1
        print(vidfile + ": " + str(currframecount))
        if currframecount == 54121:
            currframecount = 0
            chunkCount = chunkCount + 1
            vid_writer.release()
            vid_writer = cv2.VideoWriter(folderPath+'15min_'+str(chunkCount)+'.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vid_writer.write(frame)