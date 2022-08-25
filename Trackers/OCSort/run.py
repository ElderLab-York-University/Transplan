import numpy as np
import sys
import json
import pickle
import cv2
import pandas as pd
from tqdm import tqdm
from OCSort.trackers.ocsort_tracker.ocsort import OCSort


if __name__ == "__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"]
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info_imgs = [frame_height, frame_width]
    img_size = [frame_height, frame_width]

    # initialize OCSort
    det_thresh=0.75
    max_age=30
    min_hits=3 
    iou_threshold=0.3
    delta_t=3
    asso_func="iou"
    inertia=0.2
    use_byte=True
  
    tracker = OCSort(det_thresh, max_age, min_hits, 
        iou_threshold, delta_t, asso_func, inertia, use_byte)

    with open(args["DetectionPkl"],"rb") as f:
        detections=pickle.load(f)

    unique_frames = np.sort(np.unique(detections["fn"]))
    results = []

    for frame_num in tqdm(unique_frames):
        frame_mask= detections["fn"]==frame_num
        frame_detections=detections[frame_mask]
        dets = np.array([[row['x1'], row['y1'], row['x2'], row['y2'], row['score']] for _, row in frame_detections.iterrows()])

        online_targets = tracker.update(dets, info_imgs, img_size)
        for ot in online_targets:
            x1, y1, x2, y2, idd = ot[0], ot[1], ot[2], ot[3], ot[4]
            results.append([frame_num, idd, x1, y1, x2, y2])

    # write results for txt file       
    with open(args["TrackingPth"],"w") as out_file:
        for row in results:
            # print("YOO")
            print('%d,%d,%f,%f,%f,%f'%
            (row[0], row[1], row[2], row[3], row[4], row[5]),file=out_file)