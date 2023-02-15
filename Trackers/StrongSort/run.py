from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "strongsort"  # yolov5 strongsort root directory
WEIGHTS = ROOT / "weights"
sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.append(str(ROOT / "yolov8"))
sys.path.append(str(ROOT/ "trackers" / "strongsort"))



import numpy as np
import json
import pickle
import cv2
import pandas as pd
from tqdm import tqdm
import torch
from trackers.multi_tracker_zoo import create_tracker
from yolov8.ultralytics.yolo.utils.torch_utils import select_device


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

    # initialize Strong Sort
    tracking_method = "strongsort"
    tracking_config = ROOT / "trackers"/"strongsort"/"configs"/"strongsort.yaml"
    half = False
    device = select_device("0") if torch.cuda.is_available() else select_device("cpu")
    reid_weights = WEIGHTS/"osnet_x0_25_msmt17.pt"
    tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)

    with open(args["DetectionPkl"],"rb") as f:
        detections=pickle.load(f)

    unique_frames = np.sort(np.unique(detections["fn"]))
    results = []
    for frame_num in tqdm(unique_frames):
        # load one frame
        if (not cap.isOpened()):
            break
        ret, frame = cap.read()
        if not ret: continue
        # select detections 
        frame_mask= detections["fn"]==frame_num
        frame_detections=detections[frame_mask]
        dets = torch.tensor([[row['x1'], row['y1'], row['x2'], row['y2'], row['score'], row['class']] for _, row in frame_detections.iterrows()], requires_grad=False)
        # update tracker
        # dets are expected to have the following form
        # xyxys = dets[:, 0:4]
        # confs = dets[:, 4]
        # clss = dets[:, 5]
        with torch.no_grad():
            outputs  = tracker.update(dets, frame)
        # output will have the following detail
        # bbox = output[0:4]
        # id = output[4]
        # cls = output[5]
        # conf = output[6]

        for ot in outputs:
            fn = frame_num
            bbox = ot[0:4]
            idd = ot[4]
            clss = ot[5]
            score = ot[6]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            results.append([frame_num, idd, score, x1, y1, x2, y2, clss])

    # write results for txt file       
    with open(args["TrackingPth"],"w") as out_file:
        for row in results:
            # print("YOO")
            print('%d,%d,%f,%f,%f,%f,%f,%f'%
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]),file=out_file)