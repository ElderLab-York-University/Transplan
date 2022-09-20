import numpy as np
import sys
import json
import pickle
import cv2
import pandas as pd
from tqdm import tqdm
from GByteTrack.yolox.tracker.byte_tracker import BYTETracker
from GByteTrack.tools.track import make_parser
import os

# @ChangeLater
# find a better way to import meter_per_pixel from Maps 
def meter_per_pixel(center, zoom=19):
    m_per_p = 156543.03392 * np.cos(center[0] * np.pi / 180) / np.power(2, zoom)
    return m_per_p

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

    M_h = np.load(args["HomographyNPY"], allow_pickle=True)[0]
    R_h = meter_per_pixel(args["MetaData"]['center'])

    # initialize ByteTracker
    track_thresh=0.6
    track_buffer = 90
    match_thresh = 0.9
    min_box_area = 64
    ByteArgs = make_parser().parse_args(["--track_thresh", f"{track_thresh}", "--track_buffer",
        f"{track_buffer}", "--match_thresh", f"{match_thresh}", "--min-box-area", f"{min_box_area}"])
    tracker = BYTETracker(ByteArgs, M_H=M_h, R_H= R_h)

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
            fn = ot.frame_id
            idd = ot.track_id
            score = ot.score
            bbox = ot.tlbr
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            results.append([frame_num, idd, score, x1, y1, x2, y2])

    # write results for txt file       
    with open(args["TrackingPth"],"w") as out_file:
        for row in results:
            # print("YOO")
            print('%d,%d,%f,%f,%f,%f,%f'%
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=out_file)