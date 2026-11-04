import numpy as np
import sys
import json
import pickle
import cv2
import pandas as pd
from tqdm import tqdm
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.tools.track import make_parser


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

    # initialize ByteTracker
    track_thresh=0.6
    track_buffer = 90
    match_thresh = 0.9
    min_box_area = 64
    ByteArgs = make_parser().parse_args(["--track_thresh", f"{track_thresh}", "--track_buffer",
        f"{track_buffer}", "--match_thresh", f"{match_thresh}", "--min-box-area", f"{min_box_area}"])
    # Keep 90 as 90 updates: BYTETracker's frame_rate argument would otherwise
    # rescale track_buffer; both trackers use 90 *video frames* in aligned mode.
    tracker = BYTETracker(ByteArgs, frame_rate=30)

    detections=pd.read_pickle(args["DetectionPkl"])
    # with open(args["DetectionPkl"],"rb") as f:
    #     detections=pickle.load(f)

    # Default preserves the original detector-bearing-frames-only behaviour.
    # Opt-in comparison mode advances both trackers on the same video timeline.
    full_timeline = bool(args.get("BTFullTimeline", False))
    if full_timeline:
        limit = args.get("ForNFrames")
        if limit is not None:
            if int(limit) < 0:
                raise ValueError("ForNFrames must be nonnegative")
            frames = min(frames, int(limit))
        frame_numbers = range(frames)
        detections = detections.loc[
            (detections["fn"] >= 0) & (detections["fn"] < frames)
        ]
        grouped = {int(fn): group for fn, group in detections.groupby("fn")}
    else:
        frame_numbers = np.sort(np.unique(detections["fn"]))

    results = []
    for frame_num in tqdm(frame_numbers, desc="ByteTrack", unit="frame"):
        if full_timeline:
            frame_detections = grouped.get(int(frame_num))
            if frame_detections is None:
                dets = np.empty((0, 5), dtype=float)
            else:
                dets = frame_detections[["x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float)
        else:
            frame_mask = detections["fn"] == frame_num
            frame_detections = detections[frame_mask]
            dets = np.array([[row['x1'], row['y1'], row['x2'], row['y2'], row['score']]
                             for _, row in frame_detections.iterrows()])

        online_targets = tracker.update(dets, info_imgs, img_size)
        for ot in online_targets:
            fn = ot.frame_id
            idd = ot.track_id
            score = ot.score
            bbox = ot.last_tlbr
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            results.append([frame_num, idd, score, x1, y1, x2, y2])

    # write results for txt file       
    with open(args["TrackingPth"],"w") as out_file:
        for row in results:
            # print("YOO")
            print('%d,%d,%f,%f,%f,%f,%f'%
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6]),file=out_file)
    cap.release()
