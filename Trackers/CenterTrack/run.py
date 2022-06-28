import sys
import pickle as pkl
import json
import argparse
import sys
CENTERTRACK_PATH = "./Trackers/CenterTrack/CenterTrack/src/lib/"
sys.path.insert(0, CENTERTRACK_PATH)
from detector import Detector
from opts import opts
import cv2
from tqdm import tqdm
import torch

if __name__ == "__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    output_file = args["TrackingPth"]
    MODEL_PATH = "./Trackers/CenterTrack/CenterTrack/models/coco_tracking.pth"
    TASK = 'tracking' # or 'tracking,multi_pose' for pose tracking and 'tracking,ddd' for monocular 3d tracking
    GPUS = "0" if torch.cuda.is_available() else "-1" # if gpu is available use it
    print(f"device:{GPUS} in CenterTrack")

    print("this is for debugging 1")
    opt = opts().init('{} --load_model {} --gpus {}'.format(TASK, MODEL_PATH, GPUS).split(' '))
    
    detector = Detector(opt)
    print("this is for debugging 2")

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    with open(output_file, 'w') as out_file:
        for frame_num in tqdm(range(frames)):
            if (not cap.isOpened()):
                break
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                dets = detector.run(frame)['results']
                for det in dets:
                    print(f"{frame_num+1},{det['tracking_id']},{det['class']},{det['score']},{det['bbox'][0]},{det['bbox'][1]},{det['bbox'][2]},{det['bbox'][3]}",file=out_file)
                    # fn,id,class,score, bbox(4 numbers)

    
