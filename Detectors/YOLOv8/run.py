# Some basic setup:
import torch
import numpy as np
import os, json, cv2, random
import os
from tqdm import tqdm
import pandas as pd
import sys

# import YOLO from ultralytics
from ultralytics import YOLO

classes_to_keep = [2, 5, 7] #3-1:car, 6-1:bus, 8-1:truck

if __name__=="__main__":
    print("in the run.py for YOLOv8 ")
   # decide which device to run the model on (GPU/CPU)
    n_GPUs = 1*torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda:0" if n_GPUs > 0 else "cpu" # for now lets use only one GPU

    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main cod
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"]

    model = YOLO("yolov8x.pt")
    model.to(device)
    results = model.predict(source=video_path, stream=True, verbose=False) # stream=True return a generator for memory efficiency
    print("len results is ......")

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # clear the output file
    with open(text_result_path, "w"):
        pass
    with open(text_result_path, "a") as f:
        for fn, result in tqdm(enumerate(results), total=length):
            try:
                res = result.cpu().numpy()
                boxes = res.boxes.xyxy
                confs = res.boxes.conf
                cls = res.boxes.cls
                for cl, conf, box in zip(cls, confs, boxes):
                    if cl in classes_to_keep:
                        f.write(f"{fn+1} {cl} {conf} " + " ".join(map(str, box)) + "\n")
            except: pass