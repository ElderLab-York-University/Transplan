# Some basic setup:
import torch
import os
from tqdm import tqdm
import glob
import cv2
import json
import sys
# Setup detectron2 logger
import pandas as pd
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict

# choose to run on CPU to GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f'device: {device_name}')
classes_to_keep = [2, 5, 7] #3-1:car, 6-1:bus, 8-1:truck


if __name__=="__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"] 

    # os.system(f"python3 ./Detectors/YOLOv5/yolov5/detect.py --source {video_path} --save-txt --save-conf")
    # directory_in_str="./Detectors/YOLOv5/yolov5/runs/detect/exp/labels/*.txt"
    # files=[]
    # for filepath in glob.iglob(directory_in_str):
    #     files.append(filepath)

    # files=sorted(files,key=lambda x: int(x.split('_')[1].partition(".")[0]))
    # # print(files)
    # vid = cv2.VideoCapture(video_path)
    # height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

        
    # i=0
    # with open(text_result_path, "a") as text_file:
    #     for file in files:
    #         f=open(file)
    #         for line in f:
    #             nums=line.split(" ")
                
    #             if(int(nums[0]) in classes_to_keep):
    #                 center_x=float(nums[1])*width
    #                 center_y=float(nums[2])*height
    #                 size_x=float(nums[3])*width
    #                 size_y=float(nums[4])*height

    #                 x_1=round(center_x-size_x/2,5)
    #                 y_1=round(center_y-size_y/2,5)
    #                 x_2=round(center_x+size_x/2,5)
    #                 y_2=round(center_y+size_y/2,5)

    #                 l=str(i) +" " + str(nums[0]) + " " + str(nums[-1][0:-2])+" "+ str(x_1) + " " + str(y_1) + " " + str(x_2) + " " + str(y_2)+"\n"
    #                 print(l)
    #                 text_file.write(l)
    #             # print(str(i)+" " +line)
    #         i=i+1


    # # os.system(f"cp {text_result_path} {args}")






    # os.system(f"rm -rf ./Detectors/YOLOv5/yolov5/runs/detect")

  

  # video_path = "./../Dataset/GX010069.avi"
