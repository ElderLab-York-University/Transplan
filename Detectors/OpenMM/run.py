# Some basic setup:
import numpy
import torch
import os
# from tqdm import tqdm
import sys
import glob
import json
import cv2
# Setup detectron2 logger
import pandas as pd
from mmdet.apis import init_detector, inference_detector
import mmcv

if __name__ == "__main__":
  config_file = './Detectors/OpenMM/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
  checkpoint_file = './Detectors/OpenMM/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f'device: {device_name}')

  args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
  video_path = args["Video"]
  print(video_path)
  text_result_path = args["DetectionDetectorPath"] 
  print(text_result_path)
  model = init_detector(config_file, checkpoint_file, device=device_name)
  video = mmcv.VideoReader(video_path)
  i=0
  with open (text_result_path,"w") as f: 
      for frame in video:
          result = inference_detector(model, frame)
          a=0
          for res in result:
              # print(res)
              for r in res:
                  if(r[4]>0.5):
                    #   print(str(i) + " " + str(a) + " " + str(r[4]) + " " + str(r[0])+ " " + str(r[1]) + " " + str(r[2])+ " " + str(r[3]))
                      f.write(str(i) + " " + str(a) + " " + str(r[4]) + " " + str(r[0])+ " " + str(r[1]) + " " + str(r[2])+ " " + str(r[3]) +'\n')
              a=a+1
          i=i+1

