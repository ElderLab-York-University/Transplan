import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np
import mmcv
import torch
import sys
import json
sys.path.append("./Detectors/InternImage/InternImage/detection")
from ops_dcnv3 import modules as mod
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

def getbboxes(result):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    return [bboxes, labels]


if __name__ == "__main__":
  args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code

  config_file = args["MMDetConfig"]
  checkpoint_file =  args["MMDetCheckPoint"]
 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f'device: {device_name}')

  video_path = args["Video"]
#   print(video_path)
  text_result_path = args["DetectionDetectorPath"] 
#   print(text_result_path)
  model = init_detector(config_file, checkpoint_file, device=device_name)
  video = mmcv.VideoReader(video_path)
  i=0
  with open (text_result_path,"w") as f: 
      for frame in tqdm(video):
          with torch.no_grad():
            result = inference_detector(model, frame)
          res= getbboxes(result)
          bboxes=res[0]
          labels=res[1]
          for box,label in zip(bboxes,labels):
                r=box
                f.write(str(i) + " " + str(label) + " " + str(r[4]) + " " + str(r[0])+ " " + str(r[1]) + " " + str(r[2])+ " " + str(r[3]) +'\n')
          i=i+1

