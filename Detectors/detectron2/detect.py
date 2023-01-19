# Some basic setup:
# import torch
import os
from tqdm import tqdm

# Setup detectron2 logger

# import some common libraries
import numpy as np
import os, json, cv2, random
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# import some common detectron2 utilities
import pandas as pd
from Libs import *
from Utils import *

def detect(args,*oargs):
  setup(args)
  env_name = args.Detector
  exec_path = "./Detectors/detectron2/run.py"
  conda_pyrun(env_name, exec_path, args)

def df(args):
  file_path = args.DetectionDetectorPath
  num_header_lines = 3
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
  with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines[num_header_lines::]:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2 = int(float(splits[0])), int(float(splits[1])), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"].append(x1)
      data["y1"].append(y1)
      data["x2"].append(x2)
      data["y2"].append(y2)
  return pd.DataFrame.from_dict(data)

def df_txt(df,text_result_path):
  # store a modified version of detection df to the same txt file
  # used in the post processig part of the detection
  # df is in the same format specified in the df function
  with open(text_result_path, "w") as text_file:
    pass

  with open(text_result_path, "w") as text_file:
    for i, row in tqdm(df.iterrows()):
      frame_num, clss, score, x1, y1, x2, y2 = row["fn"], row['class'], row["score"], row["x1"], row["y1"], row["x2"], row["y2"]
      text_file.write(f"{frame_num} {clss} {score} {x1} {y1} {x2} {y2}\n")
    
def setup(args):
    env_name = args.Detector
    src_url = "https://github.com/facebookresearch/detectron2.git"
    rep_path = "./Detectors/detectron2/detectron2"
    # checkpoint_name="faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    print(env_name)
    if not "detectron2" in os.listdir("./Detectors/detectron2/"):
      os.system(f"git clone {src_url} {rep_path}")
    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        # install library on conda env
        print("here I am 1")
        # os.system(f"conda run -n {env_name} pip3 install pytorch torchvision torchaudio -c pytorch -y")
        os.system(f"conda run -n {env_name} pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html")
        
        os.system(f'conda install -n {env_name} cudatoolkit=10.2 -y')
        # os.system(f"conda clean --packages --tarballs")
        print("here I am 2")
        os.system(f"conda run -n {args.Detector} python -m pip install pyyaml==5.1 opencv-python cython pandas tqdm")
        print("here I am 2.5")
        cwd = os.getcwd()
        os.chdir("./Detectors/detectron2")
        os.system(f"conda run -n {args.Detector} python -m pip install -e detectron2")
        os.chdir(cwd)
        # os.system(f"conda run -n {args.Detector} python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")        
        print("here I am 3")
        # os.system(f"conda run -n {args.Detector} mim install mmcv-full")
        # print("here I am 4")
        # os.system(f"conda run -n {args.Detector} pip3 install -r ./Detectors/OpenMM/mmdetection/requirements/build.txt")
        # os.system(f"cd ./Detectors/OpenMM/mmdetection")
        # os.system(f"conda run -n {args.Detector} pip3 install  -v -e .  ")
        # os.system(f"conda run -n {args.Detector} pip3 install -e ./Detectors/OpenMM/mmdetection/")
        # print("HERE I AM 01010")
        # os.system(f"conda run -n {args.Detector} python3 Detectors/OpenMM/mmdetection/setup.py develop")
        # os.system(f"conda run -n {args.Detector} pip3 install mmdet")
        # print("YOOOOO")
        



  

  # video_path = "./../Dataset/GX010069.avi"
