# Some basic setup:
from tabnanny import check
import os
from tqdm import tqdm
import glob
import cv2
# Setup detectron2 logger
import pandas as pd
# from mmdet.apis import init_detector, inference_detector
# import mmcv
from Libs import *
from Utils import *

# choose to run on CPU to GPU


# model_weight_url = ""
# if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
#   os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
# model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"

def detect(args,*oargs):
  setup(args)
  env_name = args.Detector
  exec_path = "./Detectors/InternImage/run.py"
  conda_pyrun(env_name, exec_path, args)

def df(args):
  file_path = args.DetectionDetectorPath
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
  with open(file_path, "r+") as f:
    lines = f.readlines()
    for line in lines:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2 = float(splits[0]), float(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])
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
    src_url = "https://github.com/OpenGVLab/InternImage.git"
    rep_path = "./Detectors/InternImage/InternImage"
    if not "InternImage" in os.listdir("./Detectors/InternImage/"):
      os.system(f"git clone {src_url} {rep_path}")
      if not "checkpoint_dir" in os.listdir("./Detectors/InternImage/InternImage"):
        os.system(f"mkdir ./Detectors/InternImage/InternImage/checkpoint_dir")
        os.system(f"wget -c https://github.com/OpenGVLab/InternImage/releases/download/det_model/cascade_internimage_xl_fpn_1x_coco.pth\
        -O ./Detectors/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_1x_coco.pth")

        os.system(f"wget -c  https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth\
        -O ./Detectors/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_3x_coco.pth")


    if not env_name in get_conda_envs():
        make_conda_env(env_name, libs="python=3.7")
        # install library on conda env
        print("here I am 1")
        os.system(f"conda run --live-stream -n {args.Detector} conda install pip")
        os.system(f"conda run -n {args.Detector} pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html")
        print("here I am 2")
        os.system(f"conda run --live-stream -n {args.Detector} pip install -U openmim")
        print("here I am 2.3")
        os.system(f"conda run --live-stream -n {args.Detector} mim install mmcv-full==1.5.0")
        print("here I am 2.6")
        os.system(f"conda run --live-stream -n {args.Detector} pip install timm==0.6.11 mmdet==2.28.1")
        print("here I am 3")
        os.system(f"conda run --live-stream -n {args.Detector} pip install opencv-python termcolor yacs pyyaml scipy tqdm")
        print("here I am 4")
        os.system(f"conda run --live-stream -n {args.Detector} python3 ./Detectors/InternImage/InternImage/detection/ops_dcnv3/setup.py build install")        
        os.system(f"conda run --live-stream -n {args.Detector} python3 ./Detectors/InternImage/InternImage/detection/ops_dcnv3/test.py")        
   
        



  

  # video_path = "./../Dataset/GX010069.avi"
