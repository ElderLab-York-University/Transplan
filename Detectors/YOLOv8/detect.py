# Some basic setup:
# import torch
import os
from tqdm import tqdm
import glob
import cv2
# Setup detectron2 logger
import pandas as pd
from Libs import *
from Utils import *

# choose to run on CPU to GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f'device: {device_name}')



def detect(args,*oargs):
	setup(args)
	env_name = args.Detector
	exec_path = "./Detectors/YOLOv8/run.py"
	conda_pyrun(env_name, exec_path, args)

def df(args):
	file_path = args.DetectionDetectorPath
	data = {}
	data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
	with open(file_path, "r+") as f:
		lines = f.readlines()
		for line in lines:
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
	with open(text_result_path, "w") as text_file:
		pass

	with open(text_result_path, "w") as text_file:
		for i, row in tqdm(df.iterrows()):
			frame_num, clss, score, x1, y1, x2, y2 = row["fn"], row['class'], row["score"], row["x1"], row["y1"], row["x2"], row["y2"]
			text_file.write(f"{frame_num} {clss} {score} {x1} {y1} {x2} {y2}\n")
	
def setup(args):
	env_name = args.Detector
	src_url = "https://github.com/ultralytics/ultralytics"
	rep_path = "./Detectors/YOLOv8/ultralytics"
	# checkpoint_name="faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

	print(env_name)
	if not "ultralytics" in os.listdir("./Detectors/YOLOv8/"):
		os.system(f"git clone {src_url} {rep_path}")
	if not env_name in get_conda_envs():
		initial_directory = os.getcwd()
		make_conda_env(env_name, libs="python=3.7")
		# install library on conda env
		print("here I am 1")
		os.chdir(rep_path)
		# os.system(f"conda install -n {env_name} pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y")
		# os.system(f"conda clean --packages --tarballs")
		os.system(f"conda run -n {env_name} --live-stream pip install -e '.[dev]' ")
		print("here I am 2")
		os.chdir(initial_directory)
		# os.system(f"conda run -n {args.Detector} pip3 install openmim")

		# print("here I am 3")
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

