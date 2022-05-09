# Some basic setup:
import torch
import os
from tqdm import tqdm

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pandas as pd

# choose to run on CPU to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'device: {device_name}')

config_path = "./Detectors/detectron2/detectron2-main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

model_weight_url = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
  os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"

def df(args):
  file_path = args.DetectionDetectorPath
  num_header_lines = 3
  data = {}
  data["fn"], data["class"], data["score"], data["x1"], data["y1"], data["x2"], data["y2"] = [], [], [], [], [], [], []
  with open(file_path, "r") as f:
    lines = f.readlines()
    for line in lines[num_header_lines::]:
      splits = line.split()
      fn , clss, score, x1, y1, x2, y2 = int(splits[0]), int(splits[1]), float(splits[2]), float(splits[3]), float(splits[4]), float(splits[5]), float(splits[6])
      data["fn"].append(fn)
      data["class"].append(clss)
      data["score"].append(score)
      data["x1"].append(x1)
      data["y1"].append(y1)
      data["x2"].append(x2)
      data["y2"].append(y2)
  return pd.DataFrame.from_dict(data)
    
def detect(args):
  video_path = args.Video
  text_result_path = args.DetectionPath
  

  # video_path = "./../Dataset/GX010069.avi"
  # text_result_path = "./../Results/GX010069_detections_detectron2.txt"

  #### for now assume that annotated video willl be performed in another subtask
  #### Sa
  # annotated_video_path = "./../Results/GX010069_detections_detectron2.MP4"


  # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
  cfg = get_cfg()
  cfg.merge_from_file(config_path)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg.MODEL.WEIGHTS = model_weight_path
  cfg.MODEL.DEVICE= device_name 
  predictor = DefaultPredictor(cfg)

  # create the VideoCapture Object
  cap = cv2.VideoCapture(video_path)
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  # out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc('X','2','6','4'), fps, (frame_width,frame_height))
  
  # Read until video is completed
  for frame_num in tqdm(range(frames)):
      if (not cap.isOpened()):
          break
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret:
          outputs = predictor(frame)

          # plot boxes in the image and store to video
          color_mode = detectron2.utils.visualizer.ColorMode(1)
          v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),instance_mode=color_mode)
          out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
          annotated_frame = out.get_image()[:, :, ::-1]
          # out_cap.write(annotated_frame)

          # get boxes and store them in a text file
          classes = outputs['instances'].pred_classes.to("cpu")
          scores = outputs["instances"].scores.to("cpu")
          boxes = outputs['instances'].pred_boxes.to("cpu")
          with open(text_result_path, "a") as text_file:
            for clss, score, box in zip(classes, scores, boxes):
              text_file.write(f"{frame_num} {clss} {score} " + " ".join(map(str, box.numpy())) + "\n")

          # Display the resulting frame
          # cv2.imshow('Frame',annotated_frame)
          # cv2.waitKey()
  os.system(f"cp {text_result_path} {args.DetectionDetectorPath}")

  # # When everything done, release the video capture object
  cap.release()
  # out_cap.release()

  # # Closes all the frames
  cv2.destroyAllWindows()
