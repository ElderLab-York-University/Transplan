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

# choose to run on CPU to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

# os.system(f'wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg')
# im = cv2.imread("./input.jpg")
# cv2.imshow("just a test",im)
# cv2.waitKey()
config_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/TransPlan Local/detectron2-main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
model_weight_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/TransPlan Local/model_final_280758.pkl"
video_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Dataset/GX010069.MP4"
annotated_video_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/GX010069_detectron2_annotated_only_test.MP4"
text_result_path = "/home/sajjad/Desktop/Transplan/TransPlan Project/Results/detection_result_detectron2_only_test.txt"

with open(text_result_path, "w") as text_file:
  text_file.write("--------------------------------\n")
  text_file.write("video path: " + video_path + "\n")
  text_file.write("frame# class# score x1 y1 x2 y2 \n")
  text_file.write("--------------------------------\n")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(config_path)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_weight_path
cfg.MODEL.DEVICE= device_name 
predictor = DefaultPredictor(cfg)
# outputs = predictor(im)

# # visualize
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('just a test',out.get_image()[:, :, ::-1])
# cv2.waitKey()

# create the VideoCapture Object
cap = cv2.VideoCapture(video_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc('X','2','6','4'), fps, (frame_width,frame_height))

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
        out_cap.write(annotated_frame)

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

# When everything done, release the video capture object
cap.release()
out_cap.release()

# Closes all the frames
cv2.destroyAllWindows()
