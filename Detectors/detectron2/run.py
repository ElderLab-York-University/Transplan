# # Some basic setup:
# import torch
# import os
# from tqdm import tqdm
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

# import sys
# import glob
# import json
# import cv2
# # Setup detectron2 logger
# import pandas as pd
# import warnings
# warnings.resetwarnings()
# warnings.simplefilter('ignore', UserWarning)
# # from mmdet.apis import init_detector, inference_detector

# # see the list of MS_COCO class dict in the link below 
# # https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
# # the output of detectron2 is the class numbers + 1
# classes_to_keep = [2, 5, 7] #3-1:car, 6-1:bus, 8-1:truck



# if __name__ == "__main__":
#   # choose to run on CPU to GPU
#   args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#   device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
#   print(f'device: {device_name}')

#   config_path = "./Detectors/detectron2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

#   model_weight_url = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#   if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
#     os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
#   model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"

#   video_path = args["Video"]
#   text_result_path = args["DetectionDetectorPath"]
  

#   # video_path = "./../Dataset/GX010069.avi"
#   # text_result_path = "./../Results/GX010069_detections_detectron2.txt"

#   #### for now assume that annotated video willl be performed in another subtask
#   #### Sa
#   # annotated_video_path = "./../Results/GX010069_detections_detectron2.MP4"


#   # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
#   # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#   cfg = get_cfg()
#   cfg.merge_from_file(config_path)
#   cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
#   cfg.MODEL.WEIGHTS = model_weight_path
#   cfg.MODEL.DEVICE= device_name

#   predictor = DefaultPredictor(cfg)

#   # create the VideoCapture Object
#   cap = cv2.VideoCapture(video_path)
#   # Check if camera opened successfully
#   if (cap.isOpened()== False): 
#     print("Error opening video stream or file")

#   frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#   fps = int(cap.get(cv2.CAP_PROP_FPS))
#   frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#   frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#   # out_cap = cv2.VideoWriter(annotated_video_path,cv2.VideoWriter_fourcc('X','2','6','4'), fps, (frame_width,frame_height))
#   with open(text_result_path, "w") as text_file:
#     pass
#   # Read until video is completed
#   for frame_num in tqdm(range(frames)):
#       if (not cap.isOpened()):
#           break
#       # Capture frame-by-frame
#       ret, frame = cap.read()
#       if ret:
#           outputs = predictor(frame)

#           # plot boxes in the image and store to video
#           color_mode = detectron2.utils.visualizer.ColorMode(1)
#           v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),instance_mode=color_mode)
#           out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#           annotated_frame = out.get_image()[:, :, ::-1]
#           # out_cap.write(annotated_frame)

#           # get boxes and store them in a text file
#           classes = outputs['instances'].pred_classes.to("cpu")
#           scores = outputs["instances"].scores.to("cpu")
#           boxes = outputs['instances'].pred_boxes.to("cpu")
#           with open(text_result_path, "a") as text_file:
#             for clss, score, box in zip(classes, scores, boxes):
#               if clss in classes_to_keep:
#                 text_file.write(f"{frame_num} {clss} {score} " + " ".join(map(str, box.numpy())) + "\n")

#           # Display the resulting frame
#           # cv2.imshow('Frame',annotated_frame)
#           # cv2.waitKey()
#   # os.system(f"cp {text_result_path} {args.DetectionDetectorPath}")

#   # # When everything done, release the video capture object
#   cap.release()
#   # out_cap.release()

#   # # Closes all the frames
#   cv2.destroyAllWindows()


# TESTING with a new CODE--------------------------------------------------------------------------------------------------
# Some basic setup:
import torch
import os
from tqdm.auto import tqdm
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


# define a function to create models
def make_detectron2_model(device_name):
  # config_path = "./Detectors/detectron2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
  # model_weight_url = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
  # if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
  #   os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
  # model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"
  # cfg = get_cfg()
  # cfg.merge_from_file(config_path)
  # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
  # cfg.MODEL.WEIGHTS = model_weight_path
  # cfg.MODEL.DEVICE= device_name
  # model = DefaultPredictor(cfg)
  # return model

  # --------------------------------
  config_path = "./Detectors/detectron2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
  model_weight_url = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
  if "model_temp_280758.pkl" not in os.listdir("./Detectors/detectron2/"):
    os.system(f"wget {model_weight_url} -O ./Detectors/detectron2/model_temp_280758.pkl")
  model_weight_path = "./Detectors/detectron2/model_temp_280758.pkl"
  cfg = get_cfg()
  cfg.merge_from_file(config_path)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set threshold for this model

  model = build_model(cfg) # returns a torch.nn.Module
  DetectionCheckpointer(model).load(model_weight_path) # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
  model.train(False) # inference mode
  return model


import threading
import logging
import multiprocessing
ctx = multiprocessing.get_context('spawn')
import concurrent.futures

import sys
import glob
import json
import cv2
# Setup detectron2 logger
import pandas as pd
import numpy as np
# from mmdet.apis import init_detector, inference_detector

# see the list of MS_COCO class dict in the link below 
# https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
# the output of detectron2 is the class numbers + 1
classes_to_keep = [2, 5, 7] #3-1:car, 6-1:bus, 8-1:truck
Num_Proc = 1*torch.cuda.device_count() if torch.cuda.is_available() else 1
Batch_Size = 1
pbars = []
for i in range(max(1, Num_Proc)):
    pbars.append(tqdm(desc=f"proc:{i} device:{i%torch.cuda.device_count() if torch.cuda.is_available() else 0}"))

device_names = []
if torch.cuda.device_count() >= 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  for i in range(Num_Proc):
    device_names.append(f"cuda:{int(i%torch.cuda.device_count())}")
  device_ids = range(Num_Proc)
else:
  print("only using cpu")
  for i in range(Num_Proc):
    device_names.append(f"cpu")
  device_ids = range(Num_Proc)

models = {}
models_locks = {}
for device_name in [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]:
  models[device_name] = make_detectron2_model(device_name)
  models_locks[device_name] = threading.Lock()

results_lock = threading.Lock()


def write_outputs_to_file(outputs, frame_nums_inserted, text_file):
    for output, frame_number_i in zip(outputs, frame_nums_inserted):
        classes = output['instances'].pred_classes.to("cpu")
        scores = output["instances"].scores.to("cpu")
        boxes = output['instances'].pred_boxes.to("cpu")
        with results_lock:
            for clss, score, box in zip(classes, scores, boxes):
              if clss in classes_to_keep:
                text_file.write(f"{frame_number_i} {clss} {score} " + " ".join(map(str, box.numpy())) + "\n")

def detect_multi_process(device_name, proc_id, num_procs, video_path, Batch_Size, text_file):
  global pbars, models, models_locks
  pbar=pbars[proc_id]
  # create local video_capture for each thread/process
  cap = cv2.VideoCapture(video_path)
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  first_frame_to_capture = int(proc_id/num_procs * frames)
  last_frame_to_capture = int((proc_id+1)/num_procs * frames)
  cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_to_capture)
  pbar.reset(total = last_frame_to_capture - first_frame_to_capture)
  predictor = models[device_name]
  model_lock = models_locks[device_name]
  

  for frame_number_i in range(first_frame_to_capture, last_frame_to_capture, Batch_Size):
    inputs = []
    frame_nums_inserted = []
    temp_i = frame_number_i
    for _ in range(Batch_Size):
      ret, frame = cap.read()
      if not ret:
        temp_i += 1
        continue
      # print(frame.shape)
      # frame = frame.reshape(3, frame_height, frame_width)
      # # print(frame.shape)
      frame = torch.from_numpy(frame)
      frame = np.transpose(frame,(2,0,1))
      # print(frame.shape)
      inputs.append({"image":frame})
      # input=frame
      frame_nums_inserted.append(temp_i)
      temp_i += 1

    with model_lock:
      with torch.no_grad():
        outputs = predictor(inputs)
    write_outputs_to_file(outputs, frame_nums_inserted, text_file)
    
    pbar.update(Batch_Size)

  cap.release()


if __name__ == "__main__":
  # choose to run on CPU to GPU
  args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main cod
  video_path = args["Video"]
  text_result_path = args["DetectionDetectorPath"]


  with open(text_result_path, "w") as text_file:
      pass
  text_file = open(text_result_path, "a")

  print(f"device ids:{device_ids}")

  processes = []
  for device_name, device_id in zip(device_names, device_ids):
    processes.append(threading.Thread(target=detect_multi_process, args=(device_name,device_id, len(device_names),video_path, Batch_Size,text_file,)))

  # start threads 
  for i, p in enumerate(processes):
    p.start()
    print(f"`Process {i} started")

  # wait for threads to end
  for p in processes:
    p.join()
    
  # # Closes all the frames
  text_file.close()
  cv2.destroyAllWindows()
