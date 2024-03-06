import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList

from mmdet3d.apis import MonoDet3DInferencer
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type

import copy
import json
from tqdm import tqdm
import sys
import torch
import mmcv


def get_full_frame_result(frame, model, test_pipeline):
    data = []
    # replace the img_path in data_info with img
    cam2img = [[721.5377, 0.0, 609.5593, 44.85728],
               [0.0, 721.5377, 172.854, 0.2163791],
               [0.0, 0.0, 1.0, 0.002745884],
               [0.0, 0.0, 0.0, 1.0]]
    data_ = {"img":frame, "cam2img":cam2img}
    data_ = test_pipeline(data_)
    data.append(data_)
    collate_data = pseudo_collate(data)
    with torch.no_grad():
        results = model.test_step(collate_data)
    return results[0]

def preds_2D_from_result(result):
    if 'pred_instances' in result:
        instances = result.pred_instances
        scores = instances.scores
        bboxes = instances.bboxes # x1, y1, x2, y2
        labels = instances.labels
    return bboxes, labels, scores

if __name__ == "__main__":
    args = json.loads(sys.argv[-1])
    # config and check point will come from args
    config_file     = args["MMDetConfig"]
    checkpoint_file = args["MMDetCheckPoint"]
    # select compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')
    # set video and result path
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"] 

    # init model structure from config file
    # load weights from checkpoint_file
    # put model on device
    model = init_model(config = config_file, checkpoint=checkpoint_file, device=device_name)
    
    # change test pipeline to load from ndarray images
    # using monodet only for _init_pipeline
    mono_det_inferencer_obj = MonoDet3DInferencer(model=config_file, weights=checkpoint_file, device=device_name)
    test_pipeline = mono_det_inferencer_obj._init_pipeline(copy.deepcopy(model.cfg))

    # process frame by frame
    video_reader = mmcv.VideoReader(video_path)
    fn=0
    with open (text_result_path,"w") as f: 
        for frame in tqdm(video_reader):
            if(fn%args['FrameRate']==0):
                result = get_full_frame_result(frame, model, test_pipeline)
                bboxes, labels, scores = preds_2D_from_result(result)
                for box, label, score in zip(bboxes,labels, scores):
                        r=box
                        f.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            fn+=1