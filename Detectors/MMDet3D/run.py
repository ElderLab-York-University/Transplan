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

from mmdet3d.structures import points_cam2img

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
    camera_name=args['Dataset'].split("/")[-1][-3:]    
    intrinsic_file=args['INTRINSICS_PATH']
    with open(intrinsic_file,'r') as f:
      intrinsic_data=json.load(f)
      intrinsic_mat=np.array(intrinsic_data[camera_name]['intrinsic_matrix'])
      intrinsic_mat=intrinsic_mat[0:3,0:3]

    cam2img = torch.tensor(intrinsic_mat, dtype=torch.float)
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
        exists=True
        try:
            instances.scores
        except Exception as e:
            exists=False
        if(exists):
            scores = instances.scores
            bboxes = instances.bboxes # x1, y1, x2, y2
            labels = instances.labels
        else:
            return None, None, None
    return bboxes, labels, scores
def preds_3D_from_result(result):
    if 'pred_instances_3d' in result:
        instances = result.pred_instances_3d
        scores = instances.scores_3d
        bboxes = instances.bboxes_3d # x1, y1, x2, y2
        labels = instances.labels_3d
        corners = bboxes.corners.reshape(-1, 3)

    return bboxes, labels, scores, corners

def iou(box_a, box_b):

    box_a_top_right_corner = [box_a[1]+box_a[4], box_a[2]+box_a[5]]
    box_b_top_right_corner = [box_b[1]+box_b[4], box_b[2]+box_b[5]]

    box_a_area = (box_a[4]) * (box_a[5])
    box_b_area = (box_b[4]) * (box_b[5])

    xi = max(box_a[1], box_b[1])
    yi = max(box_a[2], box_b[2])

    corner_x_i = min(box_a_top_right_corner[0], box_b_top_right_corner[0])
    corner_y_i = min(box_a_top_right_corner[1], box_b_top_right_corner[1])

    intersection_area = max(0, corner_x_i - xi) * max(0, corner_y_i - yi)

    intersection_l_min = max(box_a[3], box_b[3])
    intersection_l_max = min(box_a[3]+box_a[6], box_b[3]+box_b[6])
    intersection_length = intersection_l_max - intersection_l_min

    iou = (intersection_area * intersection_length) / float(box_a_area * box_a[6] + box_b_area * box_b[6]
                                                            - intersection_area * intersection_length + 1e-5)

    return iou


def nms(original_boxes, iou_threshold=1):

    boxes_probability_sorted = original_boxes[np.flip(np.argsort(original_boxes[:, 0]))]
    box_indices = np.arange(0, len(boxes_probability_sorted))
    suppressed_box_indices = []
    tmp_suppress = []

    while len(box_indices) > 0:

        if box_indices[0] not in suppressed_box_indices:
            selected_box = box_indices[0]
            tmp_suppress = []

            for i in range(len(box_indices)):
                if box_indices[i] != selected_box:
                    selected_iou = iou(boxes_probability_sorted[selected_box], boxes_probability_sorted[box_indices[i]])
                    if selected_iou > iou_threshold:
                        suppressed_box_indices.append(box_indices[i])
                        tmp_suppress.append(i)

        box_indices = np.delete(box_indices, tmp_suppress, axis=0)
        box_indices = box_indices[1:]

    # preserved_boxes = np.delete(boxes_probability_sorted, suppressed_box_indices, axis=0)
    indicies=np.arange(original_boxes.shape[0])
    non_suppresed_box_indicies=np.delete(indicies,suppressed_box_indices)    
    return non_suppresed_box_indicies, suppressed_box_indices

def transform_box(box, corner ):
    corners_3d = corner
    num_bbox = box.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    camera_name=args['Dataset'].split("/")[-1][-3:]    
    intrinsic_file=args['INTRINSICS_PATH']
    with open(intrinsic_file,'r') as f:
      intrinsic_data=json.load(f)
      intrinsic_mat=np.array(intrinsic_data[camera_name]['intrinsic_matrix'])
      intrinsic_mat=intrinsic_mat[0:3,0:3]

    cam2img = torch.tensor(intrinsic_mat, device="cuda:0", dtype=torch.float)

    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    transformed_corners=(uv_origin[..., :2].reshape(num_bbox,16))

    return transformed_corners

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
    if(args['Detect3D']):
        with open (text_result_path,"w") as f, open(args['Detection3DPath'], 'w') as b: 
            print(args['Detection3DPath'])
            for frame in tqdm(video_reader):
                if(fn%args['FrameRate']==0):
                    result = get_full_frame_result(frame, model, test_pipeline)
                    bboxes, labels, scores = preds_2D_from_result(result)
                    if(bboxes is not None):
                        for box, label, score in zip(bboxes,labels, scores):
                                r=box
                                f.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                    bboxes_3d, labels_3d, scores_3d, corners = preds_3D_from_result(result)
                    # bboxes_nms= np.zeros((bboxes_3d.shape[0],7))
                    # bboxes_nms[:,0] = scores_3d.cpu().numpy()
                    # bboxes_nms[:,1:]=  bboxes_3d.cpu().numpy()[:, 0:6]
                    # non_suppresed_box_indicies,suppresed_indicies= nms(bboxes_nms)
                    # bboxes_3d= bboxes_3d[non_suppresed_box_indicies]
                    # corners=bboxes_3d.corners.reshape(-1, 3)
                    transformed_boxes_3d=transform_box(bboxes_3d, corners)

                    for box, label, score, corner in zip(transformed_boxes_3d,labels_3d, scores_3d, corners):
                            r=box
                            b.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]} {box[7]} {box[8]} {box[9]} {box[10]} {box[11]} {box[12]} {box[13]} {box[14]} {box[15]}    \n")

                fn+=1
    else:
        with open (text_result_path,"w") as f: 
            for frame in tqdm(video_reader):
                if(fn%args['FrameRate']==0):
                    result = get_full_frame_result(frame, model, test_pipeline)
                    bboxes, labels, scores = preds_2D_from_result(result)
                    for box, label, score in zip(bboxes,labels, scores):
                            r=box
                            f.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                fn+=1

