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
import matplotlib.pyplot as plt
sys.path.append("./Segmenters/InternImage/InternImage/detection")
from ops_dcnv3 import modules as mod
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import pandas as pd

def get_results_path_with_frame(results_path, fn):
    splited_path = results_path.split(".")
    return ".".join(splited_path[:-1] + [str(fn)] + splited_path[-1:])

def make_data_dict():
    data_dict = {}
    data_dict["fn"]    = []
    data_dict["x1"]    = []
    data_dict["y1"]    = []
    data_dict["x2"]    = []
    data_dict["y2"]    = []
    data_dict["score"] = []
    data_dict["class"] = []
    data_dict["mask"]  = []
    return data_dict

def seperate_results(result):
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

    with_masks = []
    for res in segm_result:
        if len(res)>0:
            with_masks.append(res)
    masks = np.vstack(with_masks)

    return bboxes, labels, masks


if __name__ == "__main__":
    config_file = './Segmenters/InternImage/InternImage/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py'
    checkpoint_file = './Segmenters/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_3x_coco.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')

    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    results_path = args["SegmentPkl"]

    model = init_detector(config_file, checkpoint_file, device=device_name)
    video = mmcv.VideoReader(video_path)

    fn=0
    for frame in tqdm(video):
        data_dict = make_data_dict()
        results_path_fn = get_results_path_with_frame(results_path, fn)

        with torch.no_grad():
            result = inference_detector(model, frame)
        bboxes, labels, segms= seperate_results(result)

        for bbox, label, segm in zip(bboxes, labels, segms):
            x1, y1, x2, y2, score = bbox
            cropped_mask = segm[int(y1):int(y2), int(x1):int(x2)]
            data_dict["x1"].append(x1)
            data_dict["y1"].append(y1)
            data_dict["x2"].append(x2)
            data_dict["y2"].append(y2)
            data_dict["score"].append(score)
            data_dict["class"].append(label)
            data_dict["fn"].append(fn)
            data_dict["mask"].append(cropped_mask)

        df = pd.DataFrame.from_dict(data_dict)
        df.to_pickle(results_path_fn, protocol=4, compression=None)
        fn += 1