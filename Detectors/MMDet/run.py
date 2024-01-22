import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import json
from tqdm import tqdm
import sys
import numpy as np
import torch
import mmcv
from sahi.slicing import slice_image

# Nov 20 2023 NOTE
# as of today SAHI does not support bbox with negatiev coordinate
# while mmdet might output such bboxes in the model
# to merge it with sahi there are two options
# 1. clip bbox to be olways positive
# 2. modify sahi to accept negative bbox values
# I believe second option is better as it preserves more info about
# detected bonding boxes
# for that reason we will temporary implement the bbox shifring locally
# and we will merge to late releases of SAHI or MMDet
from typing import Sequence, Tuple
import torch
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample, SampleList
def merge_results_by_nms(results: SampleList, offsets: Sequence[Tuple[int,
                                                                      int]],
                         src_image_shape: Tuple[int, int],
                         nms_cfg: dict) -> DetDataSample:
    """Merge patch results by nms.

    Args:
        results (List[:obj:`DetDataSample`]): A list of patch results.
        offsets (Sequence[Tuple[int, int]]): Positions of the left top points
            of patches.
        src_image_shape (Tuple[int, int]): A (height, width) tuple of the large
            image's width and height.
        nms_cfg (dict): it should specify nms type and other parameters
            like `iou_threshold`.
    Returns:
        :obj:`DetDataSample`: merged results.
    """
    shifted_instances = shift_predictions(results, offsets, src_image_shape)

    _, keeps = batched_nms(
        boxes=shifted_instances.bboxes,
        scores=shifted_instances.scores,
        idxs=shifted_instances.labels,
        nms_cfg=nms_cfg)
    merged_instances = shifted_instances[keeps]

    merged_result = results[0].clone()
    merged_result.pred_instances = merged_instances
    return merged_result

def shift_predictions(det_data_samples: SampleList,
                      offsets: Sequence[Tuple[int, int]],
                      src_image_shape: Tuple[int, int]) -> SampleList:
    """Shift predictions to the original image.

    Args:
        det_data_samples (List[:obj:`DetDataSample`]): A list of patch results.
        offsets (Sequence[Tuple[int, int]]): Positions of the left top points
            of patches.
        src_image_shape (Tuple[int, int]): A (height, width) tuple of the large
            image's width and height.
    Returns:
        (List[:obj:`DetDataSample`]): shifted results.
    """
    
    assert len(det_data_samples) == len(
        offsets), 'The `results` should has the ' 'same length with `offsets`.'
    shifted_predictions = []
    for det_data_sample, offset in zip(det_data_samples, offsets):
        pred_inst = det_data_sample.pred_instances.clone()

        # Check bbox type
        if pred_inst.bboxes.size(-1) == 4:
            # Horizontal bboxes
            shifted_bboxes = shift_bboxes(pred_inst.bboxes, offset)
        elif pred_inst.bboxes.size(-1) == 5:
            # Rotated bboxes
            shifted_bboxes = shift_rbboxes(pred_inst.bboxes, offset)
        else:
            raise NotImplementedError

        # shift bboxes and masks
        pred_inst.bboxes = shifted_bboxes
        if 'masks' in det_data_sample:
            pred_inst.masks = shift_masks(pred_inst.masks, offset,
                                          src_image_shape)

        shifted_predictions.append(pred_inst.clone())

    shifted_predictions = InstanceData.cat(shifted_predictions)

    return shifted_predictions

def shift_rbboxes(bboxes: torch.Tensor, offset: Sequence[int]):
    """Shift rotated bboxes with offset.

    Args:
        bboxes (Tensor): The rotated bboxes need to be translated.
            With shape (n, 5), which means (x, y, w, h, a).
        offset (Sequence[int]): The translation offsets with shape of (2, ).
    Returns:
        Tensor: Shifted rotated bboxes.
    """
    offset_tensor = bboxes.new_tensor(offset)
    shifted_bboxes = bboxes.clone()
    shifted_bboxes[:, 0:2] = shifted_bboxes[:, 0:2] + offset_tensor
    return shifted_bboxes

def shift_bboxes(bboxes: torch.Tensor, offset: Sequence[int]):
    """Shift bboxes with offset.

    Args:
        bboxes (Tensor): The bboxes need to be translated.
            With shape (n, 4), which means (x, y, x, y).
        offset (Sequence[int]): The translation offsets with shape of (2, ).
    Returns:
        Tensor: Shifted rotated bboxes.
    """
    offset_tensor = bboxes.new_tensor(offset)
    shifted_bboxes = bboxes.clone()
    shifted_bboxes[:, 0:2] = shifted_bboxes[:, 0:2] + offset_tensor
    shifted_bboxes[:, 2:4] = shifted_bboxes[:, 2:4] + offset_tensor
    return shifted_bboxes

# actual code begis
def preds_from_result(result):
    if 'pred_instances' in result:
        instances = result.pred_instances
        scores = instances.scores
        bboxes = instances.bboxes # x1, y1, x2, y2
        labels = instances.labels
    return bboxes, labels, scores

def get_sahi_result(frame, model, test_pipeline, args):
    height, width = frame.shape[:2]
    sliced_image_object = slice_image(
        frame,
        slice_height=args["SahiPatchSize"],
        slice_width=args["SahiPatchSize"],
        auto_slice_resolution=False,
        overlap_height_ratio=args["SahiPatchOverlapRatio"],
        overlap_width_ratio=args["SahiPatchOverlapRatio"],
    )

    # if patch batch is not specified, use all the patches all the time
    if args["SahiPatchBatchSize"]:
        SahiPatchBatchSize = args["SahiPatchBatchSize"]
    else:
        SahiPatchBatchSize = len(sliced_image_object)

    slice_results = []
    start = 0
    while True:
        # prepare batch slices
        end = min(start + SahiPatchBatchSize, len(sliced_image_object))
        images = []
        for sliced_image in sliced_image_object.images[start:end]:
            images.append(sliced_image)

        # forward the model
        slice_results.extend(inference_detector(model, images, test_pipeline=test_pipeline))

        if end >= len(sliced_image_object):
            break
        start += SahiPatchBatchSize

    image_result = merge_results_by_nms(
    slice_results,
    sliced_image_object.starting_pixels,
    src_image_shape=(height, width),
    nms_cfg={
        'type': 'nms',
        'iou_threshold': args["SahiNMSTh"]
    })

    result = image_result.cpu()
    return result

def get_full_frame_result(frame, model, test_pipeline, args):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    result = result.cpu()
    return result
def get_roi_results(frame, model, test_pipeline, args):
    rois=[]
    container=np.load(args["RoiNpz"])
    data = [container[key] for key in container]
    for roi in data:
        rois.append((roi.astype(int)))
    sliced_frames=[]
    sliced_results=[]
    height, width = frame.shape[:2]    
    for roi in rois:
        # mask = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), dtype=np.uint8)
        points = roi
        points[0]= max(0, points[0])
        points[1]= max(0,points[1])
        points[2]= min(width, points[2])
        points[3]= min(height, points[3])

        # cv2.fillPoly(mask, np.int32(points), (255))
        # # print(roi[0][0])
        # rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
        # res = cv2.bitwise_and(frame,frame,mask = mask)
        sliced_frames.append(frame[points[1]: points[3], points[0]: points[2]])
    with torch.no_grad():
        result = inference_detector(model, sliced_frames,test_pipeline=test_pipeline)
    offsets=np.array(rois)[:,0:2]
    shifted=shift_predictions(result, offsets,(height,width))
    shifted_result = result[0].clone()
    shifted_result.pred_instances = shifted
    return shifted_result.cpu()
if __name__ == "__main__":
    args = json.loads(sys.argv[-1])
    # config and check point will come from args
    config_file = args["MMDetConfig"]
    checkpoint_file = args["MMDetCheckPoint"]
    # select compute device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')
    # set video and result path
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"] 

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=device_name)

    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    video_reader = mmcv.VideoReader(video_path)
    fn=0
    with open (text_result_path,"w") as f: 
        for frame in tqdm(video_reader):
            if args["SAHI"]:
                result = get_sahi_result(frame, model, test_pipeline, args)
            elif args['Rois']:
                result=get_roi_results(frame, model, test_pipeline,args)
            else:
                result = get_full_frame_result(frame, model, test_pipeline, args)

            bboxes, labels, scores = preds_from_result(result)
            for box, label, score in zip(bboxes,labels, scores):
                    r=box
                    f.write(f"{fn} {label} {score} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            fn+=1