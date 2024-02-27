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
sys.path.append("./Detectors/InternImage/InternImage/detection")
from ops_dcnv3 import modules as mod
import mmdet_custom  # noqa: F401,F403
import mmcv_custom  # noqa: F401,F403
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

def getbboxes(result):
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
    return [bboxes, labels]
# def shift_predictions(det_data_samples: SampleList,
#                       offsets: Sequence[Tuple[int, int]],
#                       src_image_shape: Tuple[int, int]) -> SampleList:
#     """Shift predictions to the original image.

#     Args:
#         det_data_samples (List[:obj:`DetDataSample`]): A list of patch results.
#         offsets (Sequence[Tuple[int, int]]): Positions of the left top points
#             of patches.
#         src_image_shape (Tuple[int, int]): A (height, width) tuple of the large
#             image's width and height.
#     Returns:
#         (List[:obj:`DetDataSample`]): shifted results.
#     """
    
#     assert len(det_data_samples) == len(
#         offsets), 'The `results` should has the ' 'same length with `offsets`.'
#     shifted_predictions = []
#     for det_data_sample, offset in zip(det_data_samples, offsets):
#         pred_inst = det_data_sample.pred_instances.clone()

#         # Check bbox type
#         if pred_inst.bboxes.size(-1) == 4:
#             # Horizontal bboxes
#             shifted_bboxes = shift_bboxes(pred_inst.bboxes, offset)
#         elif pred_inst.bboxes.size(-1) == 5:
#             # Rotated bboxes
#             shifted_bboxes = shift_rbboxes(pred_inst.bboxes, offset)
#         else:
#             raise NotImplementedError

#         # shift bboxes and masks
#         pred_inst.bboxes = shifted_bboxes
#         if 'masks' in det_data_sample:
#             pred_inst.masks = shift_masks(pred_inst.masks, offset,
#                                           src_image_shape)

#         shifted_predictions.append(pred_inst.clone())

#     shifted_predictions = InstanceData.cat(shifted_predictions)

#     return shifted_predictions
# def shift_bboxes(bboxes: torch.Tensor, offset: Sequence[int]):
#     """Shift bboxes with offset.

#     Args:
#         bboxes (Tensor): The bboxes need to be translated.
#             With shape (n, 4), which means (x, y, x, y).
#         offset (Sequence[int]): The translation offsets with shape of (2, ).
#     Returns:
#         Tensor: Shifted rotated bboxes.
#     """
#     offset_tensor = bboxes.new_tensor(offset)
#     shifted_bboxes = bboxes.clone()
#     shifted_bboxes[:, 0:2] = shifted_bboxes[:, 0:2] + offset_tensor
#     shifted_bboxes[:, 2:4] = shifted_bboxes[:, 2:4] + offset_tensor
#     return shifted_bboxes
def shift_predictions(results,offsets, img_shape):
  shifted_predictions=[]
  labels=[]
  for result,offset in zip(results,offsets):
    for i,res in enumerate(result):
      if(len(res)>0):
        offset_res=res.copy()
        offset_res[:,0]+=offset[0]
        offset_res[:,1]+=offset[1]
        offset_res[:,2]+=offset[0]
        offset_res[:,3]+=offset[1]      
        shifted_predictions.extend(offset_res)
        labels.extend([i]*offset_res.shape[0])
  return shifted_predictions,labels
def get_roi_results(frame, model, args):
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
        result = inference_detector(model, sliced_frames)
    offsets=np.array(rois)[:,0:2]
    shifted=shift_predictions(result, offsets, (height,width))    
    # shifted_result = result[0].clone()
    # shifted_result.pred_instances = shifted
    return shifted
if __name__ == "__main__":
  args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code

  config_file = args["MMDetConfig"]
  checkpoint_file =  args["MMDetCheckPoint"]
 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f'device: {device_name}')

  video_path = args["Video"]
#   print(video_path)
  text_result_path = args["DetectionDetectorPath"] 
#   print(text_result_path)
  model = init_detector(config_file, checkpoint_file, device=device_name)
  video = mmcv.VideoReader(video_path)
  i=0
  with open (text_result_path,"w") as f: 
      for frame in tqdm(video):
        if(i%6==0):
          if(args["Rois"]):
            res=get_roi_results(frame,model,args)
          else:
            with torch.no_grad():
              result = inference_detector(model, frame)
            res= getbboxes(result)
          bboxes=res[0]
          labels=res[1]
          for box,label in zip(bboxes,labels):
                r=box
                f.write(str(i) + " " + str(label) + " " + str(r[4]) + " " + str(r[0])+ " " + str(r[1]) + " " + str(r[2])+ " " + str(r[3]) +'\n')
        i=i+1

