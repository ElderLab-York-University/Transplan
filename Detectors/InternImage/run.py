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
import cv2
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


if __name__ == "__main__":
    config_file = './Detectors/InternImage/InternImage/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py'
    checkpoint_file =   "./Detectors/InternImage/InternImage/checkpoint_dir/cascade_internimage_xl_fpn_3x_coco.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'device: {device_name}')

    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    video_path = args["Video"]
    print(video_path)
    text_result_path = args["DetectionDetectorPath"] 
    print(text_result_path)
    model = init_detector(config_file, checkpoint_file, device=device_name)
    video = mmcv.VideoReader(video_path)
    i=0
    rois=[]
    if "Rois" in args and args["UseRois"]:
        container=np.load(args["Rois"])
        data = [container[key] for key in container]
        print(args['Rois'])
        for roi in data:
            rois.append((roi.astype(int)))
    with open (text_result_path,"w") as f: 
        for frame in tqdm(video):
            if i%6==0:
                if "DetectionMask" in args and args["MaskDetections"] is not False:
                    container=np.load(args["DetectionMask"])
                    data = [container[key] for key in container]
                    for m in data:
                        m=(m!=0).astype(np.uint8)
                        
                        if(m.shape!=(video.height,video.width)):
                            
                            # print(m.shape, (video.height, video.width))
                            if (m.shape[0] == video.height-1 and m.shape[1] == video.width-1):
                                m=m[0:-1, 0:-1]
                            elif(m.shape[0]== video.height and m.shape[1]==video.width-1):
                                m=m[0:, 0:-1]      
                            elif(m.shape[0]==video.height-1 and m.shape[1] ==video.width):
                                m=m[0:-1, 0:]
                            elif(m.shape[0]== video.height and m.shape[1]==video.width+1):
                                m=m[0:, 0:-1]      
                            elif(m.shape[0]== video.height+1 and m.shape[1]==video.width):
                                m=m[0:-1, 0:]
                            elif(m.shape[0]==video.height+1 and m.shape[1]==video.width+1):
                                m=m[0:-1, 0:-1]                            
                            else:
                                print(m.shape, (video.height, video.width) )
                                # print('yo')
                                # input()
                            # print(m.shape, (video.height, video.width) )

                            # print(m.shape)
                        if(m.shape ==(video.height, video.width)):
                            frame=cv2.bitwise_and(frame,frame, mask=m)
                    frame=frame.astype(np.uint8)
                if "Rois" in args and args["UseRois"] is not False:
                    
                    for roi in rois:
                        # mask = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), dtype=np.uint8)
                        points = roi
                        points[0]= max(0, points[0])
                        points[1]= max(0,points[1])
                        points[2]= min(video.width, points[2])
                        points[3]= min(video.height, points[3])
                        
                        # cv2.fillPoly(mask, np.int32(points), (255))
                        # # print(roi[0][0])
                        # rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
                        # res = cv2.bitwise_and(frame,frame,mask = mask)
                        fr = frame[points[1]: points[3], points[0]: points[2]]
                        with torch.no_grad():
                            result = inference_detector(model, fr)
                        res= getbboxes(result)
                        bboxes=res[0]
                        labels=res[1]
                        for box,label in zip(bboxes,labels):
                            r=box
                            f.write(str(i) + " " + str(label) + " " + str(r[4]) + " " + str(r[0] +points[0])+ " " + str(r[1] +points[1]) + " " + str(r[2] +points[0])+ " " + str(r[3] +points[1]) +'\n')
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
            