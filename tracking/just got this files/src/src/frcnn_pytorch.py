import cv2
import torch
import torch.nn.parallel
import torchvision
import torchvision.transforms as T
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
#import colormap
from matplotlib import cm
norm = matplotlib.colors.Normalize(vmin=0, vmax=16)

vidPath = "/home/poorna/Downloads/GXAB0755.MP4"
folders = vidPath.split('/')
filename = folders[-1].split('.')[0]
cap = cv2.VideoCapture(vidPath)
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
model = torch.nn.DataParallel(model).cuda()
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold):
    transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    #img = torch.autograd.Variable(img,volatile=True).cuda()
    print("Getting prediction")
    t = time.time()
    with torch.no_grad():
        pred = model([img]) # Pass the image to the model
    print("time taken - " + str(time.time()-t))
    return pred

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 960,600)
frame_num = 0
dont_write = 0
with open(filename+'_temp.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    while cap.isOpened():
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        frame_num+=1
        if not hasFrame:
            #cv2.waitKey()
            print('No frame!')
            continue


        img = frame
        predictions = get_prediction(frame, 0.7)
        for box, label, score in zip(predictions[0]['boxes'],predictions[0]['labels'], predictions[0]['scores']):
            if label.item() == 3 and score.item() > 0.7:
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), color=(255,0,0), thickness=5)

        cv2.imshow('frame',frame)
        cv2.waitKey(1)



writeFile.close()
