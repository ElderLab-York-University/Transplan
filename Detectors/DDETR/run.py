from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import requests

import json
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import sys

classes_to_keep = [2, 5, 7] #3-1:car, 6-1:bus, 8-1:truck


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
# model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

# inputs = image_processor(images=image, return_tensors="pt")
# outputs = model(**inputs)

# # convert outputs (bounding boxes and class logits) to COCO API
# target_sizes = torch.tensor([image.size[::-1]])
# results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
#     0
# ]
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )

if __name__ == "__main__":
    # decide which device to run the model on (GPU/CPU)
    n_GPUs = 1*torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda:0" if n_GPUs > 0 else "cpu" # for now lets use only one GPU

     # args in a dictionary here where it was a argparse.NameSpace in the main cod
    args = json.loads(sys.argv[-1])
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"]

    # create video capture objec using openCV
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_sizes = [(frame_height, frame_width)]
    Batch_size = 1

    # create object detection model
    image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    model = model.to(device)

    # reseat the text result file
    with open(text_result_path, "w") as text_file:
      pass

    with open(text_result_path, "a") as text_file:

        # process frames
        for fn in tqdm(range(1, frames+1)):
            ret, frame = cap.read()
            if not ret: continue

            inputs = image_processor(images=frame, return_tensors="pt")
            inputs = inputs.to(device)
            # print(type(inputs))
            outputs = model(**inputs)
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            for score, label, box in zip(results["scores"].to("cpu"), results["labels"].to("cpu"), results["boxes"].to("cpu")):
                if not int(label) in classes_to_keep: continue
                box = [round(i, 2) for i in box.tolist()]
                text_file.write(f"{fn } {int(label)} {score} " + " ".join(map(str, box.numpy())) + "\n")