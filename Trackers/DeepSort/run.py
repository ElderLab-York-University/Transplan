import sys
# import pickle as pickle
import json
# import pickle5 as pickle
import pandas as pd
from DeepSort.deep_sort import nn_matching
from DeepSort.deep_sort.detection import Detection
from DeepSort.deep_sort.tracker import Tracker
from DeepSort.tools import generate_detections as gdet
import cv2
from tqdm import tqdm
import numpy as np
import sys
CENTERTRACK_PATH = "./Trackers/CenterTrack/CenterTrack/src/lib/"
sys.path.insert(0, CENTERTRACK_PATH)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    # print(args)
    video_path = args["Video"]
    text_result_path = args["DetectionDetectorPath"]
    # print(detections.iloc[:,0])
    with open(args["TrackingPth"],"w") as out_file:

        max_cosine_distance = 0.5
        nn_budget = None
        model_filename = './Trackers/DeepSort/DeepSort/models/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        cap = cv2.VideoCapture(video_path)
        results=[]
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        detections_df = pd.read_pickle(args["DetectionPkl"])

        # with open(args["DetectionPkl"],"rb") as f:
        #     detections_df=pickle.load(f)

        for frame_num in tqdm(range(frames)):
            if (not cap.isOpened()):
                break
            ret, frame=cap.read()

            frame_bool= detections_df["fn"]==frame_num
            frame_detections=detections_df[frame_bool]
            bboxes=[]
            names=[]
            scores=[]

            for frame_detection in frame_detections.iterrows():
                bbox=np.array([frame_detection[1]["x1"], frame_detection[1]["y1"],frame_detection[1]["x2"]-frame_detection[1]["x1"], frame_detection[1]["y2"]-frame_detection[1]["y1"]])
                score=frame_detection[1]["score"]
                name=frame_detection[1]["class"]
                bboxes.append(bbox)
                scores.append(score)
                names.append(name)

            bboxes=np.array(bboxes)
            scores=np.array(scores)
            names=np.array(names)
            features = np.array(encoder(frame, bboxes))
            detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
            tracker.predict()
            tracker.update(detections)
            tracked_bboxes = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlwh()
                results.append([frame_num, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        for row in results:
            # print("YOO")
            print('%d,%d,%f,%f,%f,%f'%
            (row[0], row[1], row[2], row[3], row[2]+row[4], row[3]+row[5]),file=out_file)




        
