# Author: Sajjad P. Savaoji April 27 2022
# This py file will handle all the detections
from Libs import *
from Utils import *

# import all detectros here
# -------------------------- 
# import Detectors.detectron2.detect
import Detectors.OpenMM.detect
import Detectors.YOLOv5.detect
import Detectors.detectron2.detect

# --------------------------

detectors = {}
detectors["detectron2"] = Detectors.detectron2.detect
detectors["OpenMM"] = Detectors.OpenMM.detect
detectors["YOLOv5"] = Detectors.YOLOv5.detect


def detect(args):
    # check if detector names is valid
    if args.Detector not in os.listdir("./Detectors/"):
        return FailLog("Detector not recognized in ./Detectors/")

    current_detector = detectors[args.Detector]
    current_detector.detect(args)
    store_df_pickle(args)
    return SucLog("Detection files stored")

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl)

def visdetect(args):
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    detection_df = detectors[args.Detector].df(args)

    # open the original video and process it
    cap = cv2.VideoCapture(args.Video)

    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(args.VisDetectionPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue
        for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
            frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
        out_cap.write(frame)

    cap.release()
    out_cap.release()
    return SucLog("Detection visualized on the video")
        

def draw_box_on_image(img, x1, y1, x2, y2, c=(255, 0, 0), thickness=2):
    sta_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    img = cv2.rectangle(img, sta_point, end_point, c, thickness)
    return img

def detectpostproc(args):
    # args to use in this function
        # args.DetPostProc
        # args.DetTh

    # 0. load the pklfile first
    df = pd.read_pickle(args.DetectionPkl)

    # 1. condition on the post processing flags
    if not args.DetTh is None:
        df = detectionth(df, args)

    # add other post processing steps
    
    # store the edited df as txt
    detectors[args.Detector].df_txt(df, args.DetectionDetectorPath)
    # store the new txt as pkl
    store_df_pickle(args)
    return SucLog("detection post processing done")


def detectionth(df, args):
    df = df[df["score"] >= args.DetTh]
    return df