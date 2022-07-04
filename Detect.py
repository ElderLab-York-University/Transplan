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
    remove_extraneous_boxes(args)
    store_df_pickle(args)
    return SucLog("Detection files stored")

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl)

def remove_extraneous_boxes(args):
    # detection_box=[[588.280087527352, 2028-619.126914660832], [1585.43544857768, 2028-628.317286652079],[2603.26914660832, 2028-638.75],[6.98905908096299, 2028-1526.67614879650],[588.280087527352, 2028-619.126914660832]]
    detection_box=[[509.75, 2028-602.75], [1637.75, 2028-515.75],[2672.75, 2028-626.75],[2.75, 2028-1550.75],[509.75, 2028-602.75]]
    
    poly_path = mplPath.Path(np.array(detection_box))    

    text_result_path = args.DetectionDetectorPath
    newLines=[]
    with open(text_result_path, 'r') as f:
            lines=f.readlines()        
            for line in lines:
                lineArr=line.split(" ")
                [x1 , y1 , x2 , y2]= lineArr[3:]
                point1=[float(x1), 2028-float(y1)]
                point2=[float(x2), 2028-float(y2)]
                if poly_path.contains_point(point1) or poly_path.contains_point(point2): 
                    newLines.append(line.rstrip('\n'))  
             
    with open (text_result_path,"w") as f: 
        for line in newLines:
            print(line, file=f)

# 509.75	602.75
# 1637.75	515.75
# 2672.75	626.75
# 2.75	1550.75    

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
