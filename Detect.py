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
    store_df_pickle_backup(args)

    return SucLog("Detection files stored")

def store_df_pickle_backup(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPklBackUp)

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl)

def remove_out_of_ROI(df, roi):
    poly_path = mplPath.Path(np.array(roi))
    mask = []
    print(len(mask), len(df))
    for i, row in tqdm(df.iterrows(), total=len(df)):
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        p = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]
        if poly_path.contains_point(p[0]) or poly_path.contains_point(p[1]) or poly_path.contains_point(p[2]) or poly_path.contains_point(p[3]):
            mask.append(True)
        else: mask.append(False)
    return df[mask]  

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
        # args.DetMask

    # 0. load the pklfile first
    df = pd.read_pickle(args.DetectionPklBackUp)
    # 1. condition on the post processing flags
    if not args.DetTh is None:
        df = detectionth(df, args)
    if args.DetMask:
        df = remove_out_of_ROI(df, args.MetaData["roi"])
    
    # store the edited df as txt
    detectors[args.Detector].df_txt(df, args.DetectionDetectorPath)
    # store the new txt as pkl
    store_df_pickle(args)
    return SucLog("detection post processing done")


def detectionth(df, args):
    df = df[df["score"] >= args.DetTh]
    return df

def visroi(args):
    # args.MetaData.roi
    # args.HomographyStreetView
    # args.HomographyTopView
    # args.HomographyNPY
    # args.VisROIPth

    alpha = 0.6
    M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    roi_rep = []
    roi = args.MetaData["roi"]
    for p in roi:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([new_point[0], new_point[1]])
    
    img1 = cv.imread(args.HomographyStreetView)
    img2 = cv.imread(args.HomographyTopView)
    img1p = cv.imread(args.HomographyStreetView)
    img2p = cv.imread(args.HomographyTopView)
    rows1, cols1, dim1 = img1.shape
    rows2, cols2, dim2 = img2.shape

    poly_path1 = mplPath.Path(np.array(roi))
    poly_path2 = mplPath.Path(np.array(roi_rep))
    
    for i in range(rows1):
        for j in range(cols1):
            if not poly_path1.contains_point([j, i]):
                img1[i][j] = [0, 0, 0]            
    for i in range(rows2):
        for j in range(cols2):
            if not poly_path2.contains_point([j, i]):
                img2[i][j] = [0, 0, 0]

    img1 = cv.addWeighted(img1, alpha, img1p, 1 - alpha, 0)
    img2 = cv.addWeighted(img2, alpha, img2p, 1 - alpha, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    ax1.set_title("camera view ROI")
    ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    ax2.set_title("top view ROI")
    plt.savefig(args.VisROIPth)

    return SucLog("Vis ROI executed successfully")