# Author: Sajjad P. Savaoji April 27 2022
# This py file will handle all the detections
from Libs import *
from Utils import *

# import all detectros here
# -------------------------- 
import Detectors.YOLOv5.detect
import Detectors.detectron2.detect
import Detectors.YOLOv8.detect
import Detectors.DDETR.detect
import Detectors.InternImage.detect
import Detectors.GTHW7.detect
import Detectors.GTHW73D.detect
import Detectors.RTMDet.detect
import Detectors.YoloX.detect
import Detectors.CascadeRCNN.detect
import Detectors.DeformableDETR.detect
import Detectors.CenterNet.detect
# --------------------------
detectors = {}
detectors["detectron2"]     = Detectors.detectron2.detect
detectors["YOLOv5"]         = Detectors.YOLOv5.detect
detectors["YOLOv8"]         = Detectors.YOLOv8.detect
detectors["DDETR"]          = Detectors.DDETR.detect
detectors["InternImage"]    = Detectors.InternImage.detect
detectors["GTHW7"]          = Detectors.GTHW7.detect
detectors["RTMDet"]         = Detectors.RTMDet.detect
detectors["YoloX"]          = Detectors.YoloX.detect
detectors["CascadeRCNN"]    = Detectors.CascadeRCNN.detect
detectors["DeformableDETR"] = Detectors.DeformableDETR.detect
detectors["CenterNet"]      = Detectors.CenterNet.detect
detectors["GTHW73D"]        = Detectors.GTHW73D.detect

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
    df.to_pickle(args.DetectionPklBackUp, protocol=4)

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl, protocol=4)

def remove_out_of_ROI(df, roi):
    poly_path = mplPath.Path(np.array(roi))
    mask = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="rm oROI bbox"):
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        p = [(x2+x1)/2, y2]
        if poly_path.contains_point(p):
            mask.append(True)
        else: mask.append(False)
    return df[mask]  

def visdetect(args):
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")    
    if(args.Detector == "GTHW73D"):
        visdetect_3d(args)
    else:
        visdetect_2d(args)
        
def visdetect_2d(args):
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    detection_df = pd.read_pickle(args.DetectionPkl)

    # open the original video and process it
    cap = cv2.VideoCapture(args.Video)

    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_cap = cv2.VideoWriter(args.VisDetectionPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))

    if not args.ForNFrames is None:
        frames = args.ForNFrames
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

# remove this condition and add a new function
def visdetect_3d(args):
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    # parse detection df using detector module
    detection_df = pd.read_pickle(args.DetectionPkl)
    # open the original video and process it
    cap = cv2.VideoCapture(args.Video)
    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_cap = cv2.VideoWriter(args.VisDetectionPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    if not args.ForNFrames is None:
        frames = args.ForNFrames
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue
        for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
            
            frame = draw_3Dbox_on_image(frame, row)
            
        out_cap.write (frame)
    cap.release()
    out_cap.release()
    return SucLog("3D detection visualized")

# function to plot 3D bbox on image
def draw_3Dbox_on_image(frame, row):
    pts = np.array([[row.x1,row.y1],[row.x2,row.y2],[row.x3,row.y3],[row.x4,row.y4]], np.int32)
    pts = pts.reshape((-1,1,2))    
    frame=cv2.polylines(frame,[pts],True,(0,255,255))
    pts = np.array([[row.x5,row.y5],[row.x6,row.y6],[row.x7,row.y7],[row.x8,row.y8]], np.int32)
    pts = pts.reshape((-1,1,2))  
    frame=cv2.polylines(frame,[pts],True,(0,255,255))        
    # frame = cv.circle(frame, (int(row.x3),int(row.y3)), 1, (255,0,0), 10)
    frame=cv2.line(frame, (int(row.x1),int(row.y1)), (int(row.x5),int(row.y5)), (0,255,255))
    frame=cv2.line(frame, (int(row.x2),int(row.y2)), (int(row.x6),int(row.y6)), (0,255,255))
    frame=cv2.line(frame, (int(row.x3),int(row.y3)), (int(row.x7),int(row.y7)), (0,255,255))
    frame=cv2.line(frame, (int(row.x4),int(row.y4)), (int(row.x8),int(row.y8)), (0,255,255)) 
    return frame        

def draw_box_on_image(img, x1, y1, x2, y2, c=(255, 0, 0), thickness=2):
    sta_point = (int(x1), int(y1))
    end_point = (int(x2), int(y2))
    img = cv2.rectangle(img, sta_point, end_point, c, thickness)
    return img

def draw_point_on_image(img, x, y, radius = 5, c=(0, 255, 0), thickness=4):
    cv.circle(img, (int(x),int(y)), radius=radius, color=c, thickness=thickness)
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

    if args.classes_to_keep:
        df = filter_det_class(df, args)

    if args.DetMask:
        df = remove_out_of_ROI(df, args.MetaData["roi"])
    
    # store the edited df as txt
    detectors[args.Detector].df_txt(df, args.DetectionDetectorPath)
    # store the new txt as pkl
    store_df_pickle(args)
    return SucLog("detection post processing done")


def detectionth(df, args):
    print("performing thresholding")
    df = df[df["score"] >= args.DetTh]
    return df

def filter_det_class(df, args):
    mask = df["fn"] < 0
    for clss in args.classes_to_keep:
        clss_mask = df["class"] == clss
        mask = np.logical_or(mask, clss_mask)
    return df[mask]

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
    roi_group = args.MetaData["roi_group"]
    for p in roi:
        point = np.array([p[0], p[1], 1])
        new_point = M.dot(point)
        new_point /= new_point[2]
        roi_rep.append([int(new_point[0]), int(new_point[1])])
    
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

    # draw lines according to roi_group
    for i in range(0, len(roi)):
        p1 = tuple(roi[i-1])
        p2 = tuple(roi[i])
        q1 = tuple(roi_rep[i-1])
        q2 = tuple(roi_rep[i])
        group= roi_group[i-1]
        color = roi_color_dict[group]
        cv2.line(img1, p1, p2, color, 25)
        cv2.line(img2, q1, q2, color, 5)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    ax1.set_title("camera view ROI")
    ax2.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    ax2.set_title("top view ROI")
    plt.savefig(args.VisROIPth)

    return SucLog("Vis ROI executed successfully")

def format_frame_number(frame_number, number_of_frames):
    # Calculate the number of digits needed to represent number_of_frames
    num_digits = len(str(number_of_frames))
    
    # Format frame_number as a string with leading zeros
    formatted_frame_number = str(frame_number).zfill(num_digits)
    
    return formatted_frame_number

def format_frame_id(file_path):
    return hash(os.path.abspath(file_path))

def format_bbox_id(bbox_index, file_path):
    return hash(f"{file_path}.{bbox_index}")

def image_path_from_details(video_path, frame_number, number_of_frames, output_directory):
        frame_number = int(frame_number)
        number_of_frames = int(number_of_frames)
        # Construct the output file name
        extension = 'jpg'
        formated_frame_number = format_frame_number(frame_number, number_of_frames) # adds zero fills

        output_filename = f'{formated_frame_number}.{os.path.abspath(video_path).replace("/", "_")}.{extension}'  # Use 4-digit frame number
        
        # Save the frame as an image in the output directory
        output_path = os.path.join(output_directory, output_filename)
        return output_path

def extract_images(args):
    output_directory = args.ExtractedImageDirectory

    # Define the input video file path and output directory
    input_video_path = args.Video
    file_name, file_ext = os.path.splitext(args.Video)
    video_name = file_name.split("/")[-1]

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame counter
    frame_number = 0

    # Loop through the video frames
    for _ in tqdm(range(number_of_frames)):
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        
        # Save the frame as an image in the output directory
        output_path = image_path_from_details(input_video_path, frame_number, number_of_frames, output_directory)
        cv2.imwrite(output_path, frame)
        
        # Increment the frame counter
        frame_number += 1

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
    return SucLog("extracted all images")

def detections_to_coco(args_split, args_mcs):
    """
    args_split: is the args you are creating coco det for
    args_mcs: is all the vieo-args(in a nested format/one arg)
    """
    results_path = args_split.DetectionCOCO

    info = {
        "description" : f"COCO version of {args_split.Dataset}",
    }

    images_list = []
    annots_list = []
    catego_list = []
    all_category = []

    args_mc_s = flatten_args(args_mcs)

    for arg in args_mc_s:
        input_video_path = arg.Video

        file_name, file_ext = os.path.splitext(arg.Video)
        video_name = file_name.split("/")[-1]

        image_directory = arg.ExtractedImageDirectory
        # convert detection pkl to coco formated json
        detection_df = pd.read_pickle(arg.DetectionPkl)

        cap = cv2.VideoCapture(arg.Video)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))



        unique_frames = sorted(detection_df.fn.unique())
        for fn in unique_frames:
            file_path = image_path_from_details(input_video_path, int(fn),
                                                 number_of_frames, image_directory)
            prefix = args_split.Dataset
            file_name = os.path.relpath(file_path, prefix)
            
            frame_id  = format_frame_id(file_path)

            image_dict = {"file_name":file_name,
                      "height":frame_height,
                      "width":frame_width,
                      "id":frame_id,
                      }

            images_list.append(image_dict)

            detection_df_fn = detection_df[detection_df.fn==fn]

            for i, row in detection_df_fn.iterrows():
                bbox = [int(row.x1), int(row.y1), int(row.x2-row.x1), int(row.y2-row.y1)]
                category_id = int(row["class"])
                bbox_id = format_bbox_id(i, file_path)

                annot_dict = {
                          "id": bbox_id,
                          "image_id":frame_id,
                          "category_id":category_id,
                          "area": float((row.x2 - row.x1)*(row.y2 - row.y1)),
                          "bbox":bbox,
                          "iscrowd":0,
                        }

                annots_list.append(annot_dict)


        unique_categories = sorted(detection_df["class"].unique())
        for category in unique_categories:
            all_category.append(category)

    unique_categories = sorted(np.unique(all_category))
    for category in unique_categories:
        category_dict = {"id":int(category), "name":f"{int(category)}"}
        catego_list.append(category_dict)

    if args_split.KeepCOCOClasses:
        catego_list = []
        for i, coco_class in enumerate(COCO_CLASSES):
            category_dict = {"id":i, "name":coco_class}
            catego_list.append(category_dict)

    coco_annotations = {"info": info,
                        "images": images_list,
                        "annotations": annots_list,
                        "categories": catego_list}

    # open a file for writing
    with open(results_path, 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump(coco_annotations, f)

    return SucLog("annotations converted to COCO format")

def fine_tune_detector_mp(args, args_mp, args_mss, args_mcs):
    # get args_gt
    args_gt, args_mp_gt, args_mss_gt, args_mcs_gt = get_args_mp_gt(args)
    current_detector = detectors[args.Detector]
    current_detector.fine_tune(args, args_mp, args_gt, args_mp_gt)

# temporary hard code stuff
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
'scissors', 'teddy bear', 'hair drier', 'toothbrush')