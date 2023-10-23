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
import Detectors.YOLOv8.detect
import Detectors.DDETR.detect
import Detectors.InternImage.detect

# --------------------------

detectors = {}
detectors["detectron2"]  = Detectors.detectron2.detect
detectors["OpenMM"]      = Detectors.OpenMM.detect
detectors["YOLOv5"]      = Detectors.YOLOv5.detect
detectors["YOLOv8"]      = Detectors.YOLOv8.detect
detectors["DDETR"]       = Detectors.DDETR.detect
detectors["InternImage"] = Detectors.InternImage.detect

def detect(args):
    # check if detector names is valid
    if args.Detector not in os.listdir("./Detectors/"):
        return FailLog("Detector not recognized in ./Detectors/")

    current_detector = detectors[args.Detector]
    current_detector.detect(args)

    store_df_pickle(args)
    store_df_pickle_backup(args)

    return SucLog("Detection files stored")



def find_intersecting_regions(args):
    if "Rois" in args and args.UseRois:
        patches=[]
        container=np.load(args.Rois)
        data = [container[key] for key in container]
        for roi in data:
            patches.append(roi)
        patches=np.array(patches)
        intersecting_rectangles = list()
        
        for i in range(patches.shape[0]):
            x1, y1, x2, y2 = patches[i]
            for j in range(i + 1, patches.shape[0]):
                x3, y3, x4, y4 = patches[j]
                x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
                if not (x5 > x6 or y5 > y6):
                    intersecting_rectangles.append([x5, y5, x6, y6])

        intersecting_rectangles = np.array(intersecting_rectangles)
        # cap = cv2.VideoCapture(args.Video)

        # if (cap.isOpened()== False): 
        #     return FailLog("Error opening video stream or file")
        # ret, frame= cap.read()
        # for roi in patches:
        #     q=roi        
        #     frame=draw_box_on_image(frame, q[0], q[1] , q[2] ,q[3], c=[0,0,255], thickness=2)                    
        # for i in intersecting_rectangles:
        #     frame = draw_box_on_image(frame, i[0], i[1], i[2], i[3], c=[255,255,255])
            
        # cv2.imwrite("intersecting.png", frame)
        # print(intersecting_rectangles)
        # input()        
        return intersecting_rectangles
    else:
        return []

   

def get_near_patch_boxes(args,boxes, intersecting_rectangles):
    if len(intersecting_rectangles)>0:     
        tlx, tly, brx, bry = boxes['x1'].to_numpy().astype(int), boxes['y1'].to_numpy().astype(int), boxes['x2'].to_numpy().astype(int), boxes['y2'].to_numpy().astype(int)    
        rect_tlx, rect_tly = intersecting_rectangles[:, 0], intersecting_rectangles[:, 1]
        rect_brx, rect_bry = intersecting_rectangles[:, 2], intersecting_rectangles[:, 3]

        inside_tlx = np.logical_and(rect_tlx <= tlx[:, None], tlx[:, None] <= rect_brx)
        inside_tly = np.logical_and(rect_tly <= tly[:, None], tly[:, None] <= rect_bry)
        inside_brx = np.logical_and(rect_tlx <= brx[:, None], brx[:, None] <= rect_brx)
        inside_bry = np.logical_and(rect_tly <= bry[:, None], bry[:, None] <= rect_bry)

        boxes_inside = np.any(np.logical_and(np.logical_and(inside_tlx, inside_tly), np.logical_and(inside_brx, inside_bry)), axis=1)
        # flag = np.any(np.logical_or(np.logical_and(inside_tlx, inside_tly), np.logical_and(inside_brx, inside_bry)), axis=1)
        # flag[boxes_inside] = False

        return boxes_inside

def store_df_pickle_backup(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPklBackUp)

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl)

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

def visGTDet(args):
    args_gt=get_args_gt(args)
    detection_df = pd.read_pickle(args.DetectionPkl)
    gt_df= pd.read_pickle(args_gt.DetectionPkl)
    gt_frames=np.unique(gt_df['fn'])
    # open the original video and process it
    cap = cv2.VideoCapture(args.Video)

    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out_cap = cv2.VideoWriter(args.VisDetectionGTPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))
    rois=[]    
    if args.UseRois :
        container=np.load(args.Rois)
        data = [container[key] for key in container]
        for roi in data:
            
            rois.append(np.int32(roi))
    
    if not args.ForNFrames is None:
        frames = args.ForNFrames
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue
        for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
            if(frame_num in gt_frames):
                frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
        for i, row in gt_df[gt_df["fn"]==frame_num].iterrows():
            frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
            
        out_cap.write(frame)

    cap.release()
    out_cap.release()
    return SucLog("GT+Detection visualized on the video")
    
def visdetect(args):
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
    rois=[]    
    if args.UseRois:
        container=np.load(args.Rois)
        data = [container[key] for key in container]
        for roi in data:
            
            rois.append(np.int32(roi))
        
    if not args.ForNFrames is None:
        frames = args.ForNFrames
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        ret, frame = cap.read()
        if not ret: continue
        for roi in rois:
            q=roi        
            frame=draw_box_on_image(frame, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)            
        if args.VisInferenceRois:
            cv2.imwrite(args.VisInferenceRoi, frame)
            
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

# def postprocess(self, dets):    


#         # Find all the boxes which lie inside the intersecting regions

#         idxs_inside = self.get_near_patch_boxes(dets)

#         dets_inside = dets[idxs_inside]


#         dets_nms = pd.DataFrame()

#         for fn in np.unique(dets_inside['frame']):

#             det = dets_inside[dets_inside['frame'] == fn]

#             conf, boxes = det['conf'].to_numpy(), det[['x1', 'y1', 'x2', 'y2']].to_numpy()

#             ious = np.triu(self.iou(boxes))            


#             # Only keep those boxes who has certain IoU - Overlapping boxes in the intersecting regions.

#             overlapping_pairs = np.where((ious > self.iou_thr) & (ious < 1.0))

#             overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))


#             non_overlapping_pairs = np.where((ious <= self.iou_thr) & (ious > 0.0))

#             non_overlapping_pairs = np.hstack((non_overlapping_pairs[0].reshape(-1, 1), non_overlapping_pairs[1].reshape(-1, 1)))

#             conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

#             confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

#             keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]

#             keep_indices = np.unique(keep_indices)


#             low_confs_pairs = np.hstack((conf[non_overlapping_pairs[:, 0]].reshape(-1, 1), conf[non_overlapping_pairs[:, 1]].reshape(-1, 1)))

#             low_confs_sort_idxs = np.argsort(-low_confs_pairs, axis=1)

#             keep_indices_low_confs =  non_overlapping_pairs[np.arange(0, non_overlapping_pairs.shape[0]), low_confs_sort_idxs[:, 0]]

#             keep_indices_low_confs = np.unique(keep_indices_low_confs)


#             keep_indices_low_confs = keep_indices_low_confs[np.where(conf[keep_indices_low_confs] > self.conf_thr)]  

#             dets_nms = dets_nms.append(det.iloc[keep_indices])

#             dets_nms = dets_nms.append(det.iloc[keep_indices_low_confs])


#         return dets[~idxs_inside].append(dets_nms)



def get_overlapping_bboxes(df, args, intersecting_rectangles):

        idxs_inside = get_near_patch_boxes(args,df , intersecting_rectangles)
        dets_inside = df[idxs_inside]
        
 

        dets_nms = pd.DataFrame()
        for fn in np.unique(dets_inside['fn']):

            det = dets_inside[dets_inside['fn'] == fn]
            det=det[0:4]

            conf, bboxes = det['score'].to_numpy(), det[['x1', 'y1', 'x2', 'y2']].to_numpy()
            boxes=bboxes[np.newaxis, :, :]            
            x1 = np.maximum(boxes[:, :, 0][:, :, np.newaxis], boxes[:, :, 0])
            y1 = np.maximum(boxes[:, :, 1][:, :, np.newaxis], boxes[:, :, 1])
            x2 = np.minimum(boxes[:, :, 2][:, :, np.newaxis], boxes[:, :, 2])
            y2 = np.minimum(boxes[:, :, 3][:, :, np.newaxis], boxes[:, :, 3])
            
            intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_bbox1 = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
            area_bbox2 = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
            union_area = area_bbox1[:, :, np.newaxis] + area_bbox2 - intersection_area
            
            iou_scores = intersection_area / union_area

            ious = np.triu(iou_scores[0])            

            # Only keep those boxes who has certain IoU - Overlapping boxes in the intersecting regions.

            overlapping_pairs = np.where((ious > 0.5) & (ious < 1.0))

            overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))
            
            non_overlapping_pairs = np.where((ious <= 0.5) & (ious > 0.0))

            non_overlapping_pairs = np.hstack((non_overlapping_pairs[0].reshape(-1, 1), non_overlapping_pairs[1].reshape(-1, 1)))

            conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

            confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

            keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]
            
            remove_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 1]]

            keep_indices = np.unique(keep_indices)
            
            remove_indices=np.unique(remove_indices)
           

            low_confs_pairs = np.hstack((conf[non_overlapping_pairs[:, 0]].reshape(-1, 1), conf[non_overlapping_pairs[:, 1]].reshape(-1, 1)))

            low_confs_sort_idxs = np.argsort(-low_confs_pairs, axis=1)

            keep_indices_low_confs =  non_overlapping_pairs[np.arange(0, non_overlapping_pairs.shape[0]), low_confs_sort_idxs[:, 0]]

            keep_indices_low_confs = np.unique(non_overlapping_pairs)

           

            keep_indices_low_confs = keep_indices_low_confs[np.where(conf[keep_indices_low_confs] > 0.5)]  

            dets_nms = pd.concat([dets_nms,det.iloc[np.delete(np.arange(len(det)), remove_indices)]])

            # dets_nms = pd.concat([dets_nms,det.iloc[keep_indices_low_confs]])
            
            

            # overlapping_pairs = np.where((ious > 0.5) & (ious < 1.0))

            # overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))

            # conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

            # confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

            # keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]

            # keep_indices = np.unique(keep_indices)

            # dets_nms = pd.concat([dets_nms, det.iloc[keep_indices]])


        # return dets_inside[~idxs_inside].append(dets_nms)



        # return dets_inside[~idxs_inside].append(dets_nms, ignore_index=True)

 
        print('yo')
        return pd.concat([df[~idxs_inside],dets_nms])    
    # if args.Rois is not None:
    #     cap = cv2.VideoCapture(args.Video)

    #     if (cap.isOpened()== False): 
    #         return FailLog("Error opening video stream or file")
    #     frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     detection_df = df 
    #     detection_df.insert(0, 'ID', range(0, len(detection_df)))    
    #     fps = int(cap.get(cv2.CAP_PROP_FPS))
    #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     for frame_num in tqdm(range(frames)):
    #         if (not cap.isOpened()):
    #             return FailLog("Error reading the video")
    #         ret, frame = cap.read()        
    #         bounding_boxes=detection_df[detection_df["fn"]==frame_num]
    #         boxes=(bounding_boxes[['x1','y1','x2','y2']].to_numpy())
    #         Ids= bounding_boxes['ID'].to_numpy()
    #         labels=bounding_boxes[['class']].to_numpy()
    #         scores=bounding_boxes[['score']].to_numpy()
    #         boxes_in_intersection, Ids_in_intersection, scores_in_intersection= get_near_patch_boxes(args, boxes, Ids, scores, intersecting_rectangles)
    #         bboxes=boxes_in_intersection[np.newaxis, :, :]
    #         x1 = np.maximum(bboxes[:, :, 0][:, :, np.newaxis], bboxes[:, :, 0])
    #         y1 = np.maximum(bboxes[:, :, 1][:, :, np.newaxis], bboxes[:, :, 1])
    #         x2 = np.minimum(bboxes[:, :, 2][:, :, np.newaxis], bboxes[:, :, 2])
    #         y2 = np.minimum(bboxes[:, :, 3][:, :, np.newaxis], bboxes[:, :, 3])
            
    #         intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    #         area_bbox1 = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
    #         area_bbox2 = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
    #         union_area = area_bbox1[:, :, np.newaxis] + area_bbox2 - intersection_area
            
    #         iou_scores = intersection_area / union_area
    #         np.fill_diagonal(iou_scores[0], 0)        
    #         iou_idx=np.split((np.column_stack(np.where(iou_scores[0]>0.1))),2)[0]
    #         min_scores=np.argmin((scores_in_intersection[iou_idx]), axis=1)
    #         # print(iou_idx)
    #         # print(scores_in_intersection[iou_idx])
    #         remove_idx=(iou_idx[np.indices(iou_idx.shape)[0],min_scores])[:,0]
    #         Ids_remove=(Ids_in_intersection[remove_idx])
    #         detection_df=detection_df[~detection_df['ID'].isin(Ids_remove)]
    #         # print(iou_idx)
    #         # print(scores_in_intersection[iou_idx])
    #         # print(np.shape(np.argmin((scores_in_intersection[iou_idx]) , axis=1)))
    #         # print(iou_idx[np.arange(np.shape(iou_idx)[0]), np.argmin((scores_in_intersection[iou_idx]) , axis=1)])
            
    #         # for box in boxes_in_intersection:
    #         #     frame = draw_box_on_image(frame, box[0], box[1], box[2], box[3])
    #         # cv2.imwrite("Frame.png", frame)            
    #     detection_df.drop(columns=["ID"])
    #     return detection_df
    # else:
    #     return df


# def get_overlapping_bboxes(df, args, intersecting_rectangles):
#     idxs_inside = get_near_patch_boxes(args,df , intersecting_rectangles)
#     dets_inside = df[idxs_inside]
#     dets_nms = pd.DataFrame()
    
#     dets_nms=pd.concat([dets_nms, df[~idxs_inside]])
#     print(len(dets_nms))
#     for fn in np.unique(dets_inside['fn']):
#         bounding_boxes= dets_inside[dets_inside['fn'] == fn]
#         bounding_boxes=bounding_boxes[bounding_boxes['score']>=0.75]
#         conf, bboxes = bounding_boxes['score'].to_numpy(), bounding_boxes[['x1', 'y1', 'x2', 'y2']].to_numpy()
#         boxes=bboxes[np.newaxis, :, :]            
#         x1 = np.maximum(boxes[:, :, 0][:, :, np.newaxis], boxes[:, :, 0])
#         y1 = np.maximum(boxes[:, :, 1][:, :, np.newaxis], boxes[:, :, 1])
#         x2 = np.minimum(boxes[:, :, 2][:, :, np.newaxis], boxes[:, :, 2])
#         y2 = np.minimum(boxes[:, :, 3][:, :, np.newaxis], boxes[:, :, 3])
        
#         intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#         area_bbox1 = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
#         area_bbox2 = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
#         union_area = area_bbox1[:, :, np.newaxis] + area_bbox2 - intersection_area
        
#         iou_scores = intersection_area / union_area
        
#         ious=iou_scores[0]   
#         np.fill_diagonal(ious, 0)
#         iou_idx=(np.split(np.column_stack(np.where(ious>0.2)),2)[0])    
        
#         min_scores=np.argmin((conf[iou_idx]), axis=1)
#         remove_idx=np.unique(iou_idx[np.arange(iou_idx.shape[0]), min_scores])        
#         b=bounding_boxes.loc[~bounding_boxes.index.isin(remove_idx)]
#         dets_nms=pd.concat([dets_nms, b])
        
#     #         iou_idx=np.split((np.column_stack(np.where(iou_scores[0]>0.1))),2)[0]
#     #         min_scores=np.argmin((scores_in_intersection[iou_idx]), axis=1)

    
    
    
    
    
#     return dets_nms
    
    
    
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
    intersecting_rectangles=find_intersecting_regions(args)
    # print(intersecting_rectangles)
    print(len(df))
    if args.UseRois:
        df=get_overlapping_bboxes(df, args, intersecting_rectangles)
        
    print(len(df))
    
    
    detectors[args.Detector].df_txt(df, args.DetectionDetectorPath)
    
    store_df_pickle(args)
    
    
    if args.MaskGT:
        args_gt=get_args_gt(args)
        gt_df= pd.read_pickle(args_gt.DetectionPkl)
        # cap = cv2.VideoCapture(args.Video)

        # if (cap.isOpened()== False): 
        #     return FailLog("Error opening video stream or file")
        # ret, frame= cap.read()
        m= np.load(args.GTMask)
        mask=[]
        for k in m:
            mask= m[k]
            
        mask=mask.astype(np.uint8)
        mask = 255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
            
        bbox=gt_df[['x1', 'y1' ,'x2' ,'y2']].to_numpy().astype(np.int64)
        areas= (bbox[:,2] -bbox[:,0]) * (bbox[:,3] -bbox[:,1])
        bbox= bbox.tolist()
        # q=np.array([np.arange(bbox[:,0] , bbox[:,2] ),  np.arange(bbox[:,1], bbox[:,3])])
        non_zero=np.zeros((areas.shape))
        for i,b in enumerate(bbox):
            masked= mask[b[1] : b[3] , b[0]:b[2]]
            # masked_img= frame[b[1] : b[3] , b[0]:b[2]]

            non_zero[i] = np.count_nonzero(masked)
        print(len(gt_df))
        gt_df= gt_df[(non_zero/areas)> 0.7] 
        print(len(gt_df))
        gt_df.to_pickle(args_gt.MaskedGT)
        
        
    return SucLog("detection post processing done")


def detectionth(df, args):
    print("performing thresholding")
    df = df[df["score"] >= args.DetTh]
    return df

def filter_det_class(df, args):
    print("performing class filtering")
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