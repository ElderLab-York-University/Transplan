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
detectors["YoloX"]      = Detectors.YOLOv8.detect

detectors["DDETR"]       = Detectors.DDETR.detect
detectors["InternImage"] = Detectors.InternImage.detect
detectors["GTHW7"] = Detectors.InternImage.detect

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
    if args.SaveRoiResults:
        df.to_pickle(args.DetectionPklBackupRois)

def store_df_pickle(args):
    df = detectors[args.Detector].df(args)
    df.to_pickle(args.DetectionPkl)
    if args.SaveRoiResults:
        print(args.DetectionPklRois)        
        df.to_pickle(args.DetectionPklRois)
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
    if(args.EvalRois):
        detection_df=pd.read_pickle(args.DetectionPklRois)
        print(args.DetectionPklRois)
    else:
        detection_df = pd.read_pickle(args.DetectionPkl)
    if(args.MaskGT):
        gt_df= pd.read_pickle(args_gt.MaskedGT)
    else:
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
    
    # for frame_num in tqdm(range(frames)):
    #     if (not cap.isOpened()):
    #         return FailLog("Error reading the video")
    #     ret, frame = cap.read()
    #     if not ret: continue
    #     for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
    #         if(frame_num in gt_frames):
    #             frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
    #     for i, row in gt_df[gt_df["fn"]==frame_num].iterrows():
    #         frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
            
    #     out_cap.write(frame)

    cap.release()
    out_cap.release()
    out_cap = cv2.VideoWriter(args.VisDetectionGTPth,cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width,frame_height))    
    cap = cv2.VideoCapture(args.Video)
    ret, frame = cap.read()
    f1=np.array(frame)
    f2=np.array(frame)
    f3=np.array(frame)
    f4=np.array(frame)
    f5=np.array(frame)
    nondetarr=[]
    
    if (cap.isOpened()== False): 
        return FailLog("Error opening video stream or file")
    cv2.imwrite("Reference.png", frame)
    t=0
    for frame_num in tqdm(range(frames)):
        dont_use=[]    
        not_used=[]    
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        if not ret: continue
        if frame_num %6 ==0:
            for i, pred in detection_df[detection_df["fn"]==frame_num].iterrows():
                iou_idx=-1
                iou_max=0
                for d, gt in gt_df[gt_df["fn"]==frame_num].iterrows():
                    inter_left=max(gt.x1, pred.x1)
                    inter_right=min(gt.x2, pred.x2)
                    inter_top= max(gt.y1, pred.y1)
                    inter_bottom= min(gt.y2, pred.y2)    
                    inter_area= (inter_right-inter_left +1) *(inter_bottom- inter_top +1) 
                    gt_area= (gt.x2-gt.x1 +1) * (gt.y2- gt.y1 +1)
                    pred_area=(pred.x2-pred.x1 +1) * (pred.y2-pred.y1 +1)
                    iou= inter_area/float(gt_area+pred_area - inter_area) if inter_left< inter_right and inter_top < inter_bottom else 0
                    if(iou>iou_max):
                        iou_idx=d
                        iou_max=iou
                if(iou_max>=0.5):
                    dont_use.append(iou_idx)
                else:
                    t=t+1
                    not_used.append(i)
            
        
                
            # gts=gt_df[gt_df["fn"]==frame_num]
            # print(dets)
            # print(gts)
            # for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
            #     if(args.Detector=="GTHW7"):
            #         frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
            #     else:
            #         frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
        for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
            f5=draw_box_on_image(f5, row.x1, row.y1, row.x2, row.y2, c=(0,0,255))
        if frame_num%6 ==0:
            for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
                if i in not_used:
                    frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,0,255))
                    f4=draw_box_on_image(f4, row.x1, row.y1, row.x2, row.y2, c=(0,0,255))
        for i, row in gt_df[gt_df["fn"]==frame_num].iterrows():
            if i not in dont_use:
                frame=draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
                f1 = draw_box_on_image(f1, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
                f3 = draw_box_on_image(f3, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
                nondetarr.append([frame_num, row.x1, row.y1, row.x2, row.y2])
            else:
                frame=draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(255,0,0))   
                f1 = draw_box_on_image(f1, row.x1, row.y1, row.x2, row.y2, c=(255,0,0))                             
            f2 = draw_box_on_image(f2, row.x1, row.y1, row.x2, row.y2, c=(255,0,0))
        for roi in rois:
            q=roi
            frame=draw_box_on_image(frame, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     

        out_cap.write(frame)
        ret, frame = cap.read()
    print(t)
    for roi in rois:
        q=roi        
        f1=draw_box_on_image(f1, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     
        f2=draw_box_on_image(f2, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     
        f3=draw_box_on_image(f3, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     
        f4=draw_box_on_image(f4, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     
        f5=draw_box_on_image(f5, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)     
               
    cv2.imwrite("GTALL.png", f2)
    cv2.imwrite("DETGTALL.png", f1)
    cv2.imwrite("NONDETGTALL.png", f3) 
    cv2.imwrite("FPALL.png", f4    )
    cv2.imwrite("DETALL.png", f5   )
    
    nondetdf=pd.DataFrame(nondetarr,columns=['fn','x1','y1','x2','y2'])
    nondetdf.to_pickle("./NonDetDfs/NonDetGT."+    args.Video.split("/")[-1].split(".")[0]+"." + args.NumRois+".pkl")
    detection_df.to_pickle("./DetDf/DetDf."+    args.Video.split("/")[-1].split(".")[0]+"." + args.NumRois+".pkl")
    out_cap.release()
    cap.release()
    return SucLog("GT+Detection visualized on the video")
    
def visdetect(args):
    if args.Detector is None:
        return FailLog("To interpret detections you should specify detector")
    
    # parse detection df using detector module
    detection_df = pd.read_pickle(args.DetectionPkl)
    print(len(detection_df))
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
            if(args.Detector=="GTHW7"):
                frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
            else:
                frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
        out_cap.write(frame)
    cap.release()
    out_cap.release()
    cap = cv2.VideoCapture(args.Video)
    ret, frame = cap.read()    
    for frame_num in tqdm(range(frames)):
        if (not cap.isOpened()):
            return FailLog("Error reading the video")
        if not ret: continue
        if frame_num %6 ==0:
            for i, row in detection_df[detection_df["fn"]==frame_num].iterrows():
                if(args.Detector=="GTHW7"):
                    frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2, c=(0,255,0))
                else:
                    frame = draw_box_on_image(frame, row.x1, row.y1, row.x2, row.y2)
    for roi in rois:
        q=roi        
        frame=draw_box_on_image(frame, q[0], q[1] , q[2] ,q[3], c=[255,255,255], thickness=2)            
    cv2.imwrite("GTALL.png", frame)
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


def nms(df,args):
        dets_inside = df
        dets_nms = pd.DataFrame()
        for fn in np.unique(dets_inside['fn']):

            det = dets_inside[dets_inside['fn'] == fn]
            # det=det[0:4]

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
            # remove_indices= np.append(remove_indices, np.where(conf<0.9))
            # remove_indices=np.unique(remove_indices)

            low_confs_pairs = np.hstack((conf[non_overlapping_pairs[:, 0]].reshape(-1, 1), conf[non_overlapping_pairs[:, 1]].reshape(-1, 1)))

            low_confs_sort_idxs = np.argsort(-low_confs_pairs, axis=1)

            keep_indices_low_confs =  non_overlapping_pairs[np.arange(0, non_overlapping_pairs.shape[0]), low_confs_sort_idxs[:, 0]]

            keep_indices_low_confs = np.unique(non_overlapping_pairs)

        

            keep_indices_low_confs = keep_indices_low_confs[np.where(conf[keep_indices_low_confs] > 0.5)]  

            dets_nms = pd.concat([dets_nms,det.iloc[np.delete(np.arange(len(det)), remove_indices)]])

            # dets_nms = pd.concat([dets_nms,det.iloc[keep_indices_low_confs]])
            
            # dets_nms = pd.concat([dets_nms, det.iloc[keep_indices]])
            

            # overlapping_pairs = np.where((ious > 0.5) & (ious < 1.0))

            # overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))

            # conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

            # confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

            # keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]

            # keep_indices = np.unique(keep_indices)

            # dets_nms = pd.concat([dets_nms, det.iloc[keep_indices]])


        # return dets_inside[~idxs_inside].append(dets_nms)



        # return dets_inside[~idxs_inside].append(dets_nms, ignore_index=True)


        print('regular nms done')
        return dets_nms

def get_overlapping_bboxes(df, args, intersecting_rectangles):

        idxs_inside = get_near_patch_boxes(args,df , intersecting_rectangles)
        if idxs_inside is not None:
            dets_inside = df[idxs_inside]
            
    

            dets_nms = pd.DataFrame()
            for fn in np.unique(dets_inside['fn']):

                det = dets_inside[dets_inside['fn'] == fn]
                # det=det[0:4]

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

                overlapping_pairs = np.where((ious > args.NmsTh) & (ious < 1.0))

                overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))
                
                non_overlapping_pairs = np.where((ious <= args.NmsTh) & (ious > 0.0))

                non_overlapping_pairs = np.hstack((non_overlapping_pairs[0].reshape(-1, 1), non_overlapping_pairs[1].reshape(-1, 1)))

                conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

                confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

                keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]
                
                remove_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 1]]

                keep_indices = np.unique(keep_indices)
                
                remove_indices=np.unique(remove_indices)
                # remove_indices= np.append(remove_indices, np.where(conf<0.9))
                # remove_indices=np.unique(remove_indices)

                low_confs_pairs = np.hstack((conf[non_overlapping_pairs[:, 0]].reshape(-1, 1), conf[non_overlapping_pairs[:, 1]].reshape(-1, 1)))

                low_confs_sort_idxs = np.argsort(-low_confs_pairs, axis=1)

                keep_indices_low_confs =  non_overlapping_pairs[np.arange(0, non_overlapping_pairs.shape[0]), low_confs_sort_idxs[:, 0]]

                keep_indices_low_confs = np.unique(non_overlapping_pairs)

            

                keep_indices_low_confs = keep_indices_low_confs[np.where(conf[keep_indices_low_confs] > 0.5)]  

                dets_nms = pd.concat([dets_nms,det.iloc[np.delete(np.arange(len(det)), remove_indices)]])

                # dets_nms = pd.concat([dets_nms,det.iloc[keep_indices_low_confs]])
                
                # dets_nms = pd.concat([dets_nms, det.iloc[keep_indices]])
                

                # overlapping_pairs = np.where((ious > 0.5) & (ious < 1.0))

                # overlapping_pairs = np.hstack((overlapping_pairs[0].reshape(-1, 1), overlapping_pairs[1].reshape(-1, 1)))

                # conf_pairs = np.hstack((conf[overlapping_pairs[:, 0]].reshape(-1, 1), conf[overlapping_pairs[:, 1]].reshape(-1, 1)))

                # confs_sort_idxs = np.argsort(-conf_pairs, axis=1)

                # keep_indices = overlapping_pairs[np.arange(0, overlapping_pairs.shape[0]), confs_sort_idxs[:, 0]]

                # keep_indices = np.unique(keep_indices)

                # dets_nms = pd.concat([dets_nms, det.iloc[keep_indices]])


            # return dets_inside[~idxs_inside].append(dets_nms)



            # return dets_inside[~idxs_inside].append(dets_nms, ignore_index=True)

    
            print('overlap nms done')
            return pd.concat([df[~idxs_inside],dets_nms])
        else:
            print('overlap nms done')            
            return df
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
    
    if(args.EvalRois):
        df_rois=pd.read_pickle(args.DetectionPklBackupRois)    
    df = pd.read_pickle(args.DetectionPklBackUp)
    # 1. condition on the post processing flags
    if args.UseRois:
        df.to_pickle("./DetDf/DetDf.Backup."+    args.Video.split("/")[-1].split(".")[0]+"." + args.NumRois+".pkl")
    if not args.DetTh is None:
        df = detectionth(df, args)
        if(args.EvalRois):
            df_rois=detectionth(df_rois, args)  
        
    if args.classes_to_keep:
        df = filter_det_class(df, args)
        if(args.EvalRois):
            df_rois=filter_det_class(df_rois, args)  

    if args.DetMask:
        df = remove_out_of_ROI(df, args.MetaData["roi"])
        if(args.EvalRois):
            df_rois=remove_out_of_ROI(df_rois, args.MetaData["roi"])  

    
    # store the edited df as txt
    intersecting_rectangles=find_intersecting_regions(args)
    # print(intersecting_rectangles)
    print(len(df))
    if args.UseRois:
        df=get_overlapping_bboxes(df, args, intersecting_rectangles)
        if(args.EvalRois):
            df_rois=get_overlapping_bboxes(df_rois, args, intersecting_rectangles)  
        
    print(len(df))
    # df=nms(df,args)
    print(len(df))

    
    
    
    if args.MaskGT:
        args_gt=get_args_gt(args)
        gt_df= pd.read_pickle(args_gt.DetectionPklBackUp)
        cap = cv2.VideoCapture(args.Video)
        if (cap.isOpened()== False): 
            return FailLog("Error opening video stream or file")
        ret, frame= cap.read()
        m= np.load(args.GTMask)
        mask=[]
        for k in m:
            mask= m[k]
            
        mask=mask.astype(np.uint8)
        if(frame.shape[0]>mask.shape[0] and frame.shape[1]> mask.shape[1]):
            mask=cv2.copyMakeBorder(mask, int((frame.shape[0]-mask.shape[0])/2), int((frame.shape[0]-mask.shape[0])/2), int((frame.shape[1]-mask.shape[1])/2), int((frame.shape[1]-mask.shape[1])/2),cv.BORDER_CONSTANT ,0)
            # mask= np.pad(mask, [((int(frame.shape[0]-mask.shape[0])/2), int((frame.shape[0]-mask.shape[0])/2)), (int((frame.shape[1]-mask.shape[1])/2), int((frame.shape[1]-mask.shape[1])/2))], mode='constant', constant_values=0)
        
        mask = 255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
        cv2.imwrite("mask.png", mask)
        print(frame.shape)
        bbox=gt_df[['x1', 'y1' ,'x2' ,'y2']].to_numpy().astype(np.int64)
        areas= (bbox[:,2] -bbox[:,0]) * (bbox[:,3] -bbox[:,1])
        bbox= bbox.tolist()
        # q=np.array([np.arange(bbox[:,0] , bbox[:,2] ),  np.arange(bbox[:,1], bbox[:,3])])
        zero=np.zeros((areas.shape))
        for i,b in enumerate(bbox):
            masked= mask[b[1] : b[3] , b[0]:b[2]]
            # masked_img= frame[b[1] : b[3] , b[0]:b[2]]
            zero[i] = np.count_nonzero(masked==0)
            bbox_area_on_frame=(min(b[2],frame.shape[1])-b[0])* (min(b[3],frame.shape[0]) -b[1])
            bbox_area= areas[i]            
            # print(bbox_area_on_frame)
            # print(bbox_area)
            # print(b)
            bbox_on_frame=[b[0], b[1], min(b[2],frame.shape[1]), min(b[3],frame.shape[0])]
            # print(bbox_on_frame)
            assert bbox_area>=bbox_area_on_frame
            zero[i]+=bbox_area-bbox_area_on_frame
        print(len(gt_df))
        gt_df= gt_df[(zero/areas)<= 0] 
        print(len(gt_df))
        # print(gt_df)
        # input()
        
        gt_df.to_pickle(args_gt.DetectionPkl)
        gt_df.to_pickle(args_gt.MaskedGT)
        
        detections_df= df        
        
        cap = cv2.VideoCapture(args.Video)

        if (cap.isOpened()== False): 
            return FailLog("Error opening video stream or file")
        ret, frame= cap.read()
        m= np.load(args.GTMask)
        mask=[]
        for k in m:
            mask= m[k]
            
        mask=mask.astype(np.uint8)
        mask = 255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
        
        print(frame.shape)
        bbox=detections_df[['x1', 'y1' ,'x2' ,'y2']].to_numpy().astype(np.int64)
        areas= (bbox[:,2] -bbox[:,0]) * (bbox[:,3] -bbox[:,1])
        bbox= bbox.tolist()
        # q=np.array([np.arange(bbox[:,0] , bbox[:,2] ),  np.arange(bbox[:,1], bbox[:,3])])
        zero=np.zeros((areas.shape))
        for i,b in enumerate(bbox):
            masked= mask[b[1] : b[3] , b[0]:b[2]]
            # masked_img= frame[b[1] : b[3] , b[0]:b[2]]

            zero[i] = np.count_nonzero(masked==0)
            bbox_area_on_frame=(min(b[2],frame.shape[1])-b[0])* (min(b[3],frame.shape[0]) -b[1])
            bbox_area= areas[i]
            # print(bbox_area_on_frame)
            # print(bbox_area)
            # print(b)
            bbox_on_frame=[b[0], b[1], min(b[2],frame.shape[1]), min(b[3],frame.shape[0])]
            # print(bbox_on_frame)
            assert bbox_area>=bbox_area_on_frame
            zero[i]+=bbox_area-bbox_area_on_frame
        print(len(detections_df))
        detections_df= detections_df[(zero/areas)<= 0] 
        print(len(detections_df))
        detections_df.to_pickle(args.DetectionPkl)
        if args.EvalRois:
            detections_df= df_rois
            cap = cv2.VideoCapture(args.Video)

            if (cap.isOpened()== False): 
                return FailLog("Error opening video stream or file")
            ret, frame= cap.read()
            m= np.load(args.GTMask)
            mask=[]
            for k in m:
                mask= m[k]
                
            mask=mask.astype(np.uint8)
            mask = 255*((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
            
            print(frame.shape)
            bbox=detections_df[['x1', 'y1' ,'x2' ,'y2']].to_numpy().astype(np.int64)
            areas= (bbox[:,2] -bbox[:,0]) * (bbox[:,3] -bbox[:,1])
            bbox= bbox.tolist()
            # q=np.array([np.arange(bbox[:,0] , bbox[:,2] ),  np.arange(bbox[:,1], bbox[:,3])])
            zero=np.zeros((areas.shape))
            for i,b in enumerate(bbox):
                masked= mask[b[1] : b[3] , b[0]:b[2]]
                # masked_img= frame[b[1] : b[3] , b[0]:b[2]]

                zero[i] = np.count_nonzero(masked==0)
                bbox_area_on_frame=(min(b[2],frame.shape[1])-b[0])* (min(b[3],frame.shape[0]) -b[1])
                bbox_area= areas[i]
                # print(bbox_area_on_frame)
                # print(bbox_area)
                # print(b)
                bbox_on_frame=[b[0], b[1], min(b[2],frame.shape[1]), min(b[3],frame.shape[0])]
                # print(bbox_on_frame)
                assert bbox_area>=bbox_area_on_frame
                zero[i]+=bbox_area-bbox_area_on_frame
            print(len(detections_df))
            detections_df= detections_df[(zero/areas)<= 0] 
            print(len(detections_df))
            detections_df.to_pickle(args.DetectionPklRois)   
    else:
        df.to_pickle(args.DetectionPkl)
        if(args.SaveRoiResults):
            df.to_pickle(args.DetectionPklRois)
        
        
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