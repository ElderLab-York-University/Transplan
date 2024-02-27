import os

def get_sub_dirs(roots, subs_to_include = None, subs_to_exclude = ["Results"], be_inside=None, not_be_inside=None):
    subs_path = []
    for root in roots:
        subs = os.listdir(root)
        for sub in subs:
            sub_path = os.path.join(root, sub)
            if os.path.isdir(sub_path) and\
                (subs_to_exclude is None or sub not in subs_to_exclude) and\
                (subs_to_include is None or sub in subs_to_include)and\
                (not sub.startswith('.')) and\
                (be_inside is None or be_inside in sub) and\
                (not_be_inside is None or not_be_inside not in sub):
                subs_path.append(sub_path)
    return subs_path

# choose datasets/splits/segments/sources
#  set to None if want to include all
datasets     = [
                "/home/sajjad/HW7Leslie",
                # "/mnt/dataB/CityFlowV2Local",
                # "/mnt/dataB/TransPlanData/Dataset/PreProcessedMain",
                # "/run/user/1000/gvfs/sftp:host=130.63.188.39/home/sajjad/HW7Leslie",
                # "/run/user/1000/gvfs/sftp:host=130.63.188.39/mnt/dataB/CityFlowV2Local",
            ]
split_part   = ["train", "valid"]
segment_part = ["Seg01"]
source_part  = ["Seg01sc1"]
splits       = get_sub_dirs(datasets, split_part)
segments     = get_sub_dirs(splits, segment_part)
sources      = get_sub_dirs(segments, source_part)


# choose datasets/splits/segments/sources
#  set to None if want to include all
# this pathes are used to load cached counters
cached_datasets     = [
                "/home/sajjad/HW7Leslie",
                # "/mnt/dataB/CityFlowV2Local",
                # "/mnt/dataB/TransPlanData/Dataset/PreProcessedMain",
                # "/run/user/1000/gvfs/sftp:host=130.63.188.39/home/sajjad/HW7Leslie",
                # "/run/user/1000/gvfs/sftp:host=130.63.188.39/mnt/dataB/CityFlowV2Local",
            ]
cached_split_part   = ["train", "valid"]
cached_segment_part = ["Seg01"]
cached_source_part  = ["Seg01sc1"]
cached_splits       = get_sub_dirs(cached_datasets, cached_split_part)
cached_segments     = get_sub_dirs(cached_splits, cached_segment_part)
cached_sources      = get_sub_dirs(cached_segments, cached_source_part)


# choose the segmenter
# options: ["InternImage"]
segmenters = ["InternImage"]

# choose the detectors
# options: ["GTHW7", "detectron2", "OpenMM", "YOLOv5", "YOLOv8", "InternImage", "RTMDet", "DeformableDETR", "YoloX", "CenterNet", "CascadeRCNN"]
detectors = ["pgd2D"]

# choose detector version (checkpoints, ...)
# options: ["", "HW7FT"]
# options: ["", "kitti"]
det_v = "kitti"

# choose the tracker
# options: ["GTHW7", "sort", "ByteTrack",  "CenterTrack", "DeepSort", "gsort", "OCSort", "GByteTrack", "GDeepSort", "BOTSort", "StrongSort"]
trackers = ["ByteTrack"] 

# choose grandtruth detector
# Options are the same as detector
GT_det    = "GTHW7"
GT_det_3D = "GTHW73D"

# choose grandtruth tracker
# options are the same as trackers
GT_tra    = "GTHW7"
GT_tra_3D = "GTHW73D"

# choose pose estimation model
# options: ["MMPose"]
posers = ["MMPose"]

# choose the clustering algorithm
# options: ["SpectralFull", "DBSCAN", "SpectralKNN"]
clusters = []

# choose the metric for clustering and classification pqrt
# options on image :  ["kde", "roi", "knn", "cos", "tcos", "cmm", "hausdorff","ccmm", "tccmm", "ptcos"]
# options on ground:  ["gkde", "groi", "gknn", "gcos", "gtcos", "gcmm", "ghausdorff","gccmm", "gtccmm", "gptcos"]
clt_metrics = []
# cnt_metrics = ["cos", "tcos", "cmm", "hausdorff", "kde", "roi", "knn"]
cnt_metrics = ["gkde"]

# setup training hyperparameters
# train split (train_sp) and valid split(valid_sp) should be selsected
# from "splits"
train_sp     = "train"
valid_sp     = "valid"
batch_size   = 1
num_workers  = 1
epochs       = 20
val_interval = 1

# projection hypterparameters
# --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM]
# cp_methods = ["BottomPoint", "Center", "BottomSeg", "LineSeg", "BottomPoint3D"]
tp_view    =  "GoogleMap"
cp_methods = ["BottomPoint"]
bp_method  =  "Homography"
resamp_th  =  2

# set contact point for GT and GT3D
gt_cp_method   = "BottomPoint"
gt3D_cp_method = "BottomPoint3D"
gt_tp_view     =  "GoogleMap"
gt_bp_method   = "Homography"

for src, cached_cnt_pth in zip(sources, cached_sources):
    print(f"running on src:{src}")
    ########################################################
    # 0. extract images from video
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages)
    ######################################################
    # print(f" extracting images from : {src}")
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages")

    ########################################################
    # 1. estimate the Homography Metrix using Homography GUI 
    # os.system(f"python3 main.py --Dataset={src} --HomographyGUI --VisHomographyGUI --Frame=1
    #  --TopView={tp_view}")
    ########################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src} --VisHomographyGUI --TopView={tp_view}")

    ########################################################
    # 2. visualizing the region of interest 
    # os.system(f"python3 main.py --Dataset={src} --VisROI --TopView={tp_view}\
    #  --ROIFromTop --BackprojectionMethod=[Homography/DSM]")
    #######################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src} --VisROI --TopView={tp_view}\
    #             --BackprojectionMethod={bp_method} --ROIFromTop")

    #######################################################
    # 2.5 Segment Video Frames 
    #  os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segment --Segmenter={seg} 
    # --VisSegment --ForNFrames=2000")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --Segment")

    #######################################################
    # 2.6 Segment Post Processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc 
    # --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc\
    #          --VisSegment --SegTh=0.5 --classes_to_keep 2 3 5 7")

    ########################################################
    # 3. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Dataset={src}  --Detector={det}
    #  --Detect --VisDetect --ForNFrames=600 --DetectorVersion={det_v} --SAHI --SahiPatchSize=640 --SahiPatchOverlapRatio=0.25 
    #  --SahiPatchBatchSize=1 --SahiNMSTh=0.25")
    ########################################################
    for det in detectors:
        print(f"detecting ----> src:{src} det:{det}")
        os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --VisDetect")

    ########################################################
    # 3.5 run the detection post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --DetPostProc --DetTh=0.5
    #  --classes_to_keep 2 3 5 7 --VisDetect --ForNFrames=600 --SAHI --SahiPatchSize=640 --SahiPatchOverlapRatio=0.25
    #  --SahiPatchBatchSize=1 --SahiNMSTh=0.25")
    #######################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --DetPostProc\
    #                 --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                 --DetTh=0.5 --classes_to_keep 2 3 5 7")

    ########################################################
    # 3.5.5 convert detections to coco format
    # the full commonad looks like : os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL")
    ########################################################
    # for det in detectors:
    #     print(f"converting to COCO ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --ConvertDetsToCOCO")

    #######################################################
    # 3.5.6 run pose estimation and contact point estimation
    # full command would look like below
    # os.system(f"python3 main.py --Dataset={src}\
    #     --Detector={det} --DetectorVersion={det_v}\
    #     --Tracker={tra} --Poser={poser} --ExtractPose")

    # os.system(f"python3 main.py --Dataset={src}\
    #     --Detector={det} --DetectorVersion={det_v}\
    #     --Tracker={tra} --Poser={poser} --VisPose --VisPose --ForNFrames=2000")
    #######################################################
    # for det in detectors:
    #     for poser in posers:
    #         print(f"pose estimation ----> src:{src} det:{det} poser:{poser}")
    #         os.system(f"python3 main.py --Dataset={src}\
    #             --Detector={det} --DetectorVersion={det_v}\
    #             --Tracker={tra} --Poser={poser} --ExtractPose")

    #         os.system(f"python3 main.py --Dataset={src}\
    #             --Detector={det} --DetectorVersion={det_v}\
    #             --Tracker={tra} --Poser={poser} --VisPose --VisPose --ForNFrames=2000")

    ########################################################
    # 3.6 back project detections
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker=NULL --Homography
    #  --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM]
    #  --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg/BottomPoint3D]")
    ########################################################
    # for det in detectors:
    #     for seg in segmenters:
    #         for cp_method in cp_methods:
    #             print(f"detecting ----> src:{src} det:{det}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Segmenter={seg} --Homography\
    #                 --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 3.7 Vis Contact Points and BP Points
    # os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} --VisContactPoint --VisCPTop
    #  --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] 
    #  --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg] --ForNFrames=600")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --VisContactPoint\
    #          --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --VisCPTop \
    #          --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 3.8 Evaluate contact point selection method
    # os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} 
    #  --BackprojectSource=detections --TopView=w={tp_view} --BackprojectionMethod={bp_method} 
    #  --ContactPoint={cp_method} --ForNFrames=600")
    ########################################################
    # for det in detectors:
    #     for cp_method in cp_methods:
    #         print(f"cp evaluation ----> src:{src} det:{det} cp:{cp_method}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --EvalContactPoitnSelection\
    #                     --GTDetector={GT_det} --GTDetector3D={GT_det_3D}\
    #                     --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method}\
    #                     --ContactPoint={cp_method} --GTContactPoint={gt_cp_method} --GT3DContactPoint={gt3D_cp_method}")

    ########################################################
    # 4. run the tracking and backproject and convert to meter
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --Track
    # --VisTrack --VisTrackTop --VisTrajectories --ForNFrames=600
    # --Homography --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto] 
    # --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]
    # --Meter")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --Track")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --Homography --Meter\
    #              --BackprojectSource=tracks --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #              --VisTrack --ForNFrames=200\
    #              --BackprojectSource=tracks --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 5. run the track post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}
    # --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto] 
    # --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg] 
    # --TrackPostProc --TrackTh=8 --Interpolate --InterpolateTh=10 --RemoveInvalidTracks --MaskGPFrame 
    # --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --HasPointsInROI --MaskROI --CrossROI --CrossROIMulti
    # --JustEnterROI --JustExitROI --WithinROI  --ExitOrCrossROI --SelectToBeCounted\
    # --UnifyTrackClass --MovesInROI --ROIFromTop --classes_to_keep 2 3 5 7")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for cp_method in cp_methods:
    #             print(f"tracking POSTPROC ---> src:{src} det:{det} tra:{tra}")
    #             os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                         --BackprojectSource=tracks --TopView={tp_view}\
    #                         --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                         --TrackPostProc --Interpolate --InterpolateTh=1000 --RemoveInvalidTracks --MaskGPFrame\
    #                         --HasPointsInROI --UnifyTrackClass --classes_to_keep 2 3 5 7 --ResampleTH={resamp_th}\
    #                         --MovesInROI --ROIFromTop")
            
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #              --VisTrack\
    #              --BackprojectSource=tracks --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")
                 
    ########################################################
    # 5.5 Evaluate Tracking
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}
    #  --TrackEval --GTDetector={GT_det} --GTTracker={GT_tra}")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"evaluate tracking ---> src:{src} det:{det} tra:{tra} gt_det:{GT_det} gt_tra:{GT_tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #              --TrackEval --GTDetector={GT_det} --GTTracker={GT_tra}")

    ########################################################
    # 6. find optimum BW for kde fiting
    # os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} --Tracker={tra}
    #  --FindOptimalKDEBW --ResampleTH={resamp_th} --OSR=10
    # --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto] 
    # --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"finding optimal bw for kde ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                    --FindOptimalKDEBW --ResampleTH={resamp_th} --OSR=10\
    #                    --BackprojectSource=tracks --TopView={tp_view}\
    #                    --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 7. run clustering algorithm
    # apperently the clustering visulaizaiton is harcodded at the moment
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in clt_metrics:
    #             for clt in clusters:
    #                 print(f"clustering ----> det:{det} tra:{tra} met:{met} clt:{clt}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ClusteringAlgo={clt} --ClusterMetric={met} --Cluster")
    
    ########################################################
    # 8. Run the track labelling GUI / go to 9.
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    ########################################################
    # 9. Run automated track extraction and labelling
    # python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                 --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH={resamp_th}\
    #                 --BackprojectSource=tracks --TopView={tp_view}\
    #                 --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                 --GP
    #######################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                    --ExtractCommonTracks --ResampleTH={resamp_th}\
    #                    --BackprojectSource=tracks --TopView={tp_view}\
    #                    --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                    --GP --ROIFromTop")

    ########################################################
    # 10. Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #  --Count --EvalCount --CountMetric={metric}\
    #  --CacheCounter --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #  --CountVisDensity --CountVisPrompt\
    # --BackprojectSource=tracks --TopView={tp_view}\
    # --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    # --KDEBW=3.5/10 --OSR=10 --ROIFromTop")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"counting metric:{metric} det:{det} tra:{tra}")
    #             os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                         --Count --EvalCount --CountMetric={metric} --ResampleTH={resamp_th}\
    #                         --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #                         --BackprojectSource=tracks --TopView={tp_view}\
    #                         --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                         --KDEBW=3.5 --OSR=10 --ROIFromTop --GP")

    ########################################################
    # 11. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in cnt_metrics:
    #             for cp_method in cp_methods:
    #                 print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                         --CountMetric={met} --VisTrackMoI \
    #                         --BackprojectSource=tracks --TopView={tp_view} \
    #                         --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

#_______________________MULTICAMERA_______________________#
for src in segments:
    print(f"running on seg:{src}")
    ########################################################
    # 3.8 Evaluate contact point selection method
    # os.system(f"python3 main.py --Dataset={src} --Detector={det} --DetectorVersion={det_v} 
    #  --BackprojectSource=detections --TopView=w={tp_view} --BackprojectionMethod={bp_method} 
    #  --ContactPoint={cp_method} --ForNFrames=600")
    ########################################################
    # for det in detectors:
    #     for cp_method in cp_methods:
    #         print(f"cp evaluation ----> src:{src} det:{det} cp:{cp_method}")
    #         os.system(f"python3 main.py --MultiCam  --Dataset={src}  --Detector={det} --DetectorVersion={det_v} \
    #                     --EvalContactPoitnSelectionMC\
    #                     --GTDetector={GT_det} --GTDetector3D={GT_det_3D}\
    #                     --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method}\
    #                     --ContactPoint={cp_method} --GTContactPoint={gt_cp_method} --GT3DContactPoint={gt3D_cp_method}")

    ########################################################
    # 3.7 Vis Contact Points and BP Points
    # os.system(f"python3 main.py --MultiCam --Dataset={src} --Detector={det} --DetectorVersion={det_v} --VisContactPoint --VisCPTop
    #  --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] 
    #  --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg] --ForNFrames=600")
    ########################################################
    # for det in detectors:
    #     print(f"vis MC CP Top ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --VisCPTop \
    #          --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 0 perform single camera tracking evaluation on all the sources under mc folder
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --MultiCam --TrackEval --GTDetector={GT_det} --GTTracker={GT_tra}")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"evaluate tracking ---> src:{src} det:{det} tra:{tra} gt_det:{GT_det} gt_tra:{GT_tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --MultiCam  --TrackEval\
    #                    --GTDetector={GT_det} --GTTracker={GT_tra}")

    # ########################################################
    # # 1. Average Counts MC
    # # os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} -- --VisTrackTopMC")
    # ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"average MC counts ---> src:{src} det:{det} tra:{tra} cnt:{metric}")
    #             os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={metric} --AverageCountsMC --EvalCountMC")


    # ########################################################
    # # 2. Integrate Counts MC
    # # os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrackTopMC")
    # ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             for cp_method in cp_methods:
    #                 print(f"Integrate MC counts ---> src:{src} det:{det} tra:{tra} cnt:{metric}")
    #                 os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra}\
    #                             --CountMetric={metric} --IntegrateCountsMC --EvalCountMC\
    #                             --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

#_______________________MULTISEGMENT _______________________#
for split, cached_cnt_pth in zip(splits, cached_splits):
    print(f"running on split:{split} cached_cnt_path:{cached_cnt_pth}")
    ########################################################
    # 1. convert detections of all the data under split into COCO format
    # os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det} --ConvertDetsToCOCO --KeepCOCOClasses")
    ########################################################
    # for det in detectors:
    #     for cp_method in cp_methods:
    #         os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det}\
    #                 --ConvertDetsToCOCO --KeepCOCOClasses\
    #                 --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 3.8 Evaluate contact point selection method
    # os.system(f"python3 main.py --MultiSeg --Dataset={src} --Detector={det} --DetectorVersion={det_v} 
    #  --BackprojectSource=detections --TopView=w={tp_view} --BackprojectionMethod={bp_method} 
    #  --ContactPoint={cp_method} --ForNFrames=600
    # --EvalContactPoitnSelectionMC
    # --EvalContactPoitnSelection")
    ########################################################
    # for det in detectors:
    #     for cp_method in cp_methods:
    #         print(f"eval cp error ----> src:{split} det:{det} cp:{cp_method}")
    #         os.system(f"python3 main.py --MultiSeg  --Dataset={split}  --Detector={det} --DetectorVersion={det_v}\
    #                     --EvalContactPoitnSelection\
    #                     --EvalContactPoitnSelectionMC\
    #                     --GTDetector={GT_det} --GTDetector3D={GT_det_3D}\
    #                     --BackprojectSource=detections --TopView={tp_view} --BackprojectionMethod={bp_method}\
    #                     --ContactPoint={cp_method} --GTContactPoint={gt_cp_method} --GT3DContactPoint={gt3D_cp_method}")

    ########################################################
    # 2. perform single camera tracking evaluation on all the sources under ms folder(split)
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --MultiSeg  --TrackEval --GTDetector={GT_det} --GTTracker={GT_tra}\
    #  --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"evaluate tracking ---> src:{split} det:{det} tra:{tra} gt_det:{GT_det} gt_tra:{GT_tra}")
    #         os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --TrackEval\
    #                     --GTDetector={GT_det} --GTTracker={GT_tra}\
    #                     --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 3. Run automated track extraction and labelling
    # python3 main.py --Dataset={split}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --MultiSeg\
    #                 --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH={resamp_th}\
    #                 --BackprojectSource=tracks --TopView={tp_view}\
    #                 --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                 --GP --ROIFromTop
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={split}  --Detector={det} --DetectorVersion={det_v} --Tracker={tra} --MultiSeg\
    #                    --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH={resamp_th}\
    #                    --BackprojectSource=tracks --TopView={tp_view}\
    #                    --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                    --GP --ROIFromTop")
            
    ########################################################
    # 4. Run the counting Multi Segment
    # os.system(f"python3 main.py --Dataset={split} --Detector={det} --DetectorVersion={det_v}
    #  --Tracker={tra} --Count --EvalCount --CountMetric={metric} --ResampleTH={resamp_th}\
    #  --CacheCounter --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #  --CountVisDensity --CountVisPrompt\
    #  --BackprojectSource=tracks --TopView={tp_view}\
    #  --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #  --KDEBW=3.5/10 --OSR=10 --ROIFromTop --GP")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             for cp_method in cp_methods:
    #                 print(f"counting split:{split} metric:{metric} det:{det} tra:{tra}")
    #                 os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det}\
    #                             --DetectorVersion={det_v} --Tracker={tra}\
    #                             --Count --EvalCount --CountMetric={metric} --ResampleTH={resamp_th}\
    #                             --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #                             \
    #                             --BackprojectSource=tracks --TopView={tp_view}\
    #                             --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #                             --KDEBW=3.5 --OSR=10 --GP --ROIFromTop")

    ########################################################
    # 5. Run the eval counting Multi Segment from Multi Cam
    # os.system(f"python3 main.py --Dataset={split} --Detector={det} --DetectorVersion={det_v}
    #  --Tracker={tra} --Count --EvalCount --CountMetric={metric} --ResampleTH={resamp_th}\
    #  --CacheCounter --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #  --CountVisDensity --CountVisPrompt\
    #  --BackprojectSource=tracks --TopView={tp_view}\
    #  --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #  --KDEBW=3.5/10 --OSR=10 --ROIFromTop --GP")
    #######################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             for cp_method in cp_methods:
    #                 print(f"counting split:{split} metric:{metric} det:{det} tra:{tra}")
    #                 os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det}\
    #                             --DetectorVersion={det_v} --Tracker={tra}\
    #                             --EvalCountMSfromMC --CountMetric={metric} --ResampleTH={resamp_th}\
    #                             --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

    ########################################################
    # 6. Run the eval counting Multi Segment based on multi camera
    # os.system(f"python3 main.py --Dataset={split} --Detector={det} --DetectorVersion={det_v}
    #  --Tracker={tra} --Count --EvalCount --CountMetric={metric} --ResampleTH={resamp_th}\
    #  --CacheCounter --UseCachedCounter --CachedCounterPth={cached_cnt_pth}\
    #  --CountVisDensity --CountVisPrompt\
    #  --BackprojectSource=tracks --TopView={tp_view}\
    #  --BackprojectionMethod={bp_method} --ContactPoint={cp_method}\
    #  --KDEBW=3.5/10 --OSR=10 --ROIFromTop --GP")
    #######################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             for cp_method in cp_methods:
    #                 print(f"counting split:{split} metric:{metric} det:{det} tra:{tra}")
    #                 os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det}\
    #                             --DetectorVersion={det_v} --Tracker={tra}\
    #                             --EvalCountCamera --CountMetric={metric} --ResampleTH={resamp_th}\
    #                             --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={cp_method}")

#_______________________MULTIPART___________________________#
for ds in datasets:
    print(f"running on dataset:{ds}")
    # ########################################################
    # # 1. fine tune detector
    # # ########################################################
    # for det in detectors:
    #     os.system(f"python3 main.py --MultiPart --Dataset={ds} --Detector={det} --GTDetector={GT_det}\
    #                --FineTune --TrainPart={train_sp} --ValidPart={valid_sp} --BatchSize={batch_size} \
    #                --NumWorkers={num_workers} --Epochs={epochs} --ValInterval={val_interval}\
    #                --TopView={tp_view} --BackprojectionMethod={bp_method} --ContactPoint={gt_cp_method}")

    ########################################################
    # 2. perform single camera tracking evaluation on all the sources under mp folder(dataset)
    # os.system(f"python3 main.py --MultiPart --Dataset={src}  --Detector={det} --Tracker={tra} --TrackEval --GTDetector={GT_det} --GTTracker={GT_tra}")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"evaluate tracking ---> src:{ds} det:{det} tra:{tra} gt_det:{GT_det} gt_tra:{GT_tra}")
    #         os.system(f"python3 main.py --MultiPart --Dataset={ds}  --Detector={det} --Tracker={tra} --TrackEval\
    #                    --GTDetector={GT_det} --GTTracker={GT_tra}")