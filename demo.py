import os

# function to get subdirs of roots and ignoreing Results folder
def get_sub_dirs(roots, subs_to_include = None, subs_to_exclude = ["Results"]):
    subs_path = []
    for root in roots:
        subs = os.listdir(root)
        for sub in subs:
            sub_path = os.path.join(root, sub)
            if os.path.isdir(sub_path) and\
                (subs_to_exclude is None or sub not in subs_to_exclude) and\
                (subs_to_include is None or sub in subs_to_include):
                subs_path.append(sub_path)
    return subs_path

# choose datasets/splits/segments/sources
# Set to None if want to include all
# Dataset structure should be similar to below
# Data---|
#        |--Split1---|
#        |           |--Segment1---|
#        |           |             |--Source1--|
#        |           |             |           |--VideoName.ext
#________|___________|_____________|___________|--VideoName.metadata.json

datasets     = ["./DemoDataSet"]
split_part   = ["Split1"]
segment_part = ["Segment1"] 
source_part  = ["Source1"]  
splits       = get_sub_dirs(datasets, split_part)
segments     = get_sub_dirs(splits, segment_part)
sources      = get_sub_dirs(segments, source_part)

# choose the segmenter
# options: ["InternImage"]
segmenters = ["InternImage"]

# choose the detectors
# options: ["InternImage", "RTMDet", "YoloX", "DeformableDETR", "CenterNet", "CascadeRCNN", "YOLOv8"]
detectors = ["InternImage"]

# choose the tracker
# options: [sort", "CenterTrack", "DeepSort", "ByteTrack", "gsort", "OCSort", "GByteTrack", "GDeepSort", "BOTSort", "StrongSort"]
trackers = ["ByteTrack"] 

# choose the clustering algorithm
# options: ["SpectralFull", "DBSCAN", "SpectralKNN"]
clusters = ["SpectralFull"]

# choose the metric for clustering and classification pqrt
# options: ["groi", "roi", "knn", "cos", "tcos", "cmm", "hausdorff", "kde",,"ccmm", "tccmm", "ptcos", "loskde", "hmmg"]
clt_metrics = ["tcos"]
cnt_metrics = ["kde"]


for src, cached_cnt_pth in zip(sources, sources):
    print(f"running on source:{src}")
    ########################################################
    # 1. extract images from video
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages)
    ########################################################
    # print(f"extracting images from : {src}")
    # os.system(f"python3 main.py --Dataset={src} --ExtractImages")

    ########################################################
    # 2. estimate the Homography Metrix using Homography GUI 
    # os.system(f"python3 main.py --Dataset={src} --HomographyGUI --VisHomographyGUI --Frame=1 --TopView=[GoogleMap/OrthoPhoto]")
    ########################################################
    # print(f"Homography GUI ----> src:{src}")
    # os.system(f"python3 main.py --Dataset={src}  --HomographyGUI --VisHomographyGUI --TopView=GoogleMap")

    ########################################################
    # 3. visualizing the region of interest 
    # ROI is provided to pipeline via <Videoname>.metadata.json file
    # ROI is a closed polygon (for intersections that is achieved by choosing 4 points)
    # ROI coordiantes are on the video frame
    # See Docs/ROI.md for more info
    # os.system(f"python3 main.py --Dataset={src}  --VisROI --TopView=GoogleMap")
    #######################################################
    # print(f"Vis ROI ----> src:{src}")
    # os.system(f"python3 main.py --Dataset={src} --VisROI --TopView=GoogleMap")

    #######################################################
    # 4. Segment Video Frames  and visualize 
    #  os.system(f"python3 main.py --Dataset={src} --Segment --Segmenter={seg} --VisSegment --ForNFrames=2000")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src} --Segmenter={seg} --Segment --VisSegment")

    #######################################################
    # 4.5 Segment Post Processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segment Post Proc ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src} --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 5. run the detection
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Detect --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Detect --VisDetect")

    ########################################################
    # 5.1 run the detection post processing
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --DetPostProc --DetMask --DetTh=0.50 --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detect post processing ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --VisDetect --DetPostProc --DetTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 5.2 convert detections to coco format
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --ConvertDetsToCOCO")
    ########################################################
    # for det in detectors:
    #     print(f"converting to COCO ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --ConvertDetsToCOCO")

    ########################################################
    # 5.3 back project detections
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Homography --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     for seg in segmenters:
    #         print(f"back project dets ----> src:{src} det:{det}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Segmenter={seg}  --Homography --BackprojectSource=detections --TopView=GoogleMap --BackprojectionMethod=Homography --ContactPoint=BottomPoint")

    ########################################################
    # 5.4 Vis Contact Points and BP Points
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --VisContactPoint --VisCPTop --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     print(f"vis back proj dets ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --VisContactPoint --BackprojectSource=detections --TopView=GoogleMap --BackprojectionMethod=Homography --ContactPoint=BottomPoint")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --VisCPTop        --BackprojectSource=detections --TopView=GoogleMap --BackprojectionMethod=Homography --ContactPoint=BottomPoint")

    ########################################################
    # 6. run tracking
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack")

    ########################################################
    # 6.2 back project tracks and convert to meter
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks  --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg] --BackprojectionMethod=[Homography/DSM] --TopView=[GoogleMap/OrthoPhoto] --ForNFrames=1800")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking backprojection ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=GoogleMap --BackprojectionMethod=Homography --ContactPoint=BottomPoint")

    ########################################################
    # 6.5. run the track post processing
    # SHOULD HAVE THE SAME FLAGS AS BACK PROJECTION
    # --BackprojectSource=tracks  --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg] --BackprojectionMethod=[Homography/DSM] --TopView=[GoogleMap/OrthoPhoto]
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI  --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking POSTPROC ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc\
    #                 --VisTrack --VisTrajectories --VisTrackTop\
    #                 --BackprojectSource=tracks --TopView=GoogleMap --BackprojectionMethod=Homography --ContactPoint=BottomPoint\
    #                 --MaskGPFrame --HasPointsInROI")
    
    ########################################################
    # 7.1 Run the track labelling GUI or go to 7.2.
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    ########################################################
    # 7.2 Run automated track extraction and labelling
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH=2.0 --TopView=GoogleMap")

    ########################################################
    # 8.0 find optimum BW for kde fiting
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"finding optimal bw for kde ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")
    
    ########################################################
    # 8.1 Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CountVisPrompt --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --CacheCounter --CountVisDensity")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"counting metric:{metric} det:{det} tra:{tra}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --CacheCounter --CountVisDensity --ResampleTH=2.0 --TopView=GoogleMap")
    #             # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --TopView=GoogleMap")

    ########################################################
    # 9. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in cnt_metrics:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")


#_______________________MULTICAMERA_______________________#
for src in segments:
    print(f"running on seg:{src}")
    # ########################################################
    # # 1. Average Counts MC
    # # os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} -- --VisTrackTopMC")
    # ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"average MC counts ---> src:{src} det:{det} tra:{tra} cnt:{metric}")
    #             os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={metric} --AverageCountsMC --EvalCountMC")



#_______________________MULTISOURCE_______________________#
for split in splits:
    print(f"running on split:{split}")
    # ########################################################
    # # 1. convert detections of all the data under split into COCO format
    # ########################################################
    # for det in detectors:
    #     os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det} --Tracker=NULL --ConvertDetsToCOCO")