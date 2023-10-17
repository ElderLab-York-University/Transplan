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

datasets     = ["DemoDataSet"]
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
    ########################################################
    # 1. estimate the Homography Metrix using Homography GUI 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=detectron2 --Tracker=sort --HomographyGUI --VisHomographyGUI --Frame=1 --TopView=[GoogleMap/OrthoPhoto]")
    ########################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src}  --VisHomographyGUI --TopView=GoogleMap")

    ########################################################
    # 2. visualizing the region of interest 
    # ROI is provided to pipeline via <Videoname>.metadata.json file
    # ROI is a closed polygon (for intersections that is achieved by choosing 4 points)
    # ROI coordiantes are on the video frame
    # See Docs/ROI.md for more info
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI --TopView=GoogleMap")
    #######################################################
    # print(f"src:{src}")
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI --TopView=GoogleMap")

    #######################################################
    # 2.5 Segment Video Frames  and visualize
    # Segmentation 
    #  os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segment --Segmenter={seg} --VisSegment --ForNFrames=2000")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src} --Segmenter={seg} --Segment --VisSegment")

    #######################################################
    # 2.6 Segment Post Processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")
    #######################################################
    # for seg in segmenters:
    #     print(f" Segmenting ----> src:{src} seg:{seg}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --Segmenter={seg} --SegPostProc --VisSegment --SegTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 3. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --Detect --VisDetect")

    ########################################################
    # 3.5 run the detection post processing
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --DetPostProc --DetMask --DetTh=0.50 --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --DetPostProc --DetTh=0.5 --VisDetect")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisDetect --DetPostProc --DetTh=0.5 --classes_to_keep 2 5 7")

    ########################################################
    # 3.5.5 convert detections to coco format
    # the full commonad looks like : os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL")
    ########################################################
    # for det in detectors:
    #     print(f"converting to COCO ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --ConvertDetsToCOCO")

    ########################################################
    # 3.6 back project detections
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --Homography --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     for seg in segmenters:
    #         print(f"detecting ----> src:{src} det:{det}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Segmenter={seg} --Tracker=NULL --Homography --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")

    ########################################################
    # 3.7 Vis Contact Points and BP Points
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisContactPoint --BackprojectSource=detections --TopView=[GoogleMap/OrthoPhoto] --BackprojectionMethod=[Homography/DSM] --ContactPoint=[BottomPoint/Center/BottomSeg/LineSeg]")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisContactPoint --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisCPTop        --BackprojectSource=detections --TopView=OrthoPhoto --BackprojectionMethod=Homography --ContactPoint=LineSeg")

    ########################################################
    # 4. run the tracking and backproject and convert to meter
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=GoogleMap")

    ########################################################
    # 5. run the track post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI  --BackprojectSource=tracks --TopView=[GoogleMap/OrthoPhoto]")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking POSTPROC ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc  --MaskGPFrame --HasPointsInROI --BackprojectSource=tracks --TopView=GoogleMap")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop --BackprojectSource=tracks --TopView=GoogleMap")

    ########################################################
    # 6. find optimum BW for kde fiting
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"finding optimal bw for kde ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --FindOptimalKDEBW --ResampleTH=2.0 --TopView=GoogleMap")

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
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH=2.0 --TopView=GoogleMap")

    ########################################################
    # 10. Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CountVisPrompt --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --CacheCounter --CountVisDensity")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"counting metric:{metric} det:{det} tra:{tra}")
    #             # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --CacheCounter --CountVisDensity --ResampleTH=2.0 --TopView=GoogleMap")
    #             # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --TopView=GoogleMap")

    ########################################################
    # 11. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in cnt_metrics:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")


#_______________________MULTICAMERA_______________________#

# for src in segments:
#     print(f"running on seg:{src}")
    # ########################################################
    # # 0. Average Counts MC
    # # os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} -- --VisTrackTopMC")
    # ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for metric in cnt_metrics:
    #             print(f"average MC counts ---> src:{src} det:{det} tra:{tra} cnt:{metric}")
    #             os.system(f"python3 main.py --MultiCam --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={metric} --AverageCountsMC --EvalCountMC")



#_______________________MULTISOURCE_______________________#

# for split in splits:
#     # to run on split level add --MultiSeg
#     print(f"running on split:{split}")
#     ########################################################
#     # 0. convert detections of all the data under split into COCO format
#     ########################################################
#     for det in detectors:
#         os.system(f"python3 main.py --MultiSeg --Dataset={split} --Detector={det} --Tracker=NULL --ConvertDetsToCOCO")