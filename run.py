import os
# choose the dataset/video
# options : ['./../Dataset/DandasStAtNinthLineFull', './../Dataset/DandasStAtNinthLine', "./../Dataset/SOW_src1", "./../Dataset/SOW_src2", "./../Dataset/SOW_src3", "./../Dataset/SOW_src4"]
sources = [
# TransPlan Dataset
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video2",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video2",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video2",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video2",
# HW7 & Leslie  DATASET
    # "/mnt/data/HW7Leslie/Seg17sc1",
    # "/mnt/data/HW7Leslie/Seg17sc2",
    # "/mnt/data/HW7Leslie/Seg17sc3",
    # "/mnt/data/HW7Leslie/Seg17sc4"

]

cached_cnt_sources = [
# TransPlan Dataset
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/D9L_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DBR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/DWP_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/mnt/data/TransPlanData/Dataset/PreProcessedMain/ECR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
# HW7 & Leslie  DATASET
    # "/mnt/data/HW7Leslie/Seg17sc1",
    # "/mnt/data/HW7Leslie/Seg17sc2",
    # "/mnt/data/HW7Leslie/Seg17sc3",
    # "/mnt/data/HW7Leslie/Seg17sc4"
]

# choose the detectors
# options: ["detectron2", "OpenMM", "YOLOv5", "YOLOv8", "InternImage"]
detectors = ["InternImage"]

# choose the tracker
# options: ["sort", "CenterTrack", "DeepSort", "ByteTrack", "gsort", "OCSort", "GByteTrack", "GDeepSort", "BOTSort", "StrongSort"]
trackers = ["ByteTrack"] 

# choose the clustering algorithm
# options: ["SpectralFull", "DBSCAN", "SpectralKNN"]
clusters = ["SpectralFull"]

# choose the metric for clustering and classification pqrt
# options: ["cos", "tcos", "cmm", "ccmm", "tccmm", "hausdorff", "ptcos", "loskde", "kde", "hmmg", "roi"]
clt_metrics = ["tcos", "cmm"]
cnt_metrics = ["kde"]

for src, cached_cnt_pth in zip(sources, cached_cnt_sources):
    ########################################################
    # 1. estimate the Homography Metrix using Homography GUI 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=detectron2 --Tracker=sort --HomographyGUI --VisHomographyGUI")
    ########################################################
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --HomographyGUI --VisHomographyGUI")

    ########################################################
    # 1.5 visualizing the region of interest 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI")
    #######################################################
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI")

    ########################################################
    # 2. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --DetPostProc --DetMask --DetTh=0.50 --VisDetect")
    ########################################################
    # for det in detectors:
    #     print(f"detecting ----> src:{src} det:{det}")
    #     os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL  --VisDetect --DetPostProc --DetTh=0.5")

    ########################################################
    # 3. run the tracking 
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrack --Homography --Meter --VisTrajectories --VisTrackTop")

    ########################################################
    # 3.1 run the track post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI")
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"tracking POSTPROC ---> src:{src} det:{det} tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc  --MaskGPFrame --HasPointsInROI --ExitOrCrossROI")

    ########################################################
    # 4. run clustering algorithm
    # apperently the clustering visulaizaiton is harcodded at the moment
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in clt_metrics:
    #             for clt in clusters:
    #                 print(f"clustering ----> det:{det} tra:{tra} met:{met} clt:{clt}")
    #                 os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ClusteringAlgo={clt} --ClusterMetric={met} --Cluster")
    
    ########################################################
    # 5. Run the track labelling GUI / go to 6.
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackLabelingGUI --VisLabelledTrajectories")

    ########################################################
    # 6. Run automated track extraction and labelling
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         print(f"extract common tracks ----> det:{det}, tra:{tra}")
    #         os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --ExtractCommonTracks --VisLabelledTrajectories --ResampleTH=2.0")

    ########################################################
    # 7. Run the classification(counting) part
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --CountVisPrompt --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth} --CacheCounter --CountVisDensity")
    ########################################################
    for det in detectors:
        for tra in trackers:
            for metric in cnt_metrics:
                print(f"counting metric:{metric}")
                os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --CountVisDensity --CacheCounter")

    ########################################################
    # 8. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in cnt_metrics:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")