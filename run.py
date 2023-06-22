import os
# choose the dataset/video
# options : ['./../Dataset/DandasStAtNinthLineFull', './../Dataset/DandasStAtNinthLine', "./../Dataset/SOW_src1", "./../Dataset/SOW_src2", "./../Dataset/SOW_src3", "./../Dataset/SOW_src4"]
sources = [
# "/home/sajjad/Dataset/PreProcessedMain/D9L_Video1",
"/home/sajjad/Dataset/PreProcessedMain/D9L_Video2",
# "/home/sajjad/Dataset/PreProcessedMain/DBR_Video1",
"/home/sajjad/Dataset/PreProcessedMain/DBR_Video2",
# "/home/sajjad/Dataset/PreProcessedMain/DWP_Video1",
"/home/sajjad/Dataset/PreProcessedMain/DWP_Video2",
# "/home/sajjad/Dataset/PreProcessedMain/ECR_Video1",
"/home/sajjad/Dataset/PreProcessedMain/ECR_Video2",

]

cached_cnt_sources = [
    # "/home/sajjad/Dataset/PreProcessedMain/D9L_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/home/sajjad/Dataset/PreProcessedMain/D9L_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/home/sajjad/Dataset/PreProcessedMain/DBR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/home/sajjad/Dataset/PreProcessedMain/DBR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/home/sajjad/Dataset/PreProcessedMain/DWP_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/home/sajjad/Dataset/PreProcessedMain/DWP_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    # "/home/sajjad/Dataset/PreProcessedMain/ECR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
    "/home/sajjad/Dataset/PreProcessedMain/ECR_Video1/Results/Counting/video.counting.InternImage.ByteTrack.kde.cached.pkl",
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
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisHomographyGUI --HomographyGUI")

    ########################################################
    # 1.5 visualizing the region of interest 
    # os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI")
    #######################################################
    os.system(f"python3 main.py --Dataset={src}  --Detector=Null --Tracker=Null --VisROI")

    ########################################################
    # 2. run the detection
    # the full commonad looks like : os.system(f"python3 main.py --Datas`et={src}  --Detector={det} --Tracker=NULL --Detect --DetPostProc --DetMask --DetTh=0.50 --VisDetect")
    ########################################################
    for det in detectors:
        print(f"detecting ----> src:{src} det:{det}")
        os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker=NULL --VisDetect --ForNFrames=1800 --DetPostProc --DetTh=0.50")

    ########################################################
    # 3. run the tracking 
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")
    ########################################################
    for det in detectors:
        for tra in trackers:
            print(f"tracking ---> src:{src} det:{det} tra:{tra}")
            # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")
            os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Track --VisTrack --ForNFrames=1800 --Homography --Meter --VisTrajectories --VisTrackTop")

    ########################################################
    # 3.1 run the track post processing
    # os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc --TrackTh=8 --RemoveInvalidTracks --SelectDifEdgeInROI --SelectEndingInROI --SelectBeginInROI --MaskGPFrame --HasPointsInROI --MaskROI  --CrossROI --CrossROIMulti --JustEnterROI --JustExitROI --WithinROI --Interpolate --ExitOrCrossROI")
    ########################################################
    for det in detectors:
        for tra in trackers:
            print(f"tracking ---> src:{src} det:{det} tra:{tra}")
            os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --TrackPostProc  --MaskGPFrame --HasPointsInROI --ExitOrCrossROI")
            #to visualize results
            os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --VisTrack --ForNFrames=1800 --VisTrajectories --VisTrackTop")

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
                os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --Count --CountMetric={metric} --EvalCount --UseCachedCounter --CachedCounterPth={cached_cnt_pth}")

    ########################################################
    # 8. Visualizing the results on a video including track label and track id
    # can be used to monitor the pipeline in detail
    ########################################################
    # for det in detectors:
    #     for tra in trackers:
    #         for met in metrics[:1]:
    #             print(f"visualizing MOI -----> det:{det} tra:{tra} met:{met}")
    #             os.system(f"python3 main.py --Dataset={src}  --Detector={det} --Tracker={tra} --CountMetric={met} --VisTrackMoI")